#include "convert/HdrGamut.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "convert/HdrTransfer.h"
#include "util/Error.h"

// The builder follows Output/hdr-design/tc-checks/ictcp_ref.py (section 3) step by step, in its order of operations; U27 holds it to an
// independent builder in tests/ and U5 to the anchors of ictcp_ref.py and ictcp_dense.py.
namespace dlssvid::hdr {

namespace {

constexpr int kRows = GamutBounds::kRows, kCusp = GamutBounds::kCuspRow;
constexpr int kTargetColumns = 512;      // columns of the cusp axis over the whole pa range [0, 4)
constexpr int kRingN = 2048;             // ring samples per edge of the ring
constexpr int kRingSamples = 6 * kRingN;
constexpr int kMarch = 75;               // the march to the first exit: steps of 0.01 up to 0.75
constexpr double kStep = 0.01, kMarchMax = 0.75;
constexpr int kBisect = 26;
constexpr double kEpsIn = 1e-9, kEpsLms = 1e-12;
constexpr double kPi = 3.141592653589793, k2Pi = 2.0 * kPi;
constexpr double kUc = static_cast<double>(kCusp) / (kRows - 1);
constexpr double kRing[6][3] = {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}, {1, 0, 1}};  // R Y G C B M
constexpr char kRingNames[6] = {'R', 'Y', 'G', 'C', 'B', 'M'};

using M3 = std::array<double, 9>;

M3 Inv3(const M3& m) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return M3{(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
              (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
              (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
}

M3 Mul3(const M3& a, const M3& b) {
    M3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    return r;
}

void Apply3(const M3& m, const double x[3], double y[3]) {
    const double x0 = x[0], x1 = x[1], x2 = x[2];
    for (int i = 0; i < 3; ++i) y[i] = m[3 * i] * x0 + m[3 * i + 1] * x1 + m[3 * i + 2] * x2;
}

// Python's float %: the result has the divisor's sign.
double PyMod(double x, double m) {
    double r = std::fmod(x, m);
    if (r != 0.0) {
        if ((m < 0.0) != (r < 0.0)) r += m;
    } else {
        r = std::copysign(0.0, m);
    }
    return r;
}

// (x + pi) % 2 pi - pi: [-pi, pi). The hues here keep x + pi in [-2 pi, 2 pi], where one add or subtract is fmod's own result.
double WrapAngle(double x) {
    const double y = x + kPi;
    if (y >= -k2Pi && y <= k2Pi) return (y < 0.0 ? y + k2Pi : y >= k2Pi ? y - k2Pi : y + 0.0) - kPi;
    return PyMod(y, k2Pi) - kPi;
}

double AngleOfPa(double pa) {
    const double q = std::floor(pa), r = pa - q;
    return q * (kPi / 2.0) + std::atan2(r, 1.0 - r);
}

double Scale() { return kWhiteNits / 10000.0; }  // relative light (1 = 203 nits) -> normalised (1 = 10000 nits)
double E0() {
    static const double e = PqEncode(0.0);
    return e;
}
double Iw() {
    static const double e = PqEncode(Scale());
    return e;
}

double WarpU(double I, double ic) {
    return I <= ic ? (I - E0()) / (ic - E0()) * kUc : kUc + (I - ic) / (Iw() - ic) * (1.0 - kUc);
}
double WarpI(double u, double ic) {
    return u <= kUc ? E0() + (ic - E0()) * u / kUc : ic + (Iw() - ic) * (u - kUc) / (1.0 - kUc);
}

// Ic(q) of a cube's cusp axis: linear between the axis columns.
double IcOfQ(const GamutBounds& b, double q) {
    q = PyMod(q, 4.0);
    int s = 0;
    for (int i = 0; i < 6; ++i)
        if (b.vertex[i] <= q) s = i;
    const double k1 = s < 5 ? b.vertex[s + 1] : 4.0;
    const double fp = b.axisStart[s] + (q - b.vertex[s]) * b.axisCols[s] / (k1 - b.vertex[s]);
    const double j0f = std::floor(fp), t = fp - j0f;
    const int j0 = static_cast<int>(static_cast<long long>(j0f) % b.axisN);
    const int j1 = (j0 + 1) % b.axisN;
    return b.cuspI[j0] + (b.cuspI[j1] - b.cuspI[j0]) * t;
}

// The PQ EOTF on [0, Iw] by 4096 cubic Hermite segments with the exact slope (relative LMS out): the builder's 5.7 M ray points.
class PqDecodeTable {
public:
    PqDecodeTable() {
        h_ = Iw() / kSegments;
        for (int i = 0; i <= kSegments; ++i) {
            const double e = i * h_;
            v_[i] = PqDecode(e) / Scale();
            s_[i] = PqDecodeSlope(e) / Scale() * h_;
        }
    }
    double operator()(double e) const {
        const double x = e / h_;
        const int i = std::clamp(static_cast<int>(x), 0, kSegments - 1);
        const double t = x - i, t2 = t * t, t3 = t2 * t;
        return v_[i] * (2.0 * t3 - 3.0 * t2 + 1.0) + s_[i] * (t3 - 2.0 * t2 + t) + v_[i + 1] * (-2.0 * t3 + 3.0 * t2) + s_[i + 1] * (t3 - t2);
    }

private:
    static constexpr int kSegments = 4096;
    double h_ = 0;
    std::array<double, kSegments + 1> v_{}, s_{};
};

struct Cube {
    M3 A;       // relative LMS -> the cube's RGB
    M3 to2020;  // the cube's RGB -> linear BT.2020
};

void RingPointAt(const Cube& cube, double t, double p[3]) {
    const double ft = std::floor(t);
    const int sg = static_cast<int>(ft) % 6, sg1 = (sg + 1) % 6;
    const double rr = t - ft;
    double q[3];
    for (int k = 0; k < 3; ++k) q[k] = kRing[sg][k] * (1.0 - rr) + kRing[sg1][k] * rr;
    Apply3(cube.to2020, q, p);
}

// The ring's samples and, per hue bucket, the segments (sample i to i + 1) whose hue arc can hold that hue: RingPoint tests the crossing
// rule only on them - the same rule on a superset of its crossings, so the same choice as testing every segment.
struct Ring {
    static constexpr int kBuckets = 4096;
    std::vector<double> h, C;  // hue and chroma of the 6 x kRingN samples
    std::vector<std::vector<int>> bucket;

    static int BucketOf(double a) {  // any angle -> [0, kBuckets)
        const double t = (a + kPi) / k2Pi;
        const long long b = static_cast<long long>(std::floor((t - std::floor(t)) * kBuckets));
        return static_cast<int>(std::clamp<long long>(b, 0, kBuckets - 1));
    }
    void Index() {
        bucket.assign(kBuckets, {});
        constexpr double kMargin = 1e-9;
        for (int i = 0; i < kRingSamples; ++i) {
            const double a0 = h[static_cast<size_t>(i)], step = WrapAngle(h[static_cast<size_t>(i + 1 < kRingSamples ? i + 1 : 0)] - a0);
            if (std::abs(step) > 0.1) {  // never on these rings; then every bucket
                for (auto& b : bucket) b.push_back(i);
                continue;
            }
            const double lo = std::min(a0, a0 + step) - kMargin, hi = std::max(a0, a0 + step) + kMargin;
            const long long b0 = static_cast<long long>(std::floor((lo + kPi) / k2Pi * kBuckets));
            const long long b1 = static_cast<long long>(std::floor((hi + kPi) / k2Pi * kBuckets));
            for (long long b = b0; b <= b1; ++b) bucket[static_cast<size_t>(((b % kBuckets) + kBuckets) % kBuckets)].push_back(i);
        }
    }
};

// The ring point (max = 1, min = 0) of the cube with ICtCp hue theta: the largest-chroma crossing of the sampling, then 60 bisections.
void RingPoint(const Cube& cube, const Ring& ring, double theta, double& I, double& C) {
    int best = -1;
    double bestC = 0.0;
    for (const int i : ring.bucket[static_cast<size_t>(Ring::BucketOf(theta))]) {  // in index order
        const int i1 = i + 1 < kRingSamples ? i + 1 : 0;
        const double d = WrapAngle(ring.h[static_cast<size_t>(i)] - theta), d1 = WrapAngle(ring.h[static_cast<size_t>(i1)] - theta);
        if (d <= 0.0 && d1 > 0.0 && d1 - d < kPi) {
            const double c = ring.C[static_cast<size_t>(i)] + ring.C[static_cast<size_t>(i1)];
            if (best < 0 || c > bestC) best = i, bestC = c;
        }
    }
    if (best < 0) Throw("hdr::BuildSdr1GamutTables: no ring point with the hue " + std::to_string(theta));
    double lo = static_cast<double>(best) / kRingN, hi = lo + 1.0 / kRingN;
    double p[3], itp[3];
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        RingPointAt(cube, mid, p);
        Ictcp(p, itp);
        if (WrapAngle(std::atan2(itp[2], itp[1]) - theta) <= 0.0) lo = mid;
        else hi = mid;
    }
    RingPointAt(cube, 0.5 * (lo + hi), p);
    Ictcp(p, itp);
    I = itp[0];
    C = std::hypot(itp[1], itp[2]);
}

bool Inside(const Cube& cube, const PqDecodeTable& dec, double I, double c, const double a[3]) {
    double lms[3];
    for (int k = 0; k < 3; ++k) {
        const double e = I + c * a[k];
        if (!(e >= E0() - kEpsLms && e <= Iw() + kEpsLms)) return false;
        lms[k] = dec(e);
    }
    double rgb[3];
    Apply3(cube.A, lms, rgb);
    for (int k = 0; k < 3; ++k)
        if (!(rgb[k] >= -kEpsIn && rgb[k] <= 1.0 + kEpsIn)) return false;
    return true;
}

// The first exit of the constant-(I, hue) ray: march 0.01 up to 0.75, then 26 bisections.
double FirstExit(const Cube& cube, const PqDecodeTable& dec, double I, const double a[3]) {
    double out = kMarchMax;
    for (int s = 1; s <= kMarch; ++s) {
        if (!Inside(cube, dec, I, s * kStep, a)) {
            out = s * kStep;
            break;
        }
    }
    double hi = out, lo = out < kMarchMax ? out - kStep : kMarchMax;
    for (int i = 0; i < kBisect; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (Inside(cube, dec, I, mid, a)) lo = mid;
        else hi = mid;
    }
    return lo;
}

void RayDir(double theta, double a[3]) {
    const M3& inv = IctcpMatrix().ictcpInv;
    const double c = std::cos(theta), s = std::sin(theta);
    for (int k = 0; k < 3; ++k) a[k] = c * inv[3 * k + 1] + s * inv[3 * k + 2];
}

// Runs fn(i) for i in [0, n) on `threads` threads: blocks of 256 dealt round robin. Every item is independent, so the result does not
// depend on the count; the first exception is rethrown after the join.
template <typename Fn>
void ParallelFor(size_t n, int threads, const Fn& fn) {
    constexpr size_t kBlock = 256;
    const size_t blocks = (n + kBlock - 1) / kBlock;
    const int t = static_cast<int>(std::clamp<size_t>(static_cast<size_t>(std::max(threads, 1)), 1, std::max<size_t>(blocks, 1)));
    std::exception_ptr error;
    std::mutex errorMutex;
    const auto work = [&](int id) {
        try {
            for (size_t b = static_cast<size_t>(id); b < blocks; b += static_cast<size_t>(t))
                for (size_t i = b * kBlock, e = std::min(n, (b + 1) * kBlock); i < e; ++i) fn(i);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(errorMutex);
            if (!error) error = std::current_exception();
        }
    };
    if (t == 1) {
        work(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(t - 1));
        for (int i = 1; i < t; ++i) pool.emplace_back(work, i);
        work(0);
        for (auto& th : pool) th.join();
    }
    if (error) std::rethrow_exception(error);
}

// One cube in the making: the vertices in pa order, the ring samples and, for BT.709 with edge knots, the edge points of every row.
struct CubeWork {
    Cube cube;
    GamutBounds* b = nullptr;
    bool edges = false;
    std::array<std::array<double, 3>, 6> V{};  // vertices in linear BT.2020, sorted by pa
    std::array<char, 6> names{};
    Ring ring;
    std::vector<double> E, Ce;                  // [kRows][6]: relative pa and chroma of each vertex's edge knot
    std::vector<std::pair<char, int>> knotDef;  // ('e' | 'v', vertex)
    std::vector<double> qn;                     // [kRows][columns]: relative pa of the nodes
};

void SetUpAxis(CubeWork& w) {
    GamutBounds& b = *w.b;
    double pv[6];
    std::array<std::array<double, 3>, 6> V{};
    for (int m = 0; m < 6; ++m) {
        Apply3(w.cube.to2020, kRing[m], V[m].data());
        double itp[3];
        Ictcp(V[m].data(), itp);
        pv[m] = PseudoHue(itp[1], itp[2]);
    }
    std::array<int, 6> order{0, 1, 2, 3, 4, 5};
    std::sort(order.begin(), order.end(), [&](int a, int c) { return pv[a] < pv[c]; });
    b.pa0 = pv[order[0]];
    for (int m = 0; m < 6; ++m) {
        w.V[m] = V[order[m]];
        w.names[m] = kRingNames[order[m]];
        b.vertex[m] = PyMod(pv[order[m]] - b.pa0, 4.0);
    }
    int start = 0;
    for (int s = 0; s < 6; ++s) {
        const double width = (s < 5 ? b.vertex[s + 1] : 4.0) - b.vertex[s];
        b.axisCols[s] = std::max(2, static_cast<int>(std::nearbyint(kTargetColumns * width / 4.0)));
        b.axisStart[s] = start;
        start += b.axisCols[s];
    }
    b.axisN = start;
    b.cuspI.assign(static_cast<size_t>(b.axisN), 0.0);
    b.cuspC.assign(static_cast<size_t>(b.axisN), 0.0);
}

double AxisPa(const GamutBounds& b, int col) {  // relative pa of an axis column
    int s = 5;
    while (s > 0 && col < b.axisStart[s]) --s;
    const double width = (s < 5 ? b.vertex[s + 1] : 4.0) - b.vertex[s];
    return b.vertex[s] + width * (col - b.axisStart[s]) / b.axisCols[s];
}

// The point of the edge black -> vertex m (or white -> vertex m) whose u is `us`: relative pa (unwrapped to the vertex) and chroma.
void EdgePoint(const CubeWork& w, int m, bool black, double us, double& q, double& c) {
    const GamutBounds& b = *w.b;
    const double* V = w.V[m].data();
    double W[3];
    const double ones[3] = {1.0, 1.0, 1.0};
    Apply3(w.cube.to2020, ones, W);
    const auto at = [&](double t, double& uu, double& qq, double& cc) {
        double p[3], itp[3];
        for (int k = 0; k < 3; ++k) p[k] = black ? t * V[k] : (1.0 - t) * W[k] + t * V[k];
        Ictcp(p, itp);
        double qv = PyMod(PseudoHue(itp[1], itp[2]) - b.pa0, 4.0);
        qv = b.vertex[m] + (PyMod(qv - b.vertex[m] + 2.0, 4.0) - 2.0);
        uu = WarpU(itp[0], IcOfQ(b, qv));
        qq = qv;
        cc = std::hypot(itp[1], itp[2]);
    };
    double lo = 1e-12, hi = 1.0, uu = 0, qq = 0, cc = 0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        at(mid, uu, qq, cc);
        const bool goUp = black ? uu < us : uu > us;
        if (goUp) lo = mid;
        else hi = mid;
    }
    at(0.5 * (lo + hi), uu, q, c);
}

bool IsBlackEdge(char name) { return name == 'R' || name == 'G' || name == 'B'; }

// The knots of every row and the columns between them (§15.2 p. 7-8).
void SetUpKnots(CubeWork& w) {
    GamutBounds& b = *w.b;
    for (int m = 0; m < 6; ++m) {
        if (w.edges) {
            const bool black = IsBlackEdge(w.names[m]);
            const int sideRow = black ? 1 : kRows - 2;
            const double drift = w.E[static_cast<size_t>(sideRow) * 6 + m] - b.vertex[m];
            if (drift < 0.0) w.knotDef.push_back({'e', m}), w.knotDef.push_back({'v', m});
            else w.knotDef.push_back({'v', m}), w.knotDef.push_back({'e', m});
        } else {
            w.knotDef.push_back({'v', m});
        }
    }
    const int nk = static_cast<int>(w.knotDef.size());
    b.knots = nk;
    b.rowKnot.assign(static_cast<size_t>(kRows) * nk, 0.0);
    for (int i = 0; i < kRows; ++i)
        for (int idx = 0; idx < nk; ++idx) {
            const auto [kind, m] = w.knotDef[static_cast<size_t>(idx)];
            b.rowKnot[static_cast<size_t>(i) * nk + idx] = kind == 'e' ? w.E[static_cast<size_t>(i) * 6 + m] : b.vertex[m];
        }
    b.gapCols.assign(static_cast<size_t>(nk), 0);
    b.gapStart.assign(static_cast<size_t>(nk), 0);
    int start = 0;
    for (int idx = 0; idx < nk; ++idx) {
        double widest = -1e300, cuspWidth = 0.0;
        for (int i = 0; i < kRows; ++i) {
            const double* k = &b.rowKnot[static_cast<size_t>(i) * nk];
            const double width = (idx + 1 < nk ? k[idx + 1] : k[0] + 4.0) - k[idx];
            widest = std::max(widest, width);
            if (i == kCusp) cuspWidth = width;
        }
        int cols = std::max(2, static_cast<int>(std::nearbyint(kTargetColumns * widest / 4.0)));
        if (cuspWidth > 1e-12) cols = b.axisCols[static_cast<size_t>(w.knotDef[static_cast<size_t>(idx)].second)];  // a gap between vertices
        b.gapCols[static_cast<size_t>(idx)] = cols;
        b.gapStart[static_cast<size_t>(idx)] = start;
        start += cols;
    }
    b.columns = start;
    w.qn.assign(static_cast<size_t>(kRows) * b.columns, 0.0);
    for (int i = 0; i < kRows; ++i) {
        const double* k = &b.rowKnot[static_cast<size_t>(i) * nk];
        double* q = &w.qn[static_cast<size_t>(i) * b.columns];
        for (int s = 0; s < nk; ++s) {
            const double k0 = k[s], k1 = s + 1 < nk ? k[s + 1] : k[0] + 4.0;
            for (int j = 0; j < b.gapCols[static_cast<size_t>(s)]; ++j)
                q[b.gapStart[static_cast<size_t>(s)] + j] = k0 + (k1 - k0) * j / b.gapCols[static_cast<size_t>(s)];
        }
    }
    b.node.assign(static_cast<size_t>(kRows) * b.columns, 0.0);
}

double NodeTheta(const GamutBounds& b, double q) { return AngleOfPa(PyMod(b.pa0 + q, 4.0)); }

}  // namespace

const IctcpMatrices& IctcpMatrix() {
    static const IctcpMatrices m = [] {
        IctcpMatrices r;
        r.lms = M3{1688, 2146, 262, 683, 2951, 462, 99, 309, 3688};
        r.ictcp = M3{2048, 2048, 0, 6610, -13613, 7003, 17933, -17390, -543};
        for (double& x : r.lms) x /= 4096.0;
        for (double& x : r.ictcp) x /= 4096.0;
        r.ictcpInv = Inv3(r.ictcp);
        r.lmsTo2020 = Inv3(r.lms);
        r.lmsTo709 = Mul3(Bt2020To709Matrix(), r.lmsTo2020);
        return r;
    }();
    return m;
}

void Ictcp(const double l2020[3], double itp[3]) {
    const IctcpMatrices& m = IctcpMatrix();
    double lms[3], lmsp[3];
    Apply3(m.lms, l2020, lms);
    for (int k = 0; k < 3; ++k) lmsp[k] = PqEncode(lms[k] * Scale());
    Apply3(m.ictcp, lmsp, itp);
}

void LinearOfIctcp(const double itp[3], double l2020[3]) {
    const IctcpMatrices& m = IctcpMatrix();
    double lmsp[3], lms[3];
    Apply3(m.ictcpInv, itp, lmsp);
    for (int k = 0; k < 3; ++k) lms[k] = PqDecode(lmsp[k]) / Scale();
    Apply3(m.lmsTo2020, lms, l2020);
}

double PseudoHue(double ct, double cp) {
    double pa;
    if (cp >= 0.0) pa = ct >= 0.0 ? cp / (ct + cp) : 1.0 - ct / (cp - ct);
    else pa = ct < 0.0 ? 2.0 - cp / (-ct - cp) : 3.0 + ct / (ct - cp);
    if (std::isnan(pa)) pa = 0.0;
    return PyMod(pa, 4.0);
}

double GamutBounds::operator()(double I, double pa) const {
    const double q0 = PyMod(pa - pa0, 4.0);
    const double u = WarpU(I, IcOfQ(*this, q0));
    const double fi = std::clamp(u, 0.0, 1.0) * (kRows - 1);
    const int i0 = std::min(static_cast<int>(fi), kRows - 2);
    const double ti = fi - i0;
    double k[13];
    if (knots < 1 || knots > 12) Throw("hdr::GamutBounds: " + std::to_string(knots) + " knots per row");
    for (int m = 0; m < knots; ++m) {
        const double a = rowKnot[static_cast<size_t>(i0) * knots + m], c = rowKnot[static_cast<size_t>(i0 + 1) * knots + m];
        k[m] = a + (c - a) * ti;
    }
    k[knots] = k[0] + 4.0;
    const double qq = k[0] + PyMod(q0 - k[0], 4.0);
    int s = 0;
    for (int m = 1; m < knots; ++m) s += qq >= k[m] ? 1 : 0;
    const double fp = gapStart[static_cast<size_t>(s)] + (qq - k[s]) * gapCols[static_cast<size_t>(s)] / (k[s + 1] - k[s]);
    const int j0 = std::min(static_cast<int>(std::floor(fp)), columns - 1);
    const double tp = fp - j0;
    const int j1 = j0 + 1 == columns ? 0 : j0 + 1;
    const double* r0 = &node[static_cast<size_t>(i0) * columns];
    const double* r1 = r0 + columns;
    const double a = r0[j0] + (r0[j1] - r0[j0]) * tp, c = r1[j0] + (r1[j1] - r1[j0]) * tp;
    return a + (c - a) * ti;
}

int GamutBuildThreads() { return std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 8); }

GamutTables BuildSdr1GamutTables(int threads, bool edgeKnots) {
    const auto t0 = std::chrono::steady_clock::now();
    threads = std::max(threads, 1);
    GamutTables out;
    out.edgeKnots = edgeKnots;
    out.threads = threads;
    const IctcpMatrices& mx = IctcpMatrix();
    static const PqDecodeTable dec;
    M3 eye{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<CubeWork, 2> work;
    work[0].cube = Cube{mx.lmsTo709, Inv3(Bt2020To709Matrix())};
    work[0].b = &out.b709;
    work[0].edges = edgeKnots;
    work[1].cube = Cube{mx.lmsTo2020, eye};
    work[1].b = &out.b2020;
    work[1].edges = false;
    for (CubeWork& w : work) {
        SetUpAxis(w);
        w.ring.h.assign(kRingSamples, 0.0);
        w.ring.C.assign(kRingSamples, 0.0);
    }

    // 1. the ring samples of both cubes
    ParallelFor(2 * static_cast<size_t>(kRingSamples), threads, [&](size_t i) {
        CubeWork& w = work[i / kRingSamples];
        const int j = static_cast<int>(i % kRingSamples);
        double p[3], itp[3];
        RingPointAt(w.cube, static_cast<double>(j) / kRingN, p);
        Ictcp(p, itp);
        w.ring.h[static_cast<size_t>(j)] = std::atan2(itp[2], itp[1]);
        w.ring.C[static_cast<size_t>(j)] = std::hypot(itp[1], itp[2]);
    });
    for (CubeWork& w : work) w.ring.Index();
    // 2. the ring points of the cusp axes
    const size_t axis0 = static_cast<size_t>(out.b709.axisN);
    ParallelFor(axis0 + static_cast<size_t>(out.b2020.axisN), threads, [&](size_t i) {
        CubeWork& w = work[i < axis0 ? 0 : 1];
        const int col = static_cast<int>(i < axis0 ? i : i - axis0);
        GamutBounds& b = *w.b;
        RingPoint(w.cube, w.ring, NodeTheta(b, AxisPa(b, col)), b.cuspI[static_cast<size_t>(col)], b.cuspC[static_cast<size_t>(col)]);
    });
    // 3. the edge points of BT.709: rows 1..94 of the black edges, 96..126 of the white ones
    for (CubeWork& w : work) {
        w.E.assign(static_cast<size_t>(kRows) * 6, 0.0);
        w.Ce.assign(static_cast<size_t>(kRows) * 6, 0.0);
        for (int i = 0; i < kRows; ++i)
            for (int m = 0; m < 6; ++m) w.E[static_cast<size_t>(i) * 6 + m] = w.b->vertex[m];
    }
    if (edgeKnots) {
        CubeWork& w = work[0];
        std::vector<std::pair<int, int>> items;  // (vertex, row)
        for (int m = 0; m < 6; ++m) {
            const bool black = IsBlackEdge(w.names[m]);
            for (int i = black ? 1 : kCusp + 1; i < (black ? kCusp : kRows - 1); ++i) items.push_back({m, i});
        }
        ParallelFor(items.size(), threads, [&](size_t n) {
            const auto [m, i] = items[n];
            const size_t at = static_cast<size_t>(i) * 6 + m;
            EdgePoint(w, m, IsBlackEdge(w.names[m]), static_cast<double>(i) / (kRows - 1), w.E[at], w.Ce[at]);
        });
        for (int m = 0; m < 6; ++m) {
            const bool black = IsBlackEdge(w.names[m]);
            const int to = black ? 0 : kRows - 1, from = black ? 1 : kRows - 2;
            w.E[static_cast<size_t>(to) * 6 + m] = w.E[static_cast<size_t>(from) * 6 + m];
            w.Ce[static_cast<size_t>(to) * 6 + m] = 0.0;
        }
    }
    // 4. the knots, the node hues and the nodes; the ring points of the nodes on the cusp row
    for (CubeWork& w : work) SetUpKnots(w);
    const size_t n709 = static_cast<size_t>(kRows - 2) * out.b709.columns;
    const size_t nAll = n709 + static_cast<size_t>(kRows - 2) * out.b2020.columns;
    ParallelFor(nAll, threads, [&](size_t i) {
        CubeWork& w = work[i < n709 ? 0 : 1];
        GamutBounds& b = *w.b;
        const size_t local = i < n709 ? i : i - n709;
        const int row = 1 + static_cast<int>(local / b.columns), col = static_cast<int>(local % b.columns);
        const size_t at = static_cast<size_t>(row) * b.columns + col;
        const double q = w.qn[at];
        double a[3];
        RayDir(NodeTheta(b, q), a);
        const double I = WarpI(static_cast<double>(row) / (kRows - 1), IcOfQ(b, q));
        b.node[at] = FirstExit(w.cube, dec, I, a);
    });
    std::array<std::vector<double>, 2> cuspC;
    for (int c = 0; c < 2; ++c) cuspC[c].assign(static_cast<size_t>(work[c].b->columns), 0.0);
    const size_t c709 = static_cast<size_t>(out.b709.columns);
    ParallelFor(c709 + static_cast<size_t>(out.b2020.columns), threads, [&](size_t i) {
        const int c = i < c709 ? 0 : 1;
        CubeWork& w = work[c];
        const int col = static_cast<int>(i < c709 ? i : i - c709);
        double I = 0;
        RingPoint(w.cube, w.ring, NodeTheta(*w.b, w.qn[static_cast<size_t>(kCusp) * w.b->columns + col]), I,
                  cuspC[c][static_cast<size_t>(col)]);
    });
    for (int c = 0; c < 2; ++c) {
        CubeWork& w = work[c];
        GamutBounds& b = *w.b;
        for (int idx = 0; idx < b.knots; ++idx) {
            const auto [kind, m] = w.knotDef[static_cast<size_t>(idx)];
            if (kind != 'e') continue;
            for (int i = 0; i < kRows; ++i) {
                double& node = b.node[static_cast<size_t>(i) * b.columns + b.gapStart[static_cast<size_t>(idx)]];
                node = std::max(node, w.Ce[static_cast<size_t>(i) * 6 + m]);
            }
        }
        for (int col = 0; col < b.columns; ++col) {
            double& node = b.node[static_cast<size_t>(kCusp) * b.columns + col];
            node = std::max(node, cuspC[c][static_cast<size_t>(col)]);
        }
        std::fill_n(b.node.begin(), b.columns, 0.0);
        std::fill_n(b.node.begin() + static_cast<std::ptrdiff_t>(kRows - 1) * b.columns, b.columns, 0.0);
    }
    out.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

const GamutTables& Sdr1GamutTables() {
    static const GamutTables t = BuildSdr1GamutTables(GamutBuildThreads(), kGamutEdgeKnots);
    return t;
}

double CompressChroma(double c, double b709, double b2020, double knee) {
    const double kb = knee * b709;
    if (c <= kb) return c;
    const double a = (1.0 - knee) * b709;
    const double D = std::max(std::max(b2020, b709) - kb, a);
    const double d = std::min(c - kb, D);
    return kb + d / std::sqrt(1.0 + d * d * (1.0 / (a * a) - 1.0 / (D * D)));
}

namespace {

void Compressed709(double I, double ct, double cp, double cn, double C, double out[3]) {
    const IctcpMatrices& m = IctcpMatrix();
    const double f = C > 0.0 ? cn / C : 1.0;
    const double itp[3] = {I, f * ct, f * cp};
    double lmsp[3], lms[3];
    Apply3(m.ictcpInv, itp, lmsp);
    for (int k = 0; k < 3; ++k) lms[k] = PqDecode(lmsp[k]) / Scale();
    Apply3(m.lmsTo709, lms, out);
}

}  // namespace

void GamutMapIctcp(const double itp[3], double out709[3], const GamutTables& tables, double knee) {
    const double C = std::hypot(itp[1], itp[2]), pa = PseudoHue(itp[1], itp[2]);
    const double b7 = tables.b709(itp[0], pa);
    double cn = C;
    if (C > knee * b7 && b7 > 0.0) cn = CompressChroma(C, b7, tables.b2020(itp[0], pa), knee);
    Compressed709(itp[0], itp[1], itp[2], cn, C, out709);
    for (int k = 0; k < 3; ++k) out709[k] = std::clamp(out709[k], 0.0, 1.0);
}

void GamutMap709(const double l2020[3], double out709[3], const GamutTables& tables, double knee) {
    double l[3];
    for (int k = 0; k < 3; ++k) l[k] = std::clamp(l2020[k], 0.0, 1.0);
    if (l[0] == l[1] && l[1] == l[2]) {
        for (int k = 0; k < 3; ++k) out709[k] = l[k];
        return;
    }
    double itp[3];
    Ictcp(l, itp);
    const double C = std::hypot(itp[1], itp[2]), pa = PseudoHue(itp[1], itp[2]);
    const double b7 = tables.b709(itp[0], pa);
    if (C > knee * b7 && b7 > 0.0) Compressed709(itp[0], itp[1], itp[2], CompressChroma(C, b7, tables.b2020(itp[0], pa), knee), C, out709);
    else Apply3(Bt2020To709Matrix(), l, out709);
    for (int k = 0; k < 3; ++k) out709[k] = std::clamp(out709[k], 0.0, 1.0);
}

void GamutMap709(const double l2020[3], double out709[3]) { GamutMap709(l2020, out709, Sdr1GamutTables(), kGamutKnee); }

}  // namespace dlssvid::hdr

namespace dlssvid {

namespace {

std::vector<float> ToFloat(const std::vector<double>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
    return f;
}

hdrrow::Axis FloatAxis(const hdr::GamutBounds& b, std::vector<float>& cusp) {
    hdrrow::Axis a{};
    a.pa0 = static_cast<float>(b.pa0);
    for (int s = 0; s < 6; ++s) {
        a.vertex[s] = static_cast<float>(b.vertex[s]);
        const double width = (s < 5 ? b.vertex[s + 1] : 4.0) - b.vertex[s];
        a.scale[s] = static_cast<float>(b.axisCols[s] / width);
        a.start[s] = b.axisStart[s];
    }
    a.n = b.axisN;
    cusp = ToFloat(b.cuspI);
    a.cuspI = cusp.data();
    return a;
}

}  // namespace

HdrGamut::HdrGamut(const hdr::GamutTables& tables, double knee) {
    const hdr::IctcpMatrices& m = hdr::IctcpMatrix();
    const std::array<double, 9>& T = hdr::Bt2020To709Matrix();
    const double scale = hdr::kWhiteNits / 10000.0, e0 = hdr::PqEncode(0.0), iw = hdr::PqEncode(scale);
    enc_.resize(32 * 256 + 2);
    for (size_t n = 0; n < enc_.size(); ++n) {
        const double x = std::ldexp(1.0 + static_cast<double>(n % 256) / 256.0, static_cast<int>(n / 256) - 32);
        enc_[n] = static_cast<float>(hdr::PqEncode(std::min(x, 2.0) * scale));
    }
    dec_.resize(4096 + 2);
    for (size_t j = 0; j < dec_.size(); ++j) dec_[j] = static_cast<float>(hdr::PqDecode(static_cast<double>(j) * (iw / 4096.0)) / scale);
    double w709[3];
    hdr::LuminanceWeights(Gamut::Bt709, w709);
    for (int i = 0; i < 9; ++i) {
        v_.t[i] = static_cast<float>(T[i]);
        v_.mlms[i] = static_cast<float>(m.lms[i]);
        v_.a709[i] = static_cast<float>(m.lmsTo709[i]);
    }
    for (int i = 0; i < 3; ++i) {
        v_.w709[i] = static_cast<float>(w709[i]);
        v_.ict[i] = static_cast<float>(m.ictcp[3 + i]);
        v_.ict[3 + i] = static_cast<float>(m.ictcp[6 + i]);
        v_.ictInv[2 * i] = static_cast<float>(m.ictcpInv[3 * i + 1]);
        v_.ictInv[2 * i + 1] = static_cast<float>(m.ictcpInv[3 * i + 2]);
    }
    v_.knee = static_cast<float>(knee);
    v_.kneeRest = static_cast<float>(1.0 - knee);
    v_.tau = 0.47f;
    const double uc = static_cast<double>(hdr::GamutBounds::kCuspRow) / (hdr::GamutBounds::kRows - 1);
    v_.uc = static_cast<float>(uc);
    v_.ucRest = static_cast<float>(1.0 - uc);
    v_.e0 = static_cast<float>(e0);
    v_.iw = static_cast<float>(iw);
    v_.decScale = static_cast<float>(4096.0 / iw);
    v_.enc = enc_.data();
    v_.dec = dec_.data();

    // BT.709: 12 knots per row; without the edge knots every vertex is two knots with an empty gap between them
    const hdr::GamutBounds& b7 = tables.b709;
    v_.b709.axis = FloatAxis(b7, cusp709_);
    knot709_.assign(static_cast<size_t>(hdr::GamutBounds::kRows) * 12, 0.f);
    if (b7.knots == 12) {
        for (size_t i = 0; i < knot709_.size(); ++i) knot709_[i] = static_cast<float>(b7.rowKnot[i]);
        for (int c = 0; c < 12; ++c) {
            v_.b709.cols[c] = static_cast<float>(b7.gapCols[static_cast<size_t>(c)]);
            v_.b709.start[c] = b7.gapStart[static_cast<size_t>(c)];
        }
    } else if (b7.knots == 6) {
        for (int i = 0; i < hdr::GamutBounds::kRows; ++i)
            for (int k = 0; k < 6; ++k)
                knot709_[static_cast<size_t>(i) * 12 + 2 * k] = knot709_[static_cast<size_t>(i) * 12 + 2 * k + 1] =
                    static_cast<float>(b7.rowKnot[static_cast<size_t>(i) * 6 + k]);
        for (int k = 0; k < 6; ++k) {
            v_.b709.cols[2 * k] = 0.f;
            v_.b709.start[2 * k] = b7.gapStart[static_cast<size_t>(k)];
            v_.b709.cols[2 * k + 1] = static_cast<float>(b7.gapCols[static_cast<size_t>(k)]);
            v_.b709.start[2 * k + 1] = b7.gapStart[static_cast<size_t>(k)];
        }
    } else {
        Throw("HdrGamut: the BT.709 table has " + std::to_string(b7.knots) + " knots per row");
    }
    v_.b709.knot = knot709_.data();
    v_.b709.n = b7.columns;
    node709_ = ToFloat(b7.node);
    v_.b709.node = node709_.data();

    const hdr::GamutBounds& b20 = tables.b2020;
    if (b20.knots != 6 || b20.columns != b20.axisN) Throw("HdrGamut: the BT.2020 table is not on its cusp axis");
    v_.b2020.axis = FloatAxis(b20, cusp2020_);
    node2020_ = ToFloat(b20.node);
    v_.b2020.node = node2020_.data();
}

const HdrGamut& HdrGamut::Get() {
    static const HdrGamut g(hdr::Sdr1GamutTables(), hdr::kGamutKnee);
    return g;
}

}  // namespace dlssvid
