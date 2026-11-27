// TASK-0080 (HDR step TC-1): the gamut step G1 of the HDR -> SDR transfer sdr1 - perceptual, constant ICtCp hue (tc-plan §15). The
// anchors are the numbers Output/hdr-design/tc-checks/ictcp_ref.py prints; the tables are held to an independent builder below.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "convert/ColorConvert.h"
#include "convert/HdrGamut.h"
#include "convert/HdrTransfer.h"

using namespace dlssvid;

namespace {

using M3 = std::array<double, 9>;
constexpr double kPi = 3.141592653589793;

M3 Inv(const M3& m) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return M3{(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det,
              (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
}
M3 Mul(const M3& a, const M3& b) {
    M3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    return r;
}
void Ap(const M3& m, const double x[3], double y[3]) {
    const double x0 = x[0], x1 = x[1], x2 = x[2];
    for (int i = 0; i < 3; ++i) y[i] = m[3 * i] * x0 + m[3 * i + 1] * x1 + m[3 * i + 2] * x2;
}
// RGB -> XYZ of primaries and D65, written here again on purpose: the test does not borrow the code it checks
M3 Npm(const double p[3][2]) {
    M3 P{};
    for (int j = 0; j < 3; ++j) P[j] = p[j][0] / p[j][1], P[3 + j] = 1.0, P[6 + j] = (1.0 - p[j][0] - p[j][1]) / p[j][1];
    const double w[3] = {0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290};
    double s[3];
    Ap(Inv(P), w, s);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) P[3 * i + j] *= s[j];
    return P;
}
const double kP709[3][2] = {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}};
const double kP2020[3][2] = {{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}};
const double kP3[3][2] = {{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}};
M3 To2020(const double p[3][2]) { return Mul(Inv(Npm(kP2020)), Npm(p)); }  // linear RGB of `p` -> linear BT.2020

// ST 2084 and ICtCp (BT.2100), again written here
constexpr double kM1 = 2610.0 / 16384, kM2 = 2523.0 / 4096 * 128, kC1 = 3424.0 / 4096, kC2 = 2413.0 / 4096 * 32, kC3 = 2392.0 / 4096 * 32;
double PqEnc(double y) {
    const double ym = std::pow(std::max(y, 0.0), kM1);
    return std::pow((kC1 + kC2 * ym) / (1 + kC3 * ym), kM2);
}
double PqDec(double e) {
    const double p = std::pow(std::max(e, 0.0), 1 / kM2);
    return std::pow(std::max(p - kC1, 0.0) / (kC2 - kC3 * p), 1 / kM1);
}
constexpr double kScale = 203.0 / 10000.0;
const M3 kLms = [] {
    M3 m{1688, 2146, 262, 683, 2951, 462, 99, 309, 3688};
    for (double& x : m) x /= 4096;
    return m;
}();
const M3 kIct = [] {
    M3 m{2048, 2048, 0, 6610, -13613, 7003, 17933, -17390, -543};
    for (double& x : m) x /= 4096;
    return m;
}();
const M3 kIctInv = Inv(kIct);
void Itp(const double l[3], double o[3]) {
    double lms[3], p[3];
    Ap(kLms, l, lms);
    for (int k = 0; k < 3; ++k) p[k] = PqEnc(lms[k] * kScale);
    Ap(kIct, p, o);
}
double HueDeg(const double l2020[3]) {
    double o[3];
    Itp(l2020, o);
    return std::atan2(o[2], o[1]) * 180 / kPi;
}
double WrapDeg(double d) { return std::fmod(std::fmod(d + 180, 360) + 360, 360) - 180; }

// ---------------------------------------------------------------- the independent builder of tc-plan §15.2 (exact std::pow, one thread)
double Mod(double x, double m) {  // Python's %
    double r = std::fmod(x, m);
    if (r != 0 && ((m < 0) != (r < 0))) r += m;
    return r == 0 ? std::copysign(0.0, m) : r;
}
double Pa(double ct, double cp) {
    double pa = cp >= 0 ? (ct >= 0 ? cp / (ct + cp) : 1 - ct / (cp - ct)) : (ct < 0 ? 2 - cp / (-ct - cp) : 3 + ct / (ct - cp));
    return Mod(std::isnan(pa) ? 0.0 : pa, 4.0);
}
double AngleOf(double pa) {
    const double q = std::floor(pa), r = pa - q;
    return q * (kPi / 2) + std::atan2(r, 1 - r);
}
double WrapRad(double x) { return Mod(x + kPi, 2 * kPi) - kPi; }
const double E0 = PqEnc(0.0), IW = PqEnc(kScale), UC = 95.0 / 127;
const double kRing[6][3] = {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}, {1, 0, 1}};
const char kNames[6] = {'R', 'Y', 'G', 'C', 'B', 'M'};

struct RefCube {
    M3 A, to2020;
    std::array<double, 6> kv{};
    std::array<int, 6> gS{}, gst{};
    int gn = 0;
    double pa0 = 0;
    std::array<std::array<double, 3>, 6> V{};
    std::array<char, 6> names{};
    std::vector<double> h, C, Ic;
    std::vector<double> E, Ce;                  // [128][6]
    std::vector<std::pair<char, int>> knotDef;  // (kind, vertex)
    std::vector<int> S, st;
    int nh = 0;
    std::vector<double> K;  // [128][nk]

    void At(double t, double p[3]) const {
        const int sg = static_cast<int>(std::floor(t)) % 6;
        const double rr = t - std::floor(t);
        double q[3];
        for (int k = 0; k < 3; ++k) q[k] = kRing[sg][k] * (1 - rr) + kRing[(sg + 1) % 6][k] * rr;
        Ap(to2020, q, p);
    }
    void RingPoint(double th, double& I, double& c) const {
        const int n = static_cast<int>(h.size());
        int best = -1;
        for (int i = 0; i < n; ++i) {
            const double d = WrapRad(h[i] - th), d1 = WrapRad(h[(i + 1) % n] - th);
            if (d <= 0 && d1 > 0 && d1 - d < kPi && (best < 0 || C[i] + C[(i + 1) % n] > C[best] + C[(best + 1) % n])) best = i;
        }
        REQUIRE(best >= 0);
        double lo = best / 2048.0, hi = lo + 1 / 2048.0, p[3], o[3];
        for (int i = 0; i < 60; ++i) {
            const double mid = 0.5 * (lo + hi);
            At(mid, p);
            Itp(p, o);
            (WrapRad(std::atan2(o[2], o[1]) - th) <= 0 ? lo : hi) = mid;
        }
        At(0.5 * (lo + hi), p);
        Itp(p, o);
        I = o[0], c = std::hypot(o[1], o[2]);
    }
    double IcOf(double q) const {
        q = Mod(q, 4.0);
        int s = 0;
        for (int i = 0; i < 6; ++i)
            if (kv[i] <= q) s = i;
        const double k1 = s < 5 ? kv[s + 1] : 4.0;
        const double fp = gst[s] + (q - kv[s]) * gS[s] / (k1 - kv[s]);
        const int j0 = static_cast<int>(std::floor(fp)) % gn;
        return Ic[j0] + (Ic[(j0 + 1) % gn] - Ic[j0]) * (fp - std::floor(fp));
    }
    static double WarpU(double I, double ic) { return I <= ic ? (I - E0) / (ic - E0) * UC : UC + (I - ic) / (IW - ic) * (1 - UC); }
    static double WarpI(double u, double ic) { return u <= UC ? E0 + (ic - E0) * u / UC : ic + (IW - ic) * (u - UC) / (1 - UC); }
    bool Inside(double I, double c, const double a[3]) const {
        double lms[3], rgb[3];
        for (int k = 0; k < 3; ++k) {
            const double e = I + c * a[k];
            if (!(e >= E0 - 1e-12 && e <= IW + 1e-12)) return false;
            lms[k] = PqDec(e) / kScale;
        }
        Ap(A, lms, rgb);
        for (double x : rgb)
            if (!(x >= -1e-9 && x <= 1 + 1e-9)) return false;
        return true;
    }
    double FirstExit(double I, double th) const {
        const double a[3] = {std::cos(th) * kIctInv[1] + std::sin(th) * kIctInv[2], std::cos(th) * kIctInv[4] + std::sin(th) * kIctInv[5],
                             std::cos(th) * kIctInv[7] + std::sin(th) * kIctInv[8]};
        double out = 0.75;
        for (int s = 1; s <= 75; ++s)
            if (!Inside(I, s * 0.01, a)) {
                out = s * 0.01;
                break;
            }
        double hi = out, lo = out < 0.75 ? out - 0.01 : 0.75;
        for (int i = 0; i < 26; ++i) {
            const double mid = 0.5 * (lo + hi);
            (Inside(I, mid, a) ? lo : hi) = mid;
        }
        return lo;
    }
    void EdgePoint(int m, bool black, double us, double& q, double& c) const {
        double W[3];
        const double ones[3] = {1, 1, 1};
        Ap(to2020, ones, W);
        const auto at = [&](double t, double& uu, double& qq, double& cc) {
            double p[3], o[3];
            for (int k = 0; k < 3; ++k) p[k] = black ? t * V[m][k] : (1 - t) * W[k] + t * V[m][k];
            Itp(p, o);
            qq = Mod(Pa(o[1], o[2]) - pa0, 4.0);
            qq = kv[m] + (Mod(qq - kv[m] + 2.0, 4.0) - 2.0);
            uu = WarpU(o[0], IcOf(qq));
            cc = std::hypot(o[1], o[2]);
        };
        double lo = 1e-12, hi = 1, uu, qq, cc;
        for (int i = 0; i < 60; ++i) {
            const double mid = 0.5 * (lo + hi);
            at(mid, uu, qq, cc);
            ((black ? uu < us : uu > us) ? lo : hi) = mid;
        }
        at(0.5 * (lo + hi), uu, q, c);
    }

    RefCube(const M3& a, const M3& t2020, bool edges) : A(a), to2020(t2020) {
        double pv[6];
        std::array<std::array<double, 3>, 6> Vr{};
        for (int m = 0; m < 6; ++m) {
            double o[3];
            Ap(to2020, kRing[m], Vr[m].data());
            Itp(Vr[m].data(), o);
            pv[m] = Pa(o[1], o[2]);
        }
        std::array<int, 6> ord{0, 1, 2, 3, 4, 5};
        std::sort(ord.begin(), ord.end(), [&](int x, int y) { return pv[x] < pv[y]; });
        pa0 = pv[ord[0]];
        for (int m = 0; m < 6; ++m) V[m] = Vr[ord[m]], names[m] = kNames[ord[m]], kv[m] = Mod(pv[ord[m]] - pa0, 4.0);
        std::vector<double> gq;
        for (int s = 0; s < 6; ++s) {
            const double w = (s < 5 ? kv[s + 1] : 4.0) - kv[s];
            gS[s] = std::max(2, static_cast<int>(std::nearbyint(512 * w / 4)));
            gst[s] = gn;
            gn += gS[s];
            for (int j = 0; j < gS[s]; ++j) gq.push_back(kv[s] + w * j / gS[s]);
        }
        for (int j = 0; j < 6 * 2048; ++j) {
            double p[3], o[3];
            At(j / 2048.0, p);
            Itp(p, o);
            h.push_back(std::atan2(o[2], o[1]));
            C.push_back(std::hypot(o[1], o[2]));
        }
        for (double q : gq) {
            double I, c;
            RingPoint(AngleOf(Mod(pa0 + q, 4.0)), I, c);
            Ic.push_back(I);
        }
        E.assign(128 * 6, 0.0);
        Ce.assign(128 * 6, 0.0);
        for (int i = 0; i < 128; ++i)
            for (int m = 0; m < 6; ++m) E[i * 6 + m] = kv[m];
        for (int m = 0; m < 6; ++m) {
            const bool black = names[m] == 'R' || names[m] == 'G' || names[m] == 'B';
            if (edges) {
                for (int i = black ? 1 : 96; i < (black ? 95 : 127); ++i) EdgePoint(m, black, i / 127.0, E[i * 6 + m], Ce[i * 6 + m]);
                E[(black ? 0 : 127) * 6 + m] = E[(black ? 1 : 126) * 6 + m];
                const bool before = E[(black ? 1 : 126) * 6 + m] - kv[m] < 0;
                if (before) knotDef.push_back({'e', m}), knotDef.push_back({'v', m});
                else knotDef.push_back({'v', m}), knotDef.push_back({'e', m});
            } else {
                knotDef.push_back({'v', m});
            }
        }
        const int nk = static_cast<int>(knotDef.size());
        K.assign(128 * static_cast<size_t>(nk), 0.0);
        for (int i = 0; i < 128; ++i)
            for (int x = 0; x < nk; ++x) K[i * nk + x] = knotDef[x].first == 'e' ? E[i * 6 + knotDef[x].second] : kv[knotDef[x].second];
        for (int x = 0; x < nk; ++x) {
            double widest = -1e300, cusp = 0;
            for (int i = 0; i < 128; ++i) {
                const double w = (x + 1 < nk ? K[i * nk + x + 1] : K[i * nk] + 4.0) - K[i * nk + x];
                widest = std::max(widest, w);
                if (i == 95) cusp = w;
            }
            S.push_back(cusp > 1e-12 ? gS[knotDef[x].second] : std::max(2, static_cast<int>(std::nearbyint(512 * widest / 4))));
            st.push_back(nh);
            nh += S.back();
        }
    }
    int Knots() const { return static_cast<int>(knotDef.size()); }
    double NodeQ(int i, int col) const {
        const int nk = Knots();
        int x = nk - 1;
        while (x > 0 && col < st[x]) --x;
        const double k0 = K[i * nk + x], k1 = x + 1 < nk ? K[i * nk + x + 1] : K[i * nk] + 4.0;
        return k0 + (k1 - k0) * (col - st[x]) / S[x];
    }
    double Node(int i, int col) const {
        if (i == 0 || i == 127) return 0.0;
        const double q = NodeQ(i, col), th = AngleOf(Mod(pa0 + q, 4.0));
        double b = FirstExit(WarpI(i / 127.0, IcOf(q)), th);
        for (int x = 0; x < Knots(); ++x)
            if (knotDef[x].first == 'e' && st[x] == col) b = std::max(b, Ce[i * 6 + knotDef[x].second]);
        if (i == 95) {
            double I, c;
            RingPoint(th, I, c);
            b = std::max(b, c);
        }
        return b;
    }
    void NodeIPa(int i, int col, double& I, double& pa) const {
        const double q = NodeQ(i, col);
        I = WarpI(i / 127.0, IcOf(q));
        pa = Mod(pa0 + q, 4.0);
    }
};

const M3& T709() {  // linear BT.2020 -> BT.709
    static const M3 m = Mul(Inv(Npm(kP709)), Npm(kP2020));
    return m;
}

}  // namespace

TEST_CASE("Gamut step G1 keeps the ICtCp hue and intensity and lands BT.2020 primaries on the BT.709 cube", "[unit][hdr][gamut]") {
    const M3 from709 = To2020(kP709), fromP3 = To2020(kP3);
    struct Anchor {
        const char* what;
        const M3* from;  // nullptr: the input is BT.2020
        double in[3], out[3];
    };
    // linear in -> linear BT.709 out, tc-plan §15.7 (ictcp_ref.py, sections C and J)
    const Anchor anchors[] = {
        {"2020 R", nullptr, {1, 0, 0}, {1.00000, 0.06182, 0.04141}},
        {"2020 G", nullptr, {0, 1, 0}, {0.00000, 0.89747, 0.10840}},
        {"2020 B", nullptr, {0, 0, 1}, {0.00000, 0.08954, 0.25662}},
        {"P3 R (B -0.00121 clamped)", &fromP3, {1, 0, 0}, {1.00000, 0.02209, 0.00006}},
        {"2020 (1, 0.99, 0.02) beyond B2020", nullptr, {1, 0.99, 0.02}, {1.00000, 0.99968, 0.06036}},
        {"709 R", &from709, {1, 0, 0}, {0.95747, 0.01211, 0.00431}},
        {"709 B", &from709, {0, 0, 1}, {0.00000, 0.00763, 0.94861}},
        {"709 edge black-blue (0, 0, 0.94)", &from709, {0, 0, 0.94}, {0.00000, 0.00747, 0.88963}},
        {"709 edge white-yellow (1, 1, 0.03)", &from709, {1, 1, 0.03}, {0.99184, 0.99922, 0.05382}},
    };
    for (const Anchor& a : anchors) {
        INFO(a.what);
        double l[3], o[3];
        if (a.from) Ap(*a.from, a.in, l);
        else std::copy(a.in, a.in + 3, l);
        hdr::GamutMap709(l, o);
        for (int k = 0; k < 3; ++k) CHECK(std::abs(o[k] - a.out[k]) <= 1e-4);  // the float path: U6, with the reference's slope
        // the ICtCp hue of a primary is kept: the dense maximum of the reference is 0.342 deg next to the BT.709 blue vertex
        double back[3];
        const M3 inv = Inv(T709());
        Ap(inv, o, back);
        double lc[3];
        for (int k = 0; k < 3; ++k) lc[k] = std::clamp(l[k], 0.0, 1.0);
        CHECK(std::abs(WrapDeg(HueDeg(back) - HueDeg(lc))) <= 0.5);
    }

    // the BT.2020 cube's surface: hue within 0.5 deg (ICtCp, not u'v'), I within 1e-3, the output in [0, 1]; inside the knee T l bit for bit
    std::mt19937 rng(20261127);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const hdr::GamutTables& tables = hdr::Sdr1GamutTables();
    const M3 inv709 = Inv(T709());
    double worstHue = 0, worstI = 0;
    int kept = 0, compressed = 0;
    for (int n = 0; n < 20000; ++n) {
        double l[3];
        for (double& x : l) x = std::pow(uni(rng), 2.4);
        const int face = n % 6;
        l[face % 3] = face >= 3 ? 1.0 : 0.0;
        double o[3], itp[3], back[3], itpOut[3];
        hdr::GamutMap709(l, o);
        for (double x : o) REQUIRE((x >= 0.0 && x <= 1.0));
        hdr::Ictcp(l, itp);
        const double C = std::hypot(itp[1], itp[2]), b7 = tables.b709(itp[0], hdr::PseudoHue(itp[1], itp[2]));
        Ap(inv709, o, back);
        Itp(back, itpOut);
        worstI = std::max(worstI, std::abs(itpOut[0] - itp[0]));
        if (C <= hdr::kGamutKnee * b7) {
            double t[3];
            hdr::Bt2020To709(l, t);
            for (int k = 0; k < 3; ++k) REQUIRE(o[k] == std::clamp(t[k], 0.0, 1.0));
            ++kept;
        } else {
            ++compressed;
            if (C > 0.02) worstHue = std::max(worstHue, std::abs(WrapDeg(HueDeg(back) - HueDeg(l))));
        }
    }
    std::printf("G1 on the 2020 surface: %d compressed, %d kept; |dh| max %.3f deg, |dI| max %.1e\n", compressed, kept, worstHue, worstI);
    CHECK(worstHue <= 0.5);
    CHECK(worstI <= 1e-3);
    CHECK(compressed > 15000);

    // greys are themselves bit for bit
    for (double g : {0.0, 1e-6, 0.18, 0.5, 1.0}) {
        const double l[3] = {g, g, g};
        double o[3];
        hdr::GamutMap709(l, o);
        CHECK((o[0] == g && o[1] == g && o[2] == g));
    }

    // saturation ramps (24 I x 72 hues x 300 steps up to 0.98 B2020): the output chroma never decreases
    double worstDrop = 0;
    for (int i = 0; i < 24; ++i) {
        const double I = 0.05 + (IW - 0.004 - 0.05) * i / 23;
        for (int hh = 0; hh < 72; ++hh) {
            const double th = (hh * 5.0 - 180 + 0.37) * kPi / 180;
            const double end = 0.98 * tables.b2020(I, Pa(std::cos(th), std::sin(th)));
            double prev = -1;
            for (int s = 0; s < 300; ++s) {
                const double c = end * s / 299;
                const double itp[3] = {I, c * std::cos(th), c * std::sin(th)};
                double l[3], o[3], back[3], io[3];
                hdr::LinearOfIctcp(itp, l);
                hdr::GamutMap709(l, o);
                Ap(inv709, o, back);
                Itp(back, io);
                const double co = std::hypot(io[1], io[2]);
                if (prev >= 0) worstDrop = std::max(worstDrop, prev - co);
                prev = co;
            }
        }
    }
    std::printf("G1 saturation ramps: worst drop of the output chroma %.1e\n", worstDrop);
    CHECK(worstDrop <= 1e-12);
}

TEST_CASE("Gamut boundary tables match an independent double builder", "[unit][hdr][gamut]") {
    const M3 lmsTo2020 = Inv(kLms);
    const M3 eye{1, 0, 0, 0, 1, 0, 0, 0, 1};
    const RefCube r709(Mul(T709(), lmsTo2020), Inv(T709()), true), r2020(lmsTo2020, eye, false);
    const hdr::GamutTables& t = hdr::Sdr1GamutTables();
    std::printf("G1 tables: built in %.1f ms on %d threads\n", t.buildMs, t.threads);

    struct Pair {
        const char* what;
        const RefCube& ref;
        const hdr::GamutBounds& b;
        int knots, columns;
    };
    const Pair pairs[] = {{"709", r709, t.b709, 12, 578}, {"2020", r2020, t.b2020, 6, 512}};
    std::mt19937 rng(80);
    for (const Pair& p : pairs) {
        INFO(p.what);
        REQUIRE(p.b.knots == p.knots);
        REQUIRE(p.b.columns == p.columns);
        REQUIRE(p.ref.nh == p.columns);
        REQUIRE(p.b.node.size() == size_t{128} * p.columns);
        REQUIRE(p.b.rowKnot.size() == size_t{128} * p.knots);
        CHECK(std::abs(p.b.pa0 - p.ref.pa0) <= 1e-12);
        for (int s = 0; s < 6; ++s) {
            CHECK(std::abs(p.b.vertex[s] - p.ref.kv[s]) <= 1e-12);
            CHECK(p.b.axisCols[s] == p.ref.gS[s]);
        }
        for (int x = 0; x < p.knots; ++x) {
            CHECK(p.b.gapCols[x] == p.ref.S[x]);
            CHECK(p.b.gapStart[x] == p.ref.st[x]);
        }
        for (int j = 0; j < p.ref.gn; ++j) CHECK(std::abs(p.b.cuspI[j] - p.ref.Ic[j]) <= 1e-7);
        double worst = 0;
        const auto node = [&](int i, int col) {
            const double want = p.ref.Node(i, col), got = p.b.node[static_cast<size_t>(i) * p.columns + col];
            worst = std::max(worst, std::abs(want - got));
            if (std::abs(want - got) > 1e-7) FAIL_CHECK("node " << i << ", " << col << ": " << got << " vs " << want);
        };
        for (int i : {0, 1, 94, 95, 96, 126, 127}) {
            for (int x = 0; x < p.knots; ++x) CHECK(std::abs(p.b.rowKnot[static_cast<size_t>(i) * p.knots + x] - p.ref.K[i * p.knots + x]) <= 1e-7);
            for (int col = 0; col < p.columns; ++col) node(i, col);
        }
        for (int x = 0; x < p.knots; ++x)
            if (p.ref.knotDef[x].first == 'e')
                for (int i = 0; i < 128; ++i) node(i, p.ref.st[x]);
        std::uniform_int_distribution<int> row(1, 126), col(0, p.columns - 1);
        for (int n = 0; n < 2000; ++n) node(row(rng), col(rng));
        std::printf("G1 table %s: nodes within %.1e of the independent builder\n", p.what, worst);
    }

    // the floors of the curve never act: B2020 >= 1.02 B709 at every BT.709 node (measured 1.0431)
    double lowest = 1e9;
    for (int i = 1; i < 127; ++i)
        for (int c = 0; c < t.b709.columns; ++c) {
            const double b7 = t.b709.node[static_cast<size_t>(i) * t.b709.columns + c];
            if (b7 <= 0) continue;
            double I, pa;
            r709.NodeIPa(i, c, I, pa);
            lowest = std::min(lowest, t.b2020(I, pa) / b7);
        }
    std::printf("G1 tables: min B2020 / B709 over the BT.709 nodes %.4f\n", lowest);
    CHECK(lowest >= 1.02);

    // the result does not depend on the thread count
    const hdr::GamutTables one = hdr::BuildSdr1GamutTables(1), three = hdr::BuildSdr1GamutTables(3);
    std::printf("G1 tables: %.1f ms on 1 thread, %.1f ms on 3\n", one.buildMs, three.buildMs);
    CHECK(one.b709.node == three.b709.node);
    CHECK(one.b709.rowKnot == three.b709.rowKnot);
    CHECK(one.b709.cuspI == three.b709.cuspI);
    CHECK(one.b2020.node == three.b2020.node);
    CHECK(one.b2020.cuspI == three.b2020.cuspI);
    CHECK(one.b709.node == t.b709.node);
}

TEST_CASE("The level-1 early exit never skips a pixel the reference compresses", "[unit][hdr][gamut]") {
    const HdrGamut& g = HdrGamut::Get();
    const hdrrow::Step& v = g.View();
    const hdr::GamutTables& tables = hdr::Sdr1GamutTables();
    long long taken = 0, wrong = 0, total = 0;
    const auto check = [&](const float l[3]) {
        ++total;
        float t[3];
        for (int c = 0; c < 3; ++c) t[c] = (v.t[3 * c] * l[0] + v.t[3 * c + 1] * l[1]) + v.t[3 * c + 2] * l[2];
        if (!g.Level1(t)) return;
        ++taken;
        const double d[3] = {l[0], l[1], l[2]};
        double itp[3];
        hdr::Ictcp(d, itp);
        const double C = std::hypot(itp[1], itp[2]), b7 = tables.b709(itp[0], hdr::PseudoHue(itp[1], itp[2]));
        if (C > hdr::kGamutKnee * b7 && b7 > 0) ++wrong;
    };
    const M3 from709 = To2020(kP709);
    for (int a = 0; a <= 160; ++a)
        for (int b = 0; b <= 160; ++b)
            for (int c = 0; c <= 160; ++c) {
                const double vp[3] = {a / 160.0, b / 160.0, c / 160.0};
                double lin[3], l2020[3];
                for (int k = 0; k < 3; ++k) lin[k] = std::pow(vp[k], 2.4);
                Ap(from709, lin, l2020);
                float f[3], h[3];
                for (int k = 0; k < 3; ++k) f[k] = static_cast<float>(std::clamp(l2020[k], 0.0, 1.0)), h[k] = static_cast<float>(lin[k]);
                check(f);  // BT.709 content in a BT.2020 container
                check(h);  // the BT.2020 lattice
            }
    for (Curve curve : {Curve::Pq, Curve::Hlg})
        for (int a = 0; a <= 40; ++a)
            for (int b = 0; b <= 40; ++b)
                for (int c = 0; c <= 40; ++c) {
                    const double codes[3] = {a / 40.0, b / 40.0, c / 40.0};
                    double l[3];
                    hdr::LinearOfCodes(curve, Gamut::Bt2020, codes, l);
                    const float f[3] = {static_cast<float>(l[0]), static_cast<float>(l[1]), static_cast<float>(l[2])};
                    check(f);
                }
    std::mt19937 rng(28);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int n = 0; n < 1000000; ++n) {  // near black: l = s d, s in [1e-10, 1e-3]
        const double s = std::pow(10.0, -10.0 + 7.0 * uni(rng));
        const float f[3] = {static_cast<float>(s * uni(rng)), static_cast<float>(s * uni(rng)), static_cast<float>(s * uni(rng))};
        check(f);
    }
    std::printf("G1 level 1: takes %lld of %lld pixels, %lld of them compressed by the reference\n", taken, total, wrong);
    CHECK(taken > 500000);
    CHECK(wrong == 0);
}
