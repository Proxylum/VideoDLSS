#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "convert/HdrRow.h"

// TASK-0080 (HDR step TC): the gamut step G1 of the HDR -> SDR transfer `sdr1` (tc-plan §15, the user's decision 2026-11-27: perceptual,
// constant ICtCp hue). Linear BT.2020 after the EETF (1.0 = 203 nits) -> linear BT.709: I and the hue of ICtCp (BT.2100, absolute light)
// stay, only the chroma C changes - C <= k B709 keeps the colour (T l), above the knee C' = k B709 + d / sqrt(1 + d^2 (1/a^2 - 1/D^2)),
// a = (1 - k) B709, D = max(B2020 - k B709, a), d = min(C - k B709, D), so C >= B2020 lands on B709. B709 and B2020 - where the
// constant-(I, hue) ray leaves each cube - are tables, and the tables are the definition (§15.2): the reference is
// Output/hdr-design/tc-checks/ictcp_ref.py. `hdr::` is double; HdrGamut is the float copy a frame goes through.
namespace dlssvid::hdr {

// The two settings of G1 the user picks before TC-2 (tc-plan §15.13, question 5): the knee k (0.8 or 0.85) and the edge knots of the
// BT.709 table (with them pure blues and bright yellows of BT.709 keep their chroma, without them saturated colours outside BT.709
// change smoothly near those edges). Changing either after TC-2 is a new token of the transfer.
inline constexpr double kGamutKnee = 0.8;
inline constexpr bool kGamutEdgeKnots = true;

struct IctcpMatrices {
    std::array<double, 9> lms;        // linear BT.2020 -> relative LMS (BT.2100)
    std::array<double, 9> ictcp;      // L'M'S' -> ICtCp
    std::array<double, 9> ictcpInv;   // ICtCp -> L'M'S'
    std::array<double, 9> lmsTo2020;  // relative LMS -> linear BT.2020
    std::array<double, 9> lmsTo709;   // relative LMS -> linear BT.709 (T lmsTo2020)
};
const IctcpMatrices& IctcpMatrix();
// ICtCp of linear BT.2020 (1.0 = kWhiteNits) on absolute light, and back.
void Ictcp(const double l2020[3], double itp[3]);
void LinearOfIctcp(const double itp[3], double l2020[3]);
// The "diamond" pseudo-angle of (Ct, Cp) in [0, 4): one division, monotone in atan2(cp, ct); 0 at (0, 0).
double PseudoHue(double ct, double cp);

// B(I, pa) of one cube: the chroma where the constant-(I, hue) ray leaves it, tabulated by the rules of tc-plan §15.2.
struct GamutBounds {
    static constexpr int kRows = 128, kCuspRow = 95;
    double pa0 = 0;                              // pa of the ring's first vertex; every pa below is relative to it
    std::array<double, 6> vertex{};              // the cusp axis: the 6 vertices of the ring (vertex[0] = 0)
    std::array<int, 6> axisCols{}, axisStart{};  // columns of the axis's gaps
    int axisN = 0;
    std::vector<double> cuspI, cuspC;            // the ring point of each axis column
    int knots = 6;                               // 12 for BT.709 with the edge knots, 6 otherwise
    std::vector<double> rowKnot;                 // [kRows][knots]: relative pa of the knots of each row
    std::vector<int> gapCols, gapStart;          // [knots]
    int columns = 0;
    std::vector<double> node;                    // [kRows][columns]
    double operator()(double I, double pa) const;  // §15.2 p. 9: bilinear in (u, column)
};

struct GamutTables {
    GamutBounds b709, b2020;
    bool edgeKnots = true;
    double buildMs = 0;
    int threads = 0;
};
// The tables of G1 built on `threads` threads (the result does not depend on the count).
GamutTables BuildSdr1GamutTables(int threads, bool edgeKnots = kGamutEdgeKnots);
// Built once per process on the threads ForEachBand would take (half the cores, 1..8); concurrent first calls wait.
const GamutTables& Sdr1GamutTables();
int GamutBuildThreads();

// The curve of G1 (p = 2): c <= k b709 is c; b2020 is taken as max(b2020, b709).
double CompressChroma(double c, double b709, double b2020, double knee = kGamutKnee);
// G1 on (I, Ct, Cp) without clamping the input and without the grey shortcut - the reference of the float step's slope (U6).
void GamutMapIctcp(const double itp[3], double out709[3], const GamutTables& tables, double knee = kGamutKnee);
// G1 of linear BT.2020 (clamped to [0, 1]): a grey is itself, otherwise the step; the result is clamped to [0, 1].
void GamutMap709(const double l2020[3], double out709[3], const GamutTables& tables, double knee = kGamutKnee);
void GamutMap709(const double l2020[3], double out709[3]);  // Sdr1GamutTables(), kGamutKnee

}  // namespace dlssvid::hdr

namespace dlssvid {

// The float copy of G1 a frame goes through: tables, matrices and weights all come from `hdr::`, none is a literal. One per process
// for the default settings (Get), shared by PQ and HLG.
class HdrGamut {
public:
    static const HdrGamut& Get();
    HdrGamut(const hdr::GamutTables& tables, double knee);
    HdrGamut(const HdrGamut&) = delete;
    HdrGamut& operator=(const HdrGamut&) = delete;

    // Level 1 of the step: T l lies so deep in the BT.709 cube (its box measure <= tau) that the reference keeps it. Open for U28.
    inline bool Level1(const float t709[3]) const;
    // Linear BT.2020 after the EETF -> linear BT.709 in [0, 1], in place (tc-plan §15.5).
    inline void Map(float l[3]) const;
    const hdrrow::Step& View() const { return v_; }

private:
    std::vector<float> enc_, dec_, cusp709_, cusp2020_, knot709_, node709_, node2020_;
    hdrrow::Step v_{};
};

namespace hdrrow_detail {

// std::min / std::max with the operand order the AVX2 row mirrors (it gives the same bits, ±0 included).
inline float Lerp(const float* t, int n, float x) {  // x in [0, 1], n segments
    const float fx = x * static_cast<float>(n);
    const int i = std::max(std::min(static_cast<int>(fx), n - 1), 0);
    return t[i] + (t[i + 1] - t[i]) * (fx - static_cast<float>(i));
}

// A table over 32 octaves x 256 of the float's bits: x in [2^-32, 1] by its exponent and mantissa, below 2^-32 linear from `zero`.
inline float OctaveLerp(const float* t, float x, float zero) {
    constexpr float kLow = 2.3283064365386963e-10f;  // 2^-32
    if (x < kLow) return zero + (t[0] - zero) * (std::max(x, 0.f) * 4294967296.f);
    const float c = std::min(std::max(x, kLow), 1.f);
    uint32_t b;
    std::memcpy(&b, &c, sizeof b);
    const int i = std::min((static_cast<int>(b >> 23) - 95) * 256 + static_cast<int>((b >> 15) & 255u), 32 * 256);
    return t[i] + (t[i + 1] - t[i]) * (static_cast<float>(b & 0x7FFFu) * (1.f / 32768.f));
}

inline float PqDec(const hdrrow::Step& g, float e) {
    const float x = std::clamp(e, 0.f, g.iw) * g.decScale;
    const int i = std::max(std::min(static_cast<int>(x), 4095), 0);
    return g.dec[i] + (g.dec[i + 1] - g.dec[i]) * (x - static_cast<float>(i));
}

inline float PseudoHue(float ct, float cp) {  // base + num / den: the four quadrants of the diamond
    float base, num, den;
    if (cp >= 0.f) {
        if (ct >= 0.f) base = 0.f, num = cp, den = ct + cp;
        else base = 1.f, num = -ct, den = cp - ct;
    } else {
        if (ct < 0.f) base = 2.f, num = -cp, den = -ct - cp;
        else base = 3.f, num = ct, den = ct - cp;
    }
    return base + num / den;
}

// I of the ring point of the cusp axis at relative pa q, and the row coordinate u of (I, q): shared by both tables.
struct AxisAt {
    int g0, g1;
    float tg, u;
};
inline AxisAt AxisLookup(const hdrrow::Step& g, const hdrrow::Axis& a, float I, float q) {
    int s = 0;
    for (int i = 1; i < 6; ++i) s += q >= a.vertex[i] ? 1 : 0;
    const float fg = static_cast<float>(a.start[s]) + (q - a.vertex[s]) * a.scale[s];
    AxisAt r;
    r.g0 = std::max(std::min(static_cast<int>(fg), a.n - 1), 0);
    r.g1 = r.g0 + 1 == a.n ? 0 : r.g0 + 1;
    r.tg = fg - static_cast<float>(r.g0);
    const float ic = a.cuspI[r.g0] + (a.cuspI[r.g1] - a.cuspI[r.g0]) * r.tg;
    r.u = I <= ic ? (I - g.e0) / (ic - g.e0) * g.uc : g.uc + (I - ic) / (g.iw - ic) * g.ucRest;
    return r;
}

inline float RelativePa(const hdrrow::Axis& a, float pa) {
    const float q = pa - a.pa0;
    return q + (q < 0.f ? 4.f : 0.f);
}

inline float B709At(const hdrrow::Step& g, float I, float pa) {
    const hdrrow::Bounds709& b = g.b709;
    const float q = RelativePa(b.axis, pa);
    const AxisAt ax = AxisLookup(g, b.axis, I, q);
    const float fi = std::clamp(ax.u, 0.f, 1.f) * 127.f;
    const int i0 = std::max(std::min(static_cast<int>(fi), 126), 0);
    const float ti = fi - static_cast<float>(i0);
    float k[13];
    const float* k0 = b.knot + static_cast<size_t>(i0) * 12;
    for (int m = 0; m < 12; ++m) k[m] = k0[m] + (k0[m + 12] - k0[m]) * ti;  // the knots between the two rows
    k[12] = k[0] + 4.f;
    float dq = q - k[0];
    dq = dq + (dq < 0.f ? 4.f : 0.f);
    dq = dq - (dq >= 4.f ? 4.f : 0.f);
    const float qq = k[0] + dq;
    int c = 0;
    for (int m = 1; m < 12; ++m) c += qq >= k[m] ? 1 : 0;  // an empty gap is never taken
    const float fp = static_cast<float>(b.start[c]) + (qq - k[c]) * b.cols[c] / (k[c + 1] - k[c]);
    const int j0 = std::max(std::min(static_cast<int>(fp), b.n - 1), 0);
    const int j1 = j0 + 1 == b.n ? 0 : j0 + 1;
    const float tp = fp - static_cast<float>(j0);
    const float* r0 = b.node + static_cast<size_t>(i0) * b.n;
    const float* r1 = r0 + b.n;
    const float lo = r0[j0] + (r0[j1] - r0[j0]) * tp, hi = r1[j0] + (r1[j1] - r1[j0]) * tp;
    return lo + (hi - lo) * ti;
}

inline float B2020At(const hdrrow::Step& g, float I, float pa) {
    const hdrrow::Bounds2020& b = g.b2020;
    const float q = RelativePa(b.axis, pa);
    const AxisAt ax = AxisLookup(g, b.axis, I, q);
    const float fi = std::clamp(ax.u, 0.f, 1.f) * 127.f;
    const int i0 = std::max(std::min(static_cast<int>(fi), 126), 0);
    const float ti = fi - static_cast<float>(i0);
    const float* r0 = b.node + static_cast<size_t>(i0) * b.axis.n;
    const float* r1 = r0 + b.axis.n;
    const float lo = r0[ax.g0] + (r0[ax.g1] - r0[ax.g0]) * ax.tg, hi = r1[ax.g0] + (r1[ax.g1] - r1[ax.g0]) * ax.tg;
    return lo + (hi - lo) * ti;
}

}  // namespace hdrrow_detail

inline bool HdrGamut::Level1(const float t[3]) const {
    const float Y = (v_.w709[0] * t[0] + v_.w709[1] * t[1]) + v_.w709[2] * t[2];
    const float mn = std::min(t[0], std::min(t[1], t[2])), mx = std::max(t[0], std::max(t[1], t[2]));
    return mn >= 0.f && mx <= 1.f && Y - mn <= v_.tau * Y && mx - Y <= v_.tau * (1.f - Y);
}

inline void HdrGamut::Map(float l[3]) const {
    namespace d = hdrrow_detail;
    const hdrrow::Step& g = v_;
    if (l[0] == l[1] && l[1] == l[2]) return;  // a grey is itself bit for bit (U7)
    float t[3];
    for (int c = 0; c < 3; ++c) t[c] = (g.t[3 * c] * l[0] + g.t[3 * c + 1] * l[1]) + g.t[3 * c + 2] * l[2];
    if (Level1(t)) {
        l[0] = t[0], l[1] = t[1], l[2] = t[2];
        return;
    }
    const float L = d::OctaveLerp(g.enc, (g.mlms[0] * l[0] + g.mlms[1] * l[1]) + g.mlms[2] * l[2], g.e0);
    const float M = d::OctaveLerp(g.enc, (g.mlms[3] * l[0] + g.mlms[4] * l[1]) + g.mlms[5] * l[2], g.e0);
    const float S = d::OctaveLerp(g.enc, (g.mlms[6] * l[0] + g.mlms[7] * l[1]) + g.mlms[8] * l[2], g.e0);
    const float I = 0.5f * (L + M);
    const float ct = (g.ict[0] * L + g.ict[1] * M) + g.ict[2] * S, cp = (g.ict[3] * L + g.ict[4] * M) + g.ict[5] * S;
    const float C = std::sqrt(ct * ct + cp * cp);
    const float pa = C > 0.f ? d::PseudoHue(ct, cp) : 0.f;
    const float b7 = d::B709At(g, I, pa), kb = g.knee * b7;
    if (!(C > kb) || !(b7 > 0.f)) {  // level 2: inside the knee the colour is T l
        for (int c = 0; c < 3; ++c) l[c] = std::clamp(t[c], 0.f, 1.f);
        return;
    }
    const float b20 = std::max(d::B2020At(g, I, pa), b7);  // the floors are a safety net (§15.2 p. 6)
    const float a = g.kneeRest * b7, D = std::max(b20 - kb, a), dd = std::min(C - kb, D);
    const float q = (D * D - a * a) / (a * a * D * D);
    const float f = (kb + dd / std::sqrt(1.f + dd * dd * q)) / C;
    const float fct = f * ct, fcp = f * cp;
    const float Lo = d::PqDec(g, (I + g.ictInv[0] * fct) + g.ictInv[1] * fcp);
    const float Mo = d::PqDec(g, (I + g.ictInv[2] * fct) + g.ictInv[3] * fcp);
    const float So = d::PqDec(g, (I + g.ictInv[4] * fct) + g.ictInv[5] * fcp);
    for (int c = 0; c < 3; ++c) l[c] = std::clamp((g.a709[3 * c] * Lo + g.a709[3 * c + 1] * Mo) + g.a709[3 * c + 2] * So, 0.f, 1.f);
}

}  // namespace dlssvid
