#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "convert/ColorConvert.h"
#include "convert/HdrGamut.h"
#include "convert/HdrRow.h"

// TASK-0080 (HDR step TC): the HDR -> SDR transfer of the SDR reading of a PQ or HLG source (Output/hdr-design, «вариант Б»): the codes'
// light (PQ EOTF; HLG inverse OETF and OOTF at 1000 nits, gamma 1.2), the BT.2390 EETF of a 1000-nit master to SDR white 203 nits = 1.0
// on maxRGB, the gamut step G1 BT.2020 -> BT.709 (HdrGamut.h; a BT.709 source skips it), the BT.1886 inverse (gamma 2.4). `hdr::` is the
// double reference - tests pin it to absolute numbers, not to each other; the tables a frame goes through are built from it.
namespace dlssvid::hdr {

inline constexpr double kPeakNits = 1000.0;   // the master's peak the EETF takes (and HLG's display peak)
inline constexpr double kWhiteNits = 203.0;   // SDR white: 1.0 of the result (BT.2408 reference white)
inline constexpr double kHlgGamma = 1.2;      // the HLG OOTF's system gamma at 1000 nits
inline constexpr double kSdrGamma = 2.4;      // BT.1886

// SMPTE ST 2084 on normalised light (1.0 = 10000 cd/m2): the inverse EOTF (negative light is black), the EOTF (codes below PQ(0) are
// black; the code is not clamped) and its derivative.
double PqEncode(double y);
double PqDecode(double e);
double PqDecodeSlope(double e);
// SMPTE ST 2084: a code E' in [0, 1] -> cd/m2 (0..10000), and back (negative light is black).
double PqToNits(double code);
double NitsToPq(double nits);
// BT.2100 HLG inverse OETF: a code E' in [0, 1] -> scene light E in [0, 1].
double HlgToScene(double code);
// BT.2100 HLG OOTF gain at 1000 nits: display light (cd/m2) of a channel = gain(Ys) * E, Ys the scene luminance by the primaries' weights.
double HlgOotfGain(double ys);
// The BT.2390 EETF of a PQ code of a master graded to kPeakNits onto a display whose peak is kWhiteNits, black 0: the PQ code out. A code
// above the master's peak gives the display's peak.
double Eetf2390(double pqCode);
double EetfKs();         // the knee in the EETF's normalised domain: 1.5 * maxLum - 0.5
double EetfKneeCode();   // the knee as a PQ code: below it the EETF is the identity
// BT.2020 -> BT.709 linear RGB, derived from the primaries and D65 (rows sum to 1); row-major.
const std::array<double, 9>& Bt2020To709Matrix();
void Bt2020To709(const double in[3], double out[3]);
// Y of RGB -> XYZ of the primaries and D65: the HLG OOTF's Ys weights (BT.709 0.212639 0.715169 0.072192, BT.2020 0.262700 0.677998
// 0.059302).
void LuminanceWeights(Gamut gamut, double w[3]);
// BT.1886 inverse: relative light (1.0 = kWhiteNits) -> the SDR signal L^(1/2.4); negative light is 0.
double Bt1886Inverse(double relative);
// A grey's way through the transfer (R = G = B: no gamut step, Ys = E): a code of `curve` (Pq or Hlg) -> the SDR signal.
double GreyToSdr(Curve curve, double code);
// R'G'B' codes of `curve` (clamped to [0, 1]) -> linear light of the source's primaries after the EETF on maxRGB, 1.0 = 203 nits, at
// most 1: the input of the gamut step.
void LinearOfCodes(Curve curve, Gamut gamut, const double codes[3], double l[3]);
// The whole transfer: codes -> the SDR signal R'G'B' of BT.709 in [0, 1] (Gamut::Bt709 without the gamut step).
void HdrToSdr(Curve curve, Gamut gamut, const double codes[3], double out[3]);

}  // namespace dlssvid::hdr

namespace dlssvid {

// TASK-0080: the SDR signal of every integer luma code of a `curve` source at `bitDepth` and `range` (limited: below the black code 0,
// above the white code clamped) - the scene-cut thumbnails of a PQ / HLG source read their luma through it before the box mean. Built
// once per process for each (curve, depth, range); Curve::Sdr throws.
const std::vector<float>& GreyTable(Curve curve, int bitDepth, ColorRange range);

// The float transfer a frame goes through (tc-plan §3.1, §15.5): tables of 4096 segments with linear interpolation, built from `hdr::`.
// One per (curve, gamut) and process, shared by every thread.
class HdrTransfer {
public:
    static constexpr int kPoints = 4096;
    static const HdrTransfer& For(Curve curve, Gamut gamut);  // Curve::Sdr throws
    HdrTransfer(Curve curve, Gamut gamut, const HdrGamut* gamutMap);
    HdrTransfer(const HdrTransfer&) = delete;
    HdrTransfer& operator=(const HdrTransfer&) = delete;

    // Codes in [0, 1] -> linear light after the EETF, 1.0 = 203 nits, at most 1 (the input of the gamut step). Open for U6.
    inline void Linear(float r, float g, float b, float l[3]) const;
    // Codes in [0, 1] -> the SDR signal of BT.709 in [0, 1].
    inline void Apply(float r, float g, float b, float out[3]) const;
    Curve curve() const { return curve_; }
    Gamut gamut() const { return gamut_; }
    const hdrrow::Chain& View() const { return v_; }

private:
    Curve curve_;
    Gamut gamut_;
    std::vector<float> eotf_, gain_, ys02_, oetf_;
    const HdrGamut* gamutMap_ = nullptr;
    hdrrow::Chain v_{};
};

// Which row a frame of an HDR source goes through: Auto takes the AVX2 row when the CPU runs it, Portable always the portable one. Both
// give the same bits; the switch is for the tests and the bench (B1).
enum class HdrKernel : uint8_t { Auto, Portable };
void SetHdrKernel(HdrKernel kernel);
bool HdrRowsUseAvx2();

inline void HdrTransfer::Linear(float r, float g, float b, float l[3]) const {
    namespace d = hdrrow_detail;
    const float* e = v_.eotf;
    float L[3], G;
    if (!v_.hlg) {
        L[0] = d::Lerp(e, kPoints, r), L[1] = d::Lerp(e, kPoints, g), L[2] = d::Lerp(e, kPoints, b);
        G = d::Lerp(v_.gain, kPoints, std::max(r, std::max(g, b)));  // the PQ code of maxRGB
    } else {
        const float E0 = d::Lerp(e, kPoints, r), E1 = d::Lerp(e, kPoints, g), E2 = d::Lerp(e, kPoints, b);
        const float ys = (v_.w[0] * E0 + v_.w[1] * E1) + v_.w[2] * E2;
        const float k = v_.hlgScale * d::OctaveLerp(v_.ys02, ys, 0.f);  // the OOTF: (1000 / 203) Ys^0.2
        L[0] = k * E0, L[1] = k * E1, L[2] = k * E2;
        G = d::Lerp(v_.gain, kPoints, std::min(std::max(L[0], std::max(L[1], L[2])) * v_.gainScale, 1.f));
    }
    for (int c = 0; c < 3; ++c) l[c] = std::min(L[c] * G, 1.f);
}

inline void HdrTransfer::Apply(float r, float g, float b, float out[3]) const {
    float l[3];
    Linear(r, g, b, l);
    if (gamutMap_) gamutMap_->Map(l);  // Gamut::Bt2020 only
    for (int c = 0; c < 3; ++c) out[c] = hdrrow_detail::Lerp(v_.oetf, kPoints, std::sqrt(std::clamp(l[c], 0.f, 1.f)));
}

}  // namespace dlssvid
