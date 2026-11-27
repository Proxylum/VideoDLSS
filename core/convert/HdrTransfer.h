#pragma once

#include <vector>

#include "convert/ColorConvert.h"

// TASK-0080 (HDR step TC): the HDR -> SDR transfer of the SDR reading of a PQ or HLG source (Output/hdr-design, «вариант Б»): the codes'
// light (PQ EOTF; HLG inverse OETF and OOTF at 1000 nits, gamma 1.2), the BT.2390 EETF of a 1000-nit master to SDR white 203 nits = 1.0
// on maxRGB, BT.2020 -> BT.709, the BT.1886 inverse (gamma 2.4). `hdr::` is the double reference - tests pin it to absolute numbers,
// not to each other; the tables a frame goes through are built from it.
namespace dlssvid::hdr {

inline constexpr double kPeakNits = 1000.0;   // the master's peak the EETF takes (and HLG's display peak)
inline constexpr double kWhiteNits = 203.0;   // SDR white: 1.0 of the result (BT.2408 reference white)
inline constexpr double kHlgGamma = 1.2;      // the HLG OOTF's system gamma at 1000 nits
inline constexpr double kSdrGamma = 2.4;      // BT.1886

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
// BT.2020 -> BT.709 linear RGB, derived from the primaries and D65 (rows sum to 1).
void Bt2020To709(const double in[3], double out[3]);
// BT.1886 inverse: relative light (1.0 = kWhiteNits) -> the SDR signal L^(1/2.4); negative light is 0.
double Bt1886Inverse(double relative);
// A grey's way through the transfer (R = G = B: no gamut step, Ys = E): a code of `curve` (Pq or Hlg) -> the SDR signal.
double GreyToSdr(Curve curve, double code);

}  // namespace dlssvid::hdr

namespace dlssvid {

// TASK-0080: the SDR signal of every integer luma code of a `curve` source at `bitDepth` and `range` (limited: below the black code 0,
// above the white code clamped) - the scene-cut thumbnails of a PQ / HLG source read their luma through it before the box mean. Built
// once per process for each (curve, depth, range); Curve::Sdr throws.
const std::vector<float>& GreyTable(Curve curve, int bitDepth, ColorRange range);

}  // namespace dlssvid
