#include "convert/HdrTransfer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include <immintrin.h>
#include <intrin.h>

#include "util/Error.h"

namespace dlssvid::hdr {

namespace {

// ST 2084 constants
constexpr double kM1 = 2610.0 / 16384.0, kM2 = 2523.0 / 4096.0 * 128.0, kC1 = 3424.0 / 4096.0, kC2 = 2413.0 / 4096.0 * 32.0, kC3 = 2392.0 / 4096.0 * 32.0;
// BT.2100 HLG constants
constexpr double kHlgA = 0.17883277, kHlgB = 1.0 - 4.0 * kHlgA;
const double kHlgC = 0.5 - kHlgA * std::log(4.0 * kHlgA);

struct EetfConstants {
    double masterPeak;  // PQ code of kPeakNits (the source's range, black 0)
    double maxLum;      // the display's peak in the normalised domain
    double ks;
};

const EetfConstants& Eetf() {
    static const EetfConstants c = [] {
        EetfConstants e;
        e.masterPeak = NitsToPq(kPeakNits);
        e.maxLum = NitsToPq(kWhiteNits) / e.masterPeak;
        e.ks = 1.5 * e.maxLum - 0.5;
        return e;
    }();
    return c;
}

using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 Inverse(const Mat3& m) {
    const double a = m[0][0], b = m[0][1], c = m[0][2], d = m[1][0], e = m[1][1], f = m[1][2], g = m[2][0], h = m[2][1], i = m[2][2];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return Mat3{{{(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det},
                 {(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det},
                 {(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det}}};
}

// RGB -> XYZ of primaries (x, y) and D65 (SMPTE RP 177)
Mat3 Npm(const double xy[3][2]) {
    constexpr double wx = 0.3127, wy = 0.3290;
    Mat3 p{};
    for (int j = 0; j < 3; ++j) {
        const double x = xy[j][0], y = xy[j][1];
        p[0][j] = x / y;
        p[1][j] = 1.0;
        p[2][j] = (1.0 - x - y) / y;
    }
    const double w[3] = {wx / wy, 1.0, (1.0 - wx - wy) / wy};
    const Mat3 inv = Inverse(p);
    double s[3];
    for (int i = 0; i < 3; ++i) s[i] = inv[i][0] * w[0] + inv[i][1] * w[1] + inv[i][2] * w[2];
    Mat3 m{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i][j] = p[i][j] * s[j];
    return m;
}

constexpr double kP709[3][2] = {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}};
constexpr double kP2020[3][2] = {{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}};

}  // namespace

double PqEncode(double y) {
    const double ym = std::pow(std::max(y, 0.0), kM1);
    return std::pow((kC1 + kC2 * ym) / (1.0 + kC3 * ym), kM2);
}

double PqDecode(double e) {
    const double p = std::pow(std::max(e, 0.0), 1.0 / kM2);
    return std::pow(std::max(p - kC1, 0.0) / (kC2 - kC3 * p), 1.0 / kM1);
}

double PqDecodeSlope(double e) {
    if (e <= 0.0) return 0.0;
    const double p = std::pow(e, 1.0 / kM2), n = std::max(p - kC1, 0.0), d = kC2 - kC3 * p;
    if (n <= 0.0) return 0.0;
    return (1.0 / kM1) * std::pow(n / d, 1.0 / kM1 - 1.0) * (kC2 - kC3 * kC1) / (d * d) * (1.0 / kM2) * p / e;
}

double PqToNits(double code) { return 10000.0 * PqDecode(std::clamp(code, 0.0, 1.0)); }

double NitsToPq(double nits) { return PqEncode(std::max(nits, 0.0) / 10000.0); }

double HlgToScene(double code) {
    const double e = std::clamp(code, 0.0, 1.0);
    return e <= 0.5 ? e * e / 3.0 : (std::exp((e - kHlgC) / kHlgA) + kHlgB) / 12.0;
}

double HlgOotfGain(double ys) { return kPeakNits * std::pow(std::max(ys, 0.0), kHlgGamma - 1.0); }

double Eetf2390(double pqCode) {
    const EetfConstants& c = Eetf();
    double e = std::clamp(pqCode / c.masterPeak, 0.0, 1.0);  // above the master's peak: the display's peak
    if (e >= c.ks) {
        const double t = (e - c.ks) / (1.0 - c.ks), t2 = t * t, t3 = t2 * t;
        e = (2.0 * t3 - 3.0 * t2 + 1.0) * c.ks + (t3 - 2.0 * t2 + t) * (1.0 - c.ks) + (-2.0 * t3 + 3.0 * t2) * c.maxLum;
    }
    return e * c.masterPeak;  // black 0: no black-level lift (b = 0)
}

double EetfKs() { return Eetf().ks; }

double EetfKneeCode() { return Eetf().ks * Eetf().masterPeak; }

const std::array<double, 9>& Bt2020To709Matrix() {
    static const std::array<double, 9> m = [] {
        const Mat3 a = Inverse(Npm(kP709)), b = Npm(kP2020);
        std::array<double, 9> r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r[3 * i + j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        return r;
    }();
    return m;
}

void Bt2020To709(const double in[3], double out[3]) {
    const std::array<double, 9>& m = Bt2020To709Matrix();
    const double x0 = in[0], x1 = in[1], x2 = in[2];
    for (int i = 0; i < 3; ++i) out[i] = m[3 * i] * x0 + m[3 * i + 1] * x1 + m[3 * i + 2] * x2;
}

void LuminanceWeights(Gamut gamut, double w[3]) {
    const Mat3 m = Npm(gamut == Gamut::Bt2020 ? kP2020 : kP709);
    for (int j = 0; j < 3; ++j) w[j] = m[1][j];
}

double Bt1886Inverse(double relative) { return std::pow(std::max(relative, 0.0), 1.0 / kSdrGamma); }

double GreyToSdr(Curve curve, double code) {
    double pq = 0.0;
    if (curve == Curve::Pq) pq = std::clamp(code, 0.0, 1.0);
    else if (curve == Curve::Hlg) {
        const double e = HlgToScene(code);
        pq = NitsToPq(HlgOotfGain(e) * e);
    } else Throw("hdr::GreyToSdr: an SDR curve has no transfer");
    return std::min(Bt1886Inverse(PqToNits(Eetf2390(pq)) / kWhiteNits), 1.0);
}

void LinearOfCodes(Curve curve, Gamut gamut, const double codes[3], double l[3]) {
    double nits[3];
    if (curve == Curve::Pq) {
        for (int k = 0; k < 3; ++k) nits[k] = PqToNits(codes[k]);
    } else if (curve == Curve::Hlg) {
        double e[3], w[3];
        for (int k = 0; k < 3; ++k) e[k] = HlgToScene(codes[k]);
        LuminanceWeights(gamut, w);
        const double ys = e[0] * w[0] + e[1] * w[1] + e[2] * w[2];
        for (int k = 0; k < 3; ++k) nits[k] = HlgOotfGain(ys) * e[k];
    } else {
        Throw("hdr::LinearOfCodes: an SDR curve has no transfer");
    }
    const double m = std::max(nits[0], std::max(nits[1], nits[2]));
    const double gain = m > 0.0 ? PqToNits(Eetf2390(NitsToPq(m))) / m : 0.0;
    for (int k = 0; k < 3; ++k) l[k] = std::min(nits[k] * gain / kWhiteNits, 1.0);
}

void HdrToSdr(Curve curve, Gamut gamut, const double codes[3], double out[3]) {
    double l[3], l709[3];
    LinearOfCodes(curve, gamut, codes, l);
    if (gamut == Gamut::Bt2020) GamutMap709(l, l709);
    else std::copy(l, l + 3, l709);
    for (int k = 0; k < 3; ++k) out[k] = Bt1886Inverse(std::clamp(l709[k], 0.0, 1.0));
}

}  // namespace dlssvid::hdr

namespace dlssvid {

const std::vector<float>& GreyTable(Curve curve, int bitDepth, ColorRange range) {
    if (curve == Curve::Sdr) Throw("GreyTable: an SDR curve has no transfer");
    if (bitDepth < 8 || bitDepth > 16) Throw("GreyTable: bit depth " + std::to_string(bitDepth));
    static std::mutex mutex;
    static std::map<std::tuple<int, int, int>, std::vector<float>> tables;
    const std::lock_guard<std::mutex> lock(mutex);
    auto& t = tables[{static_cast<int>(curve), bitDepth, static_cast<int>(range)}];
    if (t.empty()) {
        const double scale = std::ldexp(1.0, bitDepth - 8);
        const bool limited = range == ColorRange::Limited;
        const double lo = limited ? 16.0 * scale : 0.0, span = limited ? 219.0 * scale : std::ldexp(1.0, bitDepth) - 1.0;
        const size_t n = size_t{1} << bitDepth;
        t.resize(n);
        for (size_t code = 0; code < n; ++code)
            t[code] = static_cast<float>(hdr::GreyToSdr(curve, std::clamp((static_cast<double>(code) - lo) / span, 0.0, 1.0)));
    }
    return t;
}

HdrTransfer::HdrTransfer(Curve curve, Gamut gamut, const HdrGamut* gamutMap) : curve_(curve), gamut_(gamut), gamutMap_(gamutMap) {
    if (curve == Curve::Sdr) Throw("HdrTransfer: an SDR curve has no transfer");
    eotf_.resize(kPoints + 1);
    gain_.resize(kPoints + 1);
    oetf_.resize(kPoints + 1);
    const bool hlg = curve == Curve::Hlg;
    const double knee = hdr::EetfKneeCode();
    for (int i = 0; i <= kPoints; ++i) {
        const double x = static_cast<double>(i) / kPoints;
        oetf_[static_cast<size_t>(i)] = static_cast<float>(hdr::Bt1886Inverse(x * x));
        if (!hlg) {
            eotf_[static_cast<size_t>(i)] = static_cast<float>(hdr::PqToNits(x) / hdr::kWhiteNits);
            gain_[static_cast<size_t>(i)] = static_cast<float>(x <= knee ? 1.0 : hdr::PqToNits(hdr::Eetf2390(x)) / hdr::PqToNits(x));
        } else {
            eotf_[static_cast<size_t>(i)] = static_cast<float>(hdr::HlgToScene(x));
            const double nits = x * hdr::kPeakNits, code = hdr::NitsToPq(nits);  // the table runs over the display light 0..1000 nits
            gain_[static_cast<size_t>(i)] = static_cast<float>(code <= knee ? 1.0 : hdr::PqToNits(hdr::Eetf2390(code)) / nits);
        }
    }
    v_.hlg = hlg ? 1 : 0;
    v_.eotf = eotf_.data();
    v_.gain = gain_.data();
    v_.oetf = oetf_.data();
    if (hlg) {
        ys02_.resize(32 * 256 + 2);
        for (size_t n = 0; n < ys02_.size(); ++n) {
            const double x = std::ldexp(1.0 + static_cast<double>(n % 256) / 256.0, static_cast<int>(n / 256) - 32);
            ys02_[n] = static_cast<float>(std::pow(std::min(x, 2.0), hdr::kHlgGamma - 1.0));
        }
        v_.ys02 = ys02_.data();
        double w[3];
        hdr::LuminanceWeights(gamut, w);
        for (int k = 0; k < 3; ++k) v_.w[k] = static_cast<float>(w[k]);
        v_.hlgScale = static_cast<float>(hdr::kPeakNits / hdr::kWhiteNits);
        v_.gainScale = static_cast<float>(hdr::kWhiteNits / hdr::kPeakNits);
    }
    v_.gamut = gamutMap_ ? &gamutMap_->View() : nullptr;
}

const HdrTransfer& HdrTransfer::For(Curve curve, Gamut gamut) {
    if (curve == Curve::Pq) {
        if (gamut == Gamut::Bt709) {
            static const HdrTransfer t(Curve::Pq, Gamut::Bt709, nullptr);
            return t;
        }
        static const HdrTransfer t(Curve::Pq, Gamut::Bt2020, &HdrGamut::Get());
        return t;
    }
    if (curve == Curve::Hlg) {
        if (gamut == Gamut::Bt709) {
            static const HdrTransfer t(Curve::Hlg, Gamut::Bt709, nullptr);
            return t;
        }
        static const HdrTransfer t(Curve::Hlg, Gamut::Bt2020, &HdrGamut::Get());
        return t;
    }
    Throw("HdrTransfer: an SDR curve has no transfer");
}

namespace {
std::atomic<int> g_kernel{static_cast<int>(HdrKernel::Auto)};
}  // namespace

void SetHdrKernel(HdrKernel kernel) { g_kernel.store(static_cast<int>(kernel)); }

bool HdrRowsUseAvx2() { return g_kernel.load() == static_cast<int>(HdrKernel::Auto) && hdrrow::HasAvx2(); }

}  // namespace dlssvid

namespace dlssvid::hdrrow {

// Here, not in HdrRowAvx2.cpp: that file is built with /arch:AVX2 and may use AVX in any function, this test included.
bool HasAvx2() {
    static const bool has = [] {
        int r[4] = {0, 0, 0, 0};
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const bool osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
        if (!osxsave || !avx) return false;
        if ((_xgetbv(0) & 6u) != 6u) return false;  // the OS saves the XMM and YMM state
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;
    }();
    return has;
}

}  // namespace dlssvid::hdrrow
