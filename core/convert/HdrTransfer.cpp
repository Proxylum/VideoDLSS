#include "convert/HdrTransfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

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

const Mat3& Bt2020To709Matrix() {
    static const Mat3 m = [] {
        const double p709[3][2] = {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}};
        const double p2020[3][2] = {{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}};
        const Mat3 a = Inverse(Npm(p709)), b = Npm(p2020);
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        return r;
    }();
    return m;
}

}  // namespace

double PqToNits(double code) {
    const double p = std::pow(std::clamp(code, 0.0, 1.0), 1.0 / kM2);
    return 10000.0 * std::pow(std::max(p - kC1, 0.0) / (kC2 - kC3 * p), 1.0 / kM1);
}

double NitsToPq(double nits) {
    const double y = std::pow(std::max(nits, 0.0) / 10000.0, kM1);
    return std::pow((kC1 + kC2 * y) / (1.0 + kC3 * y), kM2);
}

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

void Bt2020To709(const double in[3], double out[3]) {
    const Mat3& m = Bt2020To709Matrix();
    for (int i = 0; i < 3; ++i) out[i] = m[i][0] * in[0] + m[i][1] * in[1] + m[i][2] * in[2];
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

}  // namespace dlssvid
