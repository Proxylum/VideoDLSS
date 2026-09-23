#include "io/EncodeDefaults.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iterator>

namespace dlssvid {

namespace {

// H.264 at 30 fps: (megapixels, Mbit/s) for 720p, 1080p, 1440p, 4K and 8K; linear between the points, proportional
// beyond the ends.
constexpr double kH264At30[][2] = {{0.9216, 10.0}, {2.0736, 16.0}, {3.6864, 24.0}, {8.2944, 45.0}, {33.1776, 160.0}};

double H264At30(double megapixels) {
    const size_t n = std::size(kH264At30);
    if (megapixels <= kH264At30[0][0]) return kH264At30[0][1] * megapixels / kH264At30[0][0];
    for (size_t i = 1; i < n; ++i) {
        if (megapixels <= kH264At30[i][0]) {
            const double t = (megapixels - kH264At30[i - 1][0]) / (kH264At30[i][0] - kH264At30[i - 1][0]);
            return kH264At30[i - 1][1] + t * (kH264At30[i][1] - kH264At30[i - 1][1]);
        }
    }
    return kH264At30[n - 1][1] * megapixels / kH264At30[n - 1][0];
}

double FpsFactor(double fps) {
    if (fps <= 30.0) return 1.0;
    if (fps <= 60.0) return 1.0 + 0.5 * (fps - 30.0) / 30.0;
    return 1.5 * std::sqrt(fps / 60.0);
}

double CodecFactor(const std::string& codec) {
    if (codec.rfind("hevc", 0) == 0 || codec.rfind("h265", 0) == 0) return 0.65;
    if (codec.rfind("av1", 0) == 0) return 0.55;
    return 1.0;
}

// A number people would type: halves below 10, whole Mbit/s below 40, fives above.
double Nice(double mbps) {
    if (mbps < 10.0) return std::round(mbps * 2.0) / 2.0;
    if (mbps < 40.0) return std::round(mbps);
    return std::round(mbps / 5.0) * 5.0;
}

}  // namespace

bool LossyCodec(const std::string& codec) { return codec != "ffv1"; }

double RecommendedBitrateMbps(const std::string& codec, uint32_t width, uint32_t height, double fps) {
    if (!LossyCodec(codec) || width == 0 || height == 0) return 0.0;
    const double megapixels = static_cast<double>(width) * static_cast<double>(height) / 1e6;
    const double base = H264At30(megapixels) * FpsFactor(fps > 0.0 ? fps : 30.0) * CodecFactor(codec);
    return std::max(1.0, Nice(base));
}

std::optional<double> ParseBitrateMbps(const std::string& ffmpegValue) {
    std::string s = ffmpegValue;
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base(), s.end());
    if (s.empty()) return std::nullopt;
    char* end = nullptr;
    const double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || value < 0.0 || !std::isfinite(value)) return std::nullopt;
    std::string suffix(end);
    double scale = 1.0;  // bits per second
    if (suffix == "k" || suffix == "K") scale = 1e3;
    else if (suffix == "m" || suffix == "M") scale = 1e6;
    else if (suffix == "g" || suffix == "G") scale = 1e9;
    else if (!suffix.empty()) return std::nullopt;
    return value * scale / 1e6;
}

std::string BitrateOption(double mbps) {
    if (mbps <= 0.0) return {};
    const double rounded = std::round(mbps);
    if (std::fabs(mbps - rounded) < 1e-9) return std::to_string(static_cast<long long>(rounded)) + "M";
    return std::to_string(static_cast<long long>(std::llround(mbps * 1000.0))) + "k";
}

std::map<std::string, std::string> EffectiveCodecOptions(const std::string& codec, const std::map<std::string, std::string>& options, uint32_t width,
                                                         uint32_t height, double fps) {
    std::map<std::string, std::string> out = options;
    if (!LossyCodec(codec) || out.count("b")) return out;
    const double mbps = RecommendedBitrateMbps(codec, width, height, fps);
    if (mbps > 0.0) out["b"] = BitrateOption(mbps);
    return out;
}

}  // namespace dlssvid
