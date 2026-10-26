#include "convert/SceneCut.h"

#include <algorithm>
#include <cmath>

namespace dlssvid {

LumaThumb MakeLumaThumb(const CpuFrame& frame, const ColorInfo& info, uint32_t maxWidth) {
    LumaThumb t;
    const FrameDesc& d = frame.desc;
    if (frame.data.size() < d.ByteSize() || d.width < 2 || d.height < 2 || maxWidth < 2) return t;
    const uint32_t f = std::max(1u, (d.width + maxWidth - 1) / maxWidth);  // an integer box: every sample of a block counts once
    t.width = std::max(2u, (d.width / f) & ~1u);
    t.height = std::max(2u, (d.height / f) & ~1u);
    t.frameWidth = d.width;
    t.frameHeight = d.height;
    const int bits = d.HighBitDepth() ? d.bitDepth : 8;
    const double scale = std::ldexp(1.0, bits - 8);
    const bool limited = info.range == ColorRange::Limited;
    const double lo = limited ? 16.0 * scale : 0.0, span = limited ? 219.0 * scale : std::ldexp(1.0, bits) - 1.0;
    const size_t cols = d.PlaneCols(0);
    const uint8_t* y8 = frame.Plane(0);
    const uint16_t* y16 = d.HighBitDepth() ? frame.Plane16(0) : nullptr;
    t.y.resize(static_cast<size_t>(t.width) * t.height);
    const double area = static_cast<double>(f) * f;
    for (uint32_t ty = 0; ty < t.height; ++ty)
        for (uint32_t tx = 0; tx < t.width; ++tx) {
            double sum = 0.0;
            for (uint32_t r = ty * f; r < (ty + 1) * f && r < d.height; ++r) {
                const size_t row = static_cast<size_t>(r) * cols;
                for (uint32_t c = tx * f; c < (tx + 1) * f && c < d.width; ++c) sum += y16 ? y16[row + c] : y8[row + c];
            }
            t.y[static_cast<size_t>(ty) * t.width + tx] = static_cast<float>(std::clamp((sum / area - lo) / span, 0.0, 1.0));
        }
    return t;
}

double SceneCutScore(const LumaThumb& a, const LumaThumb& b) {
    if (a.frameWidth != b.frameWidth || a.frameHeight != b.frameHeight || a.y.size() != b.y.size() || a.Empty()) return 1.0;
    double sum = 0.0;
    for (size_t i = 0; i < a.y.size(); ++i) sum += std::fabs(static_cast<double>(a.y[i]) - b.y[i]);
    return sum / static_cast<double>(a.y.size());
}

double SceneCutDetector::Push(const CpuFrame& frame, bool& compared) {
    compared = false;
    LumaThumb t = MakeLumaThumb(frame, info_, maxWidth_);
    if (t.Empty()) return 0.0;
    compared = !prev_.Empty();
    const double s = compared ? SceneCutScore(prev_, t) : 0.0;
    prev_ = std::move(t);
    return s;
}

double SceneCutDetector::Score(const CpuFrame& frame) {
    bool compared = false;
    return Push(frame, compared);
}

bool SceneCutDetector::Next(const CpuFrame& frame, double threshold, double* score) {
    bool compared = false;
    const double s = Push(frame, compared);
    if (score) *score = s;
    return compared && s > threshold;
}

}  // namespace dlssvid
