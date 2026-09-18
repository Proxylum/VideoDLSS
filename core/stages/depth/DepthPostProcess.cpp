#include "stages/depth/DepthPostProcess.h"

#include <cmath>
#include <limits>
#include <queue>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {
bool Valid(float v) { return std::isfinite(v) && v > 0.f; }
}  // namespace

ScaleShift SolveScaleShift(const PassImage& depth, const PassImage& reference, bool shiftAllowed) {
    if (depth.width != reference.width || depth.height != reference.height) Throw("SolveScaleShift: size mismatch");
    // Least squares for r = s*d + t: minimise sum (s d + t - r)^2.
    double sd = 0, sr = 0, sdd = 0, sdr = 0;
    size_t n = 0;
    for (uint32_t y = 0; y < depth.height; ++y)
        for (uint32_t x = 0; x < depth.width; ++x) {
            const float d = depth.Get(x, y, 0), r = reference.Get(x, y, 0);
            if (!Valid(d) || !Valid(r)) continue;
            sd += d;
            sr += r;
            sdd += static_cast<double>(d) * d;
            sdr += static_cast<double>(d) * r;
            ++n;
        }
    ScaleShift ss;
    ss.samples = n;
    if (n < 2) return ss;
    if (shiftAllowed) {
        const double den = n * sdd - sd * sd;
        if (std::fabs(den) < 1e-12) return ss;
        ss.scale = static_cast<float>((n * sdr - sd * sr) / den);
        ss.shift = static_cast<float>((sr - ss.scale * sd) / n);
    } else {
        if (sdd <= 0) return ss;
        ss.scale = static_cast<float>(sdr / sdd);
        ss.shift = 0.f;
    }
    if (!std::isfinite(ss.scale) || ss.scale <= 0.f) {
        ss.scale = 1.f;
        ss.shift = 0.f;
    }
    return ss;
}

void ApplyScaleShift(PassImage& depth, const ScaleShift& ss) {
    for (uint32_t y = 0; y < depth.height; ++y)
        for (uint32_t x = 0; x < depth.width; ++x) {
            const float d = depth.Get(x, y, 0);
            if (Valid(d)) depth.Set(x, y, 0, d * ss.scale + ss.shift);
        }
}

size_t FillInvalidDepth(PassImage& depth) {
    const uint32_t w = depth.width, h = depth.height;
    std::vector<uint8_t> valid(static_cast<size_t>(w) * h);
    std::queue<size_t> q;
    size_t invalid = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            valid[i] = Valid(depth.Get(x, y, 0));
            if (valid[i]) q.push(i);
            else ++invalid;
        }
    if (invalid == 0) return 0;
    if (q.empty()) {  // nothing valid at all
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) depth.Set(x, y, 0, 1.f);
        return invalid;
    }
    const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        const size_t i = q.front();
        q.pop();
        const long x = static_cast<long>(i % w), y = static_cast<long>(i / w);
        const float v = depth.Get(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0);
        for (int k = 0; k < 4; ++k) {
            const long nx = x + dx[k], ny = y + dy[k];
            if (nx < 0 || ny < 0 || nx >= static_cast<long>(w) || ny >= static_cast<long>(h)) continue;
            const size_t ni = static_cast<size_t>(ny) * w + static_cast<size_t>(nx);
            if (valid[ni]) continue;
            depth.Set(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), 0, v);
            valid[ni] = 1;
            q.push(ni);
        }
    }
    return invalid;
}

double TemporalAlignmentError(const PassImage& prev, const PassImage& cur) {
    if (prev.width != cur.width || prev.height != cur.height) Throw("TemporalAlignmentError: size mismatch");
    double diff = 0, mean = 0;
    size_t n = 0;
    for (uint32_t y = 0; y < cur.height; ++y)
        for (uint32_t x = 0; x < cur.width; ++x) {
            const float a = prev.Get(x, y, 0), b = cur.Get(x, y, 0);
            if (!Valid(a) || !Valid(b)) continue;
            diff += std::fabs(static_cast<double>(b) - a);
            mean += a;
            ++n;
        }
    if (n == 0 || mean <= 0) return 0.0;
    return (diff / n) / (mean / n);
}

ScaleShift TemporalStabilizer::Stabilize(PassImage& depth) {
    ScaleShift ss;
    if (mode_ == Mode::None || window_ <= 0) return ss;
    if (depth.type != PixelType::F32) depth = depth.ConvertTo(PixelType::F32);
    if (!history_.empty() && (history_.front().width != depth.width || history_.front().height != depth.height)) Reset();
    if (!history_.empty()) {
        // reference = mean of the window
        PassImage ref;
        ref.Allocate(depth.width, depth.height, PixelType::F32, {"Z"});
        const float* sum = referenceSum_.As<float>();
        float* r = ref.As<float>();
        const float inv = 1.f / static_cast<float>(history_.size());
        for (size_t i = 0; i < ref.data.size() / 4; ++i) r[i] = sum[i] * inv;
        ss = SolveScaleShift(depth, ref, mode_ == Mode::ScaleShift);
        ApplyScaleShift(depth, ss);
    } else {
        referenceSum_.Allocate(depth.width, depth.height, PixelType::F32, {"Z"});
    }
    // push
    history_.push_back(depth);
    float* sum = referenceSum_.As<float>();
    const float* d = depth.As<float>();
    for (size_t i = 0; i < depth.data.size() / 4; ++i) sum[i] += Valid(d[i]) ? d[i] : 0.f;
    if (static_cast<int>(history_.size()) > window_) {
        const float* old = history_.front().As<float>();
        for (size_t i = 0; i < depth.data.size() / 4; ++i) sum[i] -= Valid(old[i]) ? old[i] : 0.f;
        history_.pop_front();
    }
    return ss;
}

}  // namespace dlssvid
