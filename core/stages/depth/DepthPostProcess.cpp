#include "stages/depth/DepthPostProcess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
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

// ---- TemporalStabilizer ------------------------------------------------------------------------

namespace {
// lower median of a non-empty vector (reorders it)
double LowerMedian(std::vector<double>& v) {
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>((v.size() - 1) / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}
}  // namespace

void TemporalStabilizer::Reset() {
    frames_.clear();
    track_.clear();
    prevLog_.clear();
    width_ = height_ = 0;
    step_ = 1;
    trackBase_ = pushedEnd_ = next_ = 0;
    pushedAny_ = closed_ = false;
}

void TemporalStabilizer::Measure(const PassImage& depth, Track& t) {
    bool restart = false;
    if (depth.width != width_ || depth.height != height_) {
        // the grid of samples is fixed per resolution; a new resolution starts a segment
        width_ = depth.width;
        height_ = depth.height;
        const double pixels = static_cast<double>(width_) * height_;
        step_ = static_cast<uint32_t>(std::max(1.0, std::ceil(std::sqrt(pixels / static_cast<double>(kMaxSamples)))));
        prevLog_.clear();
        restart = true;
    }
    std::vector<double> cur;
    cur.reserve(static_cast<size_t>((width_ + step_ - 1) / step_) * ((height_ + step_ - 1) / step_));
    size_t valid = 0;
    for (uint32_t y = 0; y < height_; y += step_)
        for (uint32_t x = 0; x < width_; x += step_) {
            const float d = depth.Get(x, y, 0);
            if (Valid(d)) {
                cur.push_back(std::log(static_cast<double>(d)));
                ++valid;
            } else {
                cur.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
    t.valid = valid >= 16;
    const Track* prev = track_.empty() ? nullptr : &track_.back();
    const bool connectable = t.valid && !restart && prev && prev->valid;
    t.segmentStart = true;
    if (mode_ == Mode::ScaleOnly) {
        t.logScale = 0.0;
        if (connectable && prevLog_.size() == cur.size()) {
            std::vector<double> diff;
            diff.reserve(cur.size());
            for (size_t i = 0; i < cur.size(); ++i)
                if (std::isfinite(cur[i]) && std::isfinite(prevLog_[i])) diff.push_back(cur[i] - prevLog_[i]);
            if (diff.size() >= std::max<size_t>(16, static_cast<size_t>(kMinOverlap * static_cast<double>(valid)))) {
                const double delta = LowerMedian(diff);
                if (std::fabs(delta) <= std::log(kBreakRatio)) {
                    t.segmentStart = false;
                    t.logScale = prev->logScale + delta;
                }
            }
        }
    } else if (mode_ == Mode::ScaleShift && t.valid) {
        std::vector<double> logs;
        logs.reserve(valid);
        for (const double l : cur)
            if (std::isfinite(l)) logs.push_back(l);
        t.loc = std::exp(LowerMedian(logs));  // the median depth
        std::vector<double> dev;
        dev.reserve(valid);
        for (const double l : cur)
            if (std::isfinite(l)) dev.push_back(std::fabs(std::exp(l) - t.loc));
        t.spread = LowerMedian(dev);  // median |d - loc|: robust to the sky / far outliers
        // a jump of the location or of the spread breaks the segment: a low-contrast frame (fog, a blank wall) has a
        // spread far below its neighbours', and stretching it to their mean would amplify its noise (review of TASK-0038)
        const auto jump = [](double a, double b) { return !(a > 0.0 && b > 0.0) || std::max(a / b, b / a) > kBreakRatio; };
        if (connectable) t.segmentStart = jump(t.loc, prev->loc) || jump(t.spread, prev->spread);
    }
    prevLog_ = std::move(cur);
}

void TemporalStabilizer::Push(int64_t index, PassImage depth) {
    if (pushedAny_ && index != pushedEnd_)
        Throw("TemporalStabilizer: frame " + std::to_string(index) + " after " + std::to_string(pushedEnd_ - 1) + " (frames must come in order)");
    if (!pushedAny_) {
        trackBase_ = next_ = index;
        pushedAny_ = true;
    }
    if (mode_ != Mode::None) {
        if (depth.type != PixelType::F32) depth = depth.ConvertTo(PixelType::F32);
        Track t;
        Measure(depth, t);
        track_.push_back(t);
    }
    frames_.emplace_back(index, std::move(depth));
    pushedEnd_ = index + 1;
}

ScaleShift TemporalStabilizer::TransformFor(int64_t t) const {
    ScaleShift ss;
    const Track& cur = TrackAt(t);
    if (!cur.valid) return ss;
    int left = 0, right = 0;
    while (left < radius_ && !TrackAt(t - left).segmentStart) ++left;
    while (right < radius_ && t + right + 1 < pushedEnd_ && !TrackAt(t + right + 1).segmentStart) ++right;
    const int r = std::min(left, right);
    if (r == 0) return ss;  // at a segment's end: unchanged (no one-sided pull towards the neighbours)
    double scale = 1.0, shift = 0.0;
    if (mode_ == Mode::ScaleOnly) {
        double sum = 0.0;
        for (int64_t k = t - r; k <= t + r; ++k) sum += TrackAt(k).logScale;
        scale = std::exp(sum / (2 * r + 1) - cur.logScale);
    } else {
        double loc = 0.0, spread = 0.0;
        for (int64_t k = t - r; k <= t + r; ++k) {
            loc += TrackAt(k).loc;
            spread += TrackAt(k).spread;
        }
        loc /= (2 * r + 1);
        spread /= (2 * r + 1);
        if (cur.spread <= 1e-12 * cur.loc) return ss;
        scale = spread / cur.spread;
        shift = loc - scale * cur.loc;
    }
    if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(shift)) return ss;
    if (scale > kBreakRatio || scale < 1.0 / kBreakRatio) {
        // alignment removes flicker, it does not re-grade a frame: a gradual drift inside a segment (each step under
        // kBreakRatio) could still ask for more over the window
        scale = std::clamp(scale, 1.0 / kBreakRatio, kBreakRatio);
        if (mode_ == Mode::ScaleShift) {
            double loc = 0.0;
            for (int64_t k = t - r; k <= t + r; ++k) loc += TrackAt(k).loc;
            shift = loc / (2 * r + 1) - scale * cur.loc;
        }
    }
    ss.scale = static_cast<float>(scale);
    ss.shift = static_cast<float>(shift);
    ss.samples = 2 * static_cast<size_t>(r) + 1;
    return ss;
}

bool TemporalStabilizer::Pop(int64_t& index, PassImage& depth, ScaleShift* applied) {
    if (frames_.empty()) return false;
    if (mode_ != Mode::None && !closed_ && pushedEnd_ <= next_ + radius_) return false;  // frame next_+N not in yet
    auto& [i, frame] = frames_.front();
    ScaleShift ss;
    if (mode_ != Mode::None) {
        ss = TransformFor(i);
        if (ss.scale != 1.f || ss.shift != 0.f) {
            float* d = frame.As<float>();
            const size_t n = frame.data.size() / sizeof(float);
            for (size_t k = 0; k < n; ++k) {
                if (!Valid(d[k])) continue;
                const float v = ss.scale * d[k] + ss.shift;
                d[k] = v > 0.f ? v : kMinAlignedDepth;
            }
        }
    }
    index = i;
    depth = std::move(frame);
    if (applied) *applied = ss;
    frames_.pop_front();
    next_ = index + 1;
    while (!track_.empty() && trackBase_ < next_ - radius_) {  // the window of the next frames reaches back N frames
        track_.pop_front();
        ++trackBase_;
    }
    return true;
}

}  // namespace dlssvid
