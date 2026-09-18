#pragma once

#include <cstdint>
#include <deque>
#include <optional>

#include "passes/PassImage.h"

namespace dlssvid {

// Least-squares scale/shift that maps `depth` onto `reference` over valid (finite, > 0) pixels.
struct ScaleShift {
    float scale = 1.f;
    float shift = 0.f;
    size_t samples = 0;
};
ScaleShift SolveScaleShift(const PassImage& depth, const PassImage& reference, bool shiftAllowed = true);
void ApplyScaleShift(PassImage& depth, const ScaleShift& ss);

// Replace non-finite / non-positive samples with the nearest valid one (BFS). Returns count filled.
size_t FillInvalidDepth(PassImage& depth);

// Temporal alignment error between consecutive frames (ТЗ §8 «метрика TAE»): mean |d_t - d_{t-1}|
// over valid pixels divided by mean(d_{t-1}). Without motion vectors (stage 3) this is the
// static-scene proxy; the stage logs it per frame and averages it in the manifest.
double TemporalAlignmentError(const PassImage& prev, const PassImage& cur);

// Temporal scale-shift stabilisation over a sliding window (ТЗ §2: ±8 frames): each new frame
// is aligned to the mean of the previous `window` aligned frames.
class TemporalStabilizer {
public:
    enum class Mode { None, ScaleShift, ScaleOnly };
    explicit TemporalStabilizer(Mode mode = Mode::ScaleShift, int window = 8) : mode_(mode), window_(window) {}
    // Aligns `depth` in place and pushes it into the window. Returns the applied transform.
    ScaleShift Stabilize(PassImage& depth);
    void Reset() { history_.clear(); }
    Mode GetMode() const { return mode_; }

private:
    Mode mode_;
    int window_;
    std::deque<PassImage> history_;
    PassImage referenceSum_;  // running sum of the window (F32)
};

}  // namespace dlssvid
