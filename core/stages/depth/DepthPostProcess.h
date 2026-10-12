#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

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

// Temporal alignment of depth over a centred window (ТЗ §2: ±N frames; TASK-0038, audit depth-7). The per-frame
// uncertainty of a monocular model is global, so each frame gets one transform:
//   ScaleOnly (metric models): a track c_t of the frame's log scale, c_t = c_{t-1} + median over pixels of
//     log(d_t / d_{t-1}) between neighbouring RAW frames (robust to moving objects under half the frame, to the depth
//     distribution, to the sky); frame t is scaled by s_t = exp(mean_{k=t-r..t+r} c_k - c_t). A linear trend (a dolly,
//     a constant bias) passes through unchanged; no recursion over aligned frames, hence no drift.
//   ScaleShift (relative models): location = median depth, spread = median |d - location| per frame; frame t is mapped
//     so that both equal their window means (MiDaS-style normalisation, robust to far outliers).
// The window shrinks symmetrically at the ends of the clip and of a segment, r = min(N, distance to the segment's ends);
// a segment breaks where neighbouring frames differ by more than kBreakRatio (a cut), at a resolution change and around
// a frame without valid depth. Frame t leaves after frame t+N came in (or on Close): the output is delayed by N frames;
// Mode::None passes frames through unchanged at once. Why metric depth gets no shift: docs/architecture.md.
class TemporalStabilizer {
public:
    enum class Mode { None, ScaleShift, ScaleOnly };
    static constexpr size_t kMaxSamples = 262144;   // statistics on a regular grid of at most this many pixels
    static constexpr double kBreakRatio = 1.5;      // a jump beyond this between neighbouring frames breaks the segment
    static constexpr double kMinOverlap = 0.25;     // ScaleOnly: share of samples valid in both frames (at least 16)
    static constexpr float kMinAlignedDepth = 1e-6f;  // ScaleShift: a valid pixel never becomes <= 0

    explicit TemporalStabilizer(Mode mode = Mode::None, int radius = 8) : mode_(mode), radius_(radius < 0 ? 0 : radius) {}
    Mode GetMode() const { return mode_; }
    int Radius() const { return radius_; }
    int Delay() const { return mode_ == Mode::None ? 0 : radius_; }

    // The next frame in order (F32, holes already filled). Throws when the index is not the previous one + 1.
    void Push(int64_t index, PassImage depth);
    // The next aligned frame, if its window is complete (or after Close). `applied`: the transform it got.
    bool Pop(int64_t& index, PassImage& depth, ScaleShift* applied = nullptr);
    void Close() { closed_ = true; }  // end of the clip: the remaining frames leave with shrunk windows
    void Reset();
    size_t Buffered() const { return frames_.size(); }

private:
    struct Track {
        double logScale = 0.0;  // ScaleOnly: c_t
        double loc = 0.0, spread = 0.0;  // ScaleShift
        bool valid = false;
        bool segmentStart = true;  // frame t is not connected to t-1
    };
    void Measure(const PassImage& depth, Track& t);
    const Track& TrackAt(int64_t index) const { return track_[static_cast<size_t>(index - trackBase_)]; }
    ScaleShift TransformFor(int64_t index) const;

    Mode mode_;
    int radius_;
    std::deque<std::pair<int64_t, PassImage>> frames_;  // waiting to leave, in order
    std::deque<Track> track_;                           // frames [trackBase_, pushedEnd_)
    std::vector<double> prevLog_;                       // log d of the previous frame on the grid (NaN: invalid)
    uint32_t width_ = 0, height_ = 0, step_ = 1;
    int64_t trackBase_ = 0, pushedEnd_ = 0, next_ = 0;
    bool pushedAny_ = false, closed_ = false;
};

}  // namespace dlssvid
