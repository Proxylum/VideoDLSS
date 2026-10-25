#pragma once

#include <cstdint>
#include <vector>

#include "convert/ColorConvert.h"
#include "pipeline/Frame.h"

namespace dlssvid {

// Scene cuts (PR-6, spec §2d): on a hard cut a temporal stage drops its history, so the last scene does not leak into the
// next one. Our detector, over the source's luma - the cut is a property of the source, whatever colour a stage reads:
// box-downsampled to <= maxWidth (an integer factor, even sides), normalised to [0, 1] by the range and the bit depth,
// score = mean |Y_t - Y_{t-1}|. A cut is a score above the threshold or another frame size.
struct LumaThumb {
    uint32_t width = 0, height = 0;          // of the thumbnail
    uint32_t frameWidth = 0, frameHeight = 0;  // of the frame it was made from: another size is a cut
    std::vector<float> y;                    // [0, 1], row-major
    bool Empty() const { return y.empty(); }
};

// Limited range: (Y - 16·2^(b-8)) / (219·2^(b-8)), full: Y / (2^b - 1), clamped to [0, 1]. An empty frame gives an empty thumb.
LumaThumb MakeLumaThumb(const CpuFrame& frame, const ColorInfo& info, uint32_t maxWidth = 640);

// mean |a - b| over the thumbnails; 1 (a cut at any threshold below 1) when the frames or the thumbnails differ in size.
double SceneCutScore(const LumaThumb& a, const LumaThumb& b);

class SceneCutDetector {
public:
    explicit SceneCutDetector(ColorInfo info, uint32_t maxWidth = 640) : info_(info), maxWidth_(maxWidth) {}
    // This frame against the last one given: true on a cut (score > threshold). The first frame after construction or Reset
    // and a frame without samples are no cut (an empty frame keeps the last thumb). `score`: the frame's score, 0 for those.
    bool Next(const CpuFrame& frame, double threshold, double* score = nullptr);
    void Reset() { prev_ = {}; }

private:
    ColorInfo info_;
    uint32_t maxWidth_;
    LumaThumb prev_;
};

}  // namespace dlssvid
