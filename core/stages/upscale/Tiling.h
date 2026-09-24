#pragma once

#include <functional>

#include "passes/PassImage.h"

namespace dlssvid {

// Whole-frame super-resolution through a fixed-size window (TensorRT models are built for one input shape): the frame
// is cut into `tile`×`tile` windows that overlap by `pad` pixels of context on every side — a window that would leave
// the frame is shifted back inside it, a frame smaller than a window is extended by reflection — `run` maps one window
// (RGB float NCHW, [0, 1], tile×tile) to its scale×tile square, and every window writes only the part of the frame it
// owns: the padding is discarded, the first and last windows own their edge. `rgb` is RGB F32 interleaved; the result
// is RGB F32 at scale×. Deterministic: the same frame gives the same result whatever the tile size, when `run` is
// translation-invariant (a nearest ×2 reassembles the frame exactly — the unit test).
using TileRunFn = std::function<void(const float* inTile, float* outTile)>;
PassImage UpscaleTiled(const PassImage& rgb, int scale, int tile, int pad, const TileRunFn& run);

}  // namespace dlssvid
