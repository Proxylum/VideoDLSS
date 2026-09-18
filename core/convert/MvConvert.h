#pragma once

#include <cstdint>

#include "passes/PassImage.h"

namespace dlssvid {

// docs/conventions.md §3. Motion vectors are 2-channel (u, v) float images in pixels.
//   mv_raw  : forward optical flow stored with frame t:   p(t+1) = p(t) + F_t(p(t))
//   mv_dlss : backward vectors stored with frame t:       p(t-1) = p(t) + B_t(p(t))
// Both use image coordinates (x right, y down) unless a manifest says y_up.

struct MvConvertOptions {
    uint32_t targetWidth = 0;   // 0 = keep the input resolution
    uint32_t targetHeight = 0;
    bool fillHoles = true;      // nearest-neighbour fill of pixels no source pixel mapped to
    int dilateRadius = 1;       // 0 = off; 1-2 px dilation towards the nearest (closest) depth
    float depthEdgeThreshold = 0.02f;  // relative depth difference that counts as an edge
};

// Inverts a flow field: given F on grid A (p_B = p_A + F(p_A)) produce G on grid B with
// G(p_B) = -F(p_A). Collisions are resolved with `depth` (smaller = closer wins); without
// depth the last writer wins. Output has the same resolution as the input; `zbuf` (optional)
// receives the splatted depth used for dilation.
PassImage InvertFlow(const PassImage& flow, const PassImage* depth, bool fillHoles, PassImage* zbuf = nullptr);

// mv_raw of frame t (+ depth_raw of frame t) -> mv_dlss for frame t+1.
PassImage ForwardFlowToBackwardMv(const PassImage& flowT, const PassImage* depthT, const MvConvertOptions& opt);
// mv_dlss of frame t (+ depth of frame t) -> mv_raw for frame t-1.
PassImage BackwardMvToForwardFlow(const PassImage& mvT, const PassImage* depthT, const MvConvertOptions& opt);

// Nearest-neighbour resample of the field and scaling of the vectors to the new pixel grid.
PassImage ScaleMv(const PassImage& mv, uint32_t targetWidth, uint32_t targetHeight);
// Each pixel takes the vector of the closest (smallest depth) pixel within `radius` if that
// pixel is closer by more than `threshold` (relative). Mirrors the MV dilation DLSS expects at depth edges.
PassImage DilateMvByDepth(const PassImage& mv, const PassImage& depth, int radius, float threshold);
PassImage FlipMvY(const PassImage& mv);
PassImage ZeroMv(uint32_t width, uint32_t height);

}  // namespace dlssvid
