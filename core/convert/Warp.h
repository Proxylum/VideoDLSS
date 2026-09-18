#pragma once

#include "passes/PassImage.h"

namespace dlssvid {

// Backward warp: out(p) = src(p + mv(p)), bilinear, mv in pixels (docs/conventions.md §3:
// mv_dlss points from the current pixel to its position in the previous frame, so
// WarpBackward(prev, mv_dlss) reconstructs the current frame from the previous one).
// Samples that fall outside the source are marked invalid (mask = 0) and copied from `src`.
struct WarpResult {
    PassImage image;  // same layout as src (F32)
    PassImage mask;   // 1 channel F32: 1 = valid sample
    double validFraction = 0.0;
};
WarpResult WarpBackward(const PassImage& src, const PassImage& mv);

// PSNR in dB over channels [0,1] (or any range given by `peak`); optional validity mask.
double Psnr(const PassImage& a, const PassImage& b, float peak = 1.f, const PassImage* mask = nullptr);

// Warp-test of ТЗ §8: previous frame warped by the current frame's backward vectors vs current frame.
double WarpPsnr(const PassImage& prev, const PassImage& cur, const PassImage& mvBackward, float peak = 1.f);

// Temporal alignment error with motion compensation: mean |d_t(p) - d_{t-1}(p + mv(p))| / mean d_{t-1}
// over valid, finite samples.
double TemporalAlignmentErrorWarped(const PassImage& prevDepth, const PassImage& curDepth, const PassImage& mvBackward);

}  // namespace dlssvid
