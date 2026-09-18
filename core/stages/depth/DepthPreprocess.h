#pragma once

#include <cstdint>
#include <vector>

#include "passes/PassImage.h"

namespace dlssvid {

// Model input geometry for a frame: shorter side = inputSize, longer side keeps the aspect
// ratio and is rounded to a multiple of `multiple` (Depth Anything convention, "lower_bound").
struct ModelInputSize {
    uint32_t width = 0;
    uint32_t height = 0;
};
ModelInputSize ComputeModelInputSize(uint32_t srcWidth, uint32_t srcHeight, int inputSize, int multiple = 14, double maxAspect = 1.78);

// Bilinear resize of a multi-channel float image (F16/F32 in, F32 out). Half-pixel centres.
PassImage ResizeBilinear(const PassImage& src, uint32_t width, uint32_t height);
// Area-average downscale (for shrinking source frames to maxInputRes before the model).
PassImage ResizeArea(const PassImage& src, uint32_t width, uint32_t height);

struct Normalization {
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};
// RGB [0,1] (H, W, 3) -> NCHW float tensor appended to `out` (3*H*W values), normalised.
void ToNchwNormalized(const PassImage& rgb, const Normalization& norm, std::vector<float>& out);

// Luminance (Rec.709 weights) of an RGB image as a single-channel F32 image.
PassImage Luminance(const PassImage& rgb);

// Guided filter (He et al.) of `src` (1 channel F32) with a 1-channel guide of the same size.
PassImage GuidedFilter(const PassImage& src, const PassImage& guide, int radius, float eps);

// Depth at model resolution -> source resolution: bilinear upsample followed by a guided
// filter driven by the colour frame's luminance (edge-aware, ТЗ §2).
PassImage UpsampleDepthGuided(const PassImage& depthLow, const PassImage& rgbFull, int radius = 8, float eps = 1e-4f);

}  // namespace dlssvid
