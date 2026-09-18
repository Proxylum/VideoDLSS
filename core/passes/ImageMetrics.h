#pragma once

#include <cstdint>

#include "passes/PassImage.h"

namespace dlssvid {

// Full-reference quality metrics between two colour images of the same size (1, 3 or 4 channels;
// integer types are scaled to [0, 1]). Y = BT.709 luma of the (display-referred) RGB values.
struct ImageMetrics {
    double psnrY = 0.0;    // dB, peak 1.0; infinity for identical images
    double psnrRgb = 0.0;  // dB over all colour samples
    double ssimY = 0.0;    // SSIM on luma, 11x11 Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03
    double mseY = 0.0;
};

ImageMetrics CompareImages(const PassImage& reference, const PassImage& test);

// Luma plane of a colour image as F32 in [0, 1].
PassImage LumaOf(const PassImage& img);

}  // namespace dlssvid
