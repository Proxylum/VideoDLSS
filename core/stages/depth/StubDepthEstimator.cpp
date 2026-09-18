#include "stages/depth/StubDepthEstimator.h"

#include "stages/depth/DepthPreprocess.h"

namespace dlssvid {

PassImage StubDepthEstimator::Expected(const PassImage& rgb) {
    PassImage luma = Luminance(rgb);
    PassImage depth = MakePassImage(PassKind::DepthRaw, rgb.width, rgb.height);
    for (uint32_t y = 0; y < rgb.height; ++y)
        for (uint32_t x = 0; x < rgb.width; ++x) depth.Set(x, y, 0, 1.f + 4.f * luma.Get(x, y, 0));
    return depth;
}

void StubDepthEstimator::Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) {
    ++calls_;
    depthOut.clear();
    for (const PassImage* f : rgb) depthOut.push_back(Expected(*f));
}

}  // namespace dlssvid
