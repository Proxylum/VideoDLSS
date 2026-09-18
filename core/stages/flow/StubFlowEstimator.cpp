#include "stages/flow/StubFlowEstimator.h"

#include "util/Error.h"

namespace dlssvid {

void StubFlowEstimator::Estimate(const FlowInput& a, const FlowInput&, PassImage& flowOut, PassImage* confidenceOut) {
    ++calls_;
    uint32_t w = 0, h = 0;
    if (a.rgb) {
        w = a.rgb->width;
        h = a.rgb->height;
    } else if (a.gpu) {
        w = a.gpu->width;
        h = a.gpu->height;
    }
    if (!w || !h) Throw("StubFlowEstimator: no input frame");
    flowOut = MakePassImage(PassKind::MvRaw, w, h);
    float* f = flowOut.As<float>();
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        f[2 * i] = dx_;
        f[2 * i + 1] = dy_;
    }
    if (confidenceOut) {
        confidenceOut->Allocate(w, h, PixelType::F32, {"A"});
        for (auto& b : confidenceOut->data) b = 0;
        float* c = confidenceOut->As<float>();
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) c[i] = 1.f;
    }
}

}  // namespace dlssvid
