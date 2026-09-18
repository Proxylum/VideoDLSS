#include "stages/upscale/RtxVsrUpscaler.h"

#include "util/Error.h"

namespace dlssvid {

UpscalerAvailability RtxVsrUpscaler::Available() {
#if defined(DLSSVID_WITH_RTX_VIDEO_SDK)
    return {true, {}};
#else
    return {false,
            "RTX Video SDK 1.1 is not integrated in this build: download it from https://developer.nvidia.com/rtx-video-sdk "
            "(NVIDIA developer account), set RTX_VIDEO_SDK_ROOT and rebuild (docs/dll-setup.md). "
            "Until then use --backend nis (NVIDIA Image Scaling) or --backend dlss."};
#endif
}

void RtxVsrUpscaler::Init(D3D12Device&, const UpscalerConfig&) { Throw("rtxvsr: " + Available().reason); }

void RtxVsrUpscaler::Evaluate(ID3D12GraphicsCommandList*, const UpscaleInputs&, ID3D12Resource*) { Throw("rtxvsr: not initialised"); }

nlohmann::json RtxVsrUpscaler::Describe() const { return {{"backend", "rtxvsr"}, {"available", false}}; }

}  // namespace dlssvid
