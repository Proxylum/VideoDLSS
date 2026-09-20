#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/D3D12Device.h"

namespace dlssvid {

// One interface for every upscaler (ТЗ §3): RTX VSR (default), DLSS SR, NIS and the naive bicubic
// baseline. The stage prepares D3D12 textures; backends only record work into a command list.
struct UpscalerConfig {
    std::string backend;  // dlss | nis | bicubic | rtxvsr
    uint32_t inputWidth = 0, inputHeight = 0;
    uint32_t outputWidth = 0, outputHeight = 0;
    float sharpness = 0.5f;               // nis: 0..1 (NVScaler slider); dlss: unused (sharpening is deprecated in NGX)
    std::string preset = "default";       // dlss: default | J | K | L | M (NGX render preset hint)
    bool artifactReductionOnly = false;   // no scaling: rtxvsr artifact reduction / nis NVSharpen
    bool useJitter = true;                // dlss: emulate camera jitter on the input (HACK, ТЗ §3)
    float jitterSign = 1.f;               // dlss: sign of the reported jitter relative to the resample shift
    std::filesystem::path dllDir;         // folder with nvngx_dlss.dll (empty: default search, see docs/dll-setup.md)
    nlohmann::json extra = nlohmann::json::object();
};

struct UpscaleInputs {
    ID3D12Resource* color = nullptr;  // RGBA16F, inputWidth x inputHeight, in COMMON (already jittered when jitter != 0)
    ID3D12Resource* depth = nullptr;  // R32F reverse-Z (depth_dlss) at the input size, optional
    ID3D12Resource* mv = nullptr;     // RG32F backward vectors in px (mv_dlss), optional
    uint32_t mvWidth = 0, mvHeight = 0;
    float jitterX = 0.f, jitterY = 0.f;  // jitter applied to `color`, input px (0 = none)
    bool reset = false;                  // first frame / scene cut: drop history
    int64_t frameIndex = 0;
    double frameTimeMs = 0.0;
};

class IUpscaler {
public:
    virtual ~IUpscaler() = default;
    virtual std::string_view Name() const = 0;
    virtual void Init(D3D12Device& device, const UpscalerConfig& config) = 0;
    // Records the upscale into `cl`. `output` is RGBA16F outputWidth x outputHeight created with
    // D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; all resources are in COMMON before and after.
    virtual void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) = 0;
    virtual bool WantsJitter() const { return false; }
    virtual bool WantsDepthAndMv() const { return false; }
    virtual nlohmann::json Describe() const = 0;
    virtual void Shutdown() {}
};

struct UpscalerAvailability {
    bool available = false;
    std::string reason;  // instruction for the user when unavailable
};
// Build-time / runtime availability without creating a device (SDK compiled in, DLL present, ...).
UpscalerAvailability UpscalerAvailable(const std::string& backend, const std::filesystem::path& dllDir = {});
std::unique_ptr<IUpscaler> CreateUpscaler(const std::string& backend);  // throws dlssvid::Error when unavailable
std::vector<std::string> UpscalerBackends();

// Output resolution: `scale` (x1.5 / x2 / x3) or an explicit target; even dimensions, aspect ratio kept,
// capped at maxW x maxH (4K in v1, ТЗ §3).
struct UpscaleTarget {
    uint32_t width = 0, height = 0;
    double scale = 1.0;  // effective width scale
    bool capped = false;
};
UpscaleTarget ResolveUpscaleTarget(uint32_t inW, uint32_t inH, double scale, uint32_t targetW = 0, uint32_t targetH = 0, uint32_t maxW = 3840,
                                   uint32_t maxH = 2160);

// Folder searched for NVIDIA DLLs: DLSSVID_NVIDIA_DLL_DIR, then <exe dir>/nvidia, then <exe dir> (docs/dll-setup.md).
std::vector<std::filesystem::path> NvidiaDllSearchPaths(const std::filesystem::path& override = {});

}  // namespace dlssvid
