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

struct PassImage;

// The `trt` backend's model when the project names none (models/registry.json, family realesrgan).
constexpr const char* kTrtDefaultUpscaleModel = "realesrgan-x2plus";
// The `worker` backend's model when the project names none (sr_worker/worker.py, family realbasicvsr).
constexpr const char* kWorkerDefaultUpscaleModel = "realbasicvsr";

// One interface for every upscaler (ТЗ §3): DLSS SR (default), NIS, the naive bicubic baseline and open-source
// models through TensorRT (`trt`). GPU backends record into a command list; CPU-side ones take the frame as an image.
// The stage prepares D3D12 textures; GPU backends only record work into a command list.
struct UpscalerConfig {
    std::string backend;  // dlss | nis | bicubic | trt | worker
    uint32_t inputWidth = 0, inputHeight = 0;
    uint32_t outputWidth = 0, outputHeight = 0;
    float sharpness = 0.5f;               // nis: 0..1 (NVScaler slider); dlss: unused (sharpening is deprecated in NGX)
    std::string preset = "default";       // dlss: default | J | K | L | M (NGX render preset hint)
    bool artifactReductionOnly = false;   // no scaling: nis NVSharpen
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
    // CPU-side backends (TensorRT models): the stage hands the RGBA16F frame over on the CPU and takes the result back
    // at the model's native scale (NativeScale), resampling to the target itself; Evaluate is not called for them.
    virtual bool ProcessesOnCpu() const { return false; }
    virtual int NativeScale() const { return 0; }
    virtual void EvaluateCpu(const PassImage& rgbaIn, PassImage& rgbaOut);
    // Video models take a window of frames at once (WindowSize > 1); the next window re-estimates the last
    // WindowOverlap frames and the stage blends them. NativeScale() == 0: the backend returns the target size itself.
    virtual int WindowSize() const { return 1; }
    virtual int WindowOverlap() const { return 0; }
    virtual void EvaluateCpuWindow(const std::vector<const PassImage*>& rgbaIn, uint32_t targetW, uint32_t targetH, std::vector<PassImage>& rgbaOut);
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
