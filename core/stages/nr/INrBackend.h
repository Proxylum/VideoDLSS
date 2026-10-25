#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/D3D12Device.h"

namespace dlssvid {

// NVIDIA driver version decoded from the DXGI user-mode driver version (D3D12Device::UmdDriverVersion):
// product.version.subversion.build, e.g. 32.0.15.9186 -> 591.86 ((15 mod 10) * 100 + 91, .86).
struct NvidiaDriverVersion {
    int major = 0, minor = 0;
    bool valid = false;
    bool AtLeast(int maj, int min) const { return valid && (major > maj || (major == maj && minor >= min)); }
    std::string ToString() const;  // "591.86", "?" when unknown
};
NvidiaDriverVersion NvidiaDriverFromUmd(uint64_t umdVersion);
inline constexpr int kNrMinDriverMajor = 616, kNrMinDriverMinor = 56;  // ТЗ §4: Neural Rendering needs >= 616.56

// GPU architecture from the adapter name ("NVIDIA GeForce RTX 4070 Ti SUPER" -> "Ada"), "" when unknown.
std::string GpuArchitectureFromName(std::string_view adapterName);
// From a CUDA compute capability (7.5 Turing, 8.x Ampere / 8.9 Ada, >= 10 Blackwell), "" when unknown.
std::string GpuArchitectureFromComputeCapability(int major, int minor);

inline constexpr const char* kNrDllName = "nvngx_dlssnr.dll";           // NGX Feature 18 snippet (user supplied)
inline constexpr const char* kNrForwarderName = "nvngx.dll_dlssvid.dll";  // our caller module (core/stages/nr/forwarder)

// Model parameters (names from OptiScaler_DLSSNR / ComfyUI-DLSS5-NR, see docs/plans/06-nr.md).
struct NrConfig {
    std::string backend = "ngx";             // ngx | stub
    uint32_t width = 0, height = 0;          // colour = output size (the model's working resolution)
    float intensity = 1.f;                   // DLSSNR.Intensity 0..2
    int style = 1;                           // DLSSNR.Style: 0 default, 1 natural, 2 cinematic (3..6 gave cinematic, PR-2 E7 - refused)
    int preset = 3;                          // DLSSNR.Hint.Render.Preset: always 3 - 0..2 gave the same picture (PR-2 E7), no setting
    float localTone = 1.f;                   // DLSSNR.LocalToneStrength 0..2
    float localStructure = 1.f;              // DLSSNR.LocalStructureStrength 0..2
    float skinStructure = -1.f;              // DLSSNR.SkinStructureStrength (-1 = model default)
    bool autoMask = false;                   // DLSSNR.UseAutoMask (the model's automatic skin mask)
    bool depthInverted = true;               // depth_dlss is reverse-Z (docs/conventions.md)
    float mvScaleX = 1.f, mvScaleY = 1.f;    // DLSSNR.MVecScale base (x NrMvScale): mv_dlss is in pixels of its own size
    bool useGuides = true;                   // false: still-image mode (no depth / mv, history reset every frame)
    std::filesystem::path dllDir;            // nvngx_dlssnr.dll folder override (docs/dll-setup.md)
    std::string paramsBlock = "capability";  // NGX parameter block: capability (OptiScaler) | alloc (ComfyUI)
    bool requireDriver = true;               // enforce driver >= 616.56
    nlohmann::json extra = nlohmann::json::object();
    // A/B switches of the standalone `dlssvid nr` (spec §2e: E3, E4, E5) - not schema keys, `process` leaves the defaults
    std::string mvScale = "unit";            // unit: mv_dlss px as is, the sub-rect carries the size | auto: x model size / mv size
    bool createWithMv = false;               // CreateFeature with zero guides bound instead of none (ComfyUI binds a zero MVec)
    uint32_t createMvWidth = 0, createMvHeight = 0;        // the zero MVec at CreateFeature (0: the model size)
    uint32_t createDepthWidth = 0, createDepthHeight = 0;  // the zero depth at CreateFeature (0: none bound)
    bool depthGuide = true;                  // false: mv_dlss to the model, depth_dlss not (--no-depth-guide)
};

struct NrInputs {
    ID3D12Resource* color = nullptr;  // RGBA16F width x height, display-referred [0,1], in COMMON
    ID3D12Resource* depth = nullptr;  // R32F reverse-Z (depth_dlss), optional
    uint32_t depthWidth = 0, depthHeight = 0;
    ID3D12Resource* mv = nullptr;     // RG32F backward vectors in px (mv_dlss), optional
    uint32_t mvWidth = 0, mvHeight = 0;
    bool reset = false;               // first frame / scene cut: drop the model's history
    int64_t frameIndex = 0;
};

// What the backend learned at Init: logged by the stage (ТЗ §4) and printed by `dlssvid nr --check`.
struct NrDiagnostics {
    std::string gpu, architecture;
    NvidiaDriverVersion driver;
    bool driverOk = false;
    std::filesystem::path dll, forwarder;
    std::string dllSha256;
    uint64_t dllSize = 0;
    std::string initResult;    // snippet Init_Ext result (NGX result string)
    int initAbi = -1;          // argument order that worked (0 info,version | 1 version,info | 2 version,params)
    std::string paramsBlock;   // capability | alloc
    std::string createResult;  // CreateFeature(18) result
    // Init's cost (spec E8 T_init, E6 VRAM per feature; `dlssvid nr --check`): -1 = not measured (Init stopped before)
    double initMs = -1.0;           // NGX core (cold only for the first user in the process), snippet + forwarder, block, Init_Ext
    double createMs = -1.0;         // CreateFeature(18), every attempt (the retry with the other block is part of the wait)
    int64_t vramInitBytes = -1;     // this process's local video memory: growth over the init above
    int64_t vramFeatureBytes = -1;  // ... over CreateFeature (our scratch textures and zero guides are made before, not counted)
    int64_t vramBudgetBytes = -1;   // DXGI budget of the process at the start (what is left for passes 2..4, 4K)
    bool ok = false;
    std::string hint;          // instruction for the user when !ok
    nlohmann::json ToJson() const;
};

// One re-creation of the feature inside the same Init (`dlssvid nr --check --check-repeat`, spec E8: N re-creations in one process -
// the preview's phase 2 condition): released and created again with the same parameters and block, never the other block.
struct NrRecreation {
    std::string result;             // CreateFeature(18) result, or "release: <result>" when the release failed (nothing created then)
    std::string releaseResult;      // ReleaseFeature result
    bool ok = false;
    double releaseMs = -1.0, createMs = -1.0;
    int64_t vramAfterRelease = -1;  // this process's local video memory (DXGI CurrentUsage), raw bytes: a drift shows either way
    int64_t vramAfterCreate = -1;
    nlohmann::json ToJson() const;
};

class INrBackend {
public:
    virtual ~INrBackend() = default;
    virtual std::string_view Name() const = 0;
    virtual void Init(D3D12Device& device, const NrConfig& config) = 0;
    // Records the model into `cl`. `output` is RGBA16F width x height created with
    // D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; all resources are in COMMON before and after.
    virtual void Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) = 0;
    virtual bool UsesGuides() const { return false; }
    virtual const NrDiagnostics& Diagnostics() const = 0;
    virtual nlohmann::json Describe() const = 0;
    virtual void Shutdown() {}
    // Releases the feature and creates it again (E8); after a failed creation the backend has no feature. Throws before Init.
    virtual NrRecreation RecreateFeature() = 0;
};

struct NrAvailability {
    bool available = false;
    std::string reason;  // instruction for the user when unavailable
};
// Availability without a device: SDK compiled in, DLL and forwarder present (ngx); stub is always available.
NrAvailability NrAvailable(const std::string& backend, const std::filesystem::path& dllDir = {});
std::unique_ptr<INrBackend> CreateNrBackend(const std::string& backend);  // throws dlssvid::Error when unknown
std::vector<std::string> NrBackends();

// nvngx_dlssnr.dll / nvngx.dll_dlssvid.dll lookup: DLSSVID_NVIDIA_DLL_DIR, <exe>/nvidia, <exe> (docs/dll-setup.md).
std::filesystem::path FindNrDll(const std::filesystem::path& dllDir = {});
std::filesystem::path FindNrForwarder(const std::filesystem::path& dllDir = {});
std::string NrDllInstruction();
int ParseNrStyle(const std::string& s);  // default | natural | cinematic | integer
// DLSSNR.MVecScale factor for a frame (A/B E3, spec §2e): "unit" {1,1} - mv_dlss pixels of its own resolution as is
// (OptiScaler: the sub-rect carries the size, a ratio on top would count it twice); "auto" - the ratio model / mv size, vectors
// in the model's pixels (ComfyUI: colour pixels). {1,1} without vectors. Throws on another mode.
std::pair<float, float> NrMvScale(const std::string& mode, uint32_t modelWidth, uint32_t modelHeight, uint32_t mvWidth, uint32_t mvHeight);

}  // namespace dlssvid
