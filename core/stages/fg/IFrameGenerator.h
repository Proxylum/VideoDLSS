#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/D3D12Device.h"
#include "passes/PassImage.h"
#include "stages/nr/INrBackend.h"  // NvidiaDriverVersion, GpuArchitectureFromName

namespace dlssvid {

// Frame generation (stage 7, ТЗ §4): one interface, three backends — `dlssg` (DLSS Frame Generation through the NGX
// API of the DLSS SDK, no swapchain), `rife` (RIFE 4.x through TensorRT, the baseline of ТЗ §8) and `blend` (linear
// blend of the neighbours: the naive baseline, deterministic, no GPU). A backend receives the previous and the
// current frame (RGBA16F textures and CPU copies), the current frame's guides, and returns `multiplier - 1`
// intermediate frames as CPU images; it executes its own GPU / TensorRT work.
struct FgConfig {
    std::string backend = "dlssg";           // dlssg | rife | blend
    uint32_t width = 0, height = 0;          // colour size
    uint32_t guideWidth = 0, guideHeight = 0;  // depth_dlss / mv_dlss size (0 = none)
    int multiplier = 2;                      // output frames per input frame: 2 | 3 | 4
    std::string backbufferFormat = "rgba16f";  // dlssg: rgba16f | rgba8 (auto-fallback to rgba8 when rgba16f is refused)
    bool depthInverted = true;               // depth_dlss is reverse-Z
    bool requireDriver = true;
    std::filesystem::path dllDir;            // nvngx_dlssg.dll folder override
    std::string model = "rife49";            // rife: registry id (rife49 | rife48 | rife47)
    bool fp16 = true;                        // rife
    std::string modelsDir;                   // rife: folder with registry.json (empty = default)
    nlohmann::json extra = nlohmann::json::object();
};

struct FgFrame {
    ID3D12Resource* texture = nullptr;  // RGBA16F width x height in COMMON (may be null for CPU-only backends)
    const PassImage* cpu = nullptr;     // RGB (F16 / F32) width x height
};

struct FgInputs {
    FgFrame prev, cur;
    ID3D12Resource* depth = nullptr;  // R32F reverse-Z of `cur`, optional
    uint32_t depthWidth = 0, depthHeight = 0;
    ID3D12Resource* mv = nullptr;     // RG32F cur -> prev in px (mv_dlss of `cur`), optional
    uint32_t mvWidth = 0, mvHeight = 0;
    bool reset = false;               // first frame / cut: prev is unrelated
    int64_t frameIndex = 0;           // index of `cur`
};

struct FgDiagnostics {
    std::string gpu, architecture;
    NvidiaDriverVersion driver;
    bool driverOk = false;
    std::filesystem::path dll;
    std::string dllSha256;
    uint64_t dllSize = 0;
    bool available = false;      // FrameGeneration.Available
    int multiFrameMax = 0;       // DLSSG.MultiFrameCountMax (intermediate frames per pair)
    std::string createResult;    // CreateFeature(FrameGeneration) result
    std::string backbufferFormat;
    std::string model;           // rife: model id / engine
    bool ok = false;
    std::string hint;
    nlohmann::json ToJson() const;
};

class IFrameGenerator {
public:
    virtual ~IFrameGenerator() = default;
    virtual std::string_view Name() const = 0;
    virtual void Init(D3D12Device& device, const FgConfig& config) = 0;
    // Produces config.multiplier - 1 frames between in.prev and in.cur (RGB F16 width x height) into `out`.
    // Called for every frame including the first (reset = true, prev == cur) so history-based backends prime.
    virtual void Generate(const FgInputs& in, std::vector<PassImage>& out) = 0;
    virtual bool UsesGuides() const { return false; }
    virtual const FgDiagnostics& Diagnostics() const = 0;
    virtual nlohmann::json Describe() const = 0;
    virtual void Shutdown() {}
};

struct FgAvailability {
    bool available = false;
    std::string reason;
};
FgAvailability FgAvailable(const std::string& backend, const std::filesystem::path& dllDir = {});
std::unique_ptr<IFrameGenerator> CreateFrameGenerator(const std::string& backend);  // throws when unknown
std::vector<std::string> FgBackends();

inline constexpr const char* kDlssgDllName = "nvngx_dlssg.dll";
std::filesystem::path FindDlssgDll(const std::filesystem::path& dllDir = {});
std::string DlssgDllInstruction();

// Frame indexing of the color_fg pass for a multiplier `mult`: real frame i -> i * mult, the k-th generated frame
// (k = 1..mult-1) between real i and i+1 -> i * mult + k; N real frames give (N - 1) * mult + 1 frames.
inline int64_t FgRealIndex(int64_t i, int mult) { return i * mult; }
inline int64_t FgInterpIndex(int64_t i, int k, int mult) { return i * mult + k; }
inline int64_t FgFrameCount(int64_t reals, int mult) { return reals <= 0 ? 0 : (reals - 1) * mult + 1; }

// Synthetic camera for DLSS-G on video (HACK, docs/architecture.md): a perspective projection with the frame's aspect
// ratio, its inverse, identity clip-to-previous-clip (the camera does not move; all motion is in the motion vectors).
struct FgCamera {
    float viewToClip[16];  // row-major (post-multiplication), as the DLSS-FG guide expects
    float clipToView[16];
    float identity[16];
    float nearPlane = 0.1f, farPlane = 1000.f, fovRadians = 0.f, aspect = 1.f;
};
FgCamera BuildFgCamera(uint32_t width, uint32_t height, float fovDegrees = 60.f, float nearPlane = 0.1f, float farPlane = 1000.f);
// 4x4 row-major multiply (tests: viewToClip * clipToView == I).
void Mul4x4(const float* a, const float* b, float* out);

}  // namespace dlssvid
