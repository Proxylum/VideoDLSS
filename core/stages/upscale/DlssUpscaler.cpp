#include "stages/upscale/DlssUpscaler.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu/ComputeKernel.h"
#include "gpu/Ngx.h"
#include "util/Error.h"
#include "util/Log.h"

#if defined(DLSSVID_WITH_DLSS)
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_params.h>
#endif

namespace dlssvid {

namespace {
const char* kDllName = "nvngx_dlss.dll";

std::filesystem::path FindDll(const std::filesystem::path& dllDir) {
    for (const auto& dir : NvidiaDllSearchPaths(dllDir))
        if (std::filesystem::exists(dir / kDllName)) return dir;
    return {};
}

std::string DllInstruction() {
    return std::string(kDllName) + " not found: copy it from <DLSS SDK>/lib/Windows_x86_64/rel/ into bin/nvidia/ next to the executable "
                                   "(or set DLSSVID_NVIDIA_DLL_DIR) — see docs/dll-setup.md";
}
}  // namespace

#if defined(DLSSVID_WITH_DLSS)

namespace {

void CheckNgx(NVSDK_NGX_Result r, const char* what) { ngx::Check(r, what); }

const char* QualityName(NVSDK_NGX_PerfQuality_Value q) {
    switch (q) {
        case NVSDK_NGX_PerfQuality_Value_DLAA: return "dlaa";
        case NVSDK_NGX_PerfQuality_Value_MaxQuality: return "quality";
        case NVSDK_NGX_PerfQuality_Value_Balanced: return "balanced";
        case NVSDK_NGX_PerfQuality_Value_MaxPerf: return "performance";
        case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return "ultra_performance";
        default: return "?";
    }
}

const char* PresetParamFor(NVSDK_NGX_PerfQuality_Value q) {
    switch (q) {
        case NVSDK_NGX_PerfQuality_Value_DLAA: return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA;
        case NVSDK_NGX_PerfQuality_Value_MaxQuality: return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality;
        case NVSDK_NGX_PerfQuality_Value_Balanced: return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced;
        case NVSDK_NGX_PerfQuality_Value_MaxPerf: return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance;
        default: return NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance;
    }
}

}  // namespace

struct DlssUpscaler::Impl {
    D3D12Device* device = nullptr;
    std::shared_ptr<ngx::Runtime> runtime;
    NVSDK_NGX_Parameter* caps = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
    int flags = 0;
    bool mvLowRes = true;
    unsigned optW = 0, optH = 0, minW = 0, minH = 0, maxW = 0, maxH = 0;
    std::filesystem::path dllDir;
    ComPtr<ID3D12Resource> fallbackDepth, fallbackMv;  // used when the frame has no depth / mv pass
    bool warnedDepth = false, warnedMv = false;
    int64_t frames = 0;
    int presetValue = 0;
};

DlssUpscaler::DlssUpscaler() = default;
DlssUpscaler::~DlssUpscaler() { Shutdown(); }

UpscalerAvailability DlssUpscaler::Available(const std::filesystem::path& dllDir) {
    if (FindDll(dllDir).empty()) return {false, DllInstruction()};
    return {true, {}};
}

void DlssUpscaler::Init(D3D12Device& device, const UpscalerConfig& config) {
    Shutdown();
    config_ = config;
    impl_ = std::make_unique<Impl>();
    Impl& im = *impl_;
    im.device = &device;
    im.dllDir = FindDll(config.dllDir);
    if (im.dllDir.empty()) Throw("dlss: " + DllInstruction());
    if (!device.IsNvidia()) Throw("dlss: needs an NVIDIA RTX GPU (current adapter: " + device.AdapterName() + ")");

    im.runtime = ngx::Runtime::Acquire(device, {im.dllDir});

    // ---- capability check ----
    CheckNgx(NVSDK_NGX_D3D12_GetCapabilityParameters(&im.caps), "GetCapabilityParameters");
    int available = 0;
    NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (!available) {
        int needsDriver = 0, major = 0, minor = 0, initResult = 0;
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &major);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minor);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
        std::string why = "dlss: DLSS Super Resolution is not available on this GPU/driver";
        if (needsDriver) why += " (driver update required, minimum " + std::to_string(major) + "." + std::to_string(minor) + ")";
        char code[32];
        std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(initResult));
        why += std::string(", FeatureInitResult ") + code;
        Throw(why);
    }

    // ---- quality mode from the scale, render range check ----
    const double ratio = static_cast<double>(config.outputWidth) / std::max(1u, config.inputWidth);
    if (std::fabs(ratio - 1.0) < 0.01) im.quality = NVSDK_NGX_PerfQuality_Value_DLAA;
    else if (ratio <= 1.55) im.quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    else if (ratio <= 1.85) im.quality = NVSDK_NGX_PerfQuality_Value_Balanced;
    else if (ratio <= 2.25) im.quality = NVSDK_NGX_PerfQuality_Value_MaxPerf;
    else im.quality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    float sharpness = 0.f;
    CheckNgx(NGX_DLSS_GET_OPTIMAL_SETTINGS(im.caps, config.outputWidth, config.outputHeight, im.quality, &im.optW, &im.optH, &im.maxW, &im.maxH, &im.minW, &im.minH,
                                          &sharpness),
             "GetOptimalSettings");
    if (im.quality != NVSDK_NGX_PerfQuality_Value_DLAA &&
        (config.inputWidth < im.minW || config.inputWidth > im.maxW || config.inputHeight < im.minH || config.inputHeight > im.maxH)) {
        Throw("dlss: input " + std::to_string(config.inputWidth) + "x" + std::to_string(config.inputHeight) + " is outside the render range " + std::to_string(im.minW) +
              "x" + std::to_string(im.minH) + " .. " + std::to_string(im.maxW) + "x" + std::to_string(im.maxH) + " that DLSS accepts for a " +
              std::to_string(config.outputWidth) + "x" + std::to_string(config.outputHeight) + " output in " + QualityName(im.quality) +
              " mode: use --scale 1.5|2|3 (or --target close to those ratios)");
    }

    // ---- feature parameters ----
    CheckNgx(NVSDK_NGX_D3D12_AllocateParameters(&im.params), "AllocateParameters");
    im.presetValue = 0;
    if (config.preset != "default" && !config.preset.empty()) {
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(config.preset[0])));
        switch (c) {
            case 'J': im.presetValue = NVSDK_NGX_DLSS_Hint_Render_Preset_J; break;
            case 'K': im.presetValue = NVSDK_NGX_DLSS_Hint_Render_Preset_K; break;
            case 'L': im.presetValue = NVSDK_NGX_DLSS_Hint_Render_Preset_L; break;
            case 'M': im.presetValue = NVSDK_NGX_DLSS_Hint_Render_Preset_M; break;
            default: Throw("dlss: --preset must be default | J | K | L | M");
        }
        NVSDK_NGX_Parameter_SetUI(im.params, PresetParamFor(im.quality), static_cast<unsigned>(im.presetValue));
    }
    const unsigned mvW = config.extra.value("mv_width", 0u), mvH = config.extra.value("mv_height", 0u);
    im.mvLowRes = !(mvW == config.outputWidth && mvH == config.outputHeight);
    im.flags = NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;  // depth_dlss is reverse-Z (docs/conventions.md)
    if (im.mvLowRes) im.flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    // LDR: decoded video is display-referred (gamma) in [0, 1] -> IsHDR stays 0 (guide §3.1.2)

    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = config.inputWidth;
    cp.Feature.InHeight = config.inputHeight;
    cp.Feature.InTargetWidth = config.outputWidth;
    cp.Feature.InTargetHeight = config.outputHeight;
    cp.Feature.InPerfQualityValue = im.quality;
    cp.InFeatureCreateFlags = im.flags;
    cp.InEnableOutputSubrects = false;
    NVSDK_NGX_Result created = NVSDK_NGX_Result_Success;
    device.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { created = NGX_D3D12_CREATE_DLSS_EXT(cl, 1, 1, &im.feature, im.params, &cp); });
    CheckNgx(created, "CreateFeature(DLSS)");

    // ---- fallbacks for frames without depth / mv passes ----
    {
        std::vector<float> depth(static_cast<size_t>(config.inputWidth) * config.inputHeight, 0.5f);
        im.fallbackDepth = device.CreateTexture2D(config.inputWidth, config.inputHeight, DXGI_FORMAT_R32_FLOAT);
        device.UploadTexture2D(im.fallbackDepth.Get(), reinterpret_cast<const uint8_t*>(depth.data()), config.inputWidth * sizeof(float));
        const uint32_t fw = im.mvLowRes ? config.inputWidth : config.outputWidth, fh = im.mvLowRes ? config.inputHeight : config.outputHeight;
        std::vector<float> mv(static_cast<size_t>(fw) * fh * 2, 0.f);
        im.fallbackMv = device.CreateTexture2D(fw, fh, DXGI_FORMAT_R32G32_FLOAT);
        device.UploadTexture2D(im.fallbackMv.Get(), reinterpret_cast<const uint8_t*>(mv.data()), fw * 2 * sizeof(float));
    }
    Log()->info("dlss: {}x{} -> {}x{} ({}, render range {}x{}..{}x{}, preset {}, mv {}res, dll {})", config.inputWidth, config.inputHeight, config.outputWidth,
                config.outputHeight, QualityName(im.quality), im.minW, im.minH, im.maxW, im.maxH, config.preset, im.mvLowRes ? "low-" : "high-", im.dllDir.string());
}

void DlssUpscaler::Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) {
    if (!impl_ || !impl_->feature) Throw("dlss: not initialised");
    Impl& im = *impl_;
    ID3D12Resource* depth = in.depth ? in.depth : im.fallbackDepth.Get();
    ID3D12Resource* mv = in.mv ? in.mv : im.fallbackMv.Get();
    if (!in.depth && !im.warnedDepth) {
        Log()->warn("dlss: no depth_dlss for frame {} — using a constant depth (run `dlssvid depth` or pass --depth-dir for better quality)", in.frameIndex);
        im.warnedDepth = true;
    }
    if (!in.mv && !im.warnedMv) {
        Log()->warn("dlss: no mv_dlss for frame {} — using zero motion vectors (run `dlssvid flow` or pass --mv-dir)", in.frameIndex);
        im.warnedMv = true;
    }
    for (ID3D12Resource* r : {in.color, depth, mv}) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    NVSDK_NGX_D3D12_DLSS_Eval_Params ep{};
    ep.Feature.pInColor = in.color;
    ep.Feature.pInOutput = output;
    ep.Feature.InSharpness = 0.f;
    ep.pInDepth = depth;
    ep.pInMotionVectors = mv;
    ep.InJitterOffsetX = config_.jitterSign * in.jitterX;
    ep.InJitterOffsetY = config_.jitterSign * in.jitterY;
    ep.InRenderSubrectDimensions = {config_.inputWidth, config_.inputHeight};
    ep.InReset = in.reset ? 1 : 0;
    ep.InMVScaleX = 1.f;
    ep.InMVScaleY = 1.f;
    ep.InFrameTimeDeltaInMsec = static_cast<float>(in.frameTimeMs);
    CheckNgx(NGX_D3D12_EVALUATE_DLSS_EXT(cl, im.feature, im.params, &ep), "EvaluateFeature(DLSS)");

    for (ID3D12Resource* r : {in.color, depth, mv}) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    ++im.frames;
}

nlohmann::json DlssUpscaler::Describe() const {
    nlohmann::json j = {{"backend", "dlss"}, {"sdk", "ngx"}, {"jitter", config_.useJitter}, {"jitter_sign", config_.jitterSign}, {"preset", config_.preset}};
    if (impl_) {
        j["quality"] = QualityName(impl_->quality);
        j["render_range"] = {impl_->minW, impl_->minH, impl_->maxW, impl_->maxH};
        j["mv_low_res"] = impl_->mvLowRes;
        j["dll_dir"] = impl_->dllDir.string();
    }
    return j;
}

void DlssUpscaler::Shutdown() {
    if (!impl_) return;
    Impl& im = *impl_;
    if (im.device) im.device->WaitIdle();
    if (im.feature) NVSDK_NGX_D3D12_ReleaseFeature(im.feature);
    if (im.params) NVSDK_NGX_D3D12_DestroyParameters(im.params);
    if (im.caps) NVSDK_NGX_D3D12_DestroyParameters(im.caps);
    im.feature = nullptr;
    im.params = im.caps = nullptr;
    im.fallbackDepth.Reset();
    im.fallbackMv.Reset();
    im.runtime.reset();
    impl_.reset();
}

#else  // !DLSSVID_WITH_DLSS

struct DlssUpscaler::Impl {};
DlssUpscaler::DlssUpscaler() = default;
DlssUpscaler::~DlssUpscaler() = default;

UpscalerAvailability DlssUpscaler::Available(const std::filesystem::path&) {
    return {false, "built without the DLSS SDK: set DLSS_SDK_ROOT (git clone https://github.com/NVIDIA/DLSS) and reconfigure (docs/dll-setup.md)"};
}
void DlssUpscaler::Init(D3D12Device&, const UpscalerConfig&) { Throw("dlss: " + Available().reason); }
void DlssUpscaler::Evaluate(ID3D12GraphicsCommandList*, const UpscaleInputs&, ID3D12Resource*) { Throw("dlss: not available"); }
nlohmann::json DlssUpscaler::Describe() const { return {{"backend", "dlss"}, {"available", false}}; }
void DlssUpscaler::Shutdown() {}

#endif

}  // namespace dlssvid
