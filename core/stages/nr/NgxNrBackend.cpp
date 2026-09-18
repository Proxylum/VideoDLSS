#include "stages/nr/NgxNrBackend.h"

#include <cstring>
#include <string>

#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

#if defined(DLSSVID_WITH_DLSS)
#include <windows.h>

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_params.h>

#include "gpu/ComputeKernel.h"
#include "gpu/Ngx.h"
#endif
#if defined(DLSSVID_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace dlssvid {

namespace {

std::string ForwarderInstruction() {
    return std::string(kNrForwarderName) + " not found next to the executable (it is built with the project, target dlssvid_nr_forwarder) — rebuild or copy it into bin/";
}

// Architecture from the CUDA compute capability of the adapter (exact), "" when CUDA cannot see it.
std::string CudaArchitecture(const D3D12Device& device) {
#if defined(DLSSVID_WITH_CUDA)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return "";
    const LUID luid = device.AdapterLuid();
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        if (std::memcmp(prop.luid, &luid, sizeof(luid)) != 0) continue;
        const std::string arch = GpuArchitectureFromComputeCapability(prop.major, prop.minor);
        return (arch.empty() ? std::string("unknown") : arch) + " (sm_" + std::to_string(prop.major * 10 + prop.minor) + ")";
    }
#else
    (void)device;
#endif
    return "";
}

}  // namespace

#if defined(DLSSVID_WITH_DLSS)

namespace {

constexpr int kFeatureId = static_cast<int>(NVSDK_NGX_Feature_Reserved18);  // NGX Feature 18 = Neural Rendering
constexpr unsigned long long kSnippetAppId = 141959980ULL;                  // application id the snippet's Init_Ext is given (as ComfyUI-DLSS5-NR)

using FwdInitFn = int(__cdecl*)(void*, int, unsigned long long, const wchar_t*, void*, int, const void*);
using FwdCreateFn = int(__cdecl*)(void*, void*, int, void*, void**);
using FwdEvaluateFn = int(__cdecl*)(void*, void*, void*, void*, void*);
using FwdReleaseFn = int(__cdecl*)(void*, void*);

void SetRes(NVSDK_NGX_Parameter* p, const char* name, ID3D12Resource* r, unsigned w, unsigned h, const char* subrect) {
    NVSDK_NGX_Parameter_SetD3d12Resource(p, name, r);
    const std::string base = std::string("DLSSNR.") + subrect + "Subrect";
    NVSDK_NGX_Parameter_SetUI(p, (base + "BaseX").c_str(), 0);
    NVSDK_NGX_Parameter_SetUI(p, (base + "BaseY").c_str(), 0);
    NVSDK_NGX_Parameter_SetUI(p, (base + "Width").c_str(), r ? w : 0);
    NVSDK_NGX_Parameter_SetUI(p, (base + "Height").c_str(), r ? h : 0);
}

// Everything the model reads: written before CreateFeature (the tuning is read then) and again before every
// EvaluateFeature (the block is the core's and shared, OptiScaler_DLSSNR). Names: DlssNr_Common.h / caller_shim.
void SetAllParams(NVSDK_NGX_Parameter* p, const NrConfig& c, ID3D12Resource* color, ID3D12Resource* output, ID3D12Resource* depth, unsigned dw, unsigned dh,
                  ID3D12Resource* mv, unsigned mw, unsigned mh, bool reset) {
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Enabled", 1);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Width", c.width);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Height", c.height);
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Hint.Render.Preset", static_cast<unsigned>(c.preset));
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.Intensity", c.intensity);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Style", static_cast<unsigned>(c.style));
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.LocalStructureStrength", c.localStructure);
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.LocalToneStrength", c.localTone);
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.SkinStructureStrength", c.skinStructure);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.UseAutoMask", c.autoMask ? 1 : 0);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.UICorrection", 0);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.DepthInverted", c.depthInverted ? 1 : 0);
    NVSDK_NGX_Parameter_SetUI(p, "DLSSNR.Reset", reset ? 1 : 0);
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.ScalingRatio", 1.f);
    SetRes(p, "DLSSNR.Color", color, c.width, c.height, "Color");
    SetRes(p, "DLSSNR.Output", output, c.width, c.height, "Output");
    NVSDK_NGX_Parameter_SetD3d12Resource(p, "DLSSNR.Backbuffer", output);
    SetRes(p, "DLSSNR.Depth", depth, dw, dh, "Depth");
    SetRes(p, "DLSSNR.MVec", mv, mw, mh, "MVec");
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.MVecScaleX", mv ? c.mvScaleX : 1.f);
    NVSDK_NGX_Parameter_SetF(p, "DLSSNR.MVecScaleY", mv ? c.mvScaleY : 1.f);
}

std::string Hex(int r) {
    char code[32];
    std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(r));
    return code;
}

}  // namespace

struct NgxNrBackend::Impl {
    D3D12Device* device = nullptr;
    std::shared_ptr<ngx::Runtime> runtime;
    HMODULE snippet = nullptr, forwarder = nullptr;
    void* snipInit = nullptr;
    void* snipCreate = nullptr;
    void* snipEvaluate = nullptr;
    void* snipRelease = nullptr;
    FwdInitFn fwdInit = nullptr;
    FwdCreateFn fwdCreate = nullptr;
    FwdEvaluateFn fwdEvaluate = nullptr;
    FwdReleaseFn fwdRelease = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    bool paramsAllocated = false;
    NVSDK_NGX_Handle* feature = nullptr;
    ComPtr<ID3D12Resource> scratchColor, scratchOutput;  // bound at CreateFeature
    bool warnedDepth = false, warnedMv = false;
    int64_t frames = 0;
};

NgxNrBackend::NgxNrBackend() = default;
NgxNrBackend::~NgxNrBackend() { Shutdown(); }

NrAvailability NgxNrBackend::Available(const std::filesystem::path& dllDir) {
    if (FindNrDll(dllDir).empty()) return {false, NrDllInstruction()};
    if (FindNrForwarder(dllDir).empty()) return {false, ForwarderInstruction()};
    return {true, {}};
}

void NgxNrBackend::Init(D3D12Device& device, const NrConfig& config) {
    Shutdown();
    config_ = config;
    diag_ = {};
    impl_ = std::make_unique<Impl>();
    Impl& im = *impl_;
    im.device = &device;

    // ---- what we are running on (ТЗ §4: logged before anything can fail) ----
    diag_.gpu = device.AdapterName();
    const std::string byName = GpuArchitectureFromName(device.AdapterName());
    const std::string byCuda = CudaArchitecture(device);
    diag_.architecture = !byCuda.empty() ? byCuda : byName.empty() ? "unknown" : byName + " (by name)";
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor);
    diag_.dll = FindNrDll(config.dllDir);
    diag_.forwarder = FindNrForwarder(config.dllDir);
    diag_.paramsBlock = config.paramsBlock;
    if (!diag_.dll.empty()) {
        std::error_code ec;
        diag_.dllSize = std::filesystem::file_size(diag_.dll, ec);
        diag_.dllSha256 = Sha256File(diag_.dll);
    }
    Log()->info("nr: GPU {} — architecture {}, driver {}{}", diag_.gpu, diag_.architecture, diag_.driver.ToString(),
                diag_.driver.valid ? (diag_.driverOk ? " (>= 616.56 ok)" : " (< 616.56!)") : " (unknown)");
    if (!diag_.dll.empty()) Log()->info("nr: {} at {} ({} bytes, sha256 {})", kNrDllName, diag_.dll.string(), diag_.dllSize, diag_.dllSha256);
    if (!diag_.forwarder.empty()) Log()->info("nr: forwarder {}", diag_.forwarder.string());

    if (!device.IsNvidia()) {
        diag_.hint = "Neural Rendering needs an NVIDIA RTX GPU (current adapter: " + device.AdapterName() + ")";
        Throw("nr: " + diag_.hint);
    }
    if (diag_.dll.empty()) {
        diag_.hint = NrDllInstruction();
        Throw("nr: " + diag_.hint);
    }
    if (diag_.forwarder.empty()) {
        diag_.hint = ForwarderInstruction();
        Throw("nr: " + diag_.hint);
    }
    if (diag_.driver.valid && !diag_.driverOk) {
        diag_.hint = "NVIDIA driver " + diag_.driver.ToString() + " is older than " + std::to_string(kNrMinDriverMajor) + "." + std::to_string(kNrMinDriverMinor) +
                     " required by Neural Rendering: update the driver (docs/dll-setup.md); --skip-driver-check tries anyway";
        if (config.requireDriver) Throw("nr: " + diag_.hint);
        Log()->warn("nr: {}", diag_.hint);
    } else if (!diag_.driver.valid) {
        Log()->warn("nr: the NVIDIA driver version could not be read; Neural Rendering needs >= 616.56");
    }

    // ---- NGX core (shared with DLSS SR), the snippet and the forwarder ----
    im.runtime = ngx::Runtime::Acquire(device, {diag_.dll.parent_path()});
    im.snippet = LoadLibraryExW(diag_.dll.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!im.snippet) {
        diag_.hint = "LoadLibrary(" + diag_.dll.string() + ") failed (Win32 error " + std::to_string(GetLastError()) +
                     "): a patched DLL is unsigned — check the antivirus quarantine and add an exclusion for bin/nvidia/ (docs/dll-setup.md)";
        Throw("nr: " + diag_.hint);
    }
    im.snipInit = reinterpret_cast<void*>(GetProcAddress(im.snippet, "NVSDK_NGX_D3D12_Init_Ext"));
    im.snipCreate = reinterpret_cast<void*>(GetProcAddress(im.snippet, "NVSDK_NGX_D3D12_CreateFeature"));
    im.snipEvaluate = reinterpret_cast<void*>(GetProcAddress(im.snippet, "NVSDK_NGX_D3D12_EvaluateFeature"));
    im.snipRelease = reinterpret_cast<void*>(GetProcAddress(im.snippet, "NVSDK_NGX_D3D12_ReleaseFeature"));
    if (!im.snipInit || !im.snipCreate || !im.snipEvaluate || !im.snipRelease) {
        diag_.hint = diag_.dll.string() + " does not export the NVSDK_NGX_D3D12_* entry points of an NGX snippet — is it really nvngx_dlssnr.dll?";
        Throw("nr: " + diag_.hint);
    }
    im.forwarder = LoadLibraryExW(diag_.forwarder.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!im.forwarder) {
        diag_.hint = "LoadLibrary(" + diag_.forwarder.string() + ") failed (Win32 error " + std::to_string(GetLastError()) + ")";
        Throw("nr: " + diag_.hint);
    }
    im.fwdInit = reinterpret_cast<FwdInitFn>(GetProcAddress(im.forwarder, "dlssvid_nr_call_init"));
    im.fwdCreate = reinterpret_cast<FwdCreateFn>(GetProcAddress(im.forwarder, "dlssvid_nr_call_create"));
    im.fwdEvaluate = reinterpret_cast<FwdEvaluateFn>(GetProcAddress(im.forwarder, "dlssvid_nr_call_evaluate"));
    im.fwdRelease = reinterpret_cast<FwdReleaseFn>(GetProcAddress(im.forwarder, "dlssvid_nr_call_release"));
    if (!im.fwdInit || !im.fwdCreate || !im.fwdEvaluate || !im.fwdRelease) {
        diag_.hint = diag_.forwarder.string() + " is missing its dlssvid_nr_call_* exports (stale build?)";
        Throw("nr: " + diag_.hint);
    }

    // ---- parameter block ----
    auto acquireParams = [&](const std::string& kind) {
        if (im.params && im.paramsAllocated) NVSDK_NGX_D3D12_DestroyParameters(im.params);
        im.params = nullptr;
        if (kind == "alloc") {
            ngx::Check(NVSDK_NGX_D3D12_AllocateParameters(&im.params), "AllocateParameters");
            im.paramsAllocated = true;
        } else {
            ngx::Check(NVSDK_NGX_D3D12_GetCapabilityParameters(&im.params), "GetCapabilityParameters");
            im.paramsAllocated = false;
        }
        diag_.paramsBlock = kind;
    };
    if (config.paramsBlock != "capability" && config.paramsBlock != "alloc") Throw("nr: --params-block must be capability | alloc");
    acquireParams(config.paramsBlock);

    // ---- the snippet's own Init_Ext: its ABI is not the public one and the references disagree, so try in turn ----
    {
        const std::wstring wdir = diag_.dll.parent_path().wstring();
        const wchar_t* paths[] = {wdir.c_str()};
        NVSDK_NGX_FeatureCommonInfo fci{};
        fci.PathListInfo.Path = paths;
        fci.PathListInfo.Length = 1;
        fci.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
        const std::wstring appData = ngx::AppDataPath().wstring();
        struct Attempt {
            int order;
            const void* fifth;
            const char* name;
        };
        const Attempt attempts[] = {{0, &fci, "(app, path, device, info, version) — ComfyUI"},
                                    {1, &fci, "(app, path, device, version, info) — public SDK"},
                                    {1, im.params, "(app, path, device, version, params) — OptiScaler"}};
        bool ok = false;
        std::string tried;
        for (const Attempt& a : attempts) {
            const int r = im.fwdInit(im.snipInit, a.order, kSnippetAppId, appData.c_str(), device.Get(), static_cast<int>(NVSDK_NGX_Version_API), a.fifth);
            tried += std::string(tried.empty() ? "" : "; ") + a.name + " -> " + ngx::ResultString(r);
            if (r == static_cast<int>(NVSDK_NGX_Result_Success)) {
                diag_.initAbi = static_cast<int>(&a - attempts);
                diag_.initResult = ngx::ResultString(r);
                Log()->info("nr: snippet Init_Ext ok with ABI {}", a.name);
                ok = true;
                break;
            }
        }
        if (!ok) {
            diag_.initResult = tried;
            diag_.hint = "the snippet's Init_Ext failed for every known argument order: " + tried + " — see the NGX log in " + ngx::AppDataPath().string();
            Throw("nr: " + diag_.hint);
        }
    }

    // ---- CreateFeature(18) with scratch textures; the result is what ТЗ §4 asks to log ----
    im.scratchColor = device.CreateTexture2D(config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    im.scratchOutput = device.CreateTexture2D(config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto create = [&]() {
        SetAllParams(im.params, config_, im.scratchColor.Get(), im.scratchOutput.Get(), nullptr, 0, 0, nullptr, 0, 0, true);
        int r = 0;
        device.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
            ComputeKernel::Transition(cl, im.scratchColor.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ComputeKernel::Transition(cl, im.scratchOutput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            r = im.fwdCreate(im.snipCreate, cl, kFeatureId, im.params, reinterpret_cast<void**>(&im.feature));
            ComputeKernel::Transition(cl, im.scratchColor.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            ComputeKernel::Transition(cl, im.scratchOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        });
        return r;
    };
    int created = create();
    diag_.createResult = ngx::ResultString(created);
    Log()->info("nr: CreateFeature(18) [{} block] -> {}", diag_.paramsBlock, diag_.createResult);
    if (created != static_cast<int>(NVSDK_NGX_Result_Success) &&
        (created == static_cast<int>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature) || created == static_cast<int>(NVSDK_NGX_Result_FAIL_InvalidParameter) ||
         created == static_cast<int>(NVSDK_NGX_Result_FAIL_MissingInput))) {
        const std::string other = diag_.paramsBlock == "capability" ? "alloc" : "capability";
        Log()->warn("nr: retrying CreateFeature(18) with the {} parameter block", other);
        im.feature = nullptr;
        acquireParams(other);
        const int again = create();
        diag_.createResult += "; " + other + " block -> " + ngx::ResultString(again);
        Log()->info("nr: CreateFeature(18) [{} block] -> {}", other, ngx::ResultString(again));
        created = again;
    }
    if (created != static_cast<int>(NVSDK_NGX_Result_Success) || !im.feature) {
        im.feature = nullptr;
        if (created == static_cast<int>(NVSDK_NGX_Result_FAIL_FeatureNotSupported)) {
            if (diag_.architecture.find("Blackwell") == std::string::npos)
                diag_.hint = "FAIL_FeatureNotSupported: the official nvngx_dlssnr.dll refuses GPUs before RTX 50 (this one: " + diag_.architecture +
                             "). Patch your copy with `dlssvid nr-patch --input <nvngx_dlssnr.dll>` (dlssnr-patcher, CUDA 13.3) and keep the result in bin/nvidia/ — docs/dll-setup.md";
            else
                diag_.hint = "FAIL_FeatureNotSupported on a Blackwell GPU: check the driver (>= 616.56) and that the DLL is a DLSS 5 nvngx_dlssnr.dll";
        } else if (created == static_cast<int>(NVSDK_NGX_Result_FAIL_PlatformError)) {
            diag_.hint = "FAIL_PlatformError: the model rejected the calling module — the forwarder " + diag_.forwarder.string() +
                         " must be the one built with this executable (its name must contain nvngx.dll)";
        } else {
            diag_.hint = "CreateFeature(18) failed with " + diag_.createResult + " — see the NGX log in " + ngx::AppDataPath().string();
        }
        Throw("nr: " + diag_.hint);
    }
    diag_.ok = true;
    Log()->info("nr: {}x{} ready (intensity {}, style {}, preset {}, guides {}, {} block)", config.width, config.height, config.intensity, config.style, config.preset,
                config.useGuides ? "on" : "off (still mode)", diag_.paramsBlock);
}

void NgxNrBackend::Evaluate(ID3D12GraphicsCommandList* cl, const NrInputs& in, ID3D12Resource* output) {
    if (!impl_ || !impl_->feature) Throw("nr: not initialised");
    Impl& im = *impl_;
    if (!in.color || !output) Throw("nr: colour input and output are required");
    ID3D12Resource* depth = config_.useGuides ? in.depth : nullptr;
    ID3D12Resource* mv = config_.useGuides ? in.mv : nullptr;
    if (config_.useGuides && !in.depth && !im.warnedDepth) {
        Log()->warn("nr: no depth_dlss for frame {} — the model runs without a depth guide (run `dlssvid depth` or pass --depth-dir)", in.frameIndex);
        im.warnedDepth = true;
    }
    if (config_.useGuides && !in.mv && !im.warnedMv) {
        Log()->warn("nr: no mv_dlss for frame {} — still-image mode, history reset every frame (run `dlssvid flow` or pass --mv-dir)", in.frameIndex);
        im.warnedMv = true;
    }
    const bool reset = in.reset || !mv;  // without motion vectors the history cannot be aligned (ComfyUI still-image mode)

    for (ID3D12Resource* r : {in.color, depth, mv})
        if (r) ComputeKernel::Transition(cl, r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    SetAllParams(im.params, config_, in.color, output, depth, in.depthWidth, in.depthHeight, mv, in.mvWidth, in.mvHeight, reset);
    const int r = im.fwdEvaluate(im.snipEvaluate, cl, im.feature, im.params, nullptr);
    if (r != static_cast<int>(NVSDK_NGX_Result_Success)) {
        for (ID3D12Resource* res : {in.color, depth, mv})
            if (res) ComputeKernel::Transition(cl, res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        Throw("nr: EvaluateFeature(18) failed on frame " + std::to_string(in.frameIndex) + ": " + ngx::ResultString(r) +
              (r == static_cast<int>(NVSDK_NGX_Result_FAIL_UnsupportedInputFormat) ? " (guide texture format: mv_dlss is RG32F, depth_dlss R32F)" : ""));
    }

    for (ID3D12Resource* res : {in.color, depth, mv})
        if (res) ComputeKernel::Transition(cl, res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    ComputeKernel::Transition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    ++im.frames;
}

nlohmann::json NgxNrBackend::Describe() const {
    nlohmann::json j = {{"backend", "ngx"},
                        {"feature", kFeatureId},
                        {"intensity", config_.intensity},
                        {"style", config_.style},
                        {"preset", config_.preset},
                        {"local_tone", config_.localTone},
                        {"local_structure", config_.localStructure},
                        {"skin_structure", config_.skinStructure},
                        {"auto_mask", config_.autoMask},
                        {"guides", config_.useGuides},
                        {"params_block", diag_.paramsBlock}};
    j["diagnostics"] = diag_.ToJson();
    return j;
}

void NgxNrBackend::Shutdown() {
    if (!impl_) return;
    Impl& im = *impl_;
    if (im.device) im.device->WaitIdle();
    if (im.feature && im.fwdRelease) im.fwdRelease(im.snipRelease, im.feature);
    im.feature = nullptr;
    if (im.params && im.paramsAllocated) NVSDK_NGX_D3D12_DestroyParameters(im.params);
    im.params = nullptr;
    im.scratchColor.Reset();
    im.scratchOutput.Reset();
    im.runtime.reset();
    if (im.forwarder) FreeLibrary(im.forwarder);
    if (im.snippet) FreeLibrary(im.snippet);
    im.forwarder = im.snippet = nullptr;
    impl_.reset();
}

#else  // !DLSSVID_WITH_DLSS

struct NgxNrBackend::Impl {};
NgxNrBackend::NgxNrBackend() = default;
NgxNrBackend::~NgxNrBackend() = default;

NrAvailability NgxNrBackend::Available(const std::filesystem::path&) {
    return {false, "built without the DLSS SDK (NGX headers): set DLSS_SDK_ROOT (git clone https://github.com/NVIDIA/DLSS) and reconfigure (docs/dll-setup.md)"};
}
void NgxNrBackend::Init(D3D12Device& device, const NrConfig& config) {
    config_ = config;
    diag_ = {};
    diag_.gpu = device.AdapterName();
    diag_.architecture = GpuArchitectureFromName(device.AdapterName());
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor);
    diag_.hint = Available().reason;
    Throw("nr: " + diag_.hint);
}
void NgxNrBackend::Evaluate(ID3D12GraphicsCommandList*, const NrInputs&, ID3D12Resource*) { Throw("nr: ngx backend not available"); }
nlohmann::json NgxNrBackend::Describe() const { return {{"backend", "ngx"}, {"available", false}}; }
void NgxNrBackend::Shutdown() {}

#endif

}  // namespace dlssvid
