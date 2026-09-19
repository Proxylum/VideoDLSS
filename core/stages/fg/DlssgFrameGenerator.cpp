#include "stages/fg/DlssgFrameGenerator.h"

#include <cfloat>
#include <cstring>
#include <vector>

#include "util/Error.h"
#include "util/Half.h"
#include "util/Log.h"
#include "util/Sha256.h"

#if defined(DLSSVID_WITH_DLSS)
#include <nvsdk_ngx_helpers_dlssg_d3d.h>

#include "gpu/ComputeKernel.h"
#include "gpu/Ngx.h"
#include "stages/upscale/BicubicUpscaler.h"
#endif

namespace dlssvid {

#if defined(DLSSVID_WITH_DLSS)

namespace {
DXGI_FORMAT FormatFor(const std::string& s) {
    if (s == "rgba16f") return DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (s == "rgba8") return DXGI_FORMAT_R8G8B8A8_UNORM;
    Throw("fg: --backbuffer-format must be rgba16f | rgba8 (got '" + s + "')");
}
}  // namespace

struct DlssgFrameGenerator::Impl {
    D3D12Device* device = nullptr;
    std::shared_ptr<ngx::Runtime> runtime;
    NVSDK_NGX_Parameter* caps = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    ComPtr<ID3D12Resource> backbuffer8, output;  // backbuffer8: RGBA8 copy of the colour when the runtime wants rgba8
    std::unique_ptr<Resampler> converter;
    ComPtr<ID3D12Resource> fallbackDepth, fallbackMv;
    uint32_t fallbackW = 0, fallbackH = 0;
    FgCamera camera;
    bool warnedDepth = false, warnedMv = false;
    int64_t frames = 0;
};

DlssgFrameGenerator::DlssgFrameGenerator() = default;
DlssgFrameGenerator::~DlssgFrameGenerator() { Shutdown(); }

FgAvailability DlssgFrameGenerator::Available(const std::filesystem::path& dllDir) {
    if (FindDlssgDll(dllDir).empty()) return {false, DlssgDllInstruction()};
    return {true, {}};
}

void DlssgFrameGenerator::Init(D3D12Device& device, const FgConfig& config) {
    Shutdown();
    config_ = config;
    diag_ = {};
    impl_ = std::make_unique<Impl>();
    Impl& im = *impl_;
    im.device = &device;
    if (config.multiplier < 2 || config.multiplier > 4) Throw("fg: --multiplier must be 2 | 3 | 4");

    diag_.gpu = device.AdapterName();
    const std::string byName = GpuArchitectureFromName(device.AdapterName());
    diag_.architecture = byName.empty() ? "unknown" : byName;
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.valid;
    diag_.dll = FindDlssgDll(config.dllDir);
    diag_.backbufferFormat = config.backbufferFormat;
    if (!diag_.dll.empty()) {
        std::error_code ec;
        diag_.dllSize = std::filesystem::file_size(diag_.dll, ec);
        diag_.dllSha256 = Sha256File(diag_.dll);
    }
    Log()->info("fg: GPU {} — architecture {}, driver {}", diag_.gpu, diag_.architecture, diag_.driver.ToString());
    if (!diag_.dll.empty()) Log()->info("fg: {} at {} ({} bytes, sha256 {})", kDlssgDllName, diag_.dll.string(), diag_.dllSize, diag_.dllSha256);
    if (!device.IsNvidia()) {
        diag_.hint = "DLSS Frame Generation needs an NVIDIA RTX 40 (or newer) GPU (current adapter: " + device.AdapterName() + ")";
        Throw("fg: " + diag_.hint);
    }
    if (diag_.dll.empty()) {
        diag_.hint = DlssgDllInstruction();
        Throw("fg: " + diag_.hint);
    }

    // ---- NGX core, capability check ----
    im.runtime = ngx::Runtime::Acquire(device, {diag_.dll.parent_path()});
    ngx::Check(NVSDK_NGX_D3D12_GetCapabilityParameters(&im.caps), "GetCapabilityParameters");
    int available = 0;
    NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &available);
    diag_.available = available != 0;
    unsigned int mfMax = 0;
    if (NVSDK_NGX_Parameter_GetUI(im.caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &mfMax) == NVSDK_NGX_Result_Success) diag_.multiFrameMax = static_cast<int>(mfMax);
    if (!diag_.available) {
        int needsDriver = 0, major = 0, minor = 0, initResult = 0;
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needsDriver);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &major);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &minor);
        NVSDK_NGX_Parameter_GetI(im.caps, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &initResult);
        diag_.hint = "DLSS Frame Generation is not available on this GPU / driver (" + diag_.architecture + ", driver " + diag_.driver.ToString() + ")";
        if (needsDriver) diag_.hint += ": update the driver to at least " + std::to_string(major) + "." + std::to_string(minor);
        diag_.hint += ", FeatureInitResult " + ngx::ResultString(initResult) + ". RTX 40 / 50 support it; use --backend rife otherwise";
        Throw("fg: " + diag_.hint);
    }
    Log()->info("fg: FrameGeneration.Available = 1, MultiFrameCountMax = {}", diag_.multiFrameMax);
    if (config.multiplier - 1 > std::max(1, diag_.multiFrameMax)) {
        diag_.hint = "--multiplier " + std::to_string(config.multiplier) + " needs " + std::to_string(config.multiplier - 1) + " generated frames per pair, this GPU / driver allows " +
                     std::to_string(std::max(1, diag_.multiFrameMax)) + " (DLSS Multi Frame Generation is RTX 50); use --multiplier 2 or --backend rife";
        Throw("fg: " + diag_.hint);
    }

    // ---- feature ----
    ngx::Check(NVSDK_NGX_D3D12_AllocateParameters(&im.params), "AllocateParameters");
    const uint32_t gw = config.guideWidth ? config.guideWidth : config.width, gh = config.guideHeight ? config.guideHeight : config.height;
    auto create = [&](DXGI_FORMAT fmt) {
        NVSDK_NGX_DLSSG_Create_Params cp{};
        cp.Width = config.width;
        cp.Height = config.height;
        cp.NativeBackbufferFormat = static_cast<unsigned int>(fmt);
        cp.RenderWidth = gw;
        cp.RenderHeight = gh;
        cp.DynamicResolutionScaling = false;
        NVSDK_NGX_Result r = NVSDK_NGX_Result_Success;
        device.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { r = NGX_D3D12_CREATE_DLSSG(cl, 1, 1, &im.feature, im.params, &cp); });
        return r;
    };
    im.format = FormatFor(config.backbufferFormat);
    NVSDK_NGX_Result created = create(im.format);
    diag_.createResult = ngx::ResultString(created);
    Log()->info("fg: CreateFeature(FrameGeneration) [{}] -> {}", config.backbufferFormat, diag_.createResult);
    if (NVSDK_NGX_FAILED(created) && im.format == DXGI_FORMAT_R16G16B16A16_FLOAT &&
        (created == NVSDK_NGX_Result_FAIL_UnsupportedFormat || created == NVSDK_NGX_Result_FAIL_UnsupportedInputFormat || created == NVSDK_NGX_Result_FAIL_InvalidParameter)) {
        Log()->warn("fg: retrying CreateFeature with an rgba8 backbuffer");
        im.feature = nullptr;
        im.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        created = create(im.format);
        diag_.createResult += "; rgba8 -> " + ngx::ResultString(created);
        diag_.backbufferFormat = "rgba8";
        Log()->info("fg: CreateFeature(FrameGeneration) [rgba8] -> {}", ngx::ResultString(created));
    }
    if (NVSDK_NGX_FAILED(created) || !im.feature) {
        im.feature = nullptr;
        diag_.hint = "CreateFeature(FrameGeneration) failed with " + diag_.createResult + " — see the NGX log in " + ngx::AppDataPath().string();
        Throw("fg: " + diag_.hint);
    }

    // ---- resources: output (backbuffer format, UAV), rgba8 backbuffer copy, guide fallbacks, camera ----
    im.output = device.CreateTexture2D(config.width, config.height, im.format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (im.format != DXGI_FORMAT_R16G16B16A16_FLOAT) {
        im.backbuffer8 = device.CreateTexture2D(config.width, config.height, im.format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        im.converter = std::make_unique<Resampler>(device);
    }
    im.fallbackW = gw;
    im.fallbackH = gh;
    {
        std::vector<float> depth(static_cast<size_t>(gw) * gh, 0.5f);
        im.fallbackDepth = device.CreateTexture2D(gw, gh, DXGI_FORMAT_R32_FLOAT);
        device.UploadTexture2D(im.fallbackDepth.Get(), reinterpret_cast<const uint8_t*>(depth.data()), gw * sizeof(float));
        std::vector<float> mv(static_cast<size_t>(gw) * gh * 2, 0.f);
        im.fallbackMv = device.CreateTexture2D(gw, gh, DXGI_FORMAT_R32G32_FLOAT);
        device.UploadTexture2D(im.fallbackMv.Get(), reinterpret_cast<const uint8_t*>(mv.data()), gw * 2 * sizeof(float));
    }
    im.camera = BuildFgCamera(config.width, config.height);
    diag_.ok = true;
    Log()->info("fg: dlssg {}x{} x{} ready (guides {}x{}, backbuffer {})", config.width, config.height, config.multiplier, gw, gh, diag_.backbufferFormat);
}

void DlssgFrameGenerator::Generate(const FgInputs& in, std::vector<PassImage>& out) {
    if (!impl_ || !impl_->feature) Throw("fg: dlssg not initialised");
    Impl& im = *impl_;
    if (!in.cur.texture) Throw("fg: dlssg needs the current frame as an RGBA16F texture");
    ID3D12Resource* depth = in.depth ? in.depth : im.fallbackDepth.Get();
    ID3D12Resource* mv = in.mv ? in.mv : im.fallbackMv.Get();
    const uint32_t dw = in.depth ? in.depthWidth : im.fallbackW, dh = in.depth ? in.depthHeight : im.fallbackH;
    const uint32_t mw = in.mv ? in.mvWidth : im.fallbackW, mh = in.mv ? in.mvHeight : im.fallbackH;
    if (!in.depth && !im.warnedDepth) {
        Log()->warn("fg: no depth_dlss for frame {} — using a constant depth (run `dlssvid depth` or pass --depth-dir)", in.frameIndex);
        im.warnedDepth = true;
    }
    if (!in.mv && !im.warnedMv) {
        Log()->warn("fg: no mv_dlss for frame {} — using zero motion vectors (run `dlssvid flow` or pass --mv-dir)", in.frameIndex);
        im.warnedMv = true;
    }
    const int count = config_.multiplier - 1;
    out.clear();
    for (int k = 1; k <= count; ++k) {
        NVSDK_NGX_D3D12_DLSSG_Eval_Params ep{};
        NVSDK_NGX_DLSSG_Opt_Eval_Params op{};
        std::memcpy(op.cameraViewToClip, im.camera.viewToClip, sizeof(op.cameraViewToClip));
        std::memcpy(op.clipToCameraView, im.camera.clipToView, sizeof(op.clipToCameraView));
        std::memcpy(op.clipToLensClip, im.camera.identity, sizeof(op.clipToLensClip));
        std::memcpy(op.clipToPrevClip, im.camera.identity, sizeof(op.clipToPrevClip));
        std::memcpy(op.prevClipToClip, im.camera.identity, sizeof(op.prevClipToClip));
        op.multiFrameCount = static_cast<unsigned>(count);
        op.multiFrameIndex = static_cast<unsigned>(k);
        op.jitterOffset[0] = op.jitterOffset[1] = 0.f;
        op.mvecScale[0] = op.mvecScale[1] = 1.f;  // mv_dlss is in pixels at the motion vector resolution (guide §5)
        op.cameraPinholeOffset[0] = op.cameraPinholeOffset[1] = 0.f;
        op.cameraPos[0] = op.cameraPos[1] = op.cameraPos[2] = 0.f;
        op.cameraUp[0] = 0.f;
        op.cameraUp[1] = 1.f;
        op.cameraUp[2] = 0.f;
        op.cameraRight[0] = 1.f;
        op.cameraRight[1] = 0.f;
        op.cameraRight[2] = 0.f;
        op.cameraFwd[0] = 0.f;
        op.cameraFwd[1] = 0.f;
        op.cameraFwd[2] = 1.f;
        op.cameraNear = im.camera.nearPlane;
        op.cameraFar = im.camera.farPlane;
        op.cameraFOV = im.camera.fovRadians;
        op.cameraAspectRatio = im.camera.aspect;
        op.colorBuffersHDR = false;
        op.depthInverted = config_.depthInverted;
        op.cameraMotionIncluded = true;  // optical flow already contains every motion
        op.reset = in.reset;
        op.automodeOverrideReset = false;
        op.notRenderingGameFrames = false;
        op.orthoProjection = false;
        op.motionVectorsInvalidValue = FLT_MAX;  // never present in mv_dlss
        op.motionVectorsDilated = false;
        op.menuDetectionEnabled = false;  // a static shot is not a menu
        op.depthSubrectSize = {dw, dh};
        op.mvecsSubrectSize = {mw, mh};
        op.backbufferSubrectSize = {config_.width, config_.height};
        op.outputInterpSubrectSize = {config_.width, config_.height};

        NVSDK_NGX_Result r = NVSDK_NGX_Result_Success;
        im.device->ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
            ID3D12Resource* backbuffer = in.cur.texture;
            if (im.backbuffer8) {  // rgba16f -> rgba8 copy through the resampler (same size: exact texel centres)
                im.converter->Run(cl, in.cur.texture, config_.width, config_.height, im.backbuffer8.Get(), config_.width, config_.height, 0.f, 0.f, Resampler::Filter::Bilinear);
                backbuffer = im.backbuffer8.Get();
            }
            for (ID3D12Resource* res : {backbuffer, depth, mv}) ComputeKernel::Transition(cl, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ComputeKernel::Transition(cl, im.output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ep.pBackbuffer = backbuffer;
            ep.pDepth = depth;
            ep.pMVecs = mv;
            ep.pOutputInterpFrame = im.output.Get();
            r = NGX_D3D12_EVALUATE_DLSSG(cl, im.feature, im.params, &ep, &op);
            for (ID3D12Resource* res : {backbuffer, depth, mv}) ComputeKernel::Transition(cl, res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            ComputeKernel::Transition(cl, im.output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        });
        if (NVSDK_NGX_FAILED(r)) Throw("fg: EvaluateFeature(FrameGeneration) failed on frame " + std::to_string(in.frameIndex) + " (" + std::to_string(k) + "/" + std::to_string(count) + "): " + ngx::ResultString(r));

        // readback -> RGB F16
        size_t pitch = 0;
        const std::vector<uint8_t> bytes = im.device->ReadbackTexture2D(im.output.Get(), pitch);
        PassImage img = MakePassImage(PassKind::ColorFg, config_.width, config_.height, PixelType::F16);
        uint16_t* dst = img.As<uint16_t>();
        const size_t n = static_cast<size_t>(config_.width) * config_.height;
        if (im.format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
            for (size_t i = 0; i < n; ++i) {
                dst[i * 3] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4 + 1];
                dst[i * 3 + 2] = src[i * 4 + 2];
            }
        } else {
            for (size_t i = 0; i < n; ++i)
                for (size_t c = 0; c < 3; ++c) dst[i * 3 + c] = FloatToHalf(bytes[i * 4 + c] / 255.f);
        }
        out.push_back(std::move(img));
    }
    ++im.frames;
}

nlohmann::json DlssgFrameGenerator::Describe() const {
    nlohmann::json j = {{"backend", "dlssg"}, {"feature", static_cast<int>(NVSDK_NGX_Feature_FrameGeneration)}, {"multiplier", config_.multiplier},
                        {"backbuffer_format", diag_.backbufferFormat},  {"guides", true},           {"camera", "synthetic (fov 60, static)"}};
    j["diagnostics"] = diag_.ToJson();
    return j;
}

void DlssgFrameGenerator::Shutdown() {
    if (!impl_) return;
    Impl& im = *impl_;
    if (im.device) im.device->WaitIdle();
    if (im.feature) NVSDK_NGX_D3D12_ReleaseFeature(im.feature);
    if (im.params) NVSDK_NGX_D3D12_DestroyParameters(im.params);
    if (im.caps) NVSDK_NGX_D3D12_DestroyParameters(im.caps);
    im.feature = nullptr;
    im.params = im.caps = nullptr;
    im.output.Reset();
    im.backbuffer8.Reset();
    im.converter.reset();
    im.fallbackDepth.Reset();
    im.fallbackMv.Reset();
    im.runtime.reset();
    impl_.reset();
}

#else  // !DLSSVID_WITH_DLSS

struct DlssgFrameGenerator::Impl {};
DlssgFrameGenerator::DlssgFrameGenerator() = default;
DlssgFrameGenerator::~DlssgFrameGenerator() = default;
FgAvailability DlssgFrameGenerator::Available(const std::filesystem::path&) {
    return {false, "built without the DLSS SDK: set DLSS_SDK_ROOT (git clone https://github.com/NVIDIA/DLSS) and reconfigure (docs/dll-setup.md)"};
}
void DlssgFrameGenerator::Init(D3D12Device& device, const FgConfig& config) {
    config_ = config;
    diag_ = {};
    diag_.gpu = device.AdapterName();
    diag_.hint = Available().reason;
    Throw("fg: " + diag_.hint);
}
void DlssgFrameGenerator::Generate(const FgInputs&, std::vector<PassImage>&) { Throw("fg: dlssg not available"); }
nlohmann::json DlssgFrameGenerator::Describe() const { return {{"backend", "dlssg"}, {"available", false}}; }
void DlssgFrameGenerator::Shutdown() {}

#endif

}  // namespace dlssvid
