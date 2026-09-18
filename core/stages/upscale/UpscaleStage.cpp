#include "stages/upscale/UpscaleStage.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "stages/upscale/Jitter.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

UpscaleStage::UpscaleStage(UpscaleStageOptions options, std::unique_ptr<IUpscaler> upscaler) : options_(std::move(options)), injected_(std::move(upscaler)) {}

UpscaleStage::~UpscaleStage() { Shutdown(); }

void UpscaleStage::ApplyParams(const nlohmann::json& p) {
    // The same keys the GUI stores in the project (stage params -> CLI options).
    if (p.contains("backend")) options_.backend = p["backend"].get<std::string>();
    if (p.contains("scale")) options_.scale = p["scale"].get<double>();
    if (p.contains("sharpness")) options_.sharpness = p["sharpness"].get<float>();
    if (p.contains("preset")) options_.preset = p["preset"].get<std::string>();
    if (p.contains("artifact_reduction_only")) options_.artifactReductionOnly = p["artifact_reduction_only"].get<bool>();
    if (p.contains("jitter")) options_.useJitter = p["jitter"].get<bool>();
    if (p.contains("jitter_sign")) options_.jitterSign = p["jitter_sign"].get<float>();
    if (p.contains("target_width")) options_.targetWidth = p["target_width"].get<uint32_t>();
    if (p.contains("target_height")) options_.targetHeight = p["target_height"].get<uint32_t>();
}

void UpscaleStage::Init(const StageConfig& config, D3D12Device& device) {
    device_ = &device;
    ApplyParams(config.params);
    if (!options_.depthDir.empty()) {
        depthReader_.emplace(PassReader::Open(options_.depthDir));
        depthReader_->Validate({0, 0, -1, PassKind::DepthDlss});
    }
    if (!options_.mvDir.empty()) {
        mvReader_.emplace(PassReader::Open(options_.mvDir));
        mvReader_->Validate({0, 0, -1, PassKind::MvDlss});
    }
    stats_ = {};
    first_ = true;
    initialized_ = true;
}

void UpscaleStage::Setup(uint32_t w, uint32_t h) {
    UpscaleTarget t;
    if (options_.artifactReductionOnly) {
        t.width = w;
        t.height = h;
        t.scale = 1.0;
    } else {
        t = ResolveUpscaleTarget(w, h, options_.scale, options_.targetWidth, options_.targetHeight, options_.maxWidth, options_.maxHeight);
        if (t.capped) Log()->warn("upscale: target capped to {}x{} (v1 limit {}x{}, ТЗ §3)", t.width, t.height, options_.maxWidth, options_.maxHeight);
    }
    inW_ = w;
    inH_ = h;
    outW_ = t.width;
    outH_ = t.height;

    std::string backend = options_.backend;
    UpscalerConfig cfg;
    cfg.inputWidth = w;
    cfg.inputHeight = h;
    cfg.outputWidth = t.width;
    cfg.outputHeight = t.height;
    cfg.sharpness = options_.sharpness;
    cfg.preset = options_.preset;
    cfg.artifactReductionOnly = options_.artifactReductionOnly;
    cfg.useJitter = options_.useJitter;
    cfg.jitterSign = options_.jitterSign;
    cfg.dllDir = options_.dllDir;
    if (mvReader_) {
        cfg.extra["mv_width"] = mvReader_->Man().width;
        cfg.extra["mv_height"] = mvReader_->Man().height;
    }
    if (injected_) {
        upscaler_ = std::move(injected_);
        backend = std::string(upscaler_->Name());
        cfg.backend = backend;
        upscaler_->Init(*device_, cfg);
    } else {
        const auto known = UpscalerBackends();
        if (std::find(known.begin(), known.end(), backend) == known.end()) Throw("upscale: unknown backend '" + backend + "' (rtxvsr | dlss | nis | bicubic)");
        const UpscalerAvailability avail = UpscalerAvailable(backend, options_.dllDir);
        if (!avail.available) {
            if (!options_.allowFallback || backend == "nis") Throw("upscale: " + avail.reason);
            Log()->warn("upscale: backend '{}' unavailable — {}. Falling back to nis.", backend, avail.reason);
            backend = "nis";
            stats_.fellBack = true;
        }
        cfg.backend = backend;
        upscaler_ = CreateUpscaler(backend);
        try {
            upscaler_->Init(*device_, cfg);
        } catch (const std::exception& e) {
            if (!options_.allowFallback || backend == "nis") throw;
            Log()->warn("upscale: backend '{}' failed to initialise ({}). Falling back to nis.", backend, e.what());
            backend = "nis";
            cfg.backend = backend;
            upscaler_ = CreateUpscaler(backend);
            upscaler_->Init(*device_, cfg);
            stats_.fellBack = true;
        }
    }
    stats_.backend = backend;
    stats_.inputWidth = w;
    stats_.inputHeight = h;
    stats_.outputWidth = t.width;
    stats_.outputHeight = t.height;
    phases_ = JitterPhaseCount(w, h, t.width, t.height);
    stats_.jitterPhases = upscaler_->WantsJitter() ? phases_ : 0;

    input_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    output_ = device_->CreateTexture2D(t.width, t.height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (upscaler_->WantsJitter()) {
        jittered_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        resampler_ = std::make_unique<Resampler>(*device_);
    }

    if (!options_.outputDir.empty()) {
        Manifest m = Manifest::ForPass(PassKind::ColorSr, t.width, t.height, options_.format);
        m.fps = options_.fps;
        m.colorspace = "srgb";
        m.sourceFile = options_.sourceFile;
        m.sourceHash = options_.sourceHash;
        m.stageParams = upscaler_->Describe();
        m.stageParams["input"] = {w, h};
        m.stageParams["scale"] = t.scale;
        m.stageParams["fallback"] = stats_.fellBack;
        m.stageParams["jitter_phases"] = stats_.jitterPhases;
        writer_ = std::make_unique<PassWriter>(OutDir(), m);
    }
    if (!options_.videoOut.empty()) {
        VideoEncoder::Options eo;
        eo.codec = options_.videoCodec;
        eo.frameRate = options_.fps.num > 0 ? options_.fps : Rational{30, 1};
        encoder_ = std::make_unique<VideoEncoder>(options_.videoOut, FrameDesc{t.width, t.height, PixelFormat::Yuv420p}, eo);
        encoder_->Open();
    }
    Log()->info("upscale: {} {}x{} -> {}x{} (x{:.2f}){}", backend, w, h, t.width, t.height, t.scale, upscaler_->WantsJitter() ? ", jitter emulation" : "");
}

void UpscaleStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("UpscaleStage::Process before Init");
    CpuFrame& frame = ctx.frame;
    if (!upscaler_) Setup(frame.desc.width, frame.desc.height);
    if (frame.desc.width != inW_ || frame.desc.height != inH_) Throw("upscale: frame size changed mid-stream");

    // colour -> RGBA16F input texture
    const PassImage rgba = Yuv420pToRgba16f(frame, options_.color);
    device_->UploadTexture2D(input_.Get(), rgba.data.data(), rgba.RowBytes());

    // depth_dlss / mv_dlss: pass folders go through the cache slot so later stages see them too
    GpuFrameCache::Slot* slot = ctx.cache.Find(frame.index);
    UpscaleInputs in;
    in.frameIndex = frame.index;
    in.reset = first_;
    in.frameTimeMs = options_.fps.num > 0 ? 1000.0 / options_.fps.ToDouble() : 0.0;
    if (upscaler_->WantsDepthAndMv() && slot) {
        if (depthReader_ && depthReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "depth_dlss", depthReader_->ReadFrame(frame.index));
        if (mvReader_ && mvReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "mv_dlss", mvReader_->ReadFrame(frame.index));
        if (options_.useCachePasses || depthReader_ || mvReader_) {
            if (auto it = slot->passes.find("depth_dlss"); it != slot->passes.end() && it->second.texture) {
                in.depth = it->second.texture.Get();
                stats_.depthUsed = true;
            }
            if (auto it = slot->passes.find("mv_dlss"); it != slot->passes.end() && it->second.texture) {
                in.mv = it->second.texture.Get();
                in.mvWidth = it->second.width;
                in.mvHeight = it->second.height;
                stats_.mvUsed = true;
            }
        }
    }

    // HACK (ТЗ §3): DLSS expects a jittered render; the video frame is resampled with a sub-pixel
    // shift (-j) so its content moves by +j, and +j is reported as the jitter offset.
    JitterOffset j;
    const bool jitter = upscaler_->WantsJitter();
    if (jitter) j = HaltonJitter(frame.index, phases_);
    in.jitterX = j.x;
    in.jitterY = j.y;

    const auto t0 = std::chrono::steady_clock::now();
    device_->ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        if (jitter) {
            resampler_->Run(cl, input_.Get(), inW_, inH_, jittered_.Get(), inW_, inH_, -j.x, -j.y, Resampler::Filter::CatmullRom);
            in.color = jittered_.Get();
        } else {
            in.color = input_.Get();
        }
        upscaler_->Evaluate(cl, in, output_.Get());
    });
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // readback RGBA16F -> RGB half pass image
    size_t pitch = 0;
    const std::vector<uint8_t> bytes = device_->ReadbackTexture2D(output_.Get(), pitch);
    PassImage rgb = MakePassImage(PassKind::ColorSr, outW_, outH_, PixelType::F16);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
    uint16_t* dst = rgb.As<uint16_t>();
    const size_t n = static_cast<size_t>(outW_) * outH_;
    for (size_t i = 0; i < n; ++i) {
        dst[i * 3] = src[i * 4];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
    if (writer_) writer_->WriteFrame(frame.index, rgb);
    if (options_.uploadToGpu && slot) ctx.cache.UploadPass(*slot, "color_sr", rgb);
    if (encoder_) {
        CpuFrame yuv;
        RgbToYuv420p(rgb, options_.color, yuv);
        yuv.index = frame.index;
        yuv.pts = frame.index;
        encoder_->WriteFrame(yuv);
    }
    ++stats_.frames;
    stats_.msSum += ms;
    first_ = false;
    if (options_.onFrame) options_.onFrame(frame.index, ms);
}

void UpscaleStage::Finish() {
    if (writer_) {
        Manifest& m = const_cast<Manifest&>(writer_->Man());
        m.stageParams["ms_per_frame"] = stats_.MeanMs();
        m.stageParams["depth_used"] = stats_.depthUsed;
        m.stageParams["mv_used"] = stats_.mvUsed;
        writer_->Finish();
    }
    if (encoder_) encoder_->Close();
}

void UpscaleStage::Shutdown() {
    if (writer_) writer_->Finish();
    writer_.reset();
    if (encoder_) {
        if (encoder_->IsOpen()) encoder_->Close();
        encoder_.reset();
    }
    if (upscaler_) upscaler_->Shutdown();
    upscaler_.reset();
    resampler_.reset();
    input_.Reset();
    jittered_.Reset();
    output_.Reset();
    initialized_ = false;
}

UpscaleRunResult RunUpscale(VideoDecoder& decoder, D3D12Device& device, UpscaleStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    options.uploadToGpu = false;  // single-stage run: nothing reads the slot
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<UpscaleStage>(options);
    UpscaleStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    CpuFrame frame;
    int64_t n = 0;
    while (maxFrames < 0 || n < maxFrames) {
        if (!decoder.NextFrame(frame)) break;
        pipeline.Cache().Acquire(frame.index, frame.desc);
        pipeline.ProcessFrame(frame);
        ++n;
        if (progress) progress(n);
    }
    pipeline.Finish();
    UpscaleRunResult r;
    r.stats = raw->GetStats();
    r.outDir = raw->OutDir();
    r.upscaler = raw->Upscaler() ? raw->Upscaler()->Describe() : nlohmann::json::object();
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
