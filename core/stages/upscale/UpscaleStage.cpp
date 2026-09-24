#include "stages/upscale/UpscaleStage.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "stages/upscale/Jitter.h"
#include "util/Error.h"
#include "util/Half.h"
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
    if (p.contains("model")) options_.model = p["model"].get<std::string>();
    if (p.contains("tile")) options_.tile = p["tile"].get<int>();
    if (p.contains("models_dir")) options_.modelsDir = p["models_dir"].get<std::string>();
    if (p.contains("window")) options_.window = p["window"].get<int>();
    if (p.contains("overlap")) options_.overlap = p["overlap"].get<int>();
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
    if (!options_.model.empty()) cfg.extra["model"] = options_.model;
    if (options_.tile > 0) cfg.extra["tile"] = options_.tile;
    if (!options_.modelsDir.empty()) cfg.extra["models_dir"] = options_.modelsDir;
    if (options_.window > 0) cfg.extra["window"] = options_.window;
    if (options_.overlap > 0) cfg.extra["overlap"] = options_.overlap;
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
        if (std::find(known.begin(), known.end(), backend) == known.end()) Throw("upscale: unknown backend '" + backend + "' (dlss | nis | bicubic | trt | worker)");
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
    native_.Reset();
    nativeW_ = nativeH_ = 0;
    if (upscaler_->ProcessesOnCpu()) {  // the model's fixed factor, then Catmull-Rom to the target when they differ
        const uint32_t s = static_cast<uint32_t>(std::max(0, upscaler_->NativeScale()));
        nativeW_ = s ? w * s : t.width;  // 0: the backend delivers the target size itself (the worker)
        nativeH_ = s ? h * s : t.height;
        if (nativeW_ != t.width || nativeH_ != t.height) {
            native_ = device_->CreateTexture2D(nativeW_, nativeH_, DXGI_FORMAT_R16G16B16A16_FLOAT);
            if (!resampler_) resampler_ = std::make_unique<Resampler>(*device_);
            Log()->info("upscale: {} works at x{} ({}x{}), resampled to the target {}x{}", backend, s, nativeW_, nativeH_, t.width, t.height);
        }
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
    cache_ = &ctx.cache;

    // colour -> RGBA16F
    const PassImage rgba = Yuv420pToRgba16f(frame, options_.color);

    if (upscaler_->ProcessesOnCpu()) {
        // CPU-side models: one frame at a time (TensorRT), or a window of frames for video models (the PyTorch worker)
        if (upscaler_->WindowSize() > 1) {
            pending_.push_back({frame.index, rgba});
            first_ = false;
            if (static_cast<int>(pending_.size()) >= upscaler_->WindowSize()) RunCpuWindow(false);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        PassImage native;
        upscaler_->EvaluateCpu(rgba, native);
        PlaceCpuResult(native);
        Emit(frame.index, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        first_ = false;
        return;
    }

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
    Emit(frame.index, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    first_ = false;
}

void UpscaleStage::PlaceCpuResult(const PassImage& native) {
    if (native.width != nativeW_ || native.height != nativeH_ || native.ChannelCount() != 4 || native.type != PixelType::F16)
        Throw("upscale: backend '" + std::string(upscaler_->Name()) + "' returned " + std::to_string(native.width) + "x" + std::to_string(native.height) + ", expected " +
              std::to_string(nativeW_) + "x" + std::to_string(nativeH_) + " RGBA16F");
    if (native_) {
        device_->UploadTexture2D(native_.Get(), native.data.data(), native.RowBytes());
        device_->ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
            resampler_->Run(cl, native_.Get(), nativeW_, nativeH_, output_.Get(), outW_, outH_, 0.f, 0.f, Resampler::Filter::CatmullRom);
        });
    } else {
        device_->UploadTexture2D(output_.Get(), native.data.data(), native.RowBytes());
    }
}

// The previous window's estimate of a frame fades into this window's (0 -> previous, 1 -> this), RGBA16F in place.
static void BlendHalf(const PassImage& prev, PassImage& cur, float t) {
    if (prev.width != cur.width || prev.height != cur.height || prev.data.size() != cur.data.size()) return;
    const uint16_t* a = prev.As<uint16_t>();
    uint16_t* b = cur.As<uint16_t>();
    const size_t n = cur.data.size() / sizeof(uint16_t);
    for (size_t i = 0; i < n; ++i) b[i] = FloatToHalf(HalfToFloat(a[i]) * (1.f - t) + HalfToFloat(b[i]) * t);
}

void UpscaleStage::RunCpuWindow(bool flush) {
    if (pending_.empty()) return;
    const int window = std::max(1, upscaler_->WindowSize());
    const int overlap = std::clamp(upscaler_->WindowOverlap(), 0, window - 1);
    if (!flush && static_cast<int>(pending_.size()) < window) return;
    const size_t n = std::min<size_t>(pending_.size(), static_cast<size_t>(window));
    std::vector<const PassImage*> in;
    for (size_t i = 0; i < n; ++i) in.push_back(&pending_[i].rgba);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<PassImage> out;
    upscaler_->EvaluateCpuWindow(in, nativeW_, nativeH_, out);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / static_cast<double>(n);
    if (out.size() != n) Throw("upscale: backend '" + std::string(upscaler_->Name()) + "' returned " + std::to_string(out.size()) + " frames for a window of " + std::to_string(n));
    for (size_t i = 0; i < n; ++i) {
        const int64_t idx = pending_[i].index;
        const auto prev = std::find_if(overlapPrev_.begin(), overlapPrev_.end(), [idx](const auto& e) { return e.first == idx; });
        if (prev != overlapPrev_.end() && overlap > 0) BlendHalf(prev->second, out[i], static_cast<float>(i + 1) / static_cast<float>(overlap + 1));
    }
    // frames the next window re-estimates are kept (not emitted yet) unless flushing
    const size_t emitCount = flush || overlap == 0 || n < static_cast<size_t>(window) ? n : n - static_cast<size_t>(overlap);
    overlapPrev_.clear();
    for (size_t i = emitCount; i < n; ++i) overlapPrev_.emplace_back(pending_[i].index, out[i]);
    for (size_t i = 0; i < emitCount; ++i) {
        PlaceCpuResult(out[i]);
        Emit(pending_[i].index, ms);
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<ptrdiff_t>(emitCount));
    if (flush && !pending_.empty()) RunCpuWindow(true);
}

void UpscaleStage::Emit(int64_t index, double ms) {
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
    if (writer_) writer_->WriteFrame(index, rgb);
    if (options_.uploadToGpu && cache_)
        if (GpuFrameCache::Slot* slot = cache_->Find(index)) cache_->UploadPass(*slot, "color_sr", rgb);
    if (encoder_) {
        CpuFrame yuv;
        RgbToYuv420p(rgb, options_.color, yuv);
        yuv.index = index;
        yuv.pts = index;
        encoder_->WriteFrame(yuv);
    }
    ++stats_.frames;
    stats_.msSum += ms;
    first_ = false;
    if (options_.onFrame) options_.onFrame(index, ms);
}

void UpscaleStage::Finish() {
    if (upscaler_ && upscaler_->ProcessesOnCpu() && upscaler_->WindowSize() > 1) RunCpuWindow(true);
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
    native_.Reset();
    pending_.clear();
    overlapPrev_.clear();
    cache_ = nullptr;
    initialized_ = false;
}

UpscaleRunResult RunUpscale(VideoDecoder& decoder, D3D12Device& device, UpscaleStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress, const nlohmann::json& params) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    options.uploadToGpu = false;  // single-stage run: nothing reads the slot
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<UpscaleStage>(options);
    UpscaleStage* raw = stage.get();
    StageConfig cfg;
    cfg.name = "upscale";
    cfg.params = params;
    pipeline.AddStage(std::move(stage), cfg);
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
