#include "stages/fg/FgStage.h"

#include <windows.h>

#include <algorithm>
#include <chrono>

#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

FgStage::FgStage(FgStageOptions options, std::unique_ptr<IFrameGenerator> generator) : options_(std::move(options)), injected_(std::move(generator)) {}

FgStage::~FgStage() { Shutdown(); }

void FgStage::ApplyParams(const nlohmann::json& p) {
    // The same keys the GUI stores in the project (stage params -> CLI options).
    if (p.contains("backend")) options_.backend = p["backend"].get<std::string>();
    if (p.contains("multiplier")) options_.multiplier = p["multiplier"].get<int>();
    if (p.contains("model")) options_.model = p["model"].get<std::string>();
    if (p.contains("fp32")) options_.fp16 = !p["fp32"].get<bool>();
    if (p.contains("backbuffer_format")) options_.backbufferFormat = p["backbuffer_format"].get<std::string>();
    if (p.contains("backbuffer-format")) options_.backbufferFormat = p["backbuffer-format"].get<std::string>();
}

void FgStage::Init(const StageConfig& config, D3D12Device& device) {
    device_ = &device;
    ApplyParams(config.params);
    if (options_.multiplier < 2 || options_.multiplier > 4) Throw("fg: --multiplier must be 2 | 3 | 4");
    if (!options_.colorDir.empty()) {
        colorReader_.emplace(PassReader::Open(options_.colorDir));
        if (colorReader_->Man().channels.size() < 3) Throw("fg: --color-dir must be a colour pass (color_nr / color_sr)");
    }
    if (!options_.depthDir.empty()) {
        depthReader_.emplace(PassReader::Open(options_.depthDir));
        depthReader_->Validate({0, 0, -1, PassKind::DepthDlss});
    }
    if (!options_.mvDir.empty()) {
        mvReader_.emplace(PassReader::Open(options_.mvDir));
        mvReader_->Validate({0, 0, -1, PassKind::MvDlss});
    }
    stats_ = {};
    stats_.multiplier = options_.multiplier;
    setup_ = disabled_ = warnedDisabled_ = false;
    curTex_ = prevTex_ = nullptr;
    initialized_ = true;
}

void FgStage::Disable(const std::string& reason) {
    disabled_ = true;
    stats_.disabled = true;
    stats_.disabledReason = reason;
    if (generator_) generator_->Shutdown();
    generator_.reset();
    Log()->warn("fg: stage disabled — {}. The pipeline continues without color_fg.", reason);
}

void FgStage::Setup(uint32_t w, uint32_t h) {
    setup_ = true;
    inW_ = w;
    inH_ = h;
    stats_.width = w;
    stats_.height = h;
    FgConfig cfg;
    cfg.backend = options_.backend;
    cfg.width = w;
    cfg.height = h;
    cfg.multiplier = options_.multiplier;
    cfg.backbufferFormat = options_.backbufferFormat;
    cfg.dllDir = options_.dllDir;
    cfg.model = options_.model;
    cfg.fp16 = options_.fp16;
    cfg.modelsDir = options_.modelsDir;
    cfg.requireDriver = options_.requireDriver;
    if (depthReader_) {
        cfg.guideWidth = depthReader_->Man().width;
        cfg.guideHeight = depthReader_->Man().height;
    } else if (mvReader_) {
        cfg.guideWidth = mvReader_->Man().width;
        cfg.guideHeight = mvReader_->Man().height;
    }
    std::string backend = options_.backend;
    try {
        if (injected_) {
            generator_ = std::move(injected_);
            backend = std::string(generator_->Name());
            cfg.backend = backend;
        } else {
            const auto known = FgBackends();
            if (std::find(known.begin(), known.end(), backend) == known.end()) Throw("fg: unknown backend '" + backend + "' (dlssg | rife | blend)");
            const FgAvailability avail = FgAvailable(backend, options_.dllDir);
            if (!avail.available) Throw("fg: " + avail.reason);
            generator_ = CreateFrameGenerator(backend);
        }
        try {
            generator_->Init(*device_, cfg);
        } catch (...) {
            diag_ = generator_->Diagnostics();
            throw;
        }
        diag_ = generator_->Diagnostics();
    } catch (const std::exception& e) {
        if (!options_.disableWhenUnavailable) throw;
        Disable(e.what());
        return;
    }
    stats_.backend = backend;
    texA_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    texB_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (!options_.outputDir.empty()) {
        Manifest m = Manifest::ForPass(PassKind::ColorFg, w, h, options_.format);
        m.fps = options_.fps.num > 0 ? Rational{options_.fps.num * options_.multiplier, options_.fps.den} : options_.fps;
        m.colorspace = "srgb";
        m.sourceFile = options_.sourceFile;
        m.sourceHash = options_.sourceHash;
        m.stageParams = generator_->Describe();
        m.stageParams["multiplier"] = options_.multiplier;
        m.stageParams["source_fps"] = options_.fps.num > 0 ? options_.fps.ToDouble() : 0.0;
        writer_ = std::make_unique<PassWriter>(OutDir(), m);
    }
    if (!options_.videoOut.empty()) {
        VideoEncoder::Options eo;
        eo.codec = options_.videoCodec;
        const Rational base = options_.fps.num > 0 ? options_.fps : Rational{30, 1};
        eo.frameRate = Rational{base.num * options_.multiplier, base.den};
        encoder_ = std::make_unique<VideoEncoder>(options_.videoOut, FrameDesc{w, h, PixelFormat::Yuv420p}, eo);
        encoder_->Open();
    }
    Log()->info("fg: {} {}x{} x{} (guides {}x{})", backend, w, h, options_.multiplier, cfg.guideWidth, cfg.guideHeight);
}

void FgStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("FgStage::Process before Init");
    if (disabled_) {
        if (!warnedDisabled_) Log()->warn("fg: skipping frames ({})", stats_.disabledReason);
        warnedDisabled_ = true;
        return;
    }
    CpuFrame& frame = ctx.frame;
    GpuFrameCache::Slot* slot = ctx.cache.Find(frame.index);
    const int64_t i = stats_.frames;  // real frame counter

    // ---- colour: pass folder, the slot's color_nr / color_sr, or the decoded frame ----
    std::string source;
    if (colorReader_) {
        if (!colorReader_->HasFrame(frame.index)) Throw("fg: " + options_.colorDir.string() + " has no frame " + std::to_string(frame.index));
        cpuCur_ = colorReader_->ReadFrame(frame.index);
        source = "pass";
    } else if (options_.useCachePasses && slot && slot->passes.count("color_nr") && slot->passes.at("color_nr").texture) {
        cpuCur_ = ctx.cache.DownloadPass(*slot, "color_nr");
        source = "slot:color_nr";
    } else if (options_.useCachePasses && slot && slot->passes.count("color_sr") && slot->passes.at("color_sr").texture) {
        cpuCur_ = ctx.cache.DownloadPass(*slot, "color_sr");
        source = "slot:color_sr";
    } else {
        cpuCur_ = Yuv420pToRgb(frame, options_.color, PixelType::F16);
        source = "video";
    }
    if (!setup_) {
        stats_.colorSource = source;
        Setup(cpuCur_.width, cpuCur_.height);
        if (disabled_) return Process(ctx);
    }
    if (cpuCur_.width != inW_ || cpuCur_.height != inH_) Throw("fg: frame size changed mid-stream");
    // GPU copies (dlssg): ping-pong so the previous frame's texture stays valid
    curTex_ = (i % 2 == 0) ? texA_.Get() : texB_.Get();
    {
        const PassImage rgba = ToRgba16f(cpuCur_);
        device_->UploadTexture2D(curTex_, rgba.data.data(), rgba.RowBytes());
    }
    const bool first = i == 0;
    if (first) {
        cpuPrev_ = cpuCur_;
        prevTex_ = curTex_;
    }

    // ---- guides of the current frame through the cache slot ----
    FgInputs in;
    in.frameIndex = i;
    in.reset = first;
    in.prev = FgFrame{prevTex_, &cpuPrev_};
    in.cur = FgFrame{curTex_, &cpuCur_};
    if (generator_->UsesGuides() && slot) {
        if (depthReader_ && depthReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "depth_dlss", depthReader_->ReadFrame(frame.index));
        if (mvReader_ && mvReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "mv_dlss", mvReader_->ReadFrame(frame.index));
        if (options_.useCachePasses || depthReader_ || mvReader_) {
            if (auto it = slot->passes.find("depth_dlss"); it != slot->passes.end() && it->second.texture) {
                in.depth = it->second.texture.Get();
                in.depthWidth = it->second.width;
                in.depthHeight = it->second.height;
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

    // ---- generate the frames between the previous and the current frame ----
    const auto t0 = std::chrono::steady_clock::now();
    generator_->Generate(in, generated_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!first) {
        if (static_cast<int>(generated_.size()) != options_.multiplier - 1)
            Throw("fg: backend returned " + std::to_string(generated_.size()) + " frames, expected " + std::to_string(options_.multiplier - 1));
        for (int k = 1; k < options_.multiplier; ++k) {
            const PassImage& g = generated_[static_cast<size_t>(k - 1)];
            if (writer_) writer_->WriteFrame(FgInterpIndex(i - 1, k, options_.multiplier), g);
            if (encoder_) {
                CpuFrame yuv;
                RgbToYuv420p(g, options_.color, yuv);
                encoder_->WriteFrame(yuv);
            }
            ++stats_.generated;
        }
    }
    PassImage real = cpuCur_.type == PixelType::F16 ? cpuCur_ : cpuCur_.ConvertTo(PixelType::F16);
    if (writer_) writer_->WriteFrame(FgRealIndex(i, options_.multiplier), real);
    if (encoder_) {
        CpuFrame yuv;
        RgbToYuv420p(real, options_.color, yuv);
        encoder_->WriteFrame(yuv);
    }
    ++stats_.frames;
    stats_.msSum += ms;
    cpuPrev_ = std::move(cpuCur_);
    cpuCur_ = {};
    prevTex_ = curTex_;
    if (options_.onFrame) options_.onFrame(i, ms);
    if (options_.crashAfter >= 0 && stats_.frames > options_.crashAfter) {
        // HACK (test only): a real process death, the way a driver / runtime fault would end the stage.
        Log()->error("fg: --crash-after {}: terminating the process (isolation test)", options_.crashAfter);
        TerminateProcess(GetCurrentProcess(), 0xC0000005u);
    }
}

void FgStage::Finish() {
    if (writer_) {
        Manifest& m = const_cast<Manifest&>(writer_->Man());
        m.stageParams["ms_per_frame"] = stats_.MeanMs();
        m.stageParams["generated"] = stats_.generated;
        m.stageParams["depth_used"] = stats_.depthUsed;
        m.stageParams["mv_used"] = stats_.mvUsed;
        m.stageParams["color_source"] = stats_.colorSource;
        writer_->Finish();
    }
    if (encoder_) encoder_->Close();
}

void FgStage::Shutdown() {
    if (writer_) writer_->Finish();
    writer_.reset();
    if (encoder_) {
        if (encoder_->IsOpen()) encoder_->Close();
        encoder_.reset();
    }
    if (generator_) generator_->Shutdown();
    generator_.reset();
    texA_.Reset();
    texB_.Reset();
    curTex_ = prevTex_ = nullptr;
    generated_.clear();
    initialized_ = false;
    setup_ = false;
}

FgRunResult RunFg(VideoDecoder& decoder, D3D12Device& device, FgStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<FgStage>(options);
    FgStage* raw = stage.get();
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
    FgRunResult r;
    r.stats = raw->GetStats();
    r.outDir = raw->OutDir();
    r.backend = raw->Generator() ? raw->Generator()->Describe() : nlohmann::json::object();
    r.diagnostics = raw->Diagnostics() ? raw->Diagnostics()->ToJson() : nlohmann::json::object();
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
