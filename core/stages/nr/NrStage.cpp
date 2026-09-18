#include "stages/nr/NrStage.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "gpu/ComputeKernel.h"
#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

const std::vector<std::string>& NrStage::MaskNames() {
    static const std::vector<std::string> names{"ui", "ignore", "face", "skin"};
    return names;
}

PassImage MaskToFloat(const PassImage& mask) {
    if (mask.Empty() || mask.channels.empty()) Throw("nr: empty mask image");
    PassImage out;
    out.Allocate(mask.width, mask.height, PixelType::F32, {"A"});
    const float scale = mask.type == PixelType::U8 ? 1.f / 255.f : mask.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    for (uint32_t y = 0; y < mask.height; ++y)
        for (uint32_t x = 0; x < mask.width; ++x) out.Set(x, y, 0, std::clamp(mask.Get(x, y, 0) * scale, 0.f, 1.f));
    return out;
}

PassImage MaskMax(const PassImage& a, const PassImage& b) {
    if (a.width != b.width || a.height != b.height) Throw("nr: masks of different sizes cannot be combined");
    PassImage out = a;
    for (uint32_t y = 0; y < a.height; ++y)
        for (uint32_t x = 0; x < a.width; ++x) out.Set(x, y, 0, std::max(a.Get(x, y, 0), b.Get(x, y, 0)));
    return out;
}

NrStage::NrStage(NrStageOptions options, std::vector<std::unique_ptr<INrBackend>> backends) : options_(std::move(options)), injected_(std::move(backends)) {}

NrStage::~NrStage() { Shutdown(); }

void NrStage::ApplyParams(const nlohmann::json& p) {
    // The same keys the GUI stores in the project (stage params -> CLI options).
    auto get = [&](const char* a, const char* b, auto& dst) {
        using T = std::decay_t<decltype(dst)>;
        if (p.contains(a)) dst = p[a].get<T>();
        else if (b && p.contains(b)) dst = p[b].get<T>();
    };
    get("backend", nullptr, options_.backend);
    get("intensity", nullptr, options_.intensity);
    get("style", nullptr, options_.style);
    get("preset", nullptr, options_.preset);
    get("local_tone", "local-tone", options_.localTone);
    get("local_structure", "local-structure", options_.localStructure);
    get("skin_structure", "skin-structure", options_.skinStructure);
    get("auto_mask", "auto-mask", options_.autoMask);
    get("passes", nullptr, options_.passes);
    get("model_scale", "model-scale", options_.modelScale);
    get("transfer", nullptr, options_.transfer);
    get("max_ratio", "max-ratio", options_.maxRatio);
    get("temporal", nullptr, options_.temporal);
    get("temporal_threshold", "temporal-threshold", options_.temporalThreshold);
    get("skin_blend", "skin-blend", options_.skinBlend);
    get("tonemap", nullptr, options_.tonemap.curve);
    get("exposure", nullptr, options_.tonemap.exposure);
    get("input_transfer", "input-transfer", options_.tonemap.inputTransfer);
    if (p.contains("no_guides") || p.contains("no-guides")) options_.useGuides = !(p.contains("no_guides") ? p["no_guides"].get<bool>() : p["no-guides"].get<bool>());
    if (p.contains("guides")) options_.useGuides = p["guides"].get<bool>();
}

void NrStage::Init(const StageConfig& config, D3D12Device& device) {
    device_ = &device;
    ApplyParams(config.params);
    if (options_.passes < 1 || options_.passes > 2) Throw("nr: --passes must be 1 or 2");
    if (options_.modelScale < 0.25 || options_.modelScale > 2.0) Throw("nr: --model-scale must be within 0.25..2");
    if (options_.temporal < 0.f || options_.temporal > 1.f) Throw("nr: --temporal must be within 0..1");
    (void)ParseNrStyle(options_.style);
    if (!TonemapCurveId(options_.tonemap.curve)) Throw("nr: --tonemap must be passthrough | aces | reinhard");
    if (!TonemapTransferId(options_.tonemap.inputTransfer)) Throw("nr: --input-transfer must be srgb | linear | pq | hlg");
    if (!options_.colorDir.empty()) {
        colorReader_.emplace(PassReader::Open(options_.colorDir));
        if (colorReader_->Man().channels.size() < 3) Throw("nr: --color-dir must be a colour pass (color_sr)");
    }
    if (!options_.depthDir.empty()) {
        depthReader_.emplace(PassReader::Open(options_.depthDir));
        depthReader_->Validate({0, 0, -1, PassKind::DepthDlss});
    }
    if (!options_.mvDir.empty()) {
        mvReader_.emplace(PassReader::Open(options_.mvDir));
        mvReader_->Validate({0, 0, -1, PassKind::MvDlss});
    }
    maskReaders_.clear();
    if (!options_.masksDir.empty()) {
        for (const auto& name : MaskNames()) {
            const auto dir = options_.masksDir / ("mask_" + name);
            if (!std::filesystem::exists(dir / Manifest::kFileName)) continue;
            PassReader r = PassReader::Open(dir);
            if (r.Man().channels.size() != 1) Throw("nr: " + dir.string() + " is not a single-channel mask pass");
            maskReaders_.emplace(name, std::move(r));
            stats_.masks.push_back(name);
        }
    }
    first_ = true;
    hasPrev_ = false;
    setup_ = false;
    disabled_ = false;
    warnedDisabled_ = false;
    initialized_ = true;
}

void NrStage::Disable(const std::string& reason) {
    disabled_ = true;
    stats_.disabled = true;
    stats_.disabledReason = reason;
    for (auto& b : backends_) b->Shutdown();
    backends_.clear();
    Log()->warn("nr: stage disabled — {}. The pipeline continues without color_nr.", reason);
}

void NrStage::Setup(uint32_t w, uint32_t h) {
    setup_ = true;
    inW_ = w;
    inH_ = h;
    workW_ = std::max(2u, static_cast<uint32_t>(std::lround(w * options_.modelScale)) & ~1u);
    workH_ = std::max(2u, static_cast<uint32_t>(std::lround(h * options_.modelScale)) & ~1u);
    reduced_ = workW_ != w || workH_ != h;
    stats_.width = w;
    stats_.height = h;
    stats_.workWidth = workW_;
    stats_.workHeight = workH_;
    stats_.passes = options_.passes;

    NrConfig cfg;
    cfg.backend = options_.backend;
    cfg.width = workW_;
    cfg.height = workH_;
    cfg.intensity = options_.intensity;
    cfg.style = ParseNrStyle(options_.style);
    cfg.preset = options_.preset;
    cfg.localTone = options_.localTone;
    cfg.localStructure = options_.localStructure;
    cfg.skinStructure = options_.skinStructure;
    cfg.autoMask = options_.autoMask;
    cfg.useGuides = options_.useGuides;
    cfg.dllDir = options_.dllDir;
    cfg.paramsBlock = options_.paramsBlock;
    cfg.requireDriver = options_.requireDriver;

    std::string backend = options_.backend;
    try {
        if (!injected_.empty()) {
            backends_ = std::move(injected_);
            backend = std::string(backends_[0]->Name());
            cfg.backend = backend;
        } else {
            const auto known = NrBackends();
            if (std::find(known.begin(), known.end(), backend) == known.end()) Throw("nr: unknown backend '" + backend + "' (ngx | stub)");
            const NrAvailability avail = NrAvailable(backend, options_.dllDir);
            if (!avail.available) Throw("nr: " + avail.reason);
            for (int p = 0; p < options_.passes; ++p) backends_.push_back(CreateNrBackend(backend));
        }
        for (auto& b : backends_) {
            try {
                b->Init(*device_, cfg);
            } catch (...) {
                diag_ = b->Diagnostics();
                throw;
            }
        }
        diag_ = backends_[0]->Diagnostics();
    } catch (const std::exception& e) {
        if (!options_.disableWhenUnavailable) throw;
        Disable(e.what());
        return;
    }
    stats_.backend = backend;
    while (backends_.size() > static_cast<size_t>(options_.passes)) backends_.pop_back();

    tonemapper_ = std::make_unique<Tonemapper>(*device_);
    compose_ = std::make_unique<NrCompose>(*device_);
    if (reduced_) resampler_ = std::make_unique<Resampler>(*device_);
    input_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    proxy_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    output_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (reduced_) work_ = device_->CreateTexture2D(workW_, workH_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    modelOut_.clear();
    for (size_t p = 0; p < backends_.size(); ++p)
        modelOut_.push_back(device_->CreateTexture2D(workW_, workH_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    if (options_.temporal > 0.f) prev_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);

    if (!options_.outputDir.empty()) {
        Manifest m = Manifest::ForPass(PassKind::ColorNr, w, h, options_.format);
        m.fps = options_.fps;
        m.colorspace = "srgb";
        m.sourceFile = options_.sourceFile;
        m.sourceHash = options_.sourceHash;
        m.stageParams = backends_[0]->Describe();
        m.stageParams["passes"] = options_.passes;
        m.stageParams["model_scale"] = options_.modelScale;
        m.stageParams["work_size"] = {workW_, workH_};
        m.stageParams["transfer"] = options_.transfer;
        m.stageParams["temporal"] = options_.temporal;
        m.stageParams["skin_blend"] = options_.skinBlend;
        m.stageParams["tonemap"] = {{"curve", options_.tonemap.curve}, {"input_transfer", options_.tonemap.inputTransfer}, {"exposure", options_.tonemap.exposure}};
        m.stageParams["guides"] = options_.useGuides;
        m.stageParams["masks"] = stats_.masks;
        writer_ = std::make_unique<PassWriter>(OutDir(), m);
    }
    if (!options_.videoOut.empty()) {
        VideoEncoder::Options eo;
        eo.codec = options_.videoCodec;
        eo.frameRate = options_.fps.num > 0 ? options_.fps : Rational{30, 1};
        encoder_ = std::make_unique<VideoEncoder>(options_.videoOut, FrameDesc{w, h, PixelFormat::Yuv420p}, eo);
        encoder_->Open();
    }
    Log()->info("nr: {} {}x{}{} — {} pass(es), guides {}, tonemap {}, temporal {}, masks [{}]", backend, w, h,
                reduced_ ? " (model at " + std::to_string(workW_) + "x" + std::to_string(workH_) + ")" : "", options_.passes, options_.useGuides ? "on" : "off",
                options_.tonemap.curve, options_.temporal, [&] {
                    std::string s;
                    for (const auto& m : stats_.masks) s += (s.empty() ? "" : ", ") + m;
                    return s;
                }());
}

void NrStage::LoadMasks(int64_t frame) {
    protectFrame_ = skinFrame_ = false;
    if (maskReaders_.empty()) return;
    std::optional<PassImage> protect, skin;
    for (auto& [name, reader] : maskReaders_) {
        if (!reader.HasFrame(frame)) continue;
        PassImage m = MaskToFloat(reader.ReadFrame(frame));
        std::optional<PassImage>& dst = (name == "ui" || name == "ignore") ? protect : skin;
        dst = dst ? MaskMax(*dst, m) : std::move(m);
    }
    auto upload = [&](const PassImage& img, ComPtr<ID3D12Resource>& tex) {
        if (maskW_ != img.width || maskH_ != img.height) {  // one size for both masks: a change re-creates them
            protect_.Reset();
            skin_.Reset();
            maskW_ = img.width;
            maskH_ = img.height;
        }
        if (!tex) tex = device_->CreateTexture2D(img.width, img.height, DXGI_FORMAT_R32_FLOAT);
        device_->UploadTexture2D(tex.Get(), img.data.data(), img.RowBytes());
    };
    if (protect) {
        upload(*protect, protect_);
        protectFrame_ = true;
    }
    if (skin) {
        upload(*skin, skin_);
        skinFrame_ = true;
    }
}

void NrStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("NrStage::Process before Init");
    if (disabled_) {
        if (!warnedDisabled_) Log()->warn("nr: skipping frames ({})", stats_.disabledReason);
        warnedDisabled_ = true;
        return;
    }
    CpuFrame& frame = ctx.frame;
    GpuFrameCache::Slot* slot = ctx.cache.Find(frame.index);

    // ---- colour: color_sr pass folder, the slot's color_sr, or the decoded frame ----
    uint32_t w = 0, h = 0;
    ID3D12Resource* colorTex = nullptr;
    PassImage colorImg;
    std::string source;
    if (colorReader_) {
        if (!colorReader_->HasFrame(frame.index)) Throw("nr: " + options_.colorDir.string() + " has no frame " + std::to_string(frame.index));
        colorImg = ToRgba16f(colorReader_->ReadFrame(frame.index));
        w = colorImg.width;
        h = colorImg.height;
        source = "color_sr";
    } else if (options_.useCachePasses && slot && slot->passes.count("color_sr") && slot->passes.at("color_sr").texture) {
        const auto& pt = slot->passes.at("color_sr");
        colorTex = pt.texture.Get();
        w = pt.width;
        h = pt.height;
        source = "slot";
    } else {
        colorImg = Yuv420pToRgba16f(frame, options_.color);
        w = colorImg.width;
        h = colorImg.height;
        source = "video";
    }
    if (!setup_) {
        stats_.colorSource = source;
        Setup(w, h);
        if (disabled_) return Process(ctx);  // logs the skip once
    }
    if (w != inW_ || h != inH_) Throw("nr: frame size changed mid-stream (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    if (!colorTex) {
        device_->UploadTexture2D(input_.Get(), colorImg.data.data(), colorImg.RowBytes());
        colorTex = input_.Get();
    }

    // ---- guides: pass folders go through the cache slot so later stages see them too ----
    NrInputs in;
    in.frameIndex = frame.index;
    in.reset = first_;
    const bool wantGuides = options_.useGuides && !backends_.empty() && backends_[0]->UsesGuides();
    if (wantGuides && slot) {
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
    LoadMasks(frame.index);

    // ---- GPU: tonemap -> [downsample] -> model passes -> resolve (+ temporal history copy) ----
    NrResolveParams rp;
    rp.width = w;
    rp.height = h;
    rp.workWidth = workW_;
    rp.workHeight = workH_;
    rp.transfer = options_.transfer;
    rp.maxRatio = options_.maxRatio;
    rp.temporal = options_.temporal;
    rp.temporalThreshold = options_.temporalThreshold;
    rp.skinBlend = options_.skinBlend;
    rp.mvWidth = in.mvWidth;
    rp.mvHeight = in.mvHeight;
    const auto t0 = std::chrono::steady_clock::now();
    device_->ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        tonemapper_->Run(cl, colorTex, proxy_.Get(), w, h, options_.tonemap);
        ID3D12Resource* modelIn = proxy_.Get();
        if (reduced_) {
            resampler_->Run(cl, proxy_.Get(), w, h, work_.Get(), workW_, workH_, 0.f, 0.f,
                            options_.modelScale < 1.0 ? Resampler::Filter::Bilinear : Resampler::Filter::CatmullRom);
            modelIn = work_.Get();
        }
        for (size_t p = 0; p < backends_.size(); ++p) {
            NrInputs pin = in;
            pin.color = modelIn;
            backends_[p]->Evaluate(cl, pin, modelOut_[p].Get());
            modelIn = modelOut_[p].Get();
        }
        NrResolveInputs ri;
        ri.original = proxy_.Get();
        ri.model = modelIn;
        ri.proxy = reduced_ ? work_.Get() : nullptr;
        ri.prev = hasPrev_ ? prev_.Get() : nullptr;
        ri.mv = options_.temporal > 0.f ? in.mv : nullptr;
        ri.protect = protectFrame_ ? protect_.Get() : nullptr;
        ri.skin = skinFrame_ ? skin_.Get() : nullptr;
        compose_->Resolve(cl, ri, output_.Get(), rp);
        if (prev_) {
            ComputeKernel::Transition(cl, output_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            ComputeKernel::Transition(cl, prev_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            cl->CopyResource(prev_.Get(), output_.Get());
            ComputeKernel::Transition(cl, output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            ComputeKernel::Transition(cl, prev_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        }
    });
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    hasPrev_ = prev_ != nullptr;

    // ---- readback RGBA16F -> RGB half pass image ----
    size_t pitch = 0;
    const std::vector<uint8_t> bytes = device_->ReadbackTexture2D(output_.Get(), pitch);
    PassImage rgb = MakePassImage(PassKind::ColorNr, w, h, PixelType::F16);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
    uint16_t* dst = rgb.As<uint16_t>();
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        dst[i * 3] = src[i * 4];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
    if (writer_) writer_->WriteFrame(frame.index, rgb);
    if (options_.uploadToGpu && slot) ctx.cache.UploadPass(*slot, "color_nr", rgb);
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

void NrStage::Finish() {
    if (writer_) {
        Manifest& m = const_cast<Manifest&>(writer_->Man());
        m.stageParams["ms_per_frame"] = stats_.MeanMs();
        m.stageParams["depth_used"] = stats_.depthUsed;
        m.stageParams["mv_used"] = stats_.mvUsed;
        m.stageParams["color_source"] = stats_.colorSource;
        writer_->Finish();
    }
    if (encoder_) encoder_->Close();
}

void NrStage::Shutdown() {
    if (writer_) writer_->Finish();
    writer_.reset();
    if (encoder_) {
        if (encoder_->IsOpen()) encoder_->Close();
        encoder_.reset();
    }
    for (auto& b : backends_) b->Shutdown();
    backends_.clear();
    tonemapper_.reset();
    resampler_.reset();
    compose_.reset();
    input_.Reset();
    proxy_.Reset();
    work_.Reset();
    output_.Reset();
    prev_.Reset();
    protect_.Reset();
    skin_.Reset();
    modelOut_.clear();
    maskW_ = maskH_ = 0;
    initialized_ = false;
    setup_ = false;
}

NrRunResult RunNr(VideoDecoder& decoder, D3D12Device& device, NrStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    options.uploadToGpu = false;  // single-stage run: nothing reads the slot
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<NrStage>(options);
    NrStage* raw = stage.get();
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
    NrRunResult r;
    r.stats = raw->GetStats();
    r.outDir = raw->OutDir();
    r.backend = raw->Backend() ? raw->Backend()->Describe() : nlohmann::json::object();
    r.diagnostics = raw->Diagnostics() ? raw->Diagnostics()->ToJson() : nlohmann::json::object();
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
