#include "stages/nr/NrStage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>

#include "gpu/ComputeKernel.h"
#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "stages/ParamSchema.h"
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

void NrStage::ApplyParams(const nlohmann::json& p) { ApplyNrParams(p, options_); }

void ApplyNrParams(const nlohmann::json& p, NrStageOptions& o) {
    // The same keys the GUI stores in the project (stage params -> CLI options).
    auto get = [&](const char* a, const char* b, auto& dst) {
        using T = std::decay_t<decltype(dst)>;
        if (p.contains(a)) dst = p[a].get<T>();
        else if (b && p.contains(b)) dst = p[b].get<T>();
    };
    get("backend", nullptr, o.backend);
    get("intensity", nullptr, o.intensity);
    get("style", nullptr, o.style);
    get("local_tone", "local-tone", o.localTone);
    get("local_structure", "local-structure", o.localStructure);
    get("skin_structure", "skin-structure", o.skinStructure);
    get("auto_mask", "auto-mask", o.autoMask);
    get("passes", nullptr, o.passes);
    get("model_scale", "model-scale", o.modelScale);
    get("transfer", nullptr, o.transfer);
    get("max_ratio", "max-ratio", o.maxRatio);
    get("temporal", nullptr, o.temporal);
    get("temporal_threshold", "temporal-threshold", o.temporalThreshold);
    get("scene_cut", "scene-cut", o.sceneCut);
    get("skin_blend", "skin-blend", o.skinBlend);
    get("tonemap", nullptr, o.tonemap.curve);
    get("exposure", nullptr, o.tonemap.exposureEv);  // stops (TASK-0043)
    get("input_transfer", "input-transfer", o.tonemap.inputTransfer);
    if (p.contains("no_guides") || p.contains("no-guides")) o.useGuides = !(p.contains("no_guides") ? p["no_guides"].get<bool>() : p["no-guides"].get<bool>());
    if (p.contains("guides")) o.useGuides = p["guides"].get<bool>();
}

nlohmann::json NrParamsFromOptions(const NrStageOptions& o) {
    std::string style = o.style;
    try {  // "Natural", "1" -> natural; an unknown value stays for the schema to name it
        static const char* named[] = {"default", "natural", "cinematic"};
        style = named[ParseNrStyle(o.style)];
    } catch (const std::exception&) {
    }
    return {{"backend", o.backend},
            {"intensity", o.intensity},
            {"style", style},
            {"passes", o.passes},
            {"model_scale", o.modelScale},
            {"transfer", o.transfer},
            {"max_ratio", o.maxRatio},
            {"temporal", o.temporal},
            {"temporal_threshold", o.temporalThreshold},
            {"scene_cut", o.sceneCut},
            {"skin_blend", o.skinBlend},
            {"local_tone", o.localTone},
            {"local_structure", o.localStructure},
            {"skin_structure", o.skinStructure},
            {"auto_mask", o.autoMask},
            {"guides", o.useGuides},
            {"tonemap", o.tonemap.curve},
            {"exposure", o.tonemap.exposureEv},
            {"input_transfer", o.tonemap.inputTransfer}};
}

NrConfig NrConfigFromOptions(const NrStageOptions& o, uint32_t workW, uint32_t workH) {
    NrConfig cfg;
    cfg.backend = o.backend;
    cfg.width = workW;
    cfg.height = workH;
    cfg.style = ParseNrStyle(o.style);  // the render preset stays NrConfig's 3: 0..2 changed nothing (PR-2 E7)
    cfg.localTone = o.localTone;
    cfg.localStructure = o.localStructure;
    cfg.skinStructure = o.skinStructure;
    cfg.autoMask = o.autoMask;
    cfg.useGuides = o.useGuides;
    cfg.dllDir = o.dllDir;
    cfg.paramsBlock = o.paramsBlock;
    cfg.requireDriver = o.requireDriver;
    cfg.mvScale = o.mvScale;
    cfg.createWithMv = o.createWithMv;
    cfg.depthGuide = o.depthGuide;
    return cfg;
}

std::pair<uint32_t, uint32_t> NrWorkSize(uint32_t w, uint32_t h, double modelScale) {
    auto side = [&](uint32_t v) { return std::max(2u, static_cast<uint32_t>(std::lround(v * modelScale)) & ~1u); };
    return {side(w), side(h)};
}

nlohmann::json NrAbJson(const NrStageOptions& o) {
    nlohmann::json j = nlohmann::json::object();
    if (o.mvScale != "unit") j["mv_scale"] = o.mvScale;
    if (o.createWithMv) j["create_with_mv"] = true;
    if (!o.depthGuide) j["depth_guide"] = false;
    return j;
}

std::vector<std::string> NrAbProblems(const NrStageOptions& o) {
    std::vector<std::string> p;
    if (o.mvScale != "unit" && o.mvScale != "auto") p.push_back("--mv-scale must be unit | auto (got '" + o.mvScale + "')");
    if (!o.useGuides && (o.mvScale != "unit" || o.createWithMv || !o.depthGuide))
        p.push_back("--mv-scale auto, --create-with-mv and --no-depth-guide change how the guides reach the model: drop --no-guides (or guides: false in --params-file)");
    if (o.intensity == 0.f && (o.mvScale != "unit" || o.createWithMv || !o.depthGuide))
        p.push_back("--mv-scale auto, --create-with-mv and --no-depth-guide change how the guides reach the model: at --intensity 0 the model is not called");
    return p;
}

void NrStage::Init(const StageConfig& config, D3D12Device& device) {
    device_ = &device;
    ApplyParams(config.params);
    // every value against the schema's range and choices, whether the options came from a project or from the CLI
    auto problems = ValidateStageParams("nr", NrParamsFromOptions(options_));
    for (auto& p : NrAbProblems(options_)) problems.push_back(std::move(p));
    if (!problems.empty()) {
        std::string all;
        for (const auto& p : problems) all += (all.empty() ? "" : "; ") + p;
        Throw("nr: " + all);
    }
    bypass_ = options_.intensity == 0.f;  // PR-7: the edit strength 0 is the frame itself - the model, its guides and masks are not needed
    stats_.bypass = bypass_;
    // PR-6 (spec §2d): a cut matters only to a history - the model's (guides: with mv it keeps one) or the flicker filter's; the bypass
    // has neither. The detector reads the decoded source frame, whatever colour the stage reads
    sceneCuts_.reset();
    stats_.sceneCutActive = options_.sceneCut > 0.f && !bypass_ && (options_.useGuides || options_.temporal > 0.f);
    if (stats_.sceneCutActive) sceneCuts_.emplace(options_.color);
    stats_.sceneCuts.clear();
    stats_.sceneCutCount = 0;
    if (!options_.colorDir.empty()) {
        colorReader_.emplace(PassReader::Open(options_.colorDir));
        if (colorReader_->Man().channels.size() < 3) Throw("nr: --color-dir must be a colour pass (color_sr)");
    }
    if (!options_.depthDir.empty() && !bypass_) {
        depthReader_.emplace(PassReader::Open(options_.depthDir));
        depthReader_->Validate({0, 0, -1, PassKind::DepthDlss});
    }
    if (!options_.mvDir.empty() && !bypass_) {
        mvReader_.emplace(PassReader::Open(options_.mvDir));
        mvReader_->Validate({0, 0, -1, PassKind::MvDlss});
    }
    maskReaders_.clear();
    if (!options_.masksDir.empty() && !bypass_) {
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

bool NrStage::CreateBackends(std::string& backend) {
    NrConfig cfg = NrConfigFromOptions(options_, workW_, workH_);
    if (options_.createWithMv) {
        // Setup runs before the first frame's guides are bound (Process): the zero guides take the sizes of the pass folders, the
        // model's without one (slot guides of a pipeline: `process` never sets the switch)
        cfg.createMvWidth = mvReader_ ? mvReader_->Man().width : workW_;
        cfg.createMvHeight = mvReader_ ? mvReader_->Man().height : workH_;
        if (options_.depthGuide && depthReader_) {
            cfg.createDepthWidth = depthReader_->Man().width;
            cfg.createDepthHeight = depthReader_->Man().height;
        }
        if (!mvReader_) Log()->warn("nr: --create-with-mv without an mv_dlss pass folder: zero vectors at CreateFeature, the frames run without vectors");
    }

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
        // PR-5 (spec §2c): every pass is a feature of its own - its VRAM, as each backend measured it around its CreateFeature
        stats_.vramMibPerPass.clear();
        for (const auto& b : backends_) {
            const int64_t bytes = b->Diagnostics().vramFeatureBytes;
            if (bytes < 0) {  // one not measured: none (a mixed list would read as the cost of fewer passes)
                stats_.vramMibPerPass.clear();
                break;
            }
            stats_.vramMibPerPass.push_back(std::round(static_cast<double>(bytes) / 1048576.0 * 10.0) / 10.0);
        }
    } catch (const std::exception& e) {
        if (!options_.disableWhenUnavailable) throw;
        Disable(e.what());
        return false;
    }
    while (backends_.size() > static_cast<size_t>(options_.passes)) backends_.pop_back();
    return true;
}

void NrStage::Setup(uint32_t w, uint32_t h) {
    setup_ = true;
    inW_ = w;
    inH_ = h;
    std::tie(workW_, workH_) = NrWorkSize(w, h, options_.modelScale);
    reduced_ = workW_ != w || workH_ != h;
    stats_.width = w;
    stats_.height = h;
    stats_.workWidth = workW_;
    stats_.workHeight = workH_;
    stats_.passes = bypass_ ? 0 : options_.passes;

    std::string backend = options_.backend;
    if (!bypass_ && !CreateBackends(backend)) return;  // PR-7: at intensity 0 no backend, no DLL; false: disabled (logged)
    stats_.backend = backend;

    tonemapper_ = std::make_unique<Tonemapper>(*device_);
    input_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    proxy_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    modelOut_.clear();
    if (!bypass_) {
        compose_ = std::make_unique<NrCompose>(*device_);
        if (reduced_) resampler_ = std::make_unique<Resampler>(*device_);
        output_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (reduced_) work_ = device_->CreateTexture2D(workW_, workH_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for (size_t p = 0; p < backends_.size(); ++p)
            modelOut_.push_back(device_->CreateTexture2D(workW_, workH_, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
        if (options_.temporal > 0.f) prev_ = device_->CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }

    if (!options_.outputDir.empty()) {
        Manifest m = Manifest::ForPass(PassKind::ColorNr, w, h, options_.format);
        m.fps = options_.fps;
        m.colorspace = "srgb";
        m.sourceFile = options_.sourceFile;
        m.sourceHash = options_.sourceHash;
        m.stageParams = bypass_ ? nlohmann::json{{"backend", options_.backend}, {"bypass", true}} : backends_[0]->Describe();
        m.stageParams["passes"] = stats_.passes;          // model passes: 0 in the bypass (the estimates read it)
        if (!stats_.vramMibPerPass.empty()) m.stageParams["vram_mib_per_pass"] = stats_.vramMibPerPass;  // PR-5: each pass's feature
        m.stageParams["intensity"] = options_.intensity;  // our edit strength after the model, not the model's (PR-7)
        if (!bypass_) m.stageParams["model_intensity"] = kNrModelIntensity;  // what the model got: also the mark of a PR-7 pass
        m.stageParams["model_scale"] = options_.modelScale;
        m.stageParams["work_size"] = {workW_, workH_};
        m.stageParams["transfer"] = options_.transfer;
        m.stageParams["temporal"] = options_.temporal;
        m.stageParams["skin_blend"] = options_.skinBlend;
        m.stageParams["tonemap"] = {{"curve", options_.tonemap.curve}, {"input_transfer", options_.tonemap.inputTransfer}, {"exposure_ev", options_.tonemap.exposureEv}};
        m.stageParams["guides"] = options_.useGuides;
        m.stageParams["masks"] = stats_.masks;
        if (nlohmann::json ab = NrAbJson(options_); !ab.empty()) m.stageParams["ab"] = std::move(ab);  // process never adopts it
        passManifest_ = std::move(m);  // the writer opens at the first output frame (OpenOutputs)
    }
    if (bypass_) {
        Log()->info("nr: {} {}x{} — intensity 0: the model is not called (tonemap {} only)", backend, w, h, options_.tonemap.curve);
        return;
    }
    Log()->info("nr: {} {}x{}{} — {} pass(es), guides {}, tonemap {}, temporal {}, masks [{}]", backend, w, h,
                reduced_ ? " (model at " + std::to_string(workW_) + "x" + std::to_string(workH_) + ")" : "", options_.passes, options_.useGuides ? "on" : "off",
                options_.tonemap.curve, options_.temporal, [&] {
                    std::string s;
                    for (const auto& m : stats_.masks) s += (s.empty() ? "" : ", ") + m;
                    return s;
                }());
}

void NrStage::OpenOutputs() {
    if (passManifest_ && !writer_) writer_ = std::make_unique<PassWriter>(OutDir(), *passManifest_);
    if (!options_.videoOut.empty() && !encoder_) {
        VideoEncoder::Options eo;
        eo.codec = options_.videoCodec;
        eo.frameRate = options_.fps.num > 0 ? options_.fps : Rational{30, 1};
        encoder_ = std::make_unique<VideoEncoder>(options_.videoOut, FrameDesc{inW_, inH_, PixelFormat::Yuv420p}, eo);
        encoder_->Open();
    }
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
        colorImg = YuvToRgba16f(frame, options_.color);
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

    // ---- scene cut (PR-6): every pass starts its history again, the flicker filter does not blend across the cut ----
    bool cut = false;
    if (sceneCuts_) {
        double score = 0.0;
        cut = sceneCuts_->Next(frame, options_.sceneCut, &score);
        if (cut) {
            Log()->info("nr: scene cut at frame {} (score {:.2f})", frame.index, score);
            if (stats_.sceneCuts.size() < kMaxSceneCuts) stats_.sceneCuts.push_back(frame.index);
            ++stats_.sceneCutCount;
            hasPrev_ = false;
        }
    }

    // ---- guides: pass folders go through the cache slot so later stages see them too ----
    NrInputs in;
    in.frameIndex = frame.index;
    in.reset = first_ || cut;
    const bool wantGuides = options_.useGuides && !backends_.empty() && backends_[0]->UsesGuides();
    if (wantGuides && slot) {
        if (options_.depthGuide && depthReader_ && depthReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "depth_dlss", depthReader_->ReadFrame(frame.index));
        if (mvReader_ && mvReader_->HasFrame(frame.index)) ctx.cache.UploadPass(*slot, "mv_dlss", mvReader_->ReadFrame(frame.index));
        if (options_.useCachePasses || depthReader_ || mvReader_) {
            if (auto it = slot->passes.find("depth_dlss"); options_.depthGuide && it != slot->passes.end() && it->second.texture) {
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
    if (in.mv && stats_.mvScaleX == 0.f) {  // A/B E3: what reaches the model, so an A/B that changes nothing is visible
        std::tie(stats_.mvScaleX, stats_.mvScaleY) = NrMvScale(options_.mvScale, workW_, workH_, in.mvWidth, in.mvHeight);
        Log()->info("nr: mv_dlss {}x{} -> model {}x{}: MVecScale {:.3f} x {:.3f} (--mv-scale {})", in.mvWidth, in.mvHeight, workW_, workH_, stats_.mvScaleX,
                    stats_.mvScaleY, options_.mvScale);
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
    rp.gain = options_.intensity;  // PR-7: the edit strength, once after the last pass (NrCompose)
    rp.mvWidth = in.mvWidth;
    rp.mvHeight = in.mvHeight;
    const auto t0 = std::chrono::steady_clock::now();
    device_->ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        tonemapper_->Run(cl, colorTex, proxy_.Get(), w, h, options_.tonemap);
        if (bypass_) return;  // intensity 0: the tone-mapped frame is the output
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
    ++stats_.frames;
    stats_.msSum += ms;
    first_ = false;
    if (frame.index < options_.writeFrom) {
        // a warm-up frame (dlssvid nr --warmup): the model's history (Reset was on the first frame) and the filter's prev_ were
        // built on the GPU above, masks applied too - the history equals a run that writes these frames; nothing is output
        ++stats_.warmup;
        if (options_.onFrame) options_.onFrame(frame.index, ms);
        return;
    }
    OpenOutputs();

    // ---- readback RGBA16F -> RGB half pass image ----
    size_t pitch = 0;
    const std::vector<uint8_t> bytes = device_->ReadbackTexture2D((bypass_ ? proxy_ : output_).Get(), pitch);
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
        yuv.pts = frame.index - options_.writeFrom;  // the video of a range starts at 0, not after F empty frames
        encoder_->WriteFrame(yuv);
    }
    if (stats_.written++ == 0) stats_.firstWritten = frame.index;
    stats_.lastWritten = frame.index;
    if (options_.onFrame) options_.onFrame(frame.index, ms);
}

void NrStage::Finish() {
    if (writer_) {
        Manifest& m = const_cast<Manifest&>(writer_->Man());
        // the stage's own GPU time (submit + wait); the wall time of a run is process's to write (StampPass) - the estimates take
        // only that: a standalone run's 16 ms recorded as ms_per_frame put the NR ETA ~25x short (D8, TASK-0049)
        m.stageParams["ms_per_frame_gpu"] = stats_.MeanMs();
        m.stageParams["depth_used"] = stats_.depthUsed;
        m.stageParams["mv_used"] = stats_.mvUsed;
        m.stageParams["color_source"] = stats_.colorSource;
        // a part of the clip (dlssvid nr --start): named for the A/B script and the settings note; process never takes it for
        // the clip's pass (PassComplete: it does not start at frame 0)
        if (options_.writeFrom > 0) m.stageParams["range"] = {{"first", stats_.firstWritten}, {"last", stats_.lastWritten}, {"warmup", stats_.warmup}};
        m.stageParams["scene_cut_threshold"] = options_.sceneCut;  // PR-6: the cuts the history was reset at (none listed: not checked)
        if (stats_.sceneCutActive) {
            m.stageParams["scene_cuts"] = stats_.sceneCuts;
            m.stageParams["scene_cut_count"] = stats_.sceneCutCount;
        }
        writer_->Finish();
    }
    if (encoder_) encoder_->Close();
}

void NrStage::Shutdown() {
    if (writer_) writer_->Finish();
    writer_.reset();
    passManifest_.reset();
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

NrRunResult RunNr(VideoDecoder& decoder, D3D12Device& device, NrStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress, const nlohmann::json& params) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    options.uploadToGpu = false;  // single-stage run: nothing reads the slot
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<NrStage>(options);
    NrStage* raw = stage.get();
    StageConfig cfg;
    cfg.name = "nr";
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
    NrRunResult r;
    r.stats = raw->GetStats();
    r.outDir = raw->OutDir();
    r.backend = raw->Backend()  ? raw->Backend()->Describe()
                : raw->Bypass() ? nlohmann::json{{"backend", r.stats.backend}, {"bypass", true}}
                                : nlohmann::json::object();
    r.diagnostics = raw->Diagnostics() ? raw->Diagnostics()->ToJson() : nlohmann::json::object();
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
