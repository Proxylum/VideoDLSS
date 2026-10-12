#include "stages/depth/DepthStage.h"

#include <algorithm>

#include "convert/DepthConvert.h"
#include "convert/MvConvert.h"
#include "convert/Warp.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "pipeline/Pipeline.h"
#include "stages/depth/DepthPreprocess.h"
#include "stages/depth/StubDepthEstimator.h"
#include "stages/depth/TrtDepthEstimator.h"
#include "stages/depth/WorkerDepthEstimator.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

// ---- factory ---------------------------------------------------------------------------

std::vector<std::string> DepthBackends() { return {"da3", "vda", "worker", "worker:da3", "worker:vda", "worker:icdepth", "worker:stub", "stub"}; }

std::unique_ptr<IDepthEstimator> CreateDepthEstimator(const std::string& backend) {
    if (backend == "stub") return std::make_unique<StubDepthEstimator>();
    if (backend == "da3" || backend == "vda") return std::make_unique<TrtDepthEstimator>(backend);
    if (backend == "worker") return std::make_unique<WorkerDepthEstimator>("da3");
    if (backend.rfind("worker:", 0) == 0) return std::make_unique<WorkerDepthEstimator>(backend.substr(7));
    std::string known;
    for (const auto& b : DepthBackends()) known += (known.empty() ? "" : ", ") + b;
    Throw("unknown depth backend '" + backend + "' (known: " + known + ")");
}

// ---- temporal alignment modes -----------------------------------------------------------

DepthStabilize ParseDepthStabilize(const std::string& s) {
    if (s == "auto" || s.empty()) return DepthStabilize::Auto;
    if (s == "none") return DepthStabilize::None;
    if (s == "scale") return DepthStabilize::Scale;
    if (s == "scale_shift") return DepthStabilize::ScaleShift;
    Throw("stabilize must be auto | none | scale | scale_shift (got '" + s + "')");
}

TemporalStabilizer::Mode ResolveStabilizeMode(DepthStabilize requested, bool metric) {
    switch (requested) {
        case DepthStabilize::None: return TemporalStabilizer::Mode::None;
        case DepthStabilize::Scale: return TemporalStabilizer::Mode::ScaleOnly;
        case DepthStabilize::ScaleShift: return TemporalStabilizer::Mode::ScaleShift;
        case DepthStabilize::Auto: break;
    }
    return metric ? TemporalStabilizer::Mode::ScaleOnly : TemporalStabilizer::Mode::ScaleShift;
}

const char* StabilizeModeName(TemporalStabilizer::Mode mode) {
    switch (mode) {
        case TemporalStabilizer::Mode::None: return "none";
        case TemporalStabilizer::Mode::ScaleOnly: return "scale";
        case TemporalStabilizer::Mode::ScaleShift: return "scale_shift";
    }
    return "none";
}

const char* DepthStabilizeName(DepthStabilize s) {
    switch (s) {
        case DepthStabilize::Auto: return "auto";
        case DepthStabilize::None: return "none";
        case DepthStabilize::Scale: return "scale";
        case DepthStabilize::ScaleShift: return "scale_shift";
    }
    return "auto";
}

// ---- stage -----------------------------------------------------------------------------

DepthStage::DepthStage(DepthStageOptions options, std::unique_ptr<IDepthEstimator> estimator)
    : options_(std::move(options)), estimator_(std::move(estimator)) {
    if (!estimator_) estimator_ = CreateDepthEstimator(options_.backend);
}

DepthStage::~DepthStage() {
    if (initialized_) {
        try {
            Shutdown();
        } catch (...) {
        }
    }
}

void DepthStage::Init(const StageConfig& config, D3D12Device&) {
    if (config.params.contains("backend")) options_.backend = config.params["backend"].get<std::string>();
    estimator_->Init(options_.estimator);
    const bool metric = estimator_->IsMetric();
    stabilizer_ = TemporalStabilizer(ResolveStabilizeMode(options_.stabilize, metric), options_.stabilizeWindow);
    if (stabilizer_.GetMode() == TemporalStabilizer::Mode::None)
        Log()->info("depth: no temporal alignment (per-frame depth as the model gives it)");
    else
        Log()->info("depth: temporal alignment '{}' over ±{} frames ({}), output delayed by {} frames", StabilizeModeName(stabilizer_.GetMode()),
                    stabilizer_.Radius(), DepthStabilizeName(options_.stabilize), stabilizer_.Delay());
    if (metric && options_.stabilize == DepthStabilize::ScaleShift)
        Log()->warn("depth: scale_shift on a metric model - the shift does not keep the metres; 'scale' is the metric mode");
    cache_ = nullptr;
    device_ = nullptr;
    if (!options_.mvDir.empty()) {
        mvReader_.emplace(PassReader::Open(options_.mvDir));
        mvReader_->Validate({0, 0, -1, PassKind::MvDlss});
    }
    initialized_ = true;
    stats_ = {};
    prevDepth_.reset();
}

void DepthStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("DepthStage::Process before Init");
    cache_ = &ctx.cache;
    device_ = &ctx.device;
    Pending p;
    p.index = ctx.frame.index;
    p.rgb = YuvToRgb(ctx.frame, options_.color, PixelType::F32);
    p.cache = &ctx.cache;
    p.device = &ctx.device;
    pending_.push_back(std::move(p));
    const int window = std::max(1, estimator_->WindowSize());
    if (static_cast<int>(pending_.size()) >= window) RunWindow(false);
}

void DepthStage::RunWindow(bool flush) {
    if (pending_.empty()) return;
    const int window = std::max(1, estimator_->WindowSize());
    const int overlap = std::clamp(estimator_->WindowOverlap(), 0, window - 1);
    if (!flush && static_cast<int>(pending_.size()) < window) return;

    std::vector<const PassImage*> rgb;
    const size_t n = std::min<size_t>(pending_.size(), static_cast<size_t>(window));
    for (size_t i = 0; i < n; ++i) rgb.push_back(&pending_[i].rgb);
    std::vector<PassImage> depth;
    estimator_->Estimate(rgb, depth);
    if (depth.size() != n) Throw("depth estimator returned " + std::to_string(depth.size()) + " maps for " + std::to_string(n) + " frames");
    ++stats_.windows;

    // Blend the overlap with the previous window's tail (linear ramp), then emit.
    for (size_t i = 0; i < n; ++i) {
        const int64_t idx = pending_[i].index;
        auto prevIt = std::find_if(overlapPrev_.begin(), overlapPrev_.end(), [idx](const auto& e) { return e.first == idx; });
        if (prevIt != overlapPrev_.end() && overlap > 0) {
            const float t = static_cast<float>(i + 1) / static_cast<float>(overlap + 1);  // 0 -> previous, 1 -> current
            PassImage& cur = depth[i];
            const PassImage& prev = prevIt->second;
            for (uint32_t y = 0; y < cur.height; ++y)
                for (uint32_t x = 0; x < cur.width; ++x) cur.Set(x, y, 0, (1 - t) * prev.Get(x, y, 0) + t * cur.Get(x, y, 0));
        }
    }
    // Frames that will be re-estimated in the next window are kept (not emitted yet) unless flushing.
    const size_t emitCount = flush || overlap == 0 || n < static_cast<size_t>(window) ? n : n - static_cast<size_t>(overlap);
    overlapPrev_.clear();
    for (size_t i = emitCount; i < n; ++i) overlapPrev_.emplace_back(pending_[i].index, depth[i]);
    for (size_t i = 0; i < emitCount; ++i) Emit(pending_[i].index, std::move(depth[i]));
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<ptrdiff_t>(emitCount));
    if (flush && !pending_.empty()) RunWindow(true);
}

void DepthStage::Emit(int64_t index, PassImage depth) {
    if (depth.type != PixelType::F32) depth = depth.ConvertTo(PixelType::F32);
    depth.channels = {"Z"};
    stabilizer_.Push(index, std::move(depth));  // holes are filled after the alignment (Finalize)
    Drain();
}

void DepthStage::Drain() {
    int64_t index = 0;
    PassImage depth;
    ScaleShift applied;
    while (stabilizer_.Pop(index, depth, &applied)) Finalize(index, std::move(depth), applied);
}

void DepthStage::Finalize(int64_t index, PassImage depth, const ScaleShift& applied) {
    // after the alignment: a filled frame without valid depth (constant 1) measured as real depth joined its neighbours'
    // windows (review of TASK-0038); the fill copies the nearest valid sample, so it commutes with the transform
    if (options_.fillHoles) stats_.holesFilled += static_cast<int64_t>(FillInvalidDepth(depth));
    if (stabilizer_.GetMode() != TemporalStabilizer::Mode::None) {
        const bool first = stats_.frames == 0;
        stats_.scaleMin = first ? applied.scale : std::min(stats_.scaleMin, applied.scale);
        stats_.scaleMax = first ? applied.scale : std::max(stats_.scaleMax, applied.scale);
        stats_.shiftMin = first ? applied.shift : std::min(stats_.shiftMin, applied.shift);
        stats_.shiftMax = first ? applied.shift : std::max(stats_.shiftMax, applied.shift);
    }
    if (prevDepth_) {
        double tae = 0.0;
        if (mvReader_ && mvReader_->HasFrame(index)) {
            PassImage mv = mvReader_->ReadFrame(index);
            if (mv.width != depth.width || mv.height != depth.height) mv = ScaleMv(mv, depth.width, depth.height);
            tae = TemporalAlignmentErrorWarped(*prevDepth_, depth, mv);
        } else {
            tae = TemporalAlignmentError(*prevDepth_, depth);
        }
        stats_.taeSum += tae;
        ++stats_.taeCount;
        Log()->debug("depth frame {}: TAE {:.4f}", index, tae);
        if (options_.onFrame) options_.onFrame(index, tae);
    } else if (options_.onFrame) {
        options_.onFrame(index, 0.0);
    }
    prevDepth_ = depth;

    if (!options_.outputDir.empty()) {
        if (!rawWriter_) {
            Manifest m = Manifest::ForPass(PassKind::DepthRaw, depth.width, depth.height, options_.format);
            m.fps = options_.fps;
            m.sourceFile = options_.sourceFile;
            m.sourceHash = options_.sourceHash;
            m.model = estimator_->Describe().value("model", estimator_->Name());
            m.depth.units = estimator_->IsMetric() ? "meters" : "relative";
            m.depth.relative = !estimator_->IsMetric();
            m.depth.zNear = options_.dlss.zNear;
            m.depth.zFar = options_.dlss.zFar;
            m.stageParams = estimator_->Describe();
            m.stageParams["stabilize"] = StabilizeModeName(stabilizer_.GetMode());
            m.stageParams["stabilize_requested"] = DepthStabilizeName(options_.stabilize);
            m.stageParams["stabilize_window"] = stabilizer_.Radius();
            m.stageParams["tae_mode"] = mvReader_ ? "warped" : "static";
            rawWriter_ = std::make_unique<PassWriter>(RawDir(), m);
            if (options_.writeDlss) {
                Manifest d = Manifest::ForPass(PassKind::DepthDlss, depth.width, depth.height, options_.format);
                d.fps = m.fps;
                d.sourceFile = m.sourceFile;
                d.sourceHash = m.sourceHash;
                d.model = m.model;
                d.depth = m.depth;
                d.stageParams = m.stageParams;
                dlssWriter_ = std::make_unique<PassWriter>(DlssDir(), d);
            }
        }
        rawWriter_->WriteFrame(index, depth);
        if (dlssWriter_) {
            DepthParams params = dlssWriter_->Man().depth;
            params.relative = !estimator_->IsMetric();
            const PassImage dlss = DepthRawToDlss(depth, params);
            const_cast<Manifest&>(dlssWriter_->Man()).depth = params;  // relative range recorded from the first frame
            const_cast<Manifest&>(rawWriter_->Man()).depth = params;
            dlssWriter_->WriteFrame(index, dlss);
        }
    }
    if (options_.uploadToGpu && cache_ && device_) {
        // the slot still holds this frame unless the estimator window / the alignment delay outlived it
        GpuFrameCache::Slot* slot = cache_->Find(index);
        if (slot) cache_->UploadPass(*slot, "depth_raw", depth);
    }
    ++stats_.frames;
}

void DepthStage::Finish() {
    RunWindow(true);
    stabilizer_.Close();
    Drain();
    if (stabilizer_.Buffered() != 0) Throw("depth: " + std::to_string(stabilizer_.Buffered()) + " frame(s) left in the temporal alignment buffer");
    const nlohmann::json scale = {stats_.scaleMin, stats_.scaleMax}, shift = {stats_.shiftMin, stats_.shiftMax};
    const bool aligned = stabilizer_.GetMode() != TemporalStabilizer::Mode::None;
    if (rawWriter_) {
        auto& sp = const_cast<Manifest&>(rawWriter_->Man()).stageParams;
        sp["tae_mean"] = stats_.MeanTae();
        sp["holes_filled"] = stats_.holesFilled;
        if (aligned) sp["stabilize_scale"] = scale;
        if (stabilizer_.GetMode() == TemporalStabilizer::Mode::ScaleShift) sp["stabilize_shift"] = shift;
        rawWriter_->Finish();
    }
    if (dlssWriter_) {
        auto& sp = const_cast<Manifest&>(dlssWriter_->Man()).stageParams;
        sp["tae_mean"] = stats_.MeanTae();
        if (aligned) sp["stabilize_scale"] = scale;
        if (stabilizer_.GetMode() == TemporalStabilizer::Mode::ScaleShift) sp["stabilize_shift"] = shift;
        dlssWriter_->Finish();
    }
    Log()->info("depth: {} frames, {} windows, mean TAE {:.4f}, holes filled {}, alignment {}{}", stats_.frames, stats_.windows, stats_.MeanTae(),
                stats_.holesFilled, StabilizeModeName(stabilizer_.GetMode()),
                aligned ? fmt::format(" (scale {:.4f}..{:.4f})", stats_.scaleMin, stats_.scaleMax) : std::string());
}

void DepthStage::Shutdown() {
    if (!initialized_) return;
    estimator_->Shutdown();
    initialized_ = false;
}

DepthRunResult RunDepth(VideoDecoder& decoder, D3D12Device& device, DepthStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<DepthStage>(options);
    DepthStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    CpuFrame frame;
    int64_t n = 0;
    while (maxFrames < 0 || n < maxFrames) {
        if (!decoder.NextFrame(frame)) break;
        GpuFrameCache::Slot& slot = pipeline.Cache().Acquire(frame.index, frame.desc);
        (void)slot;
        pipeline.ProcessFrame(frame);
        ++n;
        if (progress) progress(n);
    }
    pipeline.Finish();
    DepthRunResult r;
    r.stats = raw->GetStats();
    r.rawDir = raw->RawDir();
    r.dlssDir = raw->DlssDir();
    r.estimator = raw->Estimator().Describe();
    r.stabilize = StabilizeModeName(raw->Stabilizer().GetMode());
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
