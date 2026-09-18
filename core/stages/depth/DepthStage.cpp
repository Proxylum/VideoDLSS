#include "stages/depth/DepthStage.h"

#include <algorithm>

#include "convert/DepthConvert.h"
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

// ---- stage -----------------------------------------------------------------------------

DepthStage::DepthStage(DepthStageOptions options, std::unique_ptr<IDepthEstimator> estimator)
    : options_(std::move(options)), estimator_(std::move(estimator)), stabilizer_(options_.stabilize, options_.stabilizeWindow) {
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
    if (estimator_->IsMetric()) {
        stabilizer_ = TemporalStabilizer(options_.stabilizeMetric ? TemporalStabilizer::Mode::ScaleOnly : TemporalStabilizer::Mode::None, options_.stabilizeWindow);
    } else {
        stabilizer_ = TemporalStabilizer(options_.stabilize, options_.stabilizeWindow);
    }
    initialized_ = true;
    stats_ = {};
    prevDepth_.reset();
}

void DepthStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("DepthStage::Process before Init");
    Pending p;
    p.index = ctx.frame.index;
    p.rgb = Yuv420pToRgb(ctx.frame, options_.color, PixelType::F32);
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
    for (size_t i = 0; i < emitCount; ++i) Emit(pending_[i].index, std::move(depth[i]), pending_[i]);
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<ptrdiff_t>(emitCount));
    if (flush && !pending_.empty()) RunWindow(true);
}

void DepthStage::Emit(int64_t index, PassImage depth, Pending& p) {
    if (depth.type != PixelType::F32) depth = depth.ConvertTo(PixelType::F32);
    depth.channels = {"Z"};
    if (options_.fillHoles) stats_.holesFilled += static_cast<int64_t>(FillInvalidDepth(depth));
    stabilizer_.Stabilize(depth);
    if (prevDepth_) {
        const double tae = TemporalAlignmentError(*prevDepth_, depth);
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
            m.stageParams["stabilize"] = stabilizer_.GetMode() == TemporalStabilizer::Mode::None ? "none" : stabilizer_.GetMode() == TemporalStabilizer::Mode::ScaleOnly ? "scale" : "scale_shift";
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
    if (options_.uploadToGpu && p.cache && p.device) {
        GpuFrameCache::Slot* slot = p.cache->Find(index);
        if (slot) p.cache->UploadPass(*slot, "depth_raw", depth);
    }
    ++stats_.frames;
}

void DepthStage::Finish() {
    RunWindow(true);
    if (rawWriter_) {
        auto tae = stats_.MeanTae();
        const_cast<Manifest&>(rawWriter_->Man()).stageParams["tae_mean"] = tae;
        const_cast<Manifest&>(rawWriter_->Man()).stageParams["holes_filled"] = stats_.holesFilled;
        rawWriter_->Finish();
    }
    if (dlssWriter_) {
        const_cast<Manifest&>(dlssWriter_->Man()).stageParams["tae_mean"] = stats_.MeanTae();
        dlssWriter_->Finish();
    }
    Log()->info("depth: {} frames, {} windows, mean TAE {:.4f}, holes filled {}", stats_.frames, stats_.windows, stats_.MeanTae(), stats_.holesFilled);
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
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
