#include "stages/flow/FlowStage.h"

#include <cmath>

#include "convert/Warp.h"
#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "pipeline/Pipeline.h"
#include "stages/flow/StubFlowEstimator.h"
#include "util/Error.h"
#include "util/Log.h"

#ifdef DLSSVID_WITH_CUDA
#include <cuda_runtime_api.h>

#include "stages/flow/OfaFlowEstimator.h"
#endif
#ifdef DLSSVID_WITH_TENSORRT
#include "stages/flow/TrtFlowEstimator.h"
#endif

namespace dlssvid {

// ---- factory ---------------------------------------------------------------------------

std::vector<std::string> FlowBackends() { return {"ofa", "searaft", "stub"}; }

std::unique_ptr<IFlowEstimator> CreateFlowEstimator(const std::string& backend) {
    if (backend == "stub") return std::make_unique<StubFlowEstimator>();
#ifdef DLSSVID_WITH_CUDA
    if (backend == "ofa") return std::make_unique<OfaFlowEstimator>();
#endif
#ifdef DLSSVID_WITH_TENSORRT
    if (backend == "searaft") return std::make_unique<TrtFlowEstimator>("searaft");
#endif
    Throw("unknown or unavailable flow backend '" + backend + "' (known: ofa, searaft, stub)");
}

// ---- stage -----------------------------------------------------------------------------

FlowStage::FlowStage(FlowStageOptions options, std::unique_ptr<IFlowEstimator> estimator) : options_(std::move(options)), estimator_(std::move(estimator)) {
    if (!estimator_) estimator_ = CreateFlowEstimator(options_.backend);
}

FlowStage::~FlowStage() {
    if (initialized_) {
        try {
            Shutdown();
        } catch (...) {
        }
    }
}

void FlowStage::Init(const StageConfig& config, D3D12Device&) {
    if (config.params.contains("backend")) options_.backend = config.params["backend"].get<std::string>();
    if (!options_.depthDir.empty()) {
        depthReader_.emplace(PassReader::Open(options_.depthDir));
        depthReader_->Validate({0, 0, -1, PassKind::DepthRaw});
    }
    initialized_ = true;
    stats_ = {};
    prev_.reset();
}

std::optional<PassImage> FlowStage::DepthFor(int64_t index, GpuFrameCache* cache) const {
    if (depthReader_ && depthReader_->HasFrame(index)) return depthReader_->ReadFrame(index);
    if (options_.useCacheDepth && cache) {
        if (GpuFrameCache::Slot* slot = cache->Find(index)) {
            if (slot->passes.count("depth_raw")) return cache->DownloadPass(*slot, "depth_raw");
        }
    }
    return std::nullopt;
}

void FlowStage::EnsureWriters(uint32_t w, uint32_t h) {
    if (rawWriter_ || options_.outputDir.empty()) return;
    Manifest m = Manifest::ForPass(PassKind::MvRaw, w, h, options_.format);
    m.fps = options_.fps;
    m.sourceFile = options_.sourceFile;
    m.sourceHash = options_.sourceHash;
    m.model = estimator_->Describe().value("model", estimator_->Name());
    m.stageParams = estimator_->Describe();
    rawWriter_ = std::make_unique<PassWriter>(RawDir(), m);
    if (options_.writeDlss) {
        const uint32_t tw = options_.targetWidth ? options_.targetWidth : w, th = options_.targetHeight ? options_.targetHeight : h;
        Manifest d = Manifest::ForPass(PassKind::MvDlss, tw, th, options_.format);
        d.fps = m.fps;
        d.sourceFile = m.sourceFile;
        d.sourceHash = m.sourceHash;
        d.model = m.model;
        d.stageParams = m.stageParams;
        d.stageParams["dilate"] = options_.convert.dilateRadius;
        d.stageParams["depth_occlusion"] = !options_.depthDir.empty() || options_.useCacheDepth;
        d.mv.refWidth = tw;
        d.mv.refHeight = th;
        dlssWriter_ = std::make_unique<PassWriter>(DlssDir(), d);
    }
}

void FlowStage::Process(FrameContext& ctx) {
    if (!initialized_) Throw("FlowStage::Process before Init");
    Held cur;
    cur.index = ctx.frame.index;
    cur.cache = &ctx.cache;
    if (!ctx.frame.data.empty()) cur.rgb = Yuv420pToRgb(ctx.frame, options_.color, PixelType::F32);
    const uint32_t w = ctx.frame.desc.width, h = ctx.frame.desc.height;
    if (stats_.frames == 0) estimator_->Init(options_.estimator, w, h);
    ++stats_.frames;

#ifdef DLSSVID_WITH_CUDA
    if (ctx.gpu && ctx.gpu->Valid() && estimator_->WantsGpuFrames()) {
        // Keep a device copy of the NV12 planes: the decoder reuses its frame on the next call.
        const size_t bytes = ctx.gpu->pitch * (static_cast<size_t>(ctx.gpu->height) + ctx.gpu->height / 2);
        if (gpuCopyBytes_ != bytes) {
            for (void*& p : gpuCopy_) {
                if (p) cudaFree(p);
                p = nullptr;
            }
            gpuCopyBytes_ = bytes;
        }
        void*& dst = gpuCopy_[gpuCopySlot_];
        if (!dst && cudaMalloc(&dst, bytes) != cudaSuccess) Throw("FlowStage: cudaMalloc for the held frame failed");
        auto* base = static_cast<uint8_t*>(dst);
        if (cudaMemcpy(base, reinterpret_cast<const void*>(ctx.gpu->y), ctx.gpu->pitch * ctx.gpu->height, cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpy(base + ctx.gpu->pitch * ctx.gpu->height, reinterpret_cast<const void*>(ctx.gpu->uv), ctx.gpu->pitch * (ctx.gpu->height / 2),
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
            Throw("FlowStage: device copy of the NV12 frame failed");
        cur.gpu = *ctx.gpu;
        cur.gpu.y = reinterpret_cast<uintptr_t>(base);
        cur.gpu.uv = reinterpret_cast<uintptr_t>(base + ctx.gpu->pitch * ctx.gpu->height);
        gpuCopySlot_ ^= 1;
    }
#endif

    if (prev_) EmitPair(*prev_, cur);
    prev_ = std::move(cur);
}

void FlowStage::EmitPair(Held& prev, Held& cur) {
    FlowInput a, b;
    if (!prev.rgb.Empty()) a.rgb = &prev.rgb;
    if (!cur.rgb.Empty()) b.rgb = &cur.rgb;
    if (prev.gpu.Valid()) a.gpu = &prev.gpu;
    if (cur.gpu.Valid()) b.gpu = &cur.gpu;
    PassImage flow, conf;
    estimator_->Estimate(a, b, flow, &conf);
    if (flow.type != PixelType::F32) flow = flow.ConvertTo(PixelType::F32);
    flow.channels = Spec(PassKind::MvRaw).channels;
    ++stats_.pairs;
    double mag = 0;
    for (uint32_t y = 0; y < flow.height; ++y)
        for (uint32_t x = 0; x < flow.width; ++x) mag += std::hypot(flow.Get(x, y, 0), flow.Get(x, y, 1));
    mag /= static_cast<double>(flow.width) * flow.height;
    stats_.meanMagnitude += (mag - stats_.meanMagnitude) / static_cast<double>(stats_.pairs);

    EnsureWriters(flow.width, flow.height);
    if (rawWriter_) rawWriter_->WriteFrame(prev.index, flow);
    if (options_.uploadToGpu && prev.cache) {
        if (GpuFrameCache::Slot* slot = prev.cache->Find(prev.index)) prev.cache->UploadPass(*slot, "mv_raw", flow);
    }

    // mv_dlss for the current frame: invert the previous frame's forward flow.
    if (options_.writeDlss || options_.computeWarpPsnr || options_.uploadToGpu) {
        std::optional<PassImage> depth = prev.depth ? prev.depth : DepthFor(prev.index, prev.cache);
        MvConvertOptions opt = options_.convert;
        opt.targetWidth = options_.targetWidth;
        opt.targetHeight = options_.targetHeight;
        const PassImage mvDlss = ForwardFlowToBackwardMv(flow, depth ? &*depth : nullptr, opt);
        if (dlssWriter_) {
            if (stats_.pairs == 1) {
                // first frame of the sequence has no previous frame: zero vectors
                dlssWriter_->WriteFrame(prev.index, ZeroMv(mvDlss.width, mvDlss.height));
            }
            dlssWriter_->WriteFrame(cur.index, mvDlss);
        }
        if (options_.uploadToGpu && cur.cache) {
            if (GpuFrameCache::Slot* slot = cur.cache->Find(cur.index)) cur.cache->UploadPass(*slot, "mv_dlss", mvDlss);
        }
        if (options_.computeWarpPsnr && !prev.rgb.Empty() && !cur.rgb.Empty()) {
            const PassImage mvSrc = (mvDlss.width == cur.rgb.width && mvDlss.height == cur.rgb.height) ? mvDlss : ScaleMv(mvDlss, cur.rgb.width, cur.rgb.height);
            const double psnr = WarpPsnr(prev.rgb, cur.rgb, mvSrc);
            stats_.psnrSum += psnr;
            ++stats_.psnrCount;
            stats_.minWarpPsnr = std::min(stats_.minWarpPsnr, psnr);
            Log()->debug("flow {}->{}: mean |v| {:.2f} px, warp PSNR {:.2f} dB", prev.index, cur.index, mag, psnr);
            if (options_.onFrame) options_.onFrame(cur.index, psnr);
        }
    }
}

void FlowStage::EmitLast(Held& last) {
    if (!rawWriter_) {
        if (options_.outputDir.empty()) return;
        // single-frame clip: no pair was ever computed
        const uint32_t w = last.rgb.Empty() ? last.gpu.width : last.rgb.width, h = last.rgb.Empty() ? last.gpu.height : last.rgb.height;
        if (!w || !h) return;
        EnsureWriters(w, h);
        if (dlssWriter_) dlssWriter_->WriteFrame(last.index, ZeroMv(dlssWriter_->Man().width, dlssWriter_->Man().height));
    }
    // last frame has no next frame: zero forward flow (docs/conventions.md §3)
    PassImage zero = ZeroMv(rawWriter_->Man().width, rawWriter_->Man().height);
    zero.channels = Spec(PassKind::MvRaw).channels;
    rawWriter_->WriteFrame(last.index, zero);
}

void FlowStage::Finish() {
    if (prev_) {
        EmitLast(*prev_);
        prev_.reset();
    }
    if (rawWriter_) {
        const_cast<Manifest&>(rawWriter_->Man()).stageParams["mean_magnitude_px"] = stats_.meanMagnitude;
        rawWriter_->Finish();
    }
    if (dlssWriter_) {
        const_cast<Manifest&>(dlssWriter_->Man()).stageParams["warp_psnr_mean_db"] = stats_.MeanWarpPsnr();
        const_cast<Manifest&>(dlssWriter_->Man()).stageParams["warp_psnr_min_db"] = stats_.psnrCount ? stats_.minWarpPsnr : 0.0;
        dlssWriter_->Finish();
    }
    Log()->info("flow: {} frames, {} pairs, mean |v| {:.2f} px, warp PSNR mean {:.2f} dB (min {:.2f})", stats_.frames, stats_.pairs, stats_.meanMagnitude,
                stats_.MeanWarpPsnr(), stats_.psnrCount ? stats_.minWarpPsnr : 0.0);
}

void FlowStage::Shutdown() {
    if (!initialized_) return;
    estimator_->Shutdown();
#ifdef DLSSVID_WITH_CUDA
    for (void*& p : gpuCopy_) {
        if (p) cudaFree(p);
        p = nullptr;
    }
#endif
    initialized_ = false;
}

FlowRunResult RunFlow(VideoDecoder& decoder, D3D12Device& device, FlowStageOptions options, int64_t maxFrames, const std::function<void(int64_t)>& progress) {
    const auto& info = decoder.Info();
    options.color = ColorInfoFromStream(info);
    options.fps = info.frameRate;
    Pipeline pipeline(device, 4);
    auto stage = std::make_unique<FlowStage>(options);
    FlowStage* raw = stage.get();
    pipeline.AddStage(std::move(stage));
    pipeline.Init();
    CpuFrame frame;
    GpuFrame gpu;
    int64_t n = 0;
    const bool wantGpu = decoder.UsingHwAccel();
    while (maxFrames < 0 || n < maxFrames) {
        if (!decoder.NextFrame(frame, wantGpu ? &gpu : nullptr, true)) break;
        pipeline.Cache().Acquire(frame.index, frame.desc);
        pipeline.ProcessFrame(frame, wantGpu && gpu.Valid() ? &gpu : nullptr);
        ++n;
        if (progress) progress(n);
    }
    pipeline.Finish();
    FlowRunResult r;
    r.stats = raw->GetStats();
    r.rawDir = raw->RawDir();
    r.dlssDir = raw->DlssDir();
    r.estimator = raw->Estimator().Describe();
    pipeline.Shutdown();
    return r;
}

}  // namespace dlssvid
