#include "pipeline/Pipeline.h"

#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

Pipeline::Pipeline(D3D12Device& device, uint32_t cacheSlots)
    : device_(device), cache_(std::make_unique<GpuFrameCache>(device, cacheSlots)) {}

Pipeline::~Pipeline() {
    if (initialized_) Shutdown();
}

void Pipeline::AddStage(std::unique_ptr<IStage> stage, StageConfig config) {
    if (initialized_) Throw("Pipeline::AddStage after Init");
    if (config.name.empty()) config.name = std::string(stage->Name());
    stages_.push_back(Entry{std::move(stage), std::move(config)});
}

void Pipeline::Init() {
    for (auto& e : stages_) {
        Log()->debug("stage init: {}", e.config.name);
        e.stage->Init(e.config, device_);
    }
    initialized_ = true;
}

void Pipeline::ProcessFrame(CpuFrame& frame) {
    if (!initialized_) Throw("Pipeline::ProcessFrame before Init");
    FrameContext ctx{frame, device_, *cache_};
    for (auto& e : stages_) e.stage->Process(ctx);
}

void Pipeline::Shutdown() {
    for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) it->stage->Shutdown();
    initialized_ = false;
}

RunStats RunPipeline(Pipeline& pipeline, VideoDecoder& decoder, VideoEncoder& encoder,
                     const std::function<void(int64_t)>& progress, int64_t maxFrames) {
    RunStats stats;
    const auto t0 = std::chrono::steady_clock::now();

    if (decoder.HasAudio()) {
        encoder.AddAudioStreamCopy(decoder.AudioCodecParameters(), decoder.AudioTimeBase());
        decoder.SetAudioPacketSink([&encoder](AVPacket* pkt) { encoder.WriteAudioPacket(pkt); });
    }
    encoder.Open();

    CpuFrame frame;
    while (maxFrames < 0 || stats.framesIn < maxFrames) {
        if (!decoder.NextFrame(frame)) break;
        ++stats.framesIn;
        pipeline.ProcessFrame(frame);
        encoder.WriteFrame(frame);
        ++stats.framesOut;
        if (progress) progress(stats.framesOut);
    }
    encoder.Close();

    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return stats;
}

}  // namespace dlssvid
