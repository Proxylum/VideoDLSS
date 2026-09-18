#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "pipeline/IStage.h"

namespace dlssvid {

class D3D12Device;
class GpuFrameCache;
class VideoDecoder;
class VideoEncoder;

// Linear list of stages executed per frame. Stage 0 has no scheduler or graph yet: the
// order of AddStage() is the execution order. Each stage gets a FrameContext bound to the
// shared D3D12 device and the GPU frame cache.
class Pipeline {
public:
    explicit Pipeline(D3D12Device& device, uint32_t cacheSlots = 4);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void AddStage(std::unique_ptr<IStage> stage, StageConfig config = {});
    size_t StageCount() const { return stages_.size(); }

    void Init();
    void ProcessFrame(CpuFrame& frame);
    void Finish();  // after the last frame, before Shutdown
    void Shutdown();

    D3D12Device& Device() { return device_; }
    GpuFrameCache& Cache() { return *cache_; }

private:
    struct Entry {
        std::unique_ptr<IStage> stage;
        StageConfig config;
    };
    D3D12Device& device_;
    std::unique_ptr<GpuFrameCache> cache_;
    std::vector<Entry> stages_;
    bool initialized_ = false;
};

struct RunStats {
    int64_t framesIn = 0;
    int64_t framesOut = 0;
    double seconds = 0.0;
};

// Decode every frame, run it through the pipeline and encode it. Audio packets are copied
// through untouched. `maxFrames < 0` means all frames.
RunStats RunPipeline(Pipeline& pipeline, VideoDecoder& decoder, VideoEncoder& encoder,
                     const std::function<void(int64_t)>& progress = {}, int64_t maxFrames = -1);

}  // namespace dlssvid
