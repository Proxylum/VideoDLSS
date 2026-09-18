#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#include "gpu/GpuFrame.h"
#include "pipeline/Frame.h"

namespace dlssvid {

class D3D12Device;
class GpuFrameCache;

// Serializable per-stage configuration (see ТЗ §4 «Стадии и контракты»).
struct StageConfig {
    std::string name;
    nlohmann::json params = nlohmann::json::object();
};

// Everything a stage needs for one frame. Stages read and write passes here; in stage 0
// the only pass is the source colour frame kept on the CPU plus its GPU cache slot.
struct FrameContext {
    CpuFrame& frame;
    D3D12Device& device;
    GpuFrameCache& cache;
    const GpuFrame* gpu = nullptr;  // NVDEC frame still on the device (nullptr when decoded on the CPU)
};

class IStage {
public:
    virtual ~IStage() = default;
    virtual std::string_view Name() const = 0;
    virtual void Init(const StageConfig& config, D3D12Device& device) = 0;
    virtual void Process(FrameContext& ctx) = 0;
    // Called once after the last frame: windowed stages flush their tail here.
    virtual void Finish() {}
    virtual void Shutdown() = 0;
};

}  // namespace dlssvid
