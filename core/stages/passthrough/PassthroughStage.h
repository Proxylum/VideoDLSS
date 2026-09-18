#pragma once

#include "pipeline/IStage.h"

namespace dlssvid {

// Stage 0 reference stage: uploads the frame into its GPU cache slot and reads it back.
// Output must be bit-identical to input; this exercises the device, the cache and the
// upload/readback path that every real stage will build on.
class PassthroughStage final : public IStage {
public:
    std::string_view Name() const override { return "passthrough"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Shutdown() override;

    int64_t FramesProcessed() const { return processed_; }

private:
    bool roundTrip_ = true;  // params.gpu_roundtrip (default true)
    int64_t processed_ = 0;
};

}  // namespace dlssvid
