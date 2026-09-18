#include "stages/passthrough/PassthroughStage.h"

#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "util/Error.h"

namespace dlssvid {

void PassthroughStage::Init(const StageConfig& config, D3D12Device&) {
    roundTrip_ = config.params.value("gpu_roundtrip", true);
    processed_ = 0;
}

void PassthroughStage::Process(FrameContext& ctx) {
    if (roundTrip_) {
        GpuFrameCache::Slot& slot = ctx.cache.Acquire(ctx.frame.index, ctx.frame.desc);
        ctx.cache.Upload(slot, ctx.frame);
        CpuFrame back;
        ctx.cache.Download(slot, back);
        if (back.data != ctx.frame.data) Throw("passthrough: GPU round trip is not bit-exact");
    }
    ++processed_;
}

void PassthroughStage::Shutdown() {}

}  // namespace dlssvid
