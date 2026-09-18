#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "stages/passthrough/PassthroughStage.h"

using namespace dlssvid;

namespace {

// Records what it sees; optionally mutates the frame so ordering is observable.
class RecordingStage final : public IStage {
public:
    explicit RecordingStage(std::vector<std::string>& trace, std::string name, uint8_t add)
        : trace_(trace), name_(std::move(name)), add_(add) {}
    std::string_view Name() const override { return name_; }
    void Init(const StageConfig& cfg, D3D12Device&) override { trace_.push_back("init:" + cfg.name); }
    void Process(FrameContext& ctx) override {
        trace_.push_back("process:" + name_ + ":" + std::to_string(ctx.frame.index));
        for (auto& b : ctx.frame.data) b = static_cast<uint8_t>(b + add_);
    }
    void Shutdown() override { trace_.push_back("shutdown:" + name_); }

private:
    std::vector<std::string>& trace_;
    std::string name_;
    uint8_t add_;
};

CpuFrame Frame(int64_t index) {
    CpuFrame f;
    f.Allocate(FrameDesc{16, 8, PixelFormat::Yuv420p});
    f.index = index;
    for (size_t i = 0; i < f.data.size(); ++i) f.data[i] = static_cast<uint8_t>(i + index);
    return f;
}

}  // namespace

TEST_CASE("Pipeline runs stages in order and shuts down in reverse", "[pipeline]") {
    D3D12Device dev({true, false});
    Pipeline p(dev, 2);
    std::vector<std::string> trace;
    p.AddStage(std::make_unique<RecordingStage>(trace, "a", 1));
    p.AddStage(std::make_unique<RecordingStage>(trace, "b", 2));
    REQUIRE(p.StageCount() == 2);

    CpuFrame f = Frame(0);
    CHECK_THROWS(p.ProcessFrame(f));  // before Init
    p.Init();
    CHECK_THROWS(p.AddStage(std::make_unique<RecordingStage>(trace, "late", 0)));  // no stages after Init
    p.ProcessFrame(f);
    CpuFrame g = Frame(1);
    p.ProcessFrame(g);
    p.Shutdown();

    const std::vector<std::string> expected = {"init:a",       "init:b",       "process:a:0", "process:b:0",
                                               "process:a:1",  "process:b:1",  "shutdown:b",  "shutdown:a"};
    CHECK(trace == expected);
    CHECK(f.data[0] == 3);  // 0 + 1 + 2
}

TEST_CASE("PassthroughStage leaves frames bit-identical", "[pipeline][passthrough]") {
    D3D12Device dev;
    Pipeline p(dev, 4);
    auto stage = std::make_unique<PassthroughStage>();
    auto* raw = stage.get();
    p.AddStage(std::move(stage));
    p.Init();
    for (int64_t i = 0; i < 6; ++i) {
        CpuFrame f = Frame(i);
        const auto copy = f.data;
        p.ProcessFrame(f);
        CHECK(f.data == copy);
    }
    CHECK(raw->FramesProcessed() == 6);
    p.Shutdown();
}

TEST_CASE("PassthroughStage honours gpu_roundtrip=false", "[pipeline][passthrough]") {
    D3D12Device dev({true, false});
    Pipeline p(dev, 1);
    StageConfig cfg;
    cfg.params["gpu_roundtrip"] = false;
    p.AddStage(std::make_unique<PassthroughStage>(), cfg);
    p.Init();
    CpuFrame f = Frame(3);
    const auto copy = f.data;
    p.ProcessFrame(f);
    CHECK(f.data == copy);
    CHECK(p.Cache().Find(3) == nullptr);  // nothing was uploaded
}
