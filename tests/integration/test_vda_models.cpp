// VDA Base and Large (TASK-0039) on a short clip: through TensorRT (export with the patched attention, engine build,
// the window the registry or the caller gives) and through the PyTorch worker (SDPA attention). Skipped without
// TensorRT / an NVIDIA GPU / the cached checkpoints - nothing is downloaded by the tests. The first run exports and
// builds the engines at 252x140 with a window of 4 (minutes for Large); later runs take them from the cache.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <string>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "ml/ModelRegistry.h"
#include "passes/PassSequence.h"
#include "stages/depth/DepthStage.h"
#include "stages/depth/WorkerDepthEstimator.h"
#include "util/Subprocess.h"
#ifdef DLSSVID_WITH_TENSORRT
#include "ml/TrtLoader.h"
#endif

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

const char* CheckpointOf(const std::string& id) {
    if (id == "metric-vda-base") return "depth-anything__Metric-Video-Depth-Anything-Base__metric_video_depth_anything_vitb.pth";
    if (id == "metric-vda-large") return "depth-anything__Metric-Video-Depth-Anything-Large__metric_video_depth_anything_vitl.pth";
    return "depth-anything__Metric-Video-Depth-Anything-Small__metric_video_depth_anything_vits.pth";
}

std::filesystem::path Dir(const std::string& name) {
    const auto d = TempDir() / "vda_models" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

void CheckDepth(const std::filesystem::path& rawDir, int frames) {
    const PassReader r = PassReader::Open(rawDir);
    REQUIRE(r.Man().frameCount == frames);
    for (int i = 0; i < frames; i += frames - 1) {
        const PassImage d = r.ReadFrame(i);
        double sum = 0.0;
        for (uint32_t y = 0; y < d.height; ++y)
            for (uint32_t x = 0; x < d.width; ++x) {
                const float v = d.Get(x, y, 0);
                REQUIRE(std::isfinite(v));
                REQUIRE(v > 0.f);
                sum += v;
            }
        CHECK(sum / (static_cast<double>(d.width) * d.height) < 1000.0);  // metres, not garbage
    }
}

}  // namespace

#ifdef DLSSVID_WITH_TENSORRT
TEST_CASE("VDA Base and Large through TensorRT on a short clip with a window of 4", "[integration][depth][gpu][model]") {
    std::string reason;
    if (!trt::Available(&reason)) SKIP("TensorRT unavailable: " << reason);
    D3D12Device dev;
    if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU");
    const auto cache = ModelRegistry::DefaultRegistryPath().parent_path() / "cache";
    for (const std::string id : {"metric-vda-base", "metric-vda-large"}) {
        DYNAMIC_SECTION(id) {
            if (!std::filesystem::exists(cache / CheckpointOf(id))) SKIP(id << " checkpoint not cached (models/export/fetch.py " << id << ")");
            const auto dir = Dir(id);
            ClipSpec spec;
            spec.frames = 8;
            spec.width = 64;
            spec.height = 36;
            const auto clip = WriteClip(dir / "clip.mkv", spec);
            VideoDecoder dec(clip);
            DepthStageOptions o;
            o.backend = "vda";
            o.estimator.modelId = id;
            o.estimator.window = 4;
            o.estimator.inputSize = 140;  // 252x140: a small engine
            o.stabilize = DepthStabilize::None;
            o.outputDir = dir / "passes";
            const DepthRunResult r = RunDepth(dec, dev, o);
            CHECK(r.stats.frames == 8);
            CHECK(r.estimator["model"] == id);
            CHECK(r.estimator["window"] == 4);
            CHECK(r.estimator["overlap"] == 1);  // the registry's 8 of 32, scaled
            CHECK(r.estimator["metric"] == true);
            CHECK(r.estimator["license"] == "CC-BY-NC-4.0");
            CHECK(r.estimator["input"] == nlohmann::json::array({252, 140}));
            CHECK(r.estimator["vram_mib"].get<int64_t>() > 0);
            CheckDepth(r.rawDir, 8);
        }
    }
}
#endif

TEST_CASE("VDA Large through the PyTorch worker with a window of 4", "[integration][depth][worker][model]") {
    const auto root = WorkerDepthEstimator::FindProjectRoot();
    DepthEstimatorConfig cfg;
    cfg.modelsDir = (root / "models").string();
    const auto python = WorkerDepthEstimator::FindPython(cfg, root);
    if (Subprocess::Run({python.string(), "-c", "import torch"}) != 0) SKIP("no python with torch for the worker (" << python.string() << ")");
    if (!std::filesystem::exists(root / "models" / "cache" / CheckpointOf("metric-vda-large"))) SKIP("metric-vda-large checkpoint not cached");
    cfg.modelId = "metric-vda-large";
    cfg.window = 4;
    cfg.inputSize = 140;
    WorkerDepthEstimator w("vda");
    w.Init(cfg);
    CHECK(w.WindowSize() == 4);
    CHECK(w.WindowOverlap() == 1);
    CHECK(w.IsMetric());
    std::vector<PassImage> frames;
    for (int i = 0; i < 4; ++i) {
        PassImage f = MakePassImage(PassKind::ColorSource, 64, 36, PixelType::F32);
        for (uint32_t y = 0; y < 36; ++y)
            for (uint32_t x = 0; x < 64; ++x) {
                f.Set(x, y, 0, (x + i) / 72.f);
                f.Set(x, y, 1, y / 36.f);
                f.Set(x, y, 2, 0.5f);
            }
        frames.push_back(std::move(f));
    }
    std::vector<const PassImage*> in;
    for (const auto& f : frames) in.push_back(&f);
    std::vector<PassImage> out;
    w.Estimate(in, out);
    REQUIRE(out.size() == 4);
    for (const auto& d : out) {
        CHECK(d.width == 64);
        CHECK(d.height == 36);
        CHECK(std::isfinite(d.Get(32, 18, 0)));
        CHECK(d.Get(32, 18, 0) > 0.f);
    }
    w.Shutdown();
}
