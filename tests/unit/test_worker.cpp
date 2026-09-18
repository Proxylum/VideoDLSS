// Subprocess plumbing and the depth worker protocol with the `stub` backend (needs python + numpy).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>

#include "stages/depth/StubDepthEstimator.h"
#include "stages/depth/WorkerDepthEstimator.h"
#include "util/Subprocess.h"

using namespace dlssvid;

namespace {
bool HavePythonNumpy() {
    static const bool ok = std::system("python -c \"import numpy\" >nul 2>&1") == 0;
    return ok;
}
PassImage Frame(uint32_t w, uint32_t h, int i) {
    PassImage img = MakePassImage(PassKind::ColorSource, w, h, PixelType::F32);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            img.Set(x, y, 0, (x + i) / static_cast<float>(w + 8));
            img.Set(x, y, 1, y / static_cast<float>(h));
            img.Set(x, y, 2, 0.5f);
        }
    return img;
}
}  // namespace

TEST_CASE("QuoteArg follows Windows rules", "[worker][subprocess]") {
    CHECK(QuoteArg("plain") == "plain");
    CHECK(QuoteArg("with space") == "\"with space\"");
    CHECK(QuoteArg("say \"hi\"") == "\"say \\\"hi\\\"\"");
    CHECK(QuoteArg("trail\\") == "trail\\");
    CHECK(QuoteArg("a b\\") == "\"a b\\\\\"");
}

TEST_CASE("Subprocess round-trips lines with python", "[worker][subprocess]") {
    if (!HavePythonNumpy()) SKIP("python not on PATH");
    Subprocess p;
    p.Start({"python", "-u", "-c", "import sys\nfor line in sys.stdin:\n    print('echo:' + line.strip())\n    sys.stdout.flush()\nprint('bye')"});
    REQUIRE(p.Running());
    p.WriteLine("one");
    CHECK(p.ReadLine(5000) == std::optional<std::string>("echo:one"));
    p.WriteLine("two");
    CHECK(p.ReadLine(5000) == std::optional<std::string>("echo:two"));
    p.CloseStdin();
    CHECK(p.ReadLine(5000) == std::optional<std::string>("bye"));
    CHECK_FALSE(p.ReadLine(5000).has_value());
    CHECK(p.Wait(5000) == 0);
    CHECK(p.ExitCode() == std::optional<uint32_t>(0));
    std::string out;
    CHECK(Subprocess::Run({"python", "-c", "print(6*7)"}, &out) == 0);
    CHECK(out == "42\n");
    CHECK(Subprocess::Run({"python", "-c", "import sys; sys.exit(3)"}) == 3);
    CHECK_THROWS(Subprocess().Start({"definitely-not-a-program-xyz"}));
}

TEST_CASE("Worker stub backend matches the C++ stub and honours windows", "[worker][depth]") {
    if (!HavePythonNumpy()) SKIP("python + numpy not on PATH");
    WorkerDepthEstimator est("stub");
    DepthEstimatorConfig cfg;
    cfg.extra["python"] = "python";
    cfg.extra["window"] = 3;
    cfg.extra["overlap"] = 1;
    est.Init(cfg);
    CHECK(est.IsMetric());
    CHECK(est.WindowSize() == 3);
    CHECK(est.WindowOverlap() == 1);
    CHECK(est.Name() == "worker:stub");
    const PassImage a = Frame(24, 12, 0), b = Frame(24, 12, 1);
    std::vector<PassImage> out;
    est.Estimate({&a, &b}, out);
    REQUIRE(out.size() == 2);
    CHECK(out[0].width == 24);
    CHECK(out[0].channels == std::vector<std::string>{"Z"});
    const PassImage expA = StubDepthEstimator::Expected(a), expB = StubDepthEstimator::Expected(b);
    CHECK(PassImage::MaxAbsDiff(out[0], expA) < 2e-3);  // rgb travels as float16
    CHECK(PassImage::MaxAbsDiff(out[1], expB) < 2e-3);
    CHECK(est.Describe()["runtime"] == "pytorch (depth_worker)");
    est.Shutdown();
}

TEST_CASE("Worker reports errors and unavailable backends", "[worker][depth]") {
    if (!HavePythonNumpy()) SKIP("python + numpy not on PATH");
    WorkerDepthEstimator bad("icdepth");
    DepthEstimatorConfig cfg;
    cfg.extra["python"] = "python";
    CHECK_THROWS(bad.Init(cfg));  // no public code
    WorkerDepthEstimator unknown("nope");
    CHECK_THROWS(unknown.Init(cfg));
}
