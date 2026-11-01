// Subprocess plumbing and the depth worker protocol with the `stub` backend (needs python + numpy).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "ml/ModelRegistry.h"
#include "stages/depth/StubDepthEstimator.h"
#include "stages/depth/WorkerDepthEstimator.h"
#include "stages/upscale/WorkerUpscaler.h"
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
// DLSSVID_MODELS_DIR for one test; the value before it (the CI runner sets one) comes back afterwards.
struct ModelsDirEnv {
    std::optional<std::string> before;
    explicit ModelsDirEnv(const std::filesystem::path& dir) {
        if (const char* e = std::getenv("DLSSVID_MODELS_DIR")) before = e;
        _putenv_s("DLSSVID_MODELS_DIR", dir.string().c_str());
    }
    ~ModelsDirEnv() { _putenv_s("DLSSVID_MODELS_DIR", before ? before->c_str() : ""); }
};
// TMP and TEMP for one test (temp_directory_path reads TMP first, the children inherit both); the values before come back.
struct TempEnv {
    std::optional<std::wstring> tmp, temp;
    explicit TempEnv(const std::filesystem::path& dir) {
        if (const wchar_t* e = _wgetenv(L"TMP")) tmp = e;
        if (const wchar_t* e = _wgetenv(L"TEMP")) temp = e;
        _wputenv_s(L"TMP", dir.c_str());
        _wputenv_s(L"TEMP", dir.c_str());
    }
    ~TempEnv() {
        _wputenv_s(L"TMP", tmp ? tmp->c_str() : L"");
        _wputenv_s(L"TEMP", temp ? temp->c_str() : L"");
    }
};
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

// Review of TASK-0066: the requests carry UTF-8 paths (the scratch frames under TEMP, the models folder) and Python read
// its stdin pipe in the ANSI code page - the paths came garbled («И», UTF-8 D0 98, even as a lone surrogate: 0x98 is no
// CP1251 character), the frames were not found. PythonCommand starts it in Python's UTF-8 mode.
TEST_CASE("PythonCommand runs a script in Python's UTF-8 mode, the paths as UTF-8", "[worker][subprocess][regression]") {
    CHECK(PythonCommand("python", std::filesystem::path(u8"D:/Скрипты/w.py")) ==
          std::vector<std::string>{"python", "-X", "utf8", Utf8Arg(std::filesystem::path(u8"D:/Скрипты/w.py"))});
}

TEST_CASE("Worker stub backend takes non-ASCII paths: the scratch under a Cyrillic TEMP, a Cyrillic models folder", "[worker][depth][regression]") {
    if (!HavePythonNumpy()) SKIP("python + numpy not on PATH");
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "worker_utf8";
    std::filesystem::remove_all(root);
    const auto temp = root / u8"Временная папка И";
    std::filesystem::create_directories(temp);
    const TempEnv env(temp);
    const ModelsDirEnv models(root / u8"Модели И");  // sent in the init request
    WorkerDepthEstimator est("stub");
    DepthEstimatorConfig cfg;
    cfg.extra["python"] = "python";
    est.Init(cfg);
    const PassImage a = Frame(24, 12, 0), b = Frame(24, 12, 1);
    std::vector<PassImage> out;
    est.Estimate({&a, &b}, out);  // the frames go through .npz files in the scratch folder
    REQUIRE(out.size() == 2);
    CHECK(PassImage::MaxAbsDiff(out[1], StubDepthEstimator::Expected(b)) < 2e-3);
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

// TASK-0039: the PyTorch path checks the model and window with the same rules and registry as TensorRT, before python
// starts - a wrong id gives the family's models, not a Python KeyError after the model download.
TEST_CASE("Worker VDA refuses an unknown or foreign model and a bad window before starting python", "[worker][depth][models]") {
    DepthEstimatorConfig cfg;
    cfg.modelsDir = (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string();
    cfg.extra["python"] = "python-that-does-not-exist";  // never started
    auto initError = [&](const std::string& backend) {
        WorkerDepthEstimator w(backend);
        try {
            w.Init(cfg);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    cfg.modelId = "metric-vda-huge";
    CHECK(initError("vda").find("unknown VDA model 'metric-vda-huge'") != std::string::npos);
    cfg.modelId = "da3metric-large";
    CHECK(initError("vda").find("not a VDA model") != std::string::npos);
    cfg.modelId = "metric-vda-small";
    CHECK(initError("da3").find("not a DA3 model") != std::string::npos);
    cfg.modelId = "";
    cfg.window = 10;
    CHECK(initError("vda").find("multiple of 4") != std::string::npos);
}

// Regression (TASK-0065): without a models_dir of the stage the workers got <project root>/models, while the registry, the
// fingerprints and the TensorRT paths read DLSSVID_MODELS_DIR - a clean CI checkout downloaded RealBasicVSR again on every run
// (CI of PR #32: the download hung past the 10 min of init).
TEST_CASE("Workers take the models folder of the registry: DLSSVID_MODELS_DIR, not <project root>/models", "[worker][models][regression]") {
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "worker_models_dir" / "models";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "registry.json") << R"({"models": []})";
    const ModelsDirEnv env(dir);
    CHECK(ModelRegistry::DefaultModelsDir() == dir);

    UpscalerConfig up;
    CHECK(WorkerUpscaler::FindModelsDir(up) == dir);
    up.extra["models_dir"] = "";  // the schema's default: none
    CHECK(WorkerUpscaler::FindModelsDir(up) == dir);
    up.extra["models_dir"] = "D:/m";  // the stage's folder wins
    CHECK(WorkerUpscaler::FindModelsDir(up) == std::filesystem::path("D:/m"));

    DepthEstimatorConfig depth;
    CHECK(WorkerDepthEstimator::FindModelsDir(depth) == dir);
    depth.modelsDir = "D:/m";
    CHECK(WorkerDepthEstimator::FindModelsDir(depth) == std::filesystem::path("D:/m"));

    // and Init uses it: VDA resolves its model in that (empty) registry before python starts - <root>/models knows metric-vda-small
    depth.modelsDir.clear();
    depth.modelId = "metric-vda-small";
    depth.extra["python"] = "python-that-does-not-exist";  // never started
    WorkerDepthEstimator w("vda");
    std::string error;
    try {
        w.Init(depth);
    } catch (const std::exception& e) {
        error = e.what();
    }
    CHECK(error.find("unknown VDA model 'metric-vda-small'") != std::string::npos);
}
