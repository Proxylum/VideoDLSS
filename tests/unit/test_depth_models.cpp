// Depth models of the registry (TASK-0039): the backend's model and window, a refusal with the family's models for a
// bad id, the VRAM estimate from the registry's measurements and the advice of the plan.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>

#include "ml/ModelRegistry.h"
#include "stages/depth/DepthModels.h"

using namespace dlssvid;
using Catch::Matchers::ContainsSubstring;

namespace {

// the registry of this checkout, not DLSSVID_MODELS_DIR (on the CI runner that is another tree)
std::string SourceModels() { return (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string(); }

std::string What(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

// a registry with one VDA model whose VRAM was measured at 924x518: 16 frames 2000 MiB, 32 frames 3500 MiB
std::filesystem::path MeasuredRegistry(const char* name) {
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "depth_models" / name;
    std::filesystem::create_directories(dir);
    nlohmann::json points = nlohmann::json::array();
    points.push_back({{"window", 16}, {"input", {924, 518}}, {"mib", 2000}});
    points.push_back({{"window", 32}, {"input", {924, 518}}, {"mib", 3500}});
    points.push_back({{"window", 32}, {"input", {910, 378}}, {"mib", 2600}});
    const nlohmann::json reg = {
        {"models",
         {{{"id", "vda-measured"},
           {"stage", "depth"},
           {"license", "CC-BY-NC-4.0"},
           {"format", "onnx"},
           {"params", {{"family", "vda"}, {"encoder", "vitl"}, {"metric", true}, {"window", 32}, {"overlap", 8}, {"max_window", 32}, {"vram", {{"points", points}}}}}},
          {{"id", "vda-plain"}, {"stage", "depth"}, {"license", "Apache-2.0"}, {"format", "onnx"}, {"params", {{"family", "vda"}, {"window", 32}, {"overlap", 8}}}},
          {{"id", "metric-vda-small"}, {"stage", "depth"}, {"license", "Apache-2.0"}, {"format", "onnx"}, {"params", {{"family", "vda"}, {"window", 32}, {"overlap", 8}}}}}}};
    std::ofstream(dir / "registry.json") << reg.dump(2);
    return dir;
}

}  // namespace

TEST_CASE("ResolveDepthModel picks the backend's model and names the family's models on a bad id", "[unit][depth][models]") {
    const std::string models = SourceModels();
    const DepthModelChoice small = ResolveDepthModel(models, "vda", "");
    CHECK(small.modelId == "metric-vda-small");
    CHECK(small.family == "vda");
    CHECK(small.window == 32);
    CHECK(small.overlap == 8);
    CHECK_FALSE(small.nonCommercial);
    CHECK(small.metric);
    const DepthModelChoice large = ResolveDepthModel(models, "vda", "metric-vda-large");
    CHECK(large.nonCommercial);
    CHECK(large.license == "CC-BY-NC-4.0");
    CHECK(ResolveDepthModel(models, "worker:vda", "metric-vda-base").modelId == "metric-vda-base");
    CHECK(ResolveDepthModel(models, "worker:vda", "").modelId == "metric-vda-small");
    CHECK(ResolveDepthModel(models, "da3", "").modelId == "da3metric-large");
    CHECK(ResolveDepthModel(models, "worker", "").family == "da3");

    // a foreign family: refused with the VDA models (before: Python KeyError 'hf_file' after the export started)
    const std::string foreign = What([&] { ResolveDepthModel(models, "vda", "da3metric-large"); });
    CHECK_THAT(foreign, ContainsSubstring("not a VDA model"));
    CHECK_THAT(foreign, ContainsSubstring("metric-vda-base"));
    CHECK_THAT(What([&] { ResolveDepthModel(models, "da3", "metric-vda-small"); }), ContainsSubstring("not a DA3 model"));
    const std::string unknown = What([&] { ResolveDepthModel(models, "vda", "metric-vda-huge"); });
    CHECK_THAT(unknown, ContainsSubstring("unknown VDA model 'metric-vda-huge'"));
    CHECK_THAT(unknown, ContainsSubstring("metric-vda-large (CC-BY-NC-4.0)"));
    CHECK_THAT(unknown, !ContainsSubstring("sea-raft"));  // the family's models, not the whole registry
    CHECK_THAT(unknown, !ContainsSubstring("realesr"));

    // windows: multiples of 4 above 4, at most max_window; the overlap scales
    CHECK(ResolveDepthModel(models, "vda", "", 16).overlap == 4);
    CHECK(ResolveDepthModel(models, "vda", "", 8).overlap == 2);
    CHECK(ResolveDepthModel(models, "vda", "", 4).window == 4);
    CHECK(ResolveDepthModel(models, "vda", "", 2).window == 2);
    for (const int bad : {10, 30, 40}) CHECK_THAT(What([&] { ResolveDepthModel(models, "vda", "", bad); }), ContainsSubstring("multiple of 4"));
    // DA3 has no windows: a window left in the project after a switch from VDA is ignored with a warning, not an error
    const DepthModelChoice da3 = ResolveDepthModel(models, "da3", "", 16);
    CHECK(da3.window == 1);
    REQUIRE(da3.warnings.size() == 1);
    CHECK_THAT(da3.warnings[0], ContainsSubstring("VDA models only"));
    // backends without a family pass any id
    CHECK(ResolveDepthModel(models, "stub", "whatever").family.empty());
    CHECK(ResolveDepthModel(models, "worker:stub", "x").modelId == "x");
    CHECK(EffectiveDepthInputSize(small, 518) == 518);
    CHECK(EffectiveDepthInputSize(small, 700) == 700);
}

TEST_CASE("VRAM estimate from the registry's measurements and the advice of the plan", "[unit][depth][models]") {
    const auto dir = MeasuredRegistry("advice").string();
    DepthModelChoice c = ResolveDepthModel(dir, "vda", "vda-measured");
    CHECK(*EstimateDepthVramMiB(c, false, 924, 518) == 3500 * 1.1);
    CHECK(*EstimateDepthVramMiB(c, false, 910, 378) == 2600 * 1.1);  // the smallest measured input with as many pixels
    CHECK(*EstimateDepthVramMiB(c, false, 924, 518, false) == 3500 * 2.2);  // fp32
    CHECK(*EstimateDepthVramMiB(c, false, 1848, 1036) == 3500 * 4 * 1.1);   // larger than measured: by pixels
    c = ResolveDepthModel(dir, "vda", "vda-measured", 16);
    CHECK(*EstimateDepthVramMiB(c, false, 924, 518) == 2000 * 1.1);
    c = ResolveDepthModel(dir, "vda", "vda-measured", 8);
    CHECK(*EstimateDepthVramMiB(c, false, 924, 518) == 2000 * 1.1);  // the smallest measured window >= 8: conservative
    CHECK(!EstimateDepthVramMiB(c, true, 924, 518));                  // nothing measured for the PyTorch path
    CHECK(!EstimateDepthVramMiB(ResolveDepthModel(dir, "vda", "vda-plain"), false, 924, 518));

    const uint64_t gib = uint64_t{1} << 30;
    const DepthModelChoice w32 = ResolveDepthModel(dir, "vda", "vda-measured");
    const auto need = EstimateDepthVramMiB(w32, false, 924, 518);
    DepthAdvice a = AdviseDepth(w32, need, 2 * gib, 924, 518);
    CHECK_THAT(a.error, ContainsSubstring("видеопамяти"));
    CHECK_THAT(a.error, ContainsSubstring("--window 16"));
    REQUIRE(!a.warnings.empty());
    CHECK_THAT(a.warnings[0], ContainsSubstring("CC-BY-NC-4.0"));  // the licence, whatever the memory
    a = AdviseDepth(w32, need, 4 * gib, 924, 518);
    CHECK(a.error.empty());
    CHECK_THAT(a.warnings.back(), ContainsSubstring("впритык"));
    a = AdviseDepth(w32, need, 16 * gib, 924, 518);
    CHECK(a.error.empty());
    CHECK_THAT(a.info, ContainsSubstring("≈ 3.8 ГБ видеопамяти из 16.0"));
    a = AdviseDepth(w32, need, 0, 924, 518);  // WARP / unknown: no error
    CHECK(a.error.empty());
    CHECK_THAT(a.info, ContainsSubstring("≈ 3.8 ГБ"));
    const DepthModelChoice plain = ResolveDepthModel(dir, "vda", "vda-plain");
    a = AdviseDepth(plain, std::nullopt, 16 * gib, 924, 518);
    CHECK(a.warnings.empty());
    CHECK_THAT(a.info, ContainsSubstring("не замерена"));

    CHECK_THAT(CheckDepthVramFree(w32, 3850, 2 * gib, 16 * gib), ContainsSubstring("свободно 2.0 из 16.0"));
    CHECK(CheckDepthVramFree(w32, 3850, 8 * gib, 16 * gib).empty());
    CHECK(CheckDepthVramFree(w32, 3850, 0, 0).empty());  // unknown: not refused
    CHECK(DepthWorkspaceBytes(0) == 2 * gib);  // unknown: the TensorRT default of the project
    CHECK(DepthWorkspaceBytes(14 * gib) == 7 * gib);
    CHECK(DepthWorkspaceBytes(30 * gib) == 8 * gib);
    CHECK(SmallerDepthWindow(32) == 16);
    CHECK(SmallerDepthWindow(12) == 4);
    CHECK(SmallerDepthWindow(8) == 4);
}
