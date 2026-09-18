#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "ml/ModelRegistry.h"

using namespace dlssvid;

namespace {
std::filesystem::path Tmp(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "registry";
    std::filesystem::create_directories(d);
    return d / name;
}
}  // namespace

TEST_CASE("Project registry.json loads and has the depth models", "[registry]") {
    const auto path = std::filesystem::path(DLSSVID_SOURCE_DIR) / "models" / "registry.json";
    const ModelRegistry reg = ModelRegistry::Load(path);
    CHECK(reg.Find("da3metric-large") != nullptr);
    CHECK(reg.Get("da3metric-large").params["metric"] == true);
    CHECK(reg.Get("metric-vda-small").params["window"] == 32);
    CHECK(reg.Get("da3metric-large").license == "Apache-2.0");
    CHECK(reg.Get("icdepth").params["available"] == false);
    CHECK_THROWS(reg.Get("nope"));
    CHECK(reg.CacheDir() == path.parent_path() / "cache");
    CHECK(reg.LocalPath(reg.Get("da3metric-large")).filename() == "da3metric-large.onnx");
    // no URL -> Fetch of a missing file explains what to do
    if (!reg.IsCached(reg.Get("da3metric-large"))) CHECK_THROWS(reg.Fetch(reg.Get("da3metric-large")));
}

TEST_CASE("Registry hash verification and file naming", "[registry]") {
    const auto dir = Tmp("r1");
    std::filesystem::create_directories(dir / "cache");
    std::ofstream(dir / "cache" / "a.bin") << "hello";
    // sha256("hello")
    const char* good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    std::ofstream(dir / "registry.json") << R"({"models":[{"id":"a","stage":"depth","format":"bin","file":"a.bin","sha256":")" << good << R"("},
        {"id":"b","stage":"depth","format":"bin","url":"https://example.invalid/x/model.onnx?dl=1"},
        {"id":"c","stage":"depth","format":"bin","file":"a.bin","sha256":"00"}]})";
    const ModelRegistry reg = ModelRegistry::Load(dir / "registry.json");
    CHECK(reg.Fetch(reg.Get("a")) == dir / "cache" / "a.bin");
    CHECK(reg.Get("b").fileName == "model.onnx");
    CHECK_THROWS(reg.Fetch(reg.Get("c")));  // hash mismatch
    CHECK_THROWS(ModelRegistry::Load(dir / "missing.json"));
}
