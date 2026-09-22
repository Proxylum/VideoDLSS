// Project files (*.dlssvid.json): creation defaults, relative paths, viewport state round trip.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "passes/PassSequence.h"
#include "viewport/Project.h"

using namespace dlssvid;

namespace {
std::filesystem::path Dir(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "project" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}
}  // namespace

TEST_CASE("Project::Create fills defaults next to the video", "[viewport][project]") {
    const auto d = Dir("create");
    const Project p = Project::Create(d / "clip.mp4", {});
    CHECK(p.sourceVideo == d / "clip.mp4");
    CHECK(p.passesRoot == d / "clip_passes");
    CHECK(p.resultVideo.empty());
    REQUIRE(p.stages.size() == 5);
    CHECK(p.stages[0].name == "depth");
    CHECK(p.stages[0].params["backend"] == "da3");
    CHECK(p.stages[1].name == "flow");
    CHECK(p.stages[2].name == "upscale");
    CHECK(p.stages[2].params["backend"] == "dlss");
    CHECK(p.stages[2].params["scale"] == 2);
    CHECK(p.stages[3].name == "nr");
    CHECK(p.stages[3].params["backend"] == "ngx");
    CHECK(p.stages[3].params["passes"] == 1);
    CHECK(p.stages[4].name == "fg");
    CHECK(p.stages[4].params["backend"] == "dlssg");
    CHECK(p.stages[4].params["multiplier"] == 2);
    CHECK(p.viewport == ViewportState{});
    const Project q = Project::Create(d / "clip.mp4", d / "elsewhere");
    CHECK(q.passesRoot == d / "elsewhere");
}

TEST_CASE("Project saves relative paths and loads them back absolute", "[viewport][project]") {
    const auto d = Dir("roundtrip");
    Project p = Project::Create(d / "clip.mp4", d / "passes");
    p.resultVideo = d / "out" / "result.mp4";
    p.stages[0].enabled = false;
    p.stages[0].params["frames"] = 30;
    p.viewport.mode = ViewMode::Overlay;
    p.viewport.SetSingleSource("depth_raw");
    p.viewport.AddOverlayLayer("mv_raw")->opacity = 0.4f;
    p.viewport.wipe.enabled = true;
    p.viewport.wipe.position = 0.25f;
    p.viewport.view = ViewTransform{2.f, 100.f, 50.f};
    p.viewport.frame = 17;
    p.viewport.gridSources = {"source", "depth_raw", "mv_dlss", "color_sr"};
    p.passVersionsKeep = 3;
    p.Save(d / "sub" / "proj.dlssvid.json");
    REQUIRE(std::filesystem::exists(d / "sub" / "proj.dlssvid.json"));
    CHECK(p.file == std::filesystem::absolute(d / "sub" / "proj.dlssvid.json"));

    std::ifstream in(d / "sub" / "proj.dlssvid.json");
    nlohmann::json j;
    in >> j;
    CHECK(j["schema_version"] == 1);
    CHECK(j["source"] == "../clip.mp4");
    CHECK(j["passes"] == "../passes");
    CHECK(j["result"] == "../out/result.mp4");
    CHECK(j["stages"][0]["enabled"] == false);
    CHECK(j["viewport"]["frame"] == 17);
    CHECK(j["pass_versions_keep"] == 3);

    const Project q = Project::Load(d / "sub" / "proj.dlssvid.json");
    CHECK(q.sourceVideo == std::filesystem::absolute(d / "clip.mp4").lexically_normal());
    CHECK(q.passesRoot == std::filesystem::absolute(d / "passes").lexically_normal());
    CHECK(q.resultVideo == std::filesystem::absolute(d / "out" / "result.mp4").lexically_normal());
    CHECK(q.stages == p.stages);
    CHECK(q.viewport == p.viewport);
    CHECK(q.passVersionsKeep == 3);
    CHECK(Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1})"), d).passVersionsKeep == 2);  // default when absent

    // unsupported schema and a broken file are errors
    std::ofstream(d / "bad.json") << "{\"schema_version\": 99}";
    CHECK_THROWS(Project::Load(d / "bad.json"));
    std::ofstream(d / "broken.json") << "{";
    CHECK_THROWS(Project::Load(d / "broken.json"));
    CHECK_THROWS(Project::Load(d / "missing.json"));
}

TEST_CASE("Project::Sources lists the passes found under the root", "[viewport][project]") {
    const auto d = Dir("sources");
    {
        PassWriter w(d / "passes" / "depth_raw", Manifest::ForPass(PassKind::DepthRaw, 4, 2, FileFormat::Exr));
        w.WriteFrame(0, MakePassImage(PassKind::DepthRaw, 4, 2));
        w.Finish();
    }
    const Project p = Project::Create(d / "missing.mp4", d / "passes");
    const auto sources = p.Sources();  // the video does not exist: only the pass
    REQUIRE(sources.size() == 1);
    CHECK(sources[0].name == "depth_raw");
    CHECK(!sources[0].isVideo);
}

TEST_CASE("Project stage entries survive JSON with unknown fields", "[viewport][project]") {
    const auto d = Dir("json");
    nlohmann::json j = {{"schema_version", 1}, {"source", "a.mp4"}, {"stages", nlohmann::json::array({{{"name", "depth"}, {"params", {{"backend", "vda"}}}, {"extra", 1}}})}};
    const Project p = Project::FromJson(j, d);
    CHECK(p.sourceVideo == (d / "a.mp4").lexically_normal());
    CHECK(p.passesRoot.empty());
    REQUIRE(p.stages.size() == 1);
    CHECK(p.stages[0].enabled);
    CHECK(p.stages[0].params["backend"] == "vda");
    CHECK(p.viewport.mode == ViewMode::Single);
}
