// Project files (*.dlssvid.json): creation defaults, relative paths, viewport state round trip.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <thread>
#include <fstream>

#include "passes/PassSequence.h"
#include "stages/ParamSchema.h"
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
    p.viewport.time = 17.0;
    p.viewport.gridSources = {"source", "depth_raw", "mv_dlss", "color_sr"};
    p.passVersionsKeep = 3;
    p.Save(d / "sub" / "proj.dlssvid.json");
    REQUIRE(std::filesystem::exists(d / "sub" / "proj.dlssvid.json"));
    CHECK(p.file == std::filesystem::absolute(d / "sub" / "proj.dlssvid.json"));

    std::ifstream in(d / "sub" / "proj.dlssvid.json");
    nlohmann::json j;
    in >> j;
    CHECK(j["schema_version"] == kProjectSchemaVersion);
    CHECK(j["source"] == "../clip.mp4");
    CHECK(j["passes"] == "../passes");
    CHECK(j["result"] == "../out/result.mp4");
    CHECK(j["stages"][0]["enabled"] == false);
    CHECK(j["viewport"]["time"] == 17.0);
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

// TASK-0043: nr.exposure is in stops; a version-1 project carried a linear multiplier (an explicit 0 gave black frames).
TEST_CASE("Project v1 -> v2: nr.exposure x becomes log2(x), x <= 0 is dropped, skin_structure below 0 becomes -1", "[viewport][project][regression]") {
    const auto d = Dir("migrate_v1");
    auto nrOf = [&](const nlohmann::json& params) {
        const nlohmann::json j = {{"schema_version", 1}, {"stages", nlohmann::json::array({{{"name", "nr"}, {"params", params}}, {{"name", "fg"}, {"params", {{"exposure", 0}}}}})}};
        const Project p = Project::FromJson(j, d);
        REQUIRE(p.stages.size() == 2);
        CHECK(p.stages[1].params == nlohmann::json{{"exposure", 0}});  // only nr is migrated
        return p.stages[0].params;
    };
    CHECK(nrOf({{"exposure", 2.0}, {"intensity", 1.2}}) == nlohmann::json{{"exposure", 1.0}, {"intensity", 1.2}});
    CHECK(nrOf({{"exposure", 0.5}}) == nlohmann::json{{"exposure", -1.0}});
    CHECK(nrOf({{"exposure", 3.0}})["exposure"] == 1.58);  // log2(3) to 0.01
    CHECK(nrOf({{"exposure", 1.0}}) == nlohmann::json::object());   // the identity: the key goes
    CHECK(nrOf({{"exposure", 0.0}}) == nlohmann::json::object());   // black frames before: the identity now
    CHECK(nrOf({{"exposure", -2.0}}) == nlohmann::json::object());
    CHECK(nrOf({{"exposure", 1e6}}) == nlohmann::json{{"exposure", 8.0}});  // clamped to the range
    CHECK(nrOf({{"skin_structure", -0.5}}) == nlohmann::json{{"skin_structure", -1.0}});
    CHECK(nrOf({{"skin_structure", -1.0}}) == nlohmann::json{{"skin_structure", -1.0}});
    CHECK(nrOf({{"skin_structure", 0.4}}) == nlohmann::json{{"skin_structure", 0.4}});
    CHECK(nrOf({{"skin-structure", -0.5}}) == nlohmann::json{{"skin-structure", -1.0}});  // the alias the stage reads too (TASK-0046)
    CHECK(ValidateStageParams("nr", nrOf({{"skin-structure", -0.5}})).empty());
    // a version-2 file is taken as it is; a newer one is refused
    const nlohmann::json v2 = {{"schema_version", 2}, {"stages", nlohmann::json::array({{{"name", "nr"}, {"params", {{"exposure", 0.0}}}}})}};
    CHECK(Project::FromJson(v2, d).stages[0].params == nlohmann::json{{"exposure", 0.0}});
    CHECK_THROWS(Project::FromJson(nlohmann::json{{"schema_version", kProjectSchemaVersion + 1}}, d));
    CHECK_THROWS(Project::FromJson(nlohmann::json{{"schema_version", 0}}, d));
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

TEST_CASE("Project caches the source hash by size and time and keeps the encoder settings", "[viewport][project]") {
    const auto d = Dir("hash");
    std::ofstream(d / "clip.bin", std::ios::binary) << "video bytes";
    Project p = Project::Create(d / "clip.bin", d / "passes");
    CHECK(p.sourceHash.empty());
    const std::string h1 = p.SourceHash();
    CHECK(h1.rfind("sha256:", 0) == 0);
    CHECK(p.sourceSize == 11);
    CHECK(!p.sourceMtime.empty());
    CHECK(p.SourceHash() == h1);  // from the cache
    p.codec = "hevc_nvenc";
    p.codecOptions["b"] = "50M";
    p.Save(d / "p.dlssvid.json");
    Project q = Project::Load(d / "p.dlssvid.json");
    CHECK(q.sourceHash == h1);
    CHECK(q.sourceSize == 11);
    CHECK(q.sourceMtime == p.sourceMtime);
    CHECK(q.codec == "hevc_nvenc");
    CHECK(q.codecOptions.at("b") == "50M");
    // the file changed (another size): recomputed
    std::ofstream(d / "clip.bin", std::ios::binary) << "video bytes v2";
    const std::string h2 = q.SourceHash();
    CHECK(h2 != h1);
    CHECK(q.sourceSize == 14);
    Project none;
    CHECK(none.SourceHash().empty());
    Project missing = Project::Create(d / "nope.bin", d / "passes");
    CHECK(missing.SourceHash().empty());
    const Project defaults = Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1})"), d);
    CHECK(defaults.codec == "h264_nvenc");
    CHECK(defaults.codecOptions.empty());
    CHECK(defaults.sourceHash.empty());
}

// TASK-0034: the result's bit depth and chroma live in the project next to the codec.
TEST_CASE("Project keeps the output bit depth and chroma; older files mean auto", "[viewport][project]") {
    const auto d = Dir("format");
    Project p = Project::Create(d / "clip.mp4", d / "passes");
    CHECK(p.outputFormat == OutputFormatRequest{});  // auto / auto
    p.codec = "prores_ks";
    p.outputFormat.bitDepth = 10;
    p.outputFormat.chroma = "444";
    p.Save(d / "p.dlssvid.json");
    const Project q = Project::Load(d / "p.dlssvid.json");
    CHECK(q.codec == "prores_ks");
    CHECK(q.outputFormat.bitDepth == 10);
    CHECK(q.outputFormat.chroma == "444");
    const nlohmann::json j = p.ToJson(d);
    CHECK(j["encode"]["bit_depth"] == "10");
    CHECK(j["encode"]["chroma"] == "444");
    // a project written before the fields: auto; a number for the depth is accepted too
    const Project old = Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1, "encode": {"codec": "hevc_nvenc", "options": {}}})"), d);
    CHECK(old.outputFormat.bitDepth == 0);
    CHECK(old.outputFormat.chroma == "auto");
    const Project num = Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1, "encode": {"codec": "ffv1", "bit_depth": 12, "chroma": "422"}})"), d);
    CHECK(num.outputFormat.bitDepth == 12);
    CHECK(num.outputFormat.chroma == "422");
    CHECK_THROWS(Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1, "encode": {"codec": "ffv1", "bit_depth": "9"}})"), d));
    CHECK_THROWS(Project::FromJson(nlohmann::json::parse(R"({"schema_version": 1, "encode": {"codec": "ffv1", "chroma": "411"}})"), d));
}
