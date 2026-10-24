// Pass fingerprints (stage 9, MR A): canonical JSON, a fingerprint that reacts to every component and only to them,
// the tool identity of a stage from its parameters, the stage pass tables and the timestamps.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "pipeline/PassFingerprint.h"
#include "util/Sha256.h"

using namespace dlssvid;

TEST_CASE("CanonicalJson: key order and number spelling do not matter, types do", "[unit][fingerprint]") {
    const nlohmann::json a = nlohmann::json::parse(R"({"scale": 2.0, "backend": "dlss", "nested": {"z": 1, "a": [1.0, 2.5]}})");
    const nlohmann::json b = nlohmann::json::parse(R"({"nested": {"a": [1, 2.5], "z": 1.0}, "backend": "dlss", "scale": 2})");
    CHECK(CanonicalJson(a) == CanonicalJson(b));
    CHECK(CanonicalJson(a) == R"({"backend":"dlss","nested":{"a":[1,2.5],"z":1},"scale":2})");
    CHECK(CanonicalJson(nlohmann::json::parse(R"({"x": 1.5})")) == R"({"x":1.5})");
    CHECK(CanonicalJson(nlohmann::json::parse(R"({"x": "2"})")) != CanonicalJson(nlohmann::json::parse(R"({"x": 2})")));
    CHECK(CanonicalizeJson(nlohmann::json(2u)) == nlohmann::json(2));
    CHECK(CanonicalizeJson(nlohmann::json(-3.0)) == nlohmann::json(-3));
    CHECK(CanonicalJson(nlohmann::json()) == "null");
    CHECK(CanonicalJson(nlohmann::json::object()) == "{}");
}

TEST_CASE("ComputeFingerprint changes with every component and only with them", "[unit][fingerprint]") {
    ToolInfo tool;
    tool.app = "0.1.0";
    tool.backend = "stub";
    const std::map<std::string, std::string> inputs{{"color_sr", "sha256:aaaa"}, {"depth_dlss", "sha256:bbbb"}};
    const nlohmann::json params = {{"backend", "stub"}, {"intensity", 1.0}};
    const PassFingerprint base = ComputeFingerprint("nr", "sha256:src", params, inputs, tool);
    CHECK(base.value.rfind("sha256:", 0) == 0);
    CHECK(base.value.size() == 7 + 64);
    CHECK(base.Short().size() == 8);
    CHECK(base.stage == "nr");
    CHECK(base.sourceHash == "sha256:src");
    CHECK(base.params == nlohmann::json({{"backend", "stub"}, {"intensity", 1}}));
    CHECK(base.inputs == inputs);
    CHECK(base.tool == tool);
    // the same run spelled differently is the same fingerprint
    CHECK(ComputeFingerprint("nr", "sha256:src", {{"intensity", 1}, {"backend", "stub"}}, inputs, tool).value == base.value);
    // each component
    CHECK(ComputeFingerprint("nr", "sha256:other", params, inputs, tool).value != base.value);
    CHECK(ComputeFingerprint("fg", "sha256:src", params, inputs, tool).value != base.value);
    CHECK(ComputeFingerprint("nr", "sha256:src", {{"backend", "stub"}, {"intensity", 1.4}}, inputs, tool).value != base.value);
    CHECK(ComputeFingerprint("nr", "sha256:src", {{"backend", "stub"}}, inputs, tool).value != base.value);
    auto in2 = inputs;
    in2["color_sr"] = "sha256:cccc";
    CHECK(ComputeFingerprint("nr", "sha256:src", params, in2, tool).value != base.value);
    auto in3 = inputs;
    in3.erase("depth_dlss");
    CHECK(ComputeFingerprint("nr", "sha256:src", params, in3, tool).value != base.value);
    ToolInfo t2 = tool;
    t2.dll = "sha256:dddd";
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, t2).value != base.value);
    ToolInfo t3 = tool;
    t3.app = "0.2.0";
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, t3).value != base.value);
    ToolInfo t4 = tool;
    t4.model = "da3mono-large";
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, t4).value != base.value);
    CHECK(ShortFingerprint("") == "legacy");
    CHECK(ShortFingerprint("sha256:0123456789abcdef") == "01234567");
    CHECK(ShortFingerprint("0123456789abcdef") == "01234567");
    // TASK-0034: the decoded source layout is part of the fingerprint, except the legacy 8-bit 4:2:0 one (and an empty
    // string), so every pass of an 8-bit source keeps the fingerprint it had before the decoder kept bit depths
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, tool, kLegacySourceFormat).value == base.value);
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, tool, "").value == base.value);
    const PassFingerprint ten = ComputeFingerprint("nr", "sha256:src", params, inputs, tool, "yuv444p10");
    CHECK(ten.value != base.value);
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, tool, "yuv420p10").value != ten.value);
    CHECK(ComputeFingerprint("nr", "sha256:src", params, inputs, tool, "yuv444p10").value == ten.value);
}

TEST_CASE("ToolInfo JSON round trip and ToolForStage from parameters", "[unit][fingerprint]") {
    ToolInfo t;
    t.app = "1";
    t.backend = "ngx";
    t.model = "m";
    t.modelHash = "h";
    t.dllFile = "nvngx_dlssnr.dll";
    t.dll = "sha256:x";
    CHECK(ToolInfo::FromJson(t.ToJson()) == t);
    CHECK(ToolInfo::FromJson(nlohmann::json()) == ToolInfo{});
    CHECK(ToolInfo::FromJson(nlohmann::json::object()) == ToolInfo{});

    const ToolInfo stub = ToolForStage("nr", {{"backend", "stub"}});
    CHECK(!stub.app.empty());
    CHECK(stub.backend == "stub");
    CHECK(stub.model.empty());
    CHECK(stub.dllFile.empty());
    CHECK(stub.dll.empty());
    const ToolInfo depth = ToolForStage("depth", nlohmann::json::object());
    CHECK(depth.backend == "da3");
    CHECK(depth.model == "da3metric-large");
    CHECK(ToolForStage("depth", {{"backend", "vda"}}).model == "metric-vda-small");
    CHECK(ToolForStage("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}).model == "da3mono-large");
    CHECK(ToolForStage("depth", {{"backend", "stub"}}).model.empty());
    // TASK-0039: the VDA models; this checkout's registry, not DLSSVID_MODELS_DIR (another tree on the CI runner)
    const std::string models = (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string();
    const ToolInfo large = ToolForStage("depth", {{"backend", "vda"}, {"model", "metric-vda-large"}, {"models_dir", models}});
    CHECK(large.model == "metric-vda-large");
    CHECK(large.modelHash == "24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620");
    CHECK(ToolForStage("depth", {{"backend", "worker:vda"}}).model == "metric-vda-small");
    CHECK(ToolForStage("depth", {{"backend", "vda"}, {"models_dir", models}}).modelHash.empty());  // Small not pinned: its passes stay
    CHECK(ToolForStage("depth", {{"backend", "worker"}}).model.empty());  // DA3 worker as before
    CHECK(ToolForStage("flow", {{"backend", "searaft"}}).model == "sea-raft-spring-m");
    CHECK(ToolForStage("flow", nlohmann::json::object()).backend == "ofa");
    CHECK(ToolForStage("flow", nlohmann::json::object()).model.empty());
    const ToolInfo ngx = ToolForStage("nr", nlohmann::json::object());
    CHECK(ngx.backend == "ngx");
    CHECK(ngx.dllFile == "nvngx_dlssnr.dll");
    CHECK(ToolForStage("fg", nlohmann::json::object()).dllFile == "nvngx_dlssg.dll");
    // a stage revision (TASK-0038): depth aligned over the centred window -> passes made before are recomputed once
    const std::string depthRev = std::string(kPassToolVersion) + "+depth.2";
    CHECK(ToolForStage("depth", nlohmann::json::object()).app == depthRev);
    CHECK(ToolForStage("depth", {{"stabilize", "scale"}}).app == depthRev);
    CHECK(ToolForStage("depth", {{"stabilize", "scale_shift"}}).app == depthRev);
    CHECK(ToolForStage("depth", {{"backend", "stub"}}).app == depthRev);
    CHECK(ToolForStage("depth", {{"stabilize", "none"}}).app == kPassToolVersion);  // unaligned depth is what it was
    for (const char* stage : {"flow", "upscale", "nr", "fg"}) CHECK(ToolForStage(stage, nlohmann::json::object()).app == kPassToolVersion);
    // TASK-0042: colour passes written as PNG16 before the fix are black - a revision of their own; EXR passes stay
    for (const char* stage : {"upscale", "nr", "fg"}) {
        CHECK(ToolForStage(stage, {{"format", "png"}}).app == std::string(kPassToolVersion) + "+png.2");
        CHECK(ToolForStage(stage, {{"format", "exr"}}).app == kPassToolVersion);
    }
    CHECK_THROWS(ToolForStage("depth", {{"stabilize", 5}}));
    CHECK(ToolForStage("upscale", nlohmann::json::object()).dllFile == "nvngx_dlss.dll");
    CHECK(ToolForStage("upscale", {{"backend", "nis"}}).dllFile.empty());
    CHECK(ToolForStage("fg", {{"backend", "rife"}, {"model", "rife-v4.6"}}).model == "rife-v4.6");
    CHECK(ToolForStage("fg", {{"backend", "blend"}}).dll.empty());
    CHECK_THROWS(ToolForStage("nr", {{"backend", 5}}));
    // the same parameters give the same identity (the DLL hash, when present, is cached and stable)
    CHECK(ToolForStage("nr", nlohmann::json::object()) == ngx);
}

TEST_CASE("Stage pass tables and timestamps", "[unit][fingerprint]") {
    CHECK(StageOutputPasses("depth", nlohmann::json::object()) == std::vector<std::string>{"depth_raw", "depth_dlss"});
    CHECK(StageOutputPasses("depth", {{"no_dlss", true}}) == std::vector<std::string>{"depth_raw"});
    CHECK(StageOutputPasses("flow", nlohmann::json::object()) == std::vector<std::string>{"mv_raw", "mv_dlss"});
    CHECK(StageOutputPasses("flow", {{"no_dlss", true}}) == std::vector<std::string>{"mv_raw"});
    CHECK(StageOutputPasses("upscale", nlohmann::json::object()) == std::vector<std::string>{"color_sr"});
    CHECK(StageOutputPasses("nr", nlohmann::json::object()) == std::vector<std::string>{"color_nr"});
    CHECK(StageOutputPasses("fg", nlohmann::json::object()) == std::vector<std::string>{"color_fg"});
    CHECK_THROWS(StageOutputPasses("bogus", nlohmann::json::object()));
    CHECK_THROWS(StageOutputPasses("depth", {{"no_dlss", "yes"}}));
    CHECK(StageFamilyPasses("depth") == std::vector<std::string>{"depth_raw", "depth_dlss"});
    CHECK(StageFamilyPasses("nr") == std::vector<std::string>{"color_nr"});
    CHECK(StageFamilyPasses("mask").empty());
    CHECK(StageForPass("color_nr") == "nr");
    CHECK(StageForPass("depth_raw") == "depth");
    CHECK(StageForPass("mask_face").empty());
    CHECK(StageInputCandidates("depth").empty());
    CHECK(StageInputCandidates("flow") == std::vector<std::vector<std::string>>{{"depth_raw"}});
    CHECK(StageInputCandidates("fg").front() == std::vector<std::string>{"color_nr", "color_sr"});
    CHECK(StageInputCandidates("nr").size() == 7);

    CHECK(CompactTimestamp("2026-09-22T14:09:31Z") == "20260922-140931");
    CHECK(CompactTimestamp("yesterday").empty());
    CHECK(CompactTimestamp("").empty());
    const std::string now = NowIso8601();
    CHECK(now.size() == 20);
    CHECK(now.back() == 'Z');
    CHECK(CompactTimestamp(now).size() == 15);
    CHECK(FileTimeIso8601("Z:/no/such/file").empty());
}

TEST_CASE("Sha256FileCached matches Sha256File and follows the file", "[unit][fingerprint]") {
    const std::filesystem::path dir = std::filesystem::path(DLSSVID_TEST_TMP) / "fingerprint";
    std::filesystem::create_directories(dir);
    const auto file = dir / "blob.bin";
    std::ofstream(file, std::ios::binary) << "first content";
    const std::string h1 = Sha256FileCached(file);
    CHECK(h1 == Sha256File(file));
    CHECK(Sha256FileCached(file) == h1);
    CHECK(FileTimeIso8601(file).size() == 20);
    std::ofstream(file, std::ios::binary) << "second, longer content";
    const std::string h2 = Sha256FileCached(file);
    CHECK(h2 == Sha256File(file));
    CHECK(h2 != h1);
    CHECK_THROWS(Sha256FileCached(dir / "missing.bin"));
}
