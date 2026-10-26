// Pipeline presets over a project (PR-9): every stage card with its switch and the «Кодирование» card at once - what they hold, how
// they apply, what they leave to the project, the difference they make.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "pipeline/PassFingerprint.h"
#include "stages/ParamSchema.h"
#include "stages/StagePreset.h"
#include "viewport/PipelinePreset.h"
#include "viewport/Project.h"

using namespace dlssvid;
using Kind = PresetIssueKind;
using nlohmann::json;

namespace {

StageEntry& Stage(Project& p, const std::string& name) {
    for (auto& s : p.stages)
        if (s.name == name) return s;
    FAIL("no stage " << name);
    return p.stages.front();
}

bool HasStage(const Project& p, const std::string& name) {
    return std::any_of(p.stages.begin(), p.stages.end(), [&](const StageEntry& s) { return s.name == name; });
}

// a tuned project: engineer, machine and unknown keys, encoder options without a card row, the settings note off
Project Tuned() {
    Project p;
    p.sourceVideo = "D:/v/clip.mp4";
    p.passesRoot = "D:/v/clip_passes";
    p.resultVideo = "D:/v/clip_result.mp4";
    p.stages = Project::DefaultStages();
    Stage(p, "scenes").params = {{"backend", "luma"}, {"threshold", 0.3}};
    Stage(p, "depth").params = {{"backend", "vda"}, {"model", "metric-vda-base"}, {"z_near", 0.5}, {"max_res", 720}, {"models_dir", "D:/m"}};
    Stage(p, "upscale").params = {{"backend", "trt"}, {"scale", 2}, {"sharpness", 0.8}, {"tile", 256}};
    Stage(p, "nr").params = {{"backend", "ngx"}, {"intensity", 0.8}, {"format", "png"}, {"extra_key", 1}};
    Stage(p, "fg").enabled = false;
    p.codec = "hevc_nvenc";
    p.codecOptions = {{"b", "50M"}, {"preset", "p7"}};
    p.outputFormat = {10, "auto"};
    p.settingsNote = false;
    p.passVersionsKeep = 5;
    return p;
}

json Canon(const std::string& stage, const json& params) { return CanonicalizeJson(EffectiveStageParams(stage, params)); }

bool LeftOutHas(const std::vector<LeftOut>& l, LeftOut::Why why, const std::string& stage, const std::string& key) {
    return std::any_of(l.begin(), l.end(), [&](const LeftOut& x) { return x.why == why && x.stage == stage && x.key == key; });
}

}  // namespace

TEST_CASE("MakePipelinePreset holds every stage with enabled, its form and tuning keys and the encode card; never paths, GPU-bound keys, other codec options or the settings note",
          "[unit][presets][regression]") {
    Project p = Tuned();
    p.stages.erase(p.stages.begin());  // a v3 file written without scenes
    const Preset pp = MakePipelinePreset(" Аниме ", p);
    CHECK(pp.name == "Аниме");
    CHECK(pp.scope == PresetScope::Pipeline);
    REQUIRE(pp.stages.size() == 6);
    CHECK_FALSE(pp.stages.at("scenes").enabled);  // as its card shows it: off, the defaults
    CHECK(pp.stages.at("scenes").params == json{{"threshold", 0.24}});
    const json& nr = pp.stages.at("nr").params;
    CHECK(nr.size() == 19);
    CHECK(nr["backend"] == "ngx");
    CHECK(nr["format"] == "png");
    CHECK(nr["intensity"] == 0.8);
    CHECK_FALSE(nr.contains("extra_key"));
    const json& up = pp.stages.at("upscale").params;
    CHECK(up["sharpness"] == 0.8);
    CHECK(up["preset"] == "default");
    CHECK(up["jitter"] == true);
    CHECK(up["model"] == "realesrgan-x2plus");
    const json& depth = pp.stages.at("depth").params;
    CHECK(depth["backend"] == "vda");
    CHECK(depth["model"] == "metric-vda-base");
    CHECK(depth["z_near"] == 0.5);
    for (const auto& [stage, entry] : pp.stages)
        for (const char* machine : {"models_dir", "python", "max_res", "input_size", "tile", "fp32"}) {
            INFO(stage << "." << machine);
            CHECK_FALSE(entry.params.contains(machine));
        }
    CHECK_FALSE(pp.stages.at("fg").enabled);
    CHECK(pp.stages.at("depth").enabled);
    REQUIRE(pp.encode);
    CHECK(*pp.encode == PresetEncode{"hevc_nvenc", {10, "auto"}, "50M"});
    const std::string text = PresetToJson(pp).dump(2);
    CHECK(text.find("p7") == std::string::npos);
    CHECK(text.find("settings_note") == std::string::npos);
    CHECK(text.find("D:/m") == std::string::npos);
    // what it leaves out - named for the save dialog; keys at their defaults and the ones it holds are not
    const auto left = PipelinePresetLeftOut(p);
    CHECK(LeftOutHas(left, LeftOut::Why::CodecOption, "encode", "preset"));
    CHECK(LeftOutHas(left, LeftOut::Why::SettingsNote, "encode", "settings_note"));
    CHECK(LeftOutHas(left, LeftOut::Why::Machine, "depth", "max_res"));
    CHECK(LeftOutHas(left, LeftOut::Why::Machine, "upscale", "tile"));
    CHECK(LeftOutHas(left, LeftOut::Why::Path, "depth", "models_dir"));
    CHECK(LeftOutHas(left, LeftOut::Why::Unknown, "nr", "extra_key"));
    CHECK_FALSE(LeftOutHas(left, LeftOut::Why::Engineer, "upscale", "sharpness"));
    CHECK(left.size() == 6);
    // a codec without a bitrate: its b is left out
    Project pro = Tuned();
    pro.codec = "prores_ks";
    CHECK(EncodeOf(pro).bitrate.empty());
    CHECK(LeftOutHas(PipelinePresetLeftOut(pro), LeftOut::Why::CodecOption, "encode", "b"));
    // a card preset of the same NR: its own class of keys
    const auto cardLeft = StagePresetLeftOut("nr", Stage(p, "nr").params, PresetScope::Stage);
    CHECK(cardLeft.size() == 2);  // format png (an engineer key) and extra_key
}

TEST_CASE("ApplyPipelinePreset: stage settings as ApplyPreset, enabled replaced, the encode card replaced, other codec options and the settings note kept, the result takes the codec's container",
          "[unit][presets][regression]") {
    Project p = Tuned();
    Stage(p, "depth").params = {{"backend", "da3"}, {"max_res", 720}, {"models_dir", "D:/m"}};
    Stage(p, "fg").enabled = true;
    p.viewport.time = 2.5;  // not the default: the check below can fail
    p.viewport.mode = ViewMode::Grid;
    const Project before = p;
    Preset preset = MakePipelinePreset("Другой", Tuned());
    preset.stages.at("nr").params = {{"intensity", 0.6}, {"backend", "stub"}, {"format", "exr"}};
    preset.stages.at("depth").params = {{"backend", "vda"}, {"model", "metric-vda-large"}};
    preset.stages.at("fg").enabled = false;
    preset.encode = PresetEncode{"prores_ks", {10, "422"}, ""};
    std::vector<PresetIssue> notes;
    ApplyPipelinePreset(p, preset, &notes);
    CHECK(notes.empty());
    CHECK(Stage(p, "nr").params == json{{"backend", "stub"}, {"intensity", 0.6}, {"extra_key", 1}});  // format back to exr, the default
    CHECK(Stage(p, "depth").params == json{{"backend", "vda"}, {"model", "metric-vda-large"}, {"max_res", 720}, {"models_dir", "D:/m"}});
    CHECK_FALSE(Stage(p, "fg").enabled);
    CHECK(p.codec == "prores_ks");
    CHECK(p.outputFormat == OutputFormatRequest{10, "422"});
    CHECK_FALSE(p.codecOptions.count("b"));
    CHECK(p.codecOptions.at("preset") == "p7");
    CHECK(p.settingsNote == before.settingsNote);
    CHECK(p.resultVideo == std::filesystem::path("D:/v/clip_result.mov"));
    CHECK(p.passesRoot == before.passesRoot);
    CHECK(p.sourceVideo == before.sourceVideo);
    CHECK(p.passVersionsKeep == 5);
    CHECK(p.viewport.ToJson() == before.viewport.ToJson());
    // a bitrate back
    preset.encode = PresetEncode{"hevc_nvenc", {}, "80M"};
    ApplyPipelinePreset(p, preset);
    CHECK(p.codecOptions.at("b") == "80M");
    CHECK(p.resultVideo == std::filesystem::path("D:/v/clip_result.mp4"));
    // a key that does not apply to the backend: a note
    preset.stages.at("depth").params = {{"backend", "da3"}, {"window", 16}};
    notes.clear();
    ApplyPipelinePreset(p, preset, &notes);
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].kind == Kind::ForeignKey);
    CHECK(notes[0].key == "window");
    // an empty result path stays empty (a new project)
    Project fresh;
    fresh.stages = Project::DefaultStages();
    preset.encode = PresetEncode{"prores_ks", {}, ""};
    ApplyPipelinePreset(fresh, preset);
    CHECK(fresh.resultVideo.empty());
}

TEST_CASE("ApplyPipelinePreset: a stage or the encode missing from the preset stays as it is, with a note; a stage the project lacks is inserted with the preset's enabled",
          "[unit][presets]") {
    json j = json::parse(PresetToJson(MakePipelinePreset("Частичный", Tuned())).dump());
    j["stages"].erase("depth");
    j.erase("encode");
    j["stages"]["scenes"]["enabled"] = false;  // the opposite of the inserted stage's default (DefaultStages: on)
    const PresetLoad l = ValidatePreset(j);
    REQUIRE(l.ok());
    CHECK(l.Has(Kind::MissingStage));
    CHECK(l.Has(Kind::NoEncode));
    Project p;
    p.stages = Project::DefaultStages();
    p.stages.erase(p.stages.begin());  // no scenes
    Stage(p, "depth").params = {{"backend", "stub"}, {"stabilize", "none"}};
    p.codec = "ffv1";
    ApplyPipelinePreset(p, l.preset);
    CHECK(Stage(p, "depth").params == json{{"backend", "stub"}, {"stabilize", "none"}});
    CHECK(p.codec == "ffv1");
    REQUIRE(HasStage(p, "scenes"));
    CHECK(p.stages.front().name == "scenes");  // at its place in the run order
    CHECK_FALSE(Stage(p, "scenes").enabled);  // the preset's switch, not DefaultStages' on
    CHECK(Stage(p, "scenes").params["threshold"] == 0.3);
    // a card preset goes to its stage as a card's: the switch stays
    Project q;
    q.stages = Project::DefaultStages();
    Stage(q, "nr").enabled = false;
    Stage(q, "nr").params = {{"backend", "stub"}, {"format", "png"}};
    ApplyPipelinePreset(q, MakePreset("Карточка", "nr", {{"intensity", 0.5}}));
    CHECK_FALSE(Stage(q, "nr").enabled);
    CHECK(Stage(q, "nr").params == json{{"backend", "stub"}, {"format", "png"}, {"intensity", 0.5}});
}

TEST_CASE("Pipeline preset properties: applying the project's own preset changes nothing; DiffPipeline lists the stages in run order with enabled, params and encode",
          "[unit][presets][fingerprint]") {
    for (Project p : {Tuned(), [] {
             Project d;
             d.stages = Project::DefaultStages();
             return d;
         }()}) {
        Stage(p, "depth").params["stabilize"] = "scale";
        const Project before = p;
        ApplyPipelinePreset(p, MakePipelinePreset("Своё", p));
        CHECK(DiffPipeline(before, p).empty());
        for (const auto& s : p.stages) {
            INFO(s.name);
            const StageEntry* b = nullptr;
            for (const auto& x : before.stages)
                if (x.name == s.name) b = &x;
            REQUIRE(b);
            CHECK(Canon(s.name, s.params) == Canon(s.name, b->params));
            CHECK(s.enabled == b->enabled);
        }
        CHECK(p.codecOptions == before.codecOptions);
        CHECK(p.settingsNote == before.settingsNote);
    }
    // a DA3 model from the JSON stays while the method stays; a preset with vda takes it away (as a switch by hand)
    Project da3;
    da3.stages = Project::DefaultStages();
    Stage(da3, "depth").params = {{"backend", "da3"}, {"model", "da3mono-large"}};
    Project same = da3;
    ApplyPipelinePreset(same, MakePipelinePreset("DA3", da3));
    CHECK(Stage(same, "depth").params["model"] == "da3mono-large");
    Project vda = da3;
    Preset toVda = MakePipelinePreset("VDA", da3);
    toVda.stages.at("depth").params["backend"] = "vda";
    ApplyPipelinePreset(vda, toVda);
    CHECK_FALSE(Stage(vda, "depth").params.contains("model"));
    // the difference
    Project a;
    a.stages = Project::DefaultStages();
    Project b = a;
    Stage(b, "nr").params["intensity"] = 0.7;
    Stage(b, "nr").params["style"] = "cinematic";
    Stage(b, "fg").enabled = false;
    Stage(b, "scenes").params["threshold"] = 0.3;
    b.codec = "hevc_nvenc";
    b.outputFormat.bitDepth = 10;
    b.codecOptions["b"] = "50M";
    b.codecOptions["preset"] = "p7";  // no part of it
    const PipelineDiff d = DiffPipeline(a, b);
    REQUIRE(d.stages.size() == 3);
    CHECK(d.stages[0].stage == "scenes");
    CHECK(d.stages[1].stage == "nr");
    CHECK(d.stages[2].stage == "fg");
    REQUIRE(d.stages[1].params.size() == 2);
    CHECK(d.stages[1].params[0].key == "style");
    CHECK(d.stages[1].params[1].key == "intensity");
    CHECK(d.stages[2].enabledWas);
    CHECK_FALSE(d.stages[2].enabledNow);
    CHECK(d.stages[2].params.empty());
    REQUIRE(d.encode.size() == 3);
    CHECK(d.encode[0].field == "codec");
    CHECK(d.encode[1].field == "bit_depth");
    CHECK(d.encode[1].was == "auto");
    CHECK(d.encode[1].now == "10");
    CHECK(d.encode[2].field == "bitrate");
    CHECK(d.encode[2].was.empty());
    CHECK(DiffPipeline(b, b).empty());
    // a stage one side lacks counts as its card shows it
    Project noScenes = a;
    noScenes.stages.erase(noScenes.stages.begin());
    const PipelineDiff ns = DiffPipeline(noScenes, a);
    REQUIRE(ns.stages.size() == 1);
    CHECK(ns.stages[0].stage == "scenes");
    CHECK_FALSE(ns.stages[0].enabledWas);
    CHECK(ns.stages[0].enabledNow);
}

TEST_CASE("Pipeline preset: round trip keeps every stage and the encode; written in run order with ordered keys", "[unit][presets]") {
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "presets" / "pipeline_roundtrip";
    std::filesystem::remove_all(dir);
    const Preset pp = MakePipelinePreset("Аниме 4K", Tuned());
    const auto file = dir / "anime.json";
    SavePreset(file, pp);
    const PresetLoad l = LoadPreset(file);
    REQUIRE(l.ok());
    CHECK(l.issues.empty());
    CHECK(l.preset.scope == PresetScope::Pipeline);
    REQUIRE(l.preset.stages.size() == 6);
    for (const auto& [s, e] : pp.stages) {
        INFO(s);
        CHECK(l.preset.stages.at(s).enabled == e.enabled);
        CHECK(l.preset.stages.at(s).params == e.params);
    }
    REQUIRE(l.preset.encode);
    CHECK(*l.preset.encode == *pp.encode);
    std::ifstream in(file, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    auto at = [&](const std::string& s, size_t from = 0) { return text.find(s, from); };
    // the envelope in its order
    size_t last = 0;
    for (const char* k : {"\"format\"", "\"schema\"", "\"scope\"", "\"name\"", "\"created\"", "\"app\"", "\"stages\"", "\"encode\""}) {
        INFO(k);
        const size_t i = at(k);
        REQUIRE(i != std::string::npos);
        CHECK(i > last);
        last = i;
    }
    // the stages in run order, enabled before params
    last = at("\"stages\"");
    for (const auto& s : PipelineStageNames()) {
        INFO(s);
        const size_t i = at("\"" + s + "\": {", last);
        REQUIRE(i != std::string::npos);
        CHECK(at("\"enabled\"", i) < at("\"params\"", i));
        last = i;
    }
    // nr's keys in PresetKeys order
    const size_t nr = at("\"nr\": {");
    last = nr;
    for (const auto& k : PresetKeys("nr", "ngx", PresetScope::Pipeline)) {
        INFO(k);
        const size_t i = at("\"" + k + "\":", nr);
        REQUIRE(i != std::string::npos);
        CHECK(i > last);
        last = i;
    }
    // the encoder in its order, the bit depth as the project writes it
    last = at("\"encode\"");
    for (const char* k : {"\"codec\"", "\"bit_depth\": \"10\"", "\"chroma\"", "\"options\""}) {
        INFO(k);
        const size_t i = at(k, last);
        REQUIRE(i != std::string::npos);
        last = i;
    }
}

// Review of PR-9: the reader writes the bitrate as BitrateOption ("50000k" -> "50M"); EncodeOf now gives the same form, so a project's
// own preset, saved and read back, is equal to it (before: «Битрейт 50000k → 50M», the slot rewritten, the project dirty)
TEST_CASE("Pipeline preset: a bitrate is written as the reader gives it, so the project's own preset read back changes nothing", "[unit][presets][regression]") {
    Project p = Tuned();
    p.codecOptions = {{"b", "50000k"}, {"preset", "p7"}};
    CHECK(EncodeOf(p).bitrate == "50M");
    const json file = json::parse(PresetToJson(MakePipelinePreset("Свой", p)).dump());
    CHECK(file["encode"]["options"]["b"] == "50M");
    const PresetLoad l = ValidatePreset(file);
    REQUIRE(l.ok());
    Project after = p;
    ApplyPipelinePreset(after, l.preset);
    CHECK(DiffPipeline(p, after).empty());
    // a value the card cannot hold stays as it is: the save refuses it, a slot keeps it
    p.codecOptions = {{"b", "12.5M"}};
    CHECK(EncodeOf(p).bitrate == "12.5M");
    CHECK_THROWS_AS(SavePreset(std::filesystem::path(DLSSVID_TEST_TMP) / "presets" / "bitrate.json", MakePipelinePreset("Битрейт", p)), PresetError);
}

// Review of PR-9: a key with form lists and none for the backend (depth.model under DA3) is used by the backend - the preset skips it
// as a key without a row, and the note says so instead of «does not apply to the backend»
TEST_CASE("ApplyPipelinePreset: a list key without a list for the backend is skipped as such, the project's value kept", "[unit][presets][regression]") {
    Project p;
    p.stages = Project::DefaultStages();
    Stage(p, "depth").params = {{"backend", "da3"}, {"model", "da3metric-large"}};
    Preset preset = MakePipelinePreset("DA3", p);
    preset.stages.at("depth").params = {{"backend", "da3"}, {"model", "da3mono-large"}};
    std::vector<PresetIssue> notes;
    ApplyPipelinePreset(p, preset, &notes);
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].kind == Kind::ForeignKey);
    CHECK(notes[0].text == "depth.model has no list for the backend da3 - presets do not set it, the project's value stays");
    CHECK(notes[0].kept);
    CHECK(Stage(p, "depth").params["model"] == "da3metric-large");
    // a switch of the method in the same preset dropped the project's VDA model: the note says the backend's default runs
    Stage(p, "depth").params = {{"backend", "vda"}, {"model", "metric-vda-large"}};
    notes.clear();
    ApplyPipelinePreset(p, preset, &notes);
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].text == "depth.model has no list for the backend da3 - presets do not set it, the backend's default runs");
    CHECK_FALSE(notes[0].kept);
    CHECK_FALSE(Stage(p, "depth").params.contains("model"));
}
