// Stage presets (PR-9): what a card preset and a pipeline preset hold of a stage, how they are read, checked, written and applied,
// how a card is reset, and the folder of presets.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

#include "pipeline/PassFingerprint.h"
#include "stages/ParamSchema.h"
#include "stages/PresetLibrary.h"
#include "stages/StagePreset.h"

using namespace dlssvid;
using Kind = PresetIssueKind;
using nlohmann::json;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "presets" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::filesystem::path WriteText(const std::filesystem::path& p, const std::string& text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
    return p;
}

std::string ReadText(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

std::filesystem::path U8(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

std::string Repeat(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

json CardPreset(const std::string& stage, json params, const std::string& name = "P") {
    return {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "stage"}, {"name", name}, {"stages", {{stage, {{"params", std::move(params)}}}}}};
}

json PipelineJson() {
    json stages = json::object();
    stages["scenes"] = {{"enabled", true}, {"params", {{"threshold", 0.3}}}};
    stages["depth"] = {{"enabled", true}, {"params", {{"backend", "vda"}, {"model", "metric-vda-base"}, {"z_near", 0.5}}}};
    stages["flow"] = {{"enabled", true}, {"params", {{"backend", "stub"}}}};
    stages["upscale"] = {{"enabled", true}, {"params", {{"backend", "nis"}, {"scale", 3}, {"sharpness", 0.8}}}};
    stages["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.7}, {"format", "png"}, {"backend", "stub"}}}};
    stages["fg"] = {{"enabled", false}, {"params", {{"backend", "blend"}, {"multiplier", 3}}}};
    return {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "pipeline"}, {"name", "Аниме 4K"}, {"stages", stages},
            {"encode", {{"codec", "hevc_nvenc"}, {"bit_depth", "10"}, {"chroma", "auto"}, {"options", {{"b", "50M"}}}}}};
}

size_t Count(const PresetLoad& l, Kind k) {
    size_t n = 0;
    for (const auto& i : l.issues) n += i.kind == k;
    return n;
}
size_t Count(const std::vector<PresetIssue>& issues, Kind k) {
    size_t n = 0;
    for (const auto& i : issues) n += i.kind == k;
    return n;
}

json Canon(const std::string& stage, const json& p) { return CanonicalizeJson(EffectiveStageParams(stage, p)); }

bool Has(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

}  // namespace

TEST_CASE("Preset: round trip keeps the name, the dates and every Basic and More value; numbers canonical", "[unit][presets]") {
    const auto dir = Dir("roundtrip");
    const json cur = {{"backend", "stub"}, {"format", "png"}, {"intensity", 0.1 + 0.2}, {"style", 2}, {"local-tone", 0.5}};
    const Preset p = MakePreset("  Мягкий портрет ", "nr", cur);
    CHECK(p.name == "Мягкий портрет");
    CHECK(p.scope == PresetScope::Stage);
    CHECK(p.app == DLSSVID_VERSION);
    CHECK(p.created.size() == 20);  // 2026-10-26T14:32:05Z
    const json& params = p.stages.at("nr").params;
    CHECK(params.size() == 17);
    CHECK_FALSE(params.contains("backend"));
    CHECK_FALSE(params.contains("format"));
    CHECK(params["intensity"] == 0.3);
    CHECK(params["style"] == "cinematic");
    CHECK(params["local_tone"] == 0.5);
    CHECK(params["temporal"] == 0);  // defaults written out
    const auto file = dir / U8("Мягкий портрет.json");
    SavePreset(file, p);
    CHECK_FALSE(std::filesystem::exists(dir / U8("Мягкий портрет.json.tmp")));
    const PresetLoad l = LoadPreset(file);
    REQUIRE(l.ok());
    CHECK(l.issues.empty());
    CHECK(l.preset.name == p.name);
    CHECK(l.preset.created == p.created);
    CHECK(l.preset.app == p.app);
    CHECK(l.preset.stages.at("nr").params == params);
    const std::string text = ReadText(file);
    CHECK(text.find("0.30000000000000004") == std::string::npos);
    CHECK(text.find("\"intensity\": 0.3") != std::string::npos);
    CHECK(text.find("\"scope\": \"stage\"") != std::string::npos);
    CHECK(text.find("\"enabled\"") == std::string::npos);  // a card preset has no switch
}

TEST_CASE("Preset keys: a card preset holds the Basic and More keys of its backend, a pipeline preset adds the tuning keys; never paths or GPU-bound keys",
          "[unit][presets][regression]") {
    const std::vector<std::string> nr17 = {"style",          "intensity",   "passes",         "local_tone", "local_structure", "temporal",
                                           "temporal_threshold", "guides",  "auto_mask",      "skin_structure", "skin_blend",  "tonemap",
                                           "input_transfer", "exposure",    "model_scale",    "transfer",   "max_ratio"};
    CHECK(PresetKeys("nr", "ngx", PresetScope::Stage) == nr17);
    std::vector<std::string> nrPipeline = nr17;
    nrPipeline.push_back("backend");
    nrPipeline.push_back("format");
    CHECK(PresetKeys("nr", "ngx", PresetScope::Pipeline) == nrPipeline);
    const std::set<std::string> machine = {"models_dir", "python", "max_res", "input_size", "tile", "fp32"};
    std::set<std::string> strings[2];
    for (const StageSchema& st : StageSchemas()) {
        std::set<std::string> backends;
        if (const ParamSpec* b = st.Find("backend"))
            for (const auto& c : b->choices) backends.insert(c.value);
        for (const auto& backend : backends)
            for (const PresetScope scope : {PresetScope::Stage, PresetScope::Pipeline})
                for (const auto& k : PresetKeys(st.stage, backend, scope)) {
                    INFO(st.stage << "." << k << " under " << backend << " in " << PresetScopeName(scope));
                    const ParamSpec* spec = st.Find(k);
                    REQUIRE(spec);
                    CHECK_FALSE(machine.count(k));
                    if (scope == PresetScope::Stage) CHECK_FALSE(spec->advanced);
                    if (spec->type == ParamType::String) strings[scope == PresetScope::Pipeline].insert(st.stage + "." + k);
                }
    }
    // the String keys of presets by name: registry ids and sizes, never a path
    CHECK(strings[0] == std::set<std::string>{"depth.model"});
    CHECK(strings[1] == std::set<std::string>{"depth.model", "flow.model", "flow.target", "upscale.preset", "fg.model"});
    // a list key without a list for the backend has no row and is no preset key
    CHECK_FALSE(Has(PresetKeys("depth", "da3", PresetScope::Stage), "model"));
    CHECK(Has(PresetKeys("depth", "vda", PresetScope::Stage), "model"));
    CHECK(PresetKeys("scenes", "luma", PresetScope::Pipeline) == std::vector<std::string>{"threshold"});
    CHECK(PresetKeys("upscale", "dlss", PresetScope::Stage) == std::vector<std::string>{"scale", "backend"});
    CHECK(PresetKeys("upscale", "trt", PresetScope::Stage) == std::vector<std::string>{"scale", "backend", "model"});
    const json pp = PresetParams("upscale", {{"backend", "trt"}, {"tile", 256}, {"sharpness", 0.8}, {"models_dir", "m"}}, PresetScope::Pipeline);
    CHECK(pp["sharpness"] == 0.8);
    CHECK(pp["model"] == "realesrgan-x2plus");
    CHECK_FALSE(pp.contains("tile"));
    CHECK_FALSE(pp.contains("models_dir"));
    CHECK_FALSE(pp.contains("window"));  // RealBasicVSR's
}

TEST_CASE("Preset: refuses out-of-range values, wrong types and unknown choices, all named at once", "[unit][presets][regression]") {
    const PresetLoad l = ValidatePreset(CardPreset("nr", {{"intensity", 5}, {"passes", 9}, {"style", "x"}, {"guides", "yes"}}));
    CHECK_FALSE(l.ok());
    std::map<std::string, ParamProblem::Kind> kinds;
    for (const auto& i : l.issues)
        if (i.kind == Kind::BadValue) {
            REQUIRE(i.problem);
            CHECK(i.stage == "nr");
            kinds[i.key] = i.problem->kind;
        }
    REQUIRE(kinds.size() == 4);
    CHECK(kinds["intensity"] == ParamProblem::Kind::Range);
    CHECK(kinds["passes"] == ParamProblem::Kind::Choice);
    CHECK(kinds["style"] == ParamProblem::Kind::Choice);
    CHECK(kinds["guides"] == ParamProblem::Kind::Type);
    CHECK(l.problemsText().find("nr.intensity must be within 0.0..2.0 (got 5)") != std::string::npos);
}

TEST_CASE("Preset: a card preset drops unknown, engineer and moved keys, an unknown stage and the pipeline fields with a note", "[unit][presets]") {
    json j = CardPreset("nr", {{"intensity", 0.8}, {"foo", 1}, {"format", "png"}, {"backend", "stub"}, {"scene_cut", 0.3}});
    j["stages"]["nr"]["enabled"] = false;
    j["stages"]["export"] = {{"params", {{"a", 1}}}};
    j["encode"] = {{"codec", "ffv1"}};
    const PresetLoad l = ValidatePreset(j);
    REQUIRE(l.ok());
    CHECK(Count(l, Kind::UnknownKey) == 1);
    CHECK(Count(l, Kind::EngineerKey) == 2);
    CHECK(Count(l, Kind::MovedKey) == 1);
    CHECK(Count(l, Kind::UnknownStage) == 1);
    CHECK(Count(l, Kind::PipelineFields) == 2);
    for (const auto& i : l.issues)
        if (i.kind == Kind::MovedKey) CHECK(i.to == "scenes.threshold");
    CHECK(l.preset.stages.size() == 1);
    CHECK(l.preset.stage() == "nr");
    CHECK(l.preset.stages.at("nr").params == json{{"intensity", 0.8}});
    CHECK_FALSE(l.preset.encode);
}

TEST_CASE("Preset: a pipeline preset read for one stage keeps that stage and notes the other stages and the pipeline fields", "[unit][presets]") {
    const PresetLoad l = ValidatePreset(PipelineJson(), "nr");
    REQUIRE(l.ok());
    CHECK(l.preset.scope == PresetScope::Pipeline);
    CHECK(l.preset.stages.size() == 1);
    REQUIRE(l.preset.stages.count("nr"));
    CHECK_FALSE(l.preset.encode);
    CHECK(Count(l, Kind::OtherStages) == 1);
    CHECK(Count(l, Kind::PipelineFields) == 1);
    CHECK(Count(l, Kind::MissingStage) == 0);
    for (const auto& i : l.issues)
        if (i.kind == Kind::OtherStages) CHECK(i.stages == std::vector<std::string>{"scenes", "depth", "flow", "upscale", "fg"});
    // the entry is the pipeline's: its tuning keys stay
    CHECK(l.preset.stages.at("nr").params == json{{"intensity", 0.7}, {"format", "png"}, {"backend", "stub"}});
    CHECK(ValidatePreset(PipelineJson(), "depth").preset.stages.at("depth").params["z_near"] == 0.5);
}

TEST_CASE("Preset: normalises kebab-case, no_guides and integer styles before the checks", "[unit][presets][regression]") {
    const PresetLoad bad = ValidatePreset(CardPreset("nr", {{"local-tone", 5}}));
    CHECK_FALSE(bad.ok());
    CHECK(Count(bad, Kind::BadValue) == 1);  // refused, not dropped as unknown
    CHECK(Count(bad, Kind::UnknownKey) == 0);
    const PresetLoad l = ValidatePreset(CardPreset("nr", {{"local-tone", 0.5}, {"style", 2}, {"no_guides", true}}));
    REQUIRE(l.ok());
    CHECK(l.preset.stages.at("nr").params == json{{"local_tone", 0.5}, {"style", "cinematic"}, {"guides", false}});
}

TEST_CASE("Preset envelope: format by value, schema 1 only, newer refused, 1 MiB, name 1..120 characters", "[unit][presets]") {
    CHECK(ValidatePreset({{"format", "png"}, {"intensity", 1}}).Has(Kind::NotPreset));
    CHECK(ValidatePreset(json::array()).Has(Kind::NotJson));
    json newer = CardPreset("nr", {{"intensity", 1}}, "Новый");
    newer["schema"] = 2;
    const PresetLoad n = ValidatePreset(newer);
    REQUIRE(n.Has(Kind::NewerSchema));
    CHECK(n.issues.front().schema == 2);
    CHECK(n.preset.name == "Новый");  // the menu names a refused file
    CHECK_FALSE(n.scopeKnown);        // and lists it in both menus
    for (const json& bad : {json(0), json("1"), json(1.5)}) {
        json j = CardPreset("nr", {{"intensity", 1}});
        j["schema"] = bad;
        INFO(bad.dump());
        CHECK(ValidatePreset(j).Has(Kind::BadSchema));
    }
    json noSchema = CardPreset("nr", {{"intensity", 1}});
    noSchema.erase("schema");
    CHECK(ValidatePreset(noSchema).Has(Kind::BadSchema));
    json name = CardPreset("nr", {{"intensity", 1}}, Repeat("я", 121));
    CHECK(ValidatePreset(name).Has(Kind::BadName));
    name["name"] = Repeat("я", 120);
    CHECK(ValidatePreset(name).ok());
    name["name"] = "a\tb";
    CHECK(ValidatePreset(name).Has(Kind::BadName));
    name["name"] = "   ";
    CHECK(ValidatePreset(name).Has(Kind::BadName));
    name["name"] = 5;
    CHECK(ValidatePreset(name).Has(Kind::BadName));
    CHECK(PresetNameProblem(" Мягкий ").empty());
    // the files: too large, not JSON, missing - a PresetLoad with the issue, never an exception
    const auto dir = Dir("envelope");
    CHECK(LoadPreset(WriteText(dir / "big.json", "{\"pad\": \"" + std::string(kPresetMaxBytes, 'x') + "\"}")).Has(Kind::TooLarge));
    CHECK(LoadPreset(WriteText(dir / "broken.json", "{\"format\": ")).Has(Kind::NotJson));
    CHECK(LoadPreset(dir / "missing.json").Has(Kind::Unreadable));
    const PresetLoad unknownField = ValidatePreset([] {
        json j = CardPreset("nr", {{"intensity", 1}});
        j["future"] = 1;
        j["created"] = 5;
        return j;
    }());
    CHECK(unknownField.ok());
    CHECK(Count(unknownField, Kind::UnknownField) == 2);
}

TEST_CASE("Preset scope: required; a stage preset holds exactly one stage, a pipeline preset needs enabled and params per stage; read for one stage it keeps that stage only",
          "[unit][presets][regression]") {
    json j = CardPreset("nr", {{"intensity", 1}});
    j.erase("scope");
    CHECK(ValidatePreset(j).Has(Kind::BadScope));
    CHECK_FALSE(ValidatePreset(j).scopeKnown);
    j["scope"] = "all";
    CHECK(ValidatePreset(j).Has(Kind::BadScope));
    json two = CardPreset("nr", {{"intensity", 1}});
    two["stages"]["depth"] = {{"params", {{"backend", "vda"}}}};
    CHECK(ValidatePreset(two).Has(Kind::BadScope));
    CHECK(ValidatePreset(two).scopeKnown);
    json withExport = CardPreset("nr", {{"intensity", 1}});
    withExport["stages"]["export"] = {{"params", json::object()}};
    CHECK(ValidatePreset(withExport).ok());  // an unknown stage does not count
    json noEnabled = PipelineJson();
    noEnabled["stages"]["nr"].erase("enabled");
    CHECK(ValidatePreset(noEnabled).Has(Kind::BadStages));
    json yes = PipelineJson();
    yes["stages"]["nr"]["enabled"] = "yes";
    CHECK(ValidatePreset(yes).Has(Kind::BadStages));
    CHECK(ValidatePreset(CardPreset("nr", {{"intensity", 1}}), "depth").Has(Kind::NoStage));
    json noDepth = PipelineJson();
    noDepth["stages"].erase("depth");
    CHECK(ValidatePreset(noDepth, "depth").Has(Kind::NoStage));
    const PresetLoad missing = ValidatePreset(noDepth);
    REQUIRE(missing.ok());
    CHECK(Count(missing, Kind::MissingStage) == 1);
    CHECK_FALSE(missing.preset.stages.count("depth"));
    json noEncode = PipelineJson();
    noEncode.erase("encode");
    const PresetLoad ne = ValidatePreset(noEncode);
    CHECK(ne.ok());
    CHECK(ne.Has(Kind::NoEncode));
    CHECK_FALSE(ne.preset.encode);
    json empty = noEncode;
    empty["stages"] = json::object();
    CHECK(ValidatePreset(empty).Has(Kind::BadStages));
    json emptyCard = CardPreset("nr", {{"format", "png"}});  // nothing left of the card
    CHECK(ValidatePreset(emptyCard).Has(Kind::BadStages));
    // the pipeline: every stage with its switch, the tuning keys kept, the machine keys dropped
    json full = PipelineJson();
    full["stages"]["depth"]["params"]["max_res"] = 720;
    full["stages"]["depth"]["params"]["models_dir"] = "D:/m";
    const PresetLoad p = ValidatePreset(full);
    REQUIRE(p.ok());
    CHECK(p.preset.stages.size() == 6);
    CHECK_FALSE(p.preset.stages.at("fg").enabled);
    CHECK(p.preset.stages.at("upscale").params["sharpness"] == 0.8);
    CHECK(p.preset.stages.at("depth").params == json{{"backend", "vda"}, {"model", "metric-vda-base"}, {"z_near", 0.5}});
    CHECK(Count(p, Kind::EngineerKey) == 2);
    REQUIRE(p.preset.encode);
    CHECK(p.preset.encode->codec == "hevc_nvenc");
    CHECK(p.preset.encode->format.bitDepth == 10);
    CHECK(p.preset.encode->bitrate == "50M");
}

TEST_CASE("Pipeline preset encode: a codec outside the list, a bit depth or chroma the codec cannot encode, a bad bitrate, no codec are refused; a bitrate of ProRes and fields without a card are dropped",
          "[unit][presets][regression]") {
    auto check = [](const json& e) {
        PresetEncode s;
        return std::pair{CheckEncodeSection(e, &s), s};
    };
    const std::vector<json> bad = {{{"codec", "libx264"}},
                                   {{"codec", "h264_nvenc"}, {"bit_depth", 10}},
                                   {{"codec", "av1_nvenc"}, {"chroma", "444"}},
                                   {{"codec", "prores_ks"}, {"bit_depth", "8"}},
                                   {{"codec", "ffv1"}, {"bit_depth", 16}},
                                   {{"codec", "hevc_nvenc"}, {"options", {{"b", "fast"}}}},
                                   {{"codec", "hevc_nvenc"}, {"options", {{"b", "0"}}}},
                                   {{"codec", "hevc_nvenc"}, {"options", {{"b", "12.5M"}}}},
                                   {{"codec", "hevc_nvenc"}, {"options", {{"b", "0.5M"}}}},
                                   {{"codec", "hevc_nvenc"}, {"options", {{"b", "1000M"}}}},
                                   {{"bit_depth", "auto"}}};
    for (const json& e : bad) {
        INFO(e.dump());
        const auto issues = check(e).first;
        REQUIRE(issues.size() == 1);
        CHECK(issues[0].kind == Kind::BadValue);
        CHECK(issues[0].stage == "encode");
        CHECK(issues[0].fatal());
    }
    CHECK(check({{"codec", "h264_nvenc"}, {"bit_depth", 10}, {"chroma", "422"}, {"options", {{"b", "fast"}}}}).first.size() == 3);  // in one pass
    const auto [okIssues, ok] = check({{"codec", "hevc_nvenc"}, {"options", {{"b", "50000k"}}}});
    CHECK(okIssues.empty());
    CHECK(ok.bitrate == "50M");
    CHECK(ok.format == OutputFormatRequest{});
    const auto [proIssues, pro] = check({{"codec", "prores_ks"}, {"options", {{"b", "50M"}}}});
    REQUIRE(proIssues.size() == 1);
    CHECK(proIssues[0].kind == Kind::ForeignKey);
    CHECK(pro.bitrate.empty());
    const auto extraIssues = check({{"codec", "hevc_nvenc"}, {"settings_note", false}, {"options", {{"preset", "p7"}}}}).first;
    CHECK(Count(extraIssues, Kind::EncodeField) == 2);
    CHECK(check({{"codec", "hevc_nvenc"}, {"bit_depth", 10}}).second == check({{"codec", "hevc_nvenc"}, {"bit_depth", "10"}}).second);
    CHECK(check({{"codec", "hevc_nvenc"}, {"bit_depth", 10}}).second.format.bitDepth == 10);
    // a slot keeps what the project held, with a note
    PresetEncode slot;
    const auto slotIssues = CheckEncodeSection({{"codec", "libx264"}, {"options", {{"b", "12.5M"}}}}, &slot, PresetRead::Slot);
    CHECK(Count(slotIssues, Kind::SlotValue) == 2);
    CHECK(slot.codec == "libx264");
    CHECK(slot.bitrate == "12.5M");
    // in a preset a refused encode refuses the preset
    json p = PipelineJson();
    p["encode"]["bit_depth"] = "12";
    const PresetLoad l = ValidatePreset(p);
    CHECK_FALSE(l.ok());
    CHECK_FALSE(l.preset.encode);
}

TEST_CASE("ApplyPreset keeps engineer, unknown and listless keys, replaces the card keys, switches the backend list first; the project stays sparse; a pipeline entry also replaces the tuning keys and keeps the machine keys",
          "[unit][presets][regression]") {
    std::vector<std::string> ignored;
    const json nr = ApplyPreset("nr", {{"format", "png"}, {"backend", "stub"}, {"extra_key", 1}, {"intensity", 1.4}, {"exposure", 1}},
                                PresetParams("nr", {{"intensity", 0.7}}, PresetScope::Stage), PresetScope::Stage, &ignored);
    CHECK(nr == json{{"format", "png"}, {"backend", "stub"}, {"extra_key", 1}, {"intensity", 0.7}});
    CHECK(ignored.empty());
    // depth: no backend in the preset - vda stays; a key the preset lacks returns to its default
    CHECK(ApplyPreset("depth", {{"backend", "vda"}, {"model", "metric-vda-large"}}, {{"stabilize", "none"}}, PresetScope::Stage) ==
          json{{"backend", "vda"}, {"stabilize", "none"}});
    // a switch to da3 first drops the VDA model; a DA3 model from the JSON has no row and stays under da3
    CHECK(ApplyPreset("depth", {{"backend", "vda"}, {"model", "metric-vda-large"}}, {{"backend", "da3"}}, PresetScope::Stage) == json{{"backend", "da3"}});
    CHECK(ApplyPreset("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}, {{"backend", "da3"}, {"stabilize", "none"}}, PresetScope::Stage) ==
          json{{"backend", "da3"}, {"model", "da3mono-large"}, {"stabilize", "none"}});
    ignored.clear();
    ApplyPreset("depth", {{"backend", "da3"}}, {{"window", 16}}, PresetScope::Stage, &ignored);
    CHECK(ignored == std::vector<std::string>{"window"});
    // a vda preset over {da3, da3mono-large}: the model goes - as a switch of the method by hand (the expected loss)
    CHECK(ApplyPreset("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}, {{"backend", "vda"}}, PresetScope::Stage) == json{{"backend", "vda"}});
    // the backend is always written, also at its default
    CHECK(ApplyPreset("upscale", json::object(), {{"backend", "dlss"}, {"scale", 2}}, PresetScope::Stage) == json{{"backend", "dlss"}});
    // a pipeline entry: the tuning keys too (sharpness back to its default), the machine keys stay
    const json up = {{"backend", "trt"}, {"sharpness", 0.8}, {"tile", 256}, {"models_dir", "m"}, {"scale", 3}};
    CHECK(ApplyPreset("upscale", up, {{"backend", "trt"}, {"scale", 2}}, PresetScope::Pipeline) == json{{"backend", "trt"}, {"tile", 256}, {"models_dir", "m"}});
    CHECK(ApplyPreset("upscale", up, {{"backend", "trt"}, {"scale", 2}}, PresetScope::Stage) ==
          json{{"backend", "trt"}, {"sharpness", 0.8}, {"tile", 256}, {"models_dir", "m"}});
    // nr: a pipeline entry switches the backend and the pass format; a card preset cannot
    CHECK(ApplyPreset("nr", {{"backend", "ngx"}, {"format", "exr"}, {"intensity", 1.4}}, {{"backend", "stub"}, {"format", "png"}, {"intensity", 1.0}},
                      PresetScope::Pipeline) == json{{"backend", "stub"}, {"format", "png"}});
    ignored.clear();
    CHECK(ApplyPreset("nr", {{"backend", "ngx"}}, {{"backend", "stub"}, {"intensity", 0.5}}, PresetScope::Stage, &ignored) ==
          json{{"backend", "ngx"}, {"intensity", 0.5}});
    CHECK(ignored == std::vector<std::string>{"backend"});
}

TEST_CASE("Preset properties: applying the current settings changes nothing; reset then the before-reset preset restores the run", "[unit][presets][fingerprint]") {
    const std::vector<std::pair<std::string, json>> cases = {
        {"nr", {{"backend", "stub"}, {"format", "png"}, {"extra_key", 1}, {"intensity", 0.1 + 0.2}, {"skin-structure", 0.5}, {"style", 1}}},
        {"nr", json::object()},
        {"depth", {{"backend", "vda"}, {"model", "metric-vda-base"}, {"window", 16}, {"z_near", 0.5}}},
        {"depth", {{"backend", "da3"}, {"model", "da3mono-large"}, {"stabilize", "none"}, {"window", 8}}},
        {"upscale", {{"backend", "trt"}, {"tile", 256}, {"sharpness", 0.8}, {"scale", 3}}},
        {"scenes", {{"threshold", 0.3}}},
        {"fg", {{"backend", "rife"}, {"multiplier", 3}, {"model", "rife49"}}},
    };
    for (const auto& [stage, c] : cases) {
        for (const PresetScope scope : {PresetScope::Stage, PresetScope::Pipeline}) {
            INFO(stage << " " << c.dump() << " " << PresetScopeName(scope));
            const json P = PresetParams(stage, c, scope);
            CHECK(Canon(stage, ApplyPreset(stage, c, P, scope)) == Canon(stage, c));  // 1
            for (const auto& [k, v] : P.items()) CHECK(Has(PresetKeys(stage, StageBackend(stage, c), scope), k));  // 5
        }
        INFO(stage << " " << c.dump());
        const json P = PresetParams(stage, c, PresetScope::Stage);
        CHECK(Canon(stage, ApplyPreset(stage, ResetStageParams(stage, c), P, PresetScope::Stage)) == Canon(stage, c));  // 2
        CHECK(ApplyPreset(stage, c, json::object(), PresetScope::Stage) == ResetStageParams(stage, c));                   // 3
        // 4: what is not the card's survives the application and the reset
        const json kept = ResetStageParams(stage, c);  // named: items() of a temporary dangles in the range-for
        json applied = ApplyPreset(stage, c, P, PresetScope::Stage);
        for (const auto& [k, v] : kept.items()) CHECK(applied[k] == v);
    }
}

TEST_CASE("DiffStageParams names the changed values in schema order with each side's backend; empty when equal", "[unit][presets]") {
    const auto d = DiffStageParams("nr", {{"intensity", 1.0}}, {{"intensity", 0.7}, {"style", "cinematic"}, {"format", "png"}}, PresetScope::Stage);
    REQUIRE(d.size() == 2);
    CHECK(d[0].key == "style");
    CHECK(d[1].key == "intensity");
    CHECK(d[1].was == 1.0);
    CHECK(d[1].now == 0.7);
    CHECK(DiffStageParams("nr", {{"intensity", 0.3}}, {{"intensity", 0.1 + 0.2}}, PresetScope::Stage).empty());
    CHECK(DiffStageParams("nr", json::object(), {{"intensity", 1.0}, {"passes", 1}}, PresetScope::Stage).empty());
    const auto f = DiffStageParams("nr", {{"format", "exr"}}, {{"format", "png"}}, PresetScope::Pipeline);
    REQUIRE(f.size() == 1);
    CHECK(f[0].key == "format");
    const auto dd = DiffStageParams("depth", {{"backend", "da3"}}, {{"backend", "vda"}, {"model", "metric-vda-base"}}, PresetScope::Stage);
    REQUIRE(dd.size() == 2);
    CHECK(dd[0].key == "backend");
    CHECK(dd[1].key == "model");
    CHECK(dd[1].was == "");
    CHECK(dd[1].now == "metric-vda-base");
    CHECK(dd[1].wasBackend == "da3");
    CHECK(dd[1].nowBackend == "vda");
}

// TASK-0047: a file of one stage's settings - a bare object of the schema's keys, aliases normalised, unknown keys named for the caller
// to refuse; PR-9: or a preset of either scope, read for the stage (before: refused as «a preset file»)
TEST_CASE("ReadStageParamsFile: a bare object as before, a preset unwrapped with notes, a preset without the stage refused", "[unit][schema][presets]") {
    const auto dir = Dir("params_file");
    auto file = [&](const char* name, const std::string& text) { return WriteText(dir / name, text); };
    const StageParamsFile f = ReadStageParamsFile("nr", file("ok.json", R"({"local-tone":0.3,"no_guides":true,"style":4,"intensty":1})"));
    CHECK(f.params == json{{"local_tone", 0.3}, {"guides", false}, {"style", "4"}});
    CHECK(f.dropped == std::vector<std::string>{"intensty"});
    CHECK_FALSE(f.preset);
    auto throws = [&](const std::filesystem::path& p, const char* says) {
        INFO(PathText(p));
        try {
            ReadStageParamsFile("nr", p);
            FAIL("no exception");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find(says) != std::string::npos);
            CHECK(std::string(e.what()).find(PathText(p.filename())) != std::string::npos);
        }
    };
    throws(file("array.json", "[1]"), "JSON object");
    throws(file("broken.json", "{\"intensity\": "), "not valid JSON");
    throws(file("big.json", "{\"intensity\": 1, \"pad\": \"" + std::string(1 << 20, 'x') + "\"}"), "1 MiB");
    // PR-13 (regression): the scene-cut threshold moved to the stage scenes - named, not dropped in silence
    throws(file("moved.json", R"({"intensity":1,"scene_cut":0.3})"), "nr.scene_cut moved to scenes.threshold");
    throws(file("moved_kebab.json", R"({"scene-cut":0})"), "scenes.threshold");
    // presets: checked when read
    throws(file("preset_noschema.json", R"({"format":"dlssvid-preset","stages":{}})"), "schema");
    throws(file("preset_bad.json", CardPreset("nr", {{"intensity", 5}}).dump()), "nr.intensity must be within");
    throws(file("preset_depth.json", CardPreset("depth", {{"backend", "vda"}}, "Глубина").dump()), "has no settings for nr");
    throws(WriteText(dir / U8("预设 ×2.json"), "{"), "not valid JSON");  // a name outside the ANSI code page, in UTF-8
    const StageParamsFile card = ReadStageParamsFile("nr", file("card.json", CardPreset("nr", {{"intensity", 0.7}, {"format", "png"}}, "Мягкий").dump()));
    CHECK(card.preset);
    CHECK(card.scope == PresetScope::Stage);
    CHECK(card.presetName == "Мягкий");
    CHECK(card.params == json{{"intensity", 0.7}});
    CHECK(Count(card.notes, Kind::EngineerKey) == 1);
    const StageParamsFile pipe = ReadStageParamsFile("nr", file("pipe.json", PipelineJson().dump()));
    CHECK(pipe.preset);
    CHECK(pipe.scope == PresetScope::Pipeline);
    CHECK(pipe.params == json{{"intensity", 0.7}, {"format", "png"}, {"backend", "stub"}});
    CHECK(Count(pipe.notes, Kind::OtherStages) == 1);
    CHECK(Count(pipe.notes, Kind::PipelineFields) == 1);
}

TEST_CASE("PresetLibrary: file names, collisions, reserved names, slots per stage, unreadable and newer files listed with the reason", "[unit][presets]") {
    CHECK(PresetFileStem("a/b:c") == "a_b_c");
    CHECK(PresetFileStem("a<b>c\"d|e?f*g\\h") == "a_b_c_d_e_f_g_h");
    CHECK(PresetFileStem("a\tb") == "a_b");
    CHECK(PresetFileStem("CON") == "_CON");
    CHECK(PresetFileStem("nul.txt") == "_nul.txt");
    CHECK(PresetFileStem("Com1") == "_Com1");
    CHECK(PresetFileStem("COM0") == "COM0");
    CHECK(PresetFileStem("console") == "console");
    CHECK(PresetFileStem(" .Мягкий. ") == "Мягкий");
    CHECK(PresetFileStem("...") == "preset");
    const std::string longStem = PresetFileStem("a" + Repeat("я", 60));  // 121 bytes: cut at a code point
    CHECK(longStem.size() == 79);
    CHECK(longStem == "a" + Repeat("я", 39));
    CHECK(ReservedPresetName("Перед применением · clip"));
    CHECK(ReservedPresetName("Перед сбросом · clip"));
    CHECK_FALSE(ReservedPresetName("Перед сбросом в ч/б"));
    CHECK_FALSE(ReservedPresetName("Перед применением · "));
    CHECK(PresetNameFor("Улучшение", "clip.mp4") == "Улучшение · clip");
    CHECK(PresetNameFor("Обработка", {}) == "Обработка");
    const std::string longName = PresetNameFor("Перед применением", U8("D:/v/" + Repeat("я", 200) + ".mp4"));
    size_t points = 0;
    for (const unsigned char c : longName) points += (c & 0xC0) != 0x80;
    CHECK(points == kPresetNameMax);
    CHECK(longName.compare(longName.size() - 3, 3, "…") == 0);
    CHECK(PresetNameProblem(longName).empty());
    CHECK(PresetSlotFileName(PresetSlot::BeforeReset, "nr") == "before-reset-nr.json");

    const auto dir = Dir("library");
    const PresetLibrary lib(dir);
    const auto a = lib.Save(MakePreset("Мягкий", "nr", {{"intensity", 0.7}}));
    CHECK(a == dir / U8("Мягкий.json"));
    CHECK_THROWS(lib.Save(MakePreset("Мягкий", "nr", {{"intensity", 0.5}})));  // the name is taken: the GUI asks to replace
    CHECK(lib.Save(MakePreset("Мягкий", "nr", {{"intensity", 0.5}}), a) == a);   // replacing
    CHECK(LoadPreset(a).preset.stages.at("nr").params["intensity"] == 0.5);
    CHECK_THROWS(lib.Save(MakePreset("Перед применением · clip", "nr", {})));  // a slot's name
    // a taken stem gets « (2)» - also over a broken file, which stays as it was
    WriteText(dir / U8("Ночь.json"), "{broken");
    CHECK(lib.Save(MakePreset("Ночь", "nr", {})) == dir / U8("Ночь (2).json"));
    CHECK(ReadText(dir / U8("Ночь.json")) == "{broken");
    WriteText(dir / "newer.json", [] {
        json j = CardPreset("nr", {{"intensity", 1}}, "Из будущего");
        j["schema"] = 2;
        return j.dump();
    }());
    const PresetListing listing = lib.List(PresetScope::Stage, "nr");
    REQUIRE(listing.presets.size() == 2);
    REQUIRE(listing.refused.size() == 2);
    std::set<std::string> refused;
    for (const auto& r : listing.refused) refused.insert(PathText(r.path.filename()));
    CHECK(refused == std::set<std::string>{"newer.json", "Ночь.json"});
    for (const auto& r : listing.refused) CHECK_FALSE(r.load.issues.empty());
    CHECK(listing.hidden == 0);
    // names outside the ANSI code page of the runner: listed, read, written, named in UTF-8
    for (const char* name : {"Пресет ×2", "预设"}) {
        INFO(name);
        const auto f = lib.Save(MakePreset(name, "nr", {{"intensity", 0.9}}));
        CHECK(LoadPreset(f).ok());
        CHECK(lib.FindByName(name, PresetScope::Stage) == f);
    }
    CHECK(lib.List(PresetScope::Stage, "nr").presets.size() == 4);
    // slots: per target, read as snapshots
    CHECK(lib.SlotPath(PresetSlot::BeforeApply, "nr") == dir / "auto" / "before-apply-nr.json");
    CHECK_FALSE(lib.Slot(PresetSlot::BeforeApply, "nr"));
    lib.SaveSlot(PresetSlot::BeforeApply, "nr", MakePreset(PresetNameFor("Перед применением", "clip.mp4"), "nr", {{"intensity", 1.4}}));
    const auto slot = lib.Slot(PresetSlot::BeforeApply, "nr");
    REQUIRE(slot);
    CHECK(slot->load.ok());
    CHECK(lib.IsSlotFile(slot->path));
    CHECK_FALSE(lib.IsSlotFile(a));
    CHECK(lib.List(PresetScope::Stage, "nr").presets.size() == 4);  // slots are not on the top level
    lib.Remove(dir / U8("Пресет ×2.json"));
    CHECK_FALSE(std::filesystem::exists(dir / U8("Пресет ×2.json")));
    CHECK_THROWS(lib.Remove(dir / "nothing.json"));

    // more than 500 files: the first 500 listed, the rest counted; the names over all of them
    const auto many = Dir("library_many");
    for (int i = 0; i < 520; ++i) {
        char stem[16];
        std::snprintf(stem, sizeof stem, "p%03d", i);
        WriteText(many / (std::string(stem) + ".json"), CardPreset("nr", {{"intensity", 1}}, stem).dump());
    }
    const PresetLibrary big(many);
    const PresetListing l = big.List(PresetScope::Stage, "nr");
    CHECK(l.presets.size() == 500);
    CHECK(l.hidden == 20);
    CHECK(big.FindByName("p519", PresetScope::Stage) == many / "p519.json");
}

TEST_CASE("PresetLibrary: lists by scope, the pipeline slot, one name in both scopes", "[unit][presets]") {
    const auto dir = Dir("library_scopes");
    const PresetLibrary lib(dir);
    const auto card = lib.Save(MakePreset("Аниме", "nr", {{"intensity", 0.8}}));
    const std::string cardBytes = ReadText(card);
    Preset pipe;
    pipe.name = "Аниме";
    pipe.scope = PresetScope::Pipeline;
    pipe.stages["nr"].params = {{"intensity", 0.6}, {"format", "png"}};
    pipe.stages["fg"].enabled = false;
    pipe.stages["fg"].params = {{"multiplier", 2}, {"backend", "blend"}};
    pipe.encode = PresetEncode{"hevc_nvenc", {10, "auto"}, "50M"};
    const auto pipeFile = lib.Save(pipe);  // the same name in the other scope: its own file
    CHECK(pipeFile == dir / U8("Аниме (2).json"));
    CHECK(ReadText(card) == cardBytes);
    CHECK(lib.FindByName("Аниме", PresetScope::Stage) == card);
    CHECK(lib.FindByName("Аниме", PresetScope::Pipeline) == pipeFile);
    WriteText(dir / "broken.json", "[");
    WriteText(dir / "bad_card.json", CardPreset("nr", {{"intensity", 5}}, "Плохой").dump());
    const PresetListing stage = lib.List(PresetScope::Stage, "nr");
    const PresetListing all = lib.List(PresetScope::Pipeline);
    REQUIRE(stage.presets.size() == 1);
    CHECK(stage.presets[0].path == card);
    REQUIRE(all.presets.size() == 1);
    CHECK(all.presets[0].path == pipeFile);
    CHECK(stage.refused.size() == 2);  // the broken file in both menus, the refused card preset in its own
    CHECK(all.refused.size() == 1);
    CHECK(lib.List(PresetScope::Stage, "depth").presets.empty());
    // the pipeline's slot; no reset slot for it
    CHECK(lib.SlotPath(PresetSlot::BeforeApply, kPipelineTarget) == dir / "auto" / "before-apply-pipeline.json");
    lib.SaveSlot(PresetSlot::BeforeReset, kPipelineTarget, pipe);  // even a file there is no slot of a pipeline
    REQUIRE(std::filesystem::is_regular_file(lib.SlotPath(PresetSlot::BeforeReset, kPipelineTarget)));
    CHECK_FALSE(lib.Slot(PresetSlot::BeforeReset, kPipelineTarget));
    // a slot of a newer version is replaced
    WriteText(lib.SlotPath(PresetSlot::BeforeApply, kPipelineTarget), [] {
        json j = PipelineJson();
        j["schema"] = 2;
        return j.dump();
    }());
    CHECK_FALSE(lib.Slot(PresetSlot::BeforeApply, kPipelineTarget)->load.ok());
    // a snapshot with the project's errors is written and read back as it was, with notes; a strict read refuses it
    Preset snap = pipe;
    snap.name = "Перед применением · clip";
    snap.stages["nr"].params["intensity"] = 5;
    snap.encode->codec = "libx264";
    lib.SaveSlot(PresetSlot::BeforeApply, kPipelineTarget, snap);
    const auto slot = lib.Slot(PresetSlot::BeforeApply, kPipelineTarget);
    REQUIRE(slot);
    CHECK(slot->load.ok());
    CHECK(Count(slot->load, Kind::SlotValue) == 2);
    CHECK(slot->load.preset.stages.at("nr").params["intensity"] == 5);
    CHECK(slot->load.preset.encode->codec == "libx264");
    const PresetLoad strict = LoadPreset(slot->path);
    CHECK_FALSE(strict.ok());
    CHECK(Count(strict, Kind::BadValue) == 2);
    CHECK_THROWS_AS(SavePreset(dir / "export.json", snap), PresetError);  // an export is strict
    CHECK_FALSE(std::filesystem::exists(dir / "export.json"));
}

// Review of PR-9: the parser of nlohmann is iterative, copying and dumping a value are not - a file nested 100 000 deep in its 1 MiB
// overflowed the stack when a menu listed the folder. The read refuses it as a broken file.
TEST_CASE("Preset files nested too deep are refused as broken, without a crash", "[unit][presets][regression]") {
    const auto dir = Dir("deep");
    const std::string deep = std::string(100000, '[') + std::string(100000, ']');
    const auto file = WriteText(dir / "deep.json", R"({"format":"dlssvid-preset","schema":)" + deep + R"(,"scope":"stage"})");
    const PresetLoad l = LoadPreset(file);
    REQUIRE(l.Has(Kind::NotJson));
    CHECK(l.issues.front().text.find("nested deeper than") != std::string::npos);
    CHECK_THROWS_WITH(ReadStageParamsFile("nr", file), Catch::Matchers::ContainsSubstring("nested deeper than"));
    const auto params = WriteText(dir / "params.json", R"({"intensity":)" + deep + "}");
    CHECK_THROWS_WITH(ReadStageParamsFile("nr", params), Catch::Matchers::ContainsSubstring("nested deeper than"));
    const PresetLibrary lib(dir);
    CHECK(lib.List(PresetScope::Stage, "nr").refused.size() == 2);
    // a real preset is far from the limit
    CHECK(LoadPreset(WriteText(dir / "ok.json", PipelineJson().dump())).ok());
}

// Review of PR-9: scenes.backend is an engineer key without `tuning` because there is nothing to choose, not a key of the machine
TEST_CASE("Pipeline preset: a key with one choice is dropped as nothing to choose, not as a key of the machine", "[unit][presets][regression]") {
    json j = PipelineJson();
    j["stages"]["scenes"]["params"]["backend"] = "luma";
    j["stages"]["depth"]["params"]["max_res"] = 720;
    const PresetLoad l = ValidatePreset(j);
    REQUIRE(l.ok());
    for (const auto& i : l.issues) {
        if (i.kind != Kind::EngineerKey) continue;
        INFO(i.text);
        if (i.key == "backend") CHECK(i.text == "scenes.backend is not a preset setting (nothing to choose) - dropped");
        if (i.key == "max_res") CHECK(i.text.find("bound to the machine") != std::string::npos);
    }
    CHECK(Count(l, Kind::EngineerKey) == 2);
    CHECK(IsMachineKey(*FindStageSchema("depth")->Find("max_res")));
    CHECK(IsMachineKey(*FindStageSchema("upscale")->Find("models_dir")));
    CHECK_FALSE(IsMachineKey(*FindStageSchema("scenes")->Find("backend")));
    CHECK_FALSE(IsMachineKey(*FindStageSchema("nr")->Find("format")));
    CHECK_FALSE(IsMachineKey(*FindStageSchema("nr")->Find("intensity")));
}

// Review of PR-9: a card preset of another stage with the same name is no clash - saving an NR preset «Мягкий» next to a depth one
// made «Заменить» overwrite the depth preset, which the NR menu never lists
TEST_CASE("PresetLibrary: a card preset's name is taken per stage; another stage's preset of that name is never replaced", "[unit][presets][regression]") {
    const auto dir = Dir("library_stages");
    const PresetLibrary lib(dir);
    const auto depth = lib.Save([] {
        Preset p;
        p.name = "Мягкий";
        p.scope = PresetScope::Stage;
        p.stages["depth"].params = {{"backend", "vda"}};
        return p;
    }());
    const std::string depthBytes = ReadText(depth);
    CHECK(lib.FindByName("Мягкий", PresetScope::Stage, "depth") == depth);
    CHECK_FALSE(lib.FindByName("Мягкий", PresetScope::Stage, "nr"));
    CHECK(lib.FindByName("Мягкий", PresetScope::Stage) == depth);  // without a stage: any
    const auto nr = lib.Save(MakePreset("Мягкий", "nr", {{"intensity", 0.7}}));
    CHECK(nr == dir / U8("Мягкий (2).json"));
    CHECK(ReadText(depth) == depthBytes);
    CHECK(lib.FindByName("Мягкий", PresetScope::Stage, "nr") == nr);
    CHECK_THROWS(lib.Save(MakePreset("Мягкий", "nr", {{"intensity", 0.5}})));  // now taken in nr
}
