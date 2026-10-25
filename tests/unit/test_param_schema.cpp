// Stage parameter schema (stage 9, MR D): every stage described, defaults mirror the stages, validation catches wrong
// types / ranges / choices, effective parameters make an explicit default and an absent key the same run.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "stages/ParamSchema.h"
#include "stages/depth/DepthStage.h"
#include "stages/fg/FgStage.h"
#include "stages/flow/FlowStage.h"
#include "stages/nr/NrStage.h"
#include "stages/upscale/UpscaleStage.h"

using namespace dlssvid;

TEST_CASE("Every stage has a schema whose defaults mirror the stages and the project", "[unit][schema]") {
    REQUIRE(StageSchemas().size() == 5);
    for (const char* s : {"depth", "flow", "upscale", "nr", "fg"}) {
        REQUIRE(FindStageSchema(s) != nullptr);
        CHECK(!FindStageSchema(s)->title.empty());
        CHECK(FindStageSchema(s)->Find("backend") != nullptr);
        CHECK(FindStageSchema(s)->Find("nope") == nullptr);
    }
    CHECK(FindStageSchema("bogus") == nullptr);
    for (const ProcessStage& st : DefaultProcessStages()) {  // the project's defaults are the schema's
        const StageSchema* schema = FindStageSchema(st.name);
        REQUIRE(schema);
        for (const auto& [k, v] : st.params.items()) {
            const ParamSpec* spec = schema->Find(k);
            REQUIRE(spec);
            CHECK(spec->def == v);
        }
    }
    CHECK(FindStageSchema("nr")->Find("intensity")->def == NrStageOptions{}.intensity);
    CHECK(FindStageSchema("nr")->Find("passes")->def == NrStageOptions{}.passes);
    CHECK(FindStageSchema("nr")->Find("style")->def == NrStageOptions{}.style);
    CHECK(FindStageSchema("nr")->Find("temporal")->def == NrStageOptions{}.temporal);
    CHECK(FindStageSchema("fg")->Find("multiplier")->def == FgStageOptions{}.multiplier);
    CHECK(FindStageSchema("fg")->Find("model")->def == FgStageOptions{}.model);
    CHECK(FindStageSchema("upscale")->Find("scale")->def == UpscaleStageOptions{}.scale);
    CHECK(FindStageSchema("upscale")->Find("sharpness")->def == UpscaleStageOptions{}.sharpness);
    CHECK(FindStageSchema("depth")->Find("backend")->def == DepthStageOptions{}.backend);
    // temporal alignment (TASK-0038): the schema's default and choices are what the stage parses
    const ParamSpec* stabilize = FindStageSchema("depth")->Find("stabilize");
    REQUIRE(stabilize);
    CHECK(stabilize->def == "auto");
    CHECK(ParseDepthStabilize(stabilize->def.get<std::string>()) == DepthStageOptions{}.stabilize);
    for (const auto& c : stabilize->choices) CHECK_NOTHROW(ParseDepthStabilize(c.value));
    CHECK(FindStageSchema("depth")->Find("stabilize_window")->def == DepthStageOptions{}.stabilizeWindow);
    CHECK(FindStageSchema("flow")->Find("backend")->def == FlowStageOptions{}.backend);
    // human labels for the form, technical keys stay keys; the main form of nr is the intensity alone
    CHECK(FindStageSchema("nr")->Find("intensity")->label == "Интенсивность");
    CHECK(FindStageSchema("upscale")->Find("scale")->widget == ParamWidget::Toggle);
    CHECK(FindStageSchema("upscale")->Find("scale")->choices.size() == 3);
    CHECK(FindStageSchema("nr")->guides == std::vector<std::string>{"depth_dlss", "mv_dlss"});
}

// TASK-0048 (spec §0.1, §2e): the A/B switches of the standalone dlssvid nr are flags, not settings - no schema key, so a
// project, a preset, --param or --params-file cannot set them, and ApplyNrParams never reads them.
TEST_CASE("NR A/B switches are standalone flags: no schema key, ApplyNrParams ignores them", "[unit][schema][nr]") {
    for (const char* k : {"mv_scale", "create_with_mv", "depth_guide", "no_depth_guide"}) {
        INFO(k);
        CHECK(FindStageSchema("nr")->Find(k) == nullptr);
    }
    NrStageOptions o;
    ApplyNrParams({{"mv_scale", "auto"}, {"create_with_mv", true}, {"depth_guide", false}, {"no-depth-guide", true}}, o);
    CHECK(o.mvScale == "unit");
    CHECK_FALSE(o.createWithMv);
    CHECK(o.depthGuide);
    CHECK(NrAbJson(NrStageOptions{}).empty());
}

// TASK-0047: a file of one stage's settings (dlssvid nr --params-file) - one object of the schema's keys, aliases normalised;
// keys the schema does not have are named for the caller to refuse; other files are refused with a message naming them.
TEST_CASE("ReadStageParamsFile: one object of schema keys, aliases normalised, unknown keys named, other files refused", "[unit][schema][nr]") {
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "params_file";
    std::filesystem::create_directories(dir);
    auto file = [&](const char* name, const std::string& text) {
        const auto p = dir / name;
        std::ofstream(p, std::ios::binary) << text;
        return p;
    };
    std::vector<std::string> dropped;
    const nlohmann::json j = ReadStageParamsFile("nr", file("ok.json", R"({"local-tone":0.3,"no_guides":true,"style":4,"intensty":1})"), &dropped);
    CHECK(j == nlohmann::json{{"local_tone", 0.3}, {"guides", false}, {"style", "4"}});
    CHECK(dropped == std::vector<std::string>{"intensty"});
    auto throws = [&](const std::filesystem::path& p, const char* says) {
        INFO(p.string());
        try {
            ReadStageParamsFile("nr", p);
            FAIL("no exception");
        } catch (const std::exception& e) {
            CHECK(std::string(e.what()).find(says) != std::string::npos);
            CHECK(std::string(e.what()).find(p.filename().string()) != std::string::npos);
        }
    };
    throws(file("array.json", "[1]"), "JSON object");
    throws(file("broken.json", "{\"intensity\": "), "not valid JSON");
    throws(file("preset.json", R"({"format":"dlssvid-preset","stages":{}})"), "preset");
    throws(file("big.json", "{\"intensity\": 1, \"pad\": \"" + std::string(1 << 20, 'x') + "\"}"), "1 MiB");
}

// TASK-0046: the NR card shows every setting (before: the intensity alone, the rest in the engineer JSON) - the basic
// rows in groups, the rest under «Ещё настройки», only the backend and the pass format in the JSON editor.
TEST_CASE("NR form: groups, levels, conditions of the rows", "[unit][schema][nr][card]") {
    const StageSchema* nr = FindStageSchema("nr");
    REQUIRE(nr);
    std::vector<std::string> basic, more, engineer;
    for (const auto& p : nr->params) (p.advanced ? engineer : p.more ? more : basic).push_back(p.key);
    CHECK(basic == std::vector<std::string>{"style", "intensity", "passes", "local_tone", "local_structure", "temporal", "auto_mask", "skin_structure", "skin_blend"});
    CHECK(more == std::vector<std::string>{"temporal_threshold", "scene_cut", "guides", "tonemap", "input_transfer", "exposure", "model_scale", "transfer",
                                           "max_ratio"});
    CHECK(engineer == std::vector<std::string>{"backend", "format"});
    for (const auto& p : nr->params) {
        INFO(p.key);
        CHECK((p.advanced || !p.group.empty()));  // every form row has a group
        CHECK(!p.label.empty());
        CHECK((p.advanced || !p.hint.empty()));   // and says what it does
    }
    CHECK(nr->Find("intensity")->step == 0.05);
    CHECK(ParamValueLabel(*nr->Find("intensity"), 0.0) == "без модели");  // PR-7: 0 does not call the model
    CHECK(nr->Find("intensity")->hint.find("выше 1") != std::string::npos);
    CHECK(nr->Find("style")->choices.size() == 3);  // PR-2 E7: 3..6 = cinematic, the preset changed nothing (TASK-0052)
    CHECK(nr->Find("preset") == nullptr);

    const std::set<std::string> none, faceMask{"color_sr", "mask_face"};
    const nlohmann::json def = EffectiveStageParams("nr", nlohmann::json::object());
    // skin structure: shown, but disabled without the automatic mask (as in Visual Enhancer)
    ParamRowState st = RowState(*nr->Find("skin_structure"), def, none);
    CHECK(st.visible);
    CHECK_FALSE(st.enabled);
    CHECK(st.note == "действует только с автомаской");
    CHECK(RowState(*nr->Find("skin_structure"), EffectiveStageParams("nr", {{"auto_mask", true}}), none).enabled);
    // the strength on skin masks only when the plan reads a face or skin mask
    CHECK_FALSE(RowState(*nr->Find("skin_blend"), def, none).visible);
    CHECK(RowState(*nr->Find("skin_blend"), def, faceMask).visible);
    // the threshold of the temporal filter only with the filter on; the ratio transfer only at another model scale
    CHECK_FALSE(RowState(*nr->Find("temporal_threshold"), def, none).visible);
    CHECK(RowState(*nr->Find("temporal_threshold"), EffectiveStageParams("nr", {{"temporal", 0.3}}), none).visible);
    CHECK_FALSE(RowState(*nr->Find("transfer"), def, none).visible);
    CHECK(RowState(*nr->Find("transfer"), EffectiveStageParams("nr", {{"model_scale", 0.5}}), none).visible);
    CHECK(RowState(*nr->Find("max_ratio"), EffectiveStageParams("nr", {{"model_scale", 2}}), none).visible);
    CHECK(RowState(*nr->Find("intensity"), def, none).visible);
}

TEST_CASE("Parameter values: the special value, the zero label, the suffix, the decimals", "[unit][schema][nr][card]") {
    const StageSchema* nr = FindStageSchema("nr");
    // skin_structure: -1 (as the local structure) or 0..2 - the half-way -0.5 had no meaning
    CHECK(ValidateStageParams("nr", {{"skin_structure", -1}}).empty());
    CHECK(ValidateStageParams("nr", {{"skin_structure", 0.4}}).empty());
    CHECK(ValidateStageParams("nr", {{"skin_structure", 2}}).empty());
    CHECK_FALSE(ValidateStageParams("nr", {{"skin_structure", -0.5}}).empty());
    CHECK_FALSE(ValidateStageParams("nr", {{"skin_structure", 2.5}}).empty());
    CHECK(ParamValueLabel(*nr->Find("skin_structure"), -1.0) == "как структура");
    CHECK(ParamValueLabel(*nr->Find("skin_structure"), 0.8) == "0.8");
    CHECK(ParamValueLabel(*nr->Find("temporal"), 0.0) == "выкл");
    CHECK(ParamValueLabel(*nr->Find("temporal"), 0.4) == "0.4");
    CHECK(ParamValueLabel(*nr->Find("exposure"), 1.0) == "1 EV");
    CHECK(ParamValueLabel(*nr->Find("exposure"), -0.5) == "-0.5 EV");
    CHECK(ParamValueLabel(*nr->Find("style"), "natural") == "Естественный (1)");
    CHECK(ParamValueLabel(*nr->Find("passes"), 2) == "2");
    CHECK(nr->Find("passes")->choices.size() == 4);  // PR-5: the toggle 1 | 2 | 3 | 4
    CHECK(ParamValueLabel(*nr->Find("passes"), 4) == "4");
    CHECK(ParamDecimals(*nr->Find("intensity")) == 2);
    CHECK(ParamDecimals(*nr->Find("exposure")) == 2);  // the v1 migration stores 0.01 EV (1.5 -> 0.58)
    CHECK(ParamDecimals(*nr->Find("model_scale")) == 2);
    CHECK(ParamDecimals(*nr->Find("passes")) == 0);
    CHECK(ParamDecimals(*nr->Find("temporal_threshold")) == 2);
}

// TASK-0043 (the root of the exposure defect): every nr key at its schema default is the run without the key - the
// stage options from the effective parameters equal the options' own defaults, field by field.
TEST_CASE("NR: the options from the schema's defaults are the stage's defaults, for every key", "[unit][schema][nr][regression]") {
    const NrStageOptions def;
    NrStageOptions o;
    ApplyNrParams(EffectiveStageParams("nr", nlohmann::json::object()), o);
    CHECK(o.backend == def.backend);
    CHECK(o.intensity == def.intensity);
    CHECK(o.style == def.style);
    CHECK(o.localTone == def.localTone);
    CHECK(o.localStructure == def.localStructure);
    CHECK(o.skinStructure == def.skinStructure);
    CHECK(o.autoMask == def.autoMask);
    CHECK(o.passes == def.passes);
    CHECK(o.modelScale == def.modelScale);
    CHECK(o.transfer == def.transfer);
    CHECK(o.maxRatio == def.maxRatio);
    CHECK(o.useGuides == def.useGuides);
    CHECK(o.temporal == def.temporal);
    CHECK(o.temporalThreshold == def.temporalThreshold);
    CHECK(o.sceneCut == def.sceneCut);
    CHECK(o.skinBlend == def.skinBlend);
    CHECK(o.tonemap.curve == def.tonemap.curve);
    CHECK(o.tonemap.inputTransfer == def.tonemap.inputTransfer);
    CHECK(o.tonemap.exposureEv == def.tonemap.exposureEv);  // was 0.0 against a multiplier of 1: black frames
    CHECK(o.tonemap.IsIdentity());
    // every schema key of nr is read by ApplyNrParams (format: by ProcessRunner)
    for (const auto& spec : FindStageSchema("nr")->params) {
        if (spec.key == "format") continue;
        nlohmann::json one = EffectiveStageParams("nr", nlohmann::json::object());
        nlohmann::json changed;  // another valid value
        if (spec.type == ParamType::Bool) changed = !spec.def.get<bool>();
        else if (spec.type == ParamType::Enum) changed = spec.choices.back().value == spec.def.get<std::string>() ? spec.choices.front().value : spec.choices.back().value;
        else changed = spec.def.get<double>() == spec.max ? spec.min : spec.max;
        one[spec.key] = changed;
        NrStageOptions a, b;
        ApplyNrParams(EffectiveStageParams("nr", nlohmann::json::object()), a);
        ApplyNrParams(one, b);
        const auto key = [](const NrStageOptions& x) {
            return nlohmann::json{x.backend, x.intensity, x.style, x.localTone, x.localStructure, x.skinStructure, x.autoMask, x.passes, x.modelScale,
                                  x.transfer, x.maxRatio, x.useGuides, x.temporal, x.temporalThreshold, x.sceneCut, x.skinBlend, x.tonemap.curve,
                                  x.tonemap.inputTransfer, x.tonemap.exposureEv};
        };
        INFO("nr." << spec.key);
        CHECK(key(a) != key(b));
    }
}

// TASK-0043: the stages took kebab-case aliases and nr's inverse no_guides that the validation and the defaults did not
// see - {"model-scale": 0.5} ran at 0.5 under a fingerprint saying 1.0, {"local-tone": 5} was never range-checked.
TEST_CASE("NormalizeStageParams: aliases take the schema's key; validation and the effective parameters see them", "[unit][schema][regression]") {
    CHECK(NormalizeStageParams("nr", {{"model-scale", 0.5}}) == nlohmann::json{{"model_scale", 0.5}});
    CHECK(NormalizeStageParams("nr", {{"model-scale", 0.5}, {"model_scale", 2.0}}) == nlohmann::json{{"model_scale", 2.0}});  // the schema key wins
    CHECK(NormalizeStageParams("nr", {{"no-guides", true}}) == nlohmann::json{{"guides", false}});
    CHECK(NormalizeStageParams("nr", {{"no_guides", false}}) == nlohmann::json{{"guides", true}});
    CHECK(NormalizeStageParams("nr", {{"no_guides", true}, {"guides", true}}) == nlohmann::json{{"guides", true}});
    CHECK(NormalizeStageParams("fg", {{"backbuffer-format", "rgba8"}}) == nlohmann::json{{"backbuffer_format", "rgba8"}});
    CHECK(NormalizeStageParams("depth", {{"z-near", 1}}) == nlohmann::json{{"z-near", 1}});  // no aliases there: unknown keys stay
    CHECK(EffectiveStageParams("nr", {{"model-scale", 0.5}})["model_scale"] == 0.5);
    CHECK_FALSE(EffectiveStageParams("nr", {{"model-scale", 0.5}}).contains("model-scale"));
    CHECK_FALSE(ValidateStageParams("nr", {{"local-tone", 5.0}}).empty());  // out of range under its alias too
    CHECK_FALSE(ValidateStageParams("nr", {{"no-guides", "yes"}}).empty());  // a non-boolean inverse: guides of the wrong type
    // an alias and its key are one run: one fingerprint
    CHECK(ComputeFingerprint("nr", "sha256:x", EffectiveStageParams("nr", {{"skin-blend", 0.5}}), {}, ToolInfo{}, "").value ==
          ComputeFingerprint("nr", "sha256:x", EffectiveStageParams("nr", {{"skin_blend", 0.5}}), {}, ToolInfo{}, "").value);
}

// PR-2 E7 (TASK-0052): styles 3..6 gave the cinematic picture bit for bit and the render preset changed nothing - the schema
// keeps the three named styles and no preset; a number 0..2 takes the schema's string (TASK-0044), 3..6 are refused now.
TEST_CASE("NR styles: the three named ones; numbers 0..2 take the schema's string; 3..6 and the preset are gone", "[unit][schema][nr][regression]") {
    for (const char* s : {"natural", "cinematic", "default"}) CHECK(ValidateStageParams("nr", {{"style", s}}).empty());
    for (const char* s : {"3", "4", "6", "7", "vivid"}) CHECK_FALSE(ValidateStageParams("nr", {{"style", s}}).empty());
    CHECK(NormalizeStageParams("nr", {{"style", "1"}}) == nlohmann::json{{"style", "natural"}});
    CHECK(NormalizeStageParams("nr", {{"style", 0}}) == nlohmann::json{{"style", "default"}});
    CHECK_FALSE(ValidateStageParams("nr", {{"style", 4}}).empty());
    CHECK(ValidateStageParams("nr", {{"preset", 3}}).empty());  // an unknown key now: validation skips it, --params-file and the CLI refuse it
    std::vector<ProcessStage> stages = DefaultProcessStages();
    ApplyParamSpec(stages, "nr.style=2");
    ApplyParamSpec(stages, "nr.intensity=1.5");
    for (const auto& s : stages)
        if (s.name == "nr") {
            CHECK(s.params["style"] == "2");                                      // text, as typed: the schema's Enum
            CHECK(NormalizeStageParams("nr", s.params)["style"] == "cinematic");  // the schema names it
            CHECK(s.params["intensity"] == 1.5);      // a number stays a number
        }
    // the options a run validates name the style by its schema value
    NrStageOptions o;
    o.style = "Cinematic";
    CHECK(NrParamsFromOptions(o)["style"] == "cinematic");
    CHECK_FALSE(NrParamsFromOptions(o).contains("preset"));
    o.style = "5";
    CHECK(NrParamsFromOptions(o)["style"] == "5");  // left for the schema to refuse
    CHECK_FALSE(ValidateStageParams("nr", NrParamsFromOptions(o)).empty());
    CHECK(ValidateStageParams("nr", NrParamsFromOptions(NrStageOptions{})).empty());
}

// TASK-0043: flow.hwaccel was hashed as "cuda" for searaft / stub, which decode on the CPU: it belongs to OFA alone.
TEST_CASE("flow.hwaccel belongs to the OFA backend", "[unit][schema][regression]") {
    CHECK(EffectiveStageParams("flow", nlohmann::json::object())["hwaccel"] == "cuda");
    CHECK_FALSE(EffectiveStageParams("flow", {{"backend", "searaft"}}).contains("hwaccel"));
    CHECK_FALSE(EffectiveStageParams("flow", {{"backend", "stub"}, {"hwaccel", "cuda"}}).contains("hwaccel"));
}

TEST_CASE("ValidateStageParams catches types, ranges and choices; unknown keys pass", "[unit][schema]") {
    CHECK(ValidateStageParams("nr", {{"intensity", 1.4}}).empty());
    CHECK(ValidateStageParams("nr", nlohmann::json::object()).empty());
    CHECK(ValidateStageParams("nr", nlohmann::json()).empty());
    CHECK(ValidateStageParams("nr", {{"whatever", 1}}).empty());
    CHECK(ValidateStageParams("nr", {{"intensity", 2.2}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"intensity", "high"}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"backend", "magic"}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"backend", 5}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"passes", 1.5}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"passes", 2}}).empty());
    CHECK(ValidateStageParams("nr", {{"passes", 4}}).empty());  // PR-5: 1..4
    CHECK(ValidateStageParams("nr", {{"scene_cut", 0.0}}).empty());  // PR-6: 0 = off .. 1
    CHECK(ValidateStageParams("nr", {{"scene_cut", 1.5}}).size() == 1);
    CHECK(FindStageSchema("nr")->Find("scene_cut")->def == 0.24);
    CHECK(FindStageSchema("nr")->Find("scene_cut")->more);
    CHECK(ParamValueLabel(*FindStageSchema("nr")->Find("scene_cut"), 0.0) == "выкл");
    CHECK(static_cast<double>(NrStageOptions{}.sceneCut) == Catch::Approx(0.24));
    CHECK(ValidateStageParams("nr", {{"passes", 5}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"passes", 0}}).size() == 1);
    CHECK(ValidateStageParams("nr", {{"guides", "yes"}}).size() == 1);
    CHECK(ValidateStageParams("upscale", {{"scale", 2.5}}).size() == 1);
    CHECK(ValidateStageParams("upscale", {{"scale", 1.5}}).empty());
    CHECK(ValidateStageParams("upscale", {{"scale", 2}}).empty());
    CHECK(ValidateStageParams("fg", {{"multiplier", 5}}).size() == 1);
    CHECK(ValidateStageParams("fg", {{"multiplier", 3}}).empty());
    CHECK(ValidateStageParams("depth", {{"model", 3}}).size() == 1);
    CHECK(ValidateStageParams("bogus", nlohmann::json::object()).size() == 1);
    CHECK(ValidateStageParams("nr", nlohmann::json::array()).size() == 1);
    const auto two = ValidateStageParams("nr", {{"intensity", -1}, {"style", "loud"}});
    REQUIRE(two.size() == 2);
    CHECK(two[0].find("nr.intensity") != std::string::npos);
    CHECK(two[1].find("nr.style") != std::string::npos);
}

TEST_CASE("EffectiveStageParams fills the defaults: an explicit default and an absent key fingerprint alike", "[unit][schema]") {
    const nlohmann::json a = EffectiveStageParams("nr", {{"backend", "stub"}});
    const nlohmann::json b = EffectiveStageParams("nr", {{"backend", "stub"}, {"intensity", 1.0}, {"passes", 1}});
    CHECK(a == b);
    CHECK(a["intensity"] == 1.0);
    CHECK(a["style"] == "natural");
    CHECK(a.contains("format"));
    const nlohmann::json c = EffectiveStageParams("nr", {{"backend", "stub"}, {"custom_key", 7}});
    CHECK(c["custom_key"] == 7);  // unknown keys survive
    CHECK(EffectiveStageParams("nr", nlohmann::json()).contains("intensity"));
    CHECK(EffectiveStageParams("bogus", {{"x", 1}}) == nlohmann::json({{"x", 1}}));
    ToolInfo tool;
    tool.backend = "stub";
    CHECK(ComputeFingerprint("nr", "sha256:x", a, {}, tool).value == ComputeFingerprint("nr", "sha256:x", b, {}, tool).value);
    CHECK(ComputeFingerprint("nr", "sha256:x", a, {}, tool).value !=
          ComputeFingerprint("nr", "sha256:x", EffectiveStageParams("nr", {{"backend", "stub"}, {"intensity", 1.4}}), {}, tool).value);
    CHECK(ParamValueLabel(*FindStageSchema("upscale")->Find("backend"), "dlss") == "DLSS SR");
    CHECK(ParamValueLabel(*FindStageSchema("upscale")->Find("scale"), 2) == "×2");
    CHECK(ParamValueLabel(*FindStageSchema("upscale")->Find("scale"), 1.5) == "×1.5");
    CHECK(ParamValueLabel(*FindStageSchema("nr")->Find("intensity"), 1.4) == "1.4");
    CHECK(ParamValueLabel(*FindStageSchema("nr")->Find("intensity"), 1.0) == "1");
    CHECK(ParamValueLabel(*FindStageSchema("nr")->Find("guides"), true) == "да");
    CHECK(ParamValueLabel(*FindStageSchema("depth")->Find("model"), "da3mono-large") == "da3mono-large");
}

// TASK-0022: parameters scoped to a backend (the TensorRT upscaler's model / tile) exist for that backend only — other
// backends neither get their defaults nor keep a stale value, so their passes' fingerprints did not change.
TEST_CASE("EffectiveStageParams keeps backend-scoped parameters for their backend only", "[unit][schema]") {
    const nlohmann::json dlss = EffectiveStageParams("upscale", {{"backend", "dlss"}, {"model", "realesrgan-x2plus"}, {"tile", 256}});
    CHECK(!dlss.contains("model"));
    CHECK(!dlss.contains("tile"));
    CHECK(!dlss.contains("models_dir"));
    CHECK(dlss["backend"] == "dlss");
    const nlohmann::json bare = EffectiveStageParams("upscale", nlohmann::json::object());  // the default backend is dlss
    CHECK(!bare.contains("model"));
    const nlohmann::json trt = EffectiveStageParams("upscale", {{"backend", "trt"}});
    CHECK(trt["model"] == "realesrgan-x2plus");
    CHECK(trt["tile"] == 0);
    CHECK(trt["models_dir"] == "");
    const nlohmann::json chosen = EffectiveStageParams("upscale", {{"backend", "trt"}, {"model", "realesr-general-x4v3"}});
    CHECK(chosen["model"] == "realesr-general-x4v3");
    // the scoping is declared on the specs themselves
    const StageSchema* s = FindStageSchema("upscale");
    REQUIRE(s);
    REQUIRE(s->Find("model"));
    CHECK(s->Find("model")->backends == std::vector<std::string>{"trt"});
    CHECK(s->Find("window")->backends == std::vector<std::string>{"worker"});
    const nlohmann::json worker = EffectiveStageParams("upscale", {{"backend", "worker"}});
    CHECK(!worker.contains("model"));
    CHECK(worker["window"] == 0);
    CHECK(worker["models_dir"] == "");
    CHECK(s->Find("backend")->backends.empty());
}

// TASK-0039: the form lists the VDA models for vda and worker:vda; the key, its default and the window keep every
// fingerprint made before (the literals are EffectiveStageParams of main before the change).
TEST_CASE("Depth model: VDA lists Small / Base / Large for vda and worker:vda, the defaults and the fingerprints stay", "[unit][schema]") {
    const StageSchema& depth = *FindStageSchema("depth");
    const ParamSpec& model = *depth.Find("model");
    const auto* vda = FormChoicesFor(model, "vda");
    REQUIRE(vda != nullptr);
    REQUIRE(vda->size() == 3);
    CHECK((*vda)[0].value.empty());
    CHECK((*vda)[1].value == "metric-vda-base");
    CHECK((*vda)[2].value == "metric-vda-large");
    CHECK(FormChoicesFor(model, "worker:vda") != nullptr);
    CHECK(FormChoicesFor(model, "da3") == nullptr);
    CHECK(ParamValueLabel(model, "metric-vda-large", "vda").find("CC-BY-NC-4.0") != std::string::npos);
    CHECK(ParamValueLabel(model, "", "vda").rfind("Small", 0) == 0);
    CHECK(ParamValueLabel(model, "da3mono-large", "da3") == "da3mono-large");
    CHECK(ParamValueLabel(*depth.Find("window"), nlohmann::json(), "vda") == "по умолчанию");
    CHECK(ValidateStageParams("depth", {{"backend", "worker:vda"}}).empty());
    CHECK(ValidateStageParams("depth", {{"backend", "vda"}, {"window", 40}}).size() == 1);

    const nlohmann::json before = {{"backend", "vda"}, {"format", "exr"},   {"fp32", false}, {"input_size", 518}, {"max_res", 1080},
                                   {"model", ""},      {"models_dir", ""}, {"no_dlss", false}, {"no_fill", false}, {"python", ""},
                                   {"stabilize", "auto"}, {"stabilize_window", 8}, {"z_far", 1000.0}, {"z_near", 0.1}};
    CHECK(EffectiveStageParams("depth", {{"backend", "vda"}}) == before);  // no window key: VDA-Small passes are reused
    nlohmann::json da3 = before;
    da3["backend"] = "da3";
    CHECK(EffectiveStageParams("depth", {{"backend", "da3"}}) == da3);
    CHECK(EffectiveStageParams("depth", {{"backend", "vda"}, {"window", 0}}) == before);  // 0 = the registry's: the same run
    CHECK(EffectiveStageParams("depth", {{"backend", "vda"}, {"window", 16}})["window"] == 16);
    CHECK(!EffectiveStageParams("depth", {{"backend", "da3"}, {"window", 16}}).contains("window"));
}

TEST_CASE("ParamsForBackendChange drops a model of the previous backend's list", "[unit][schema]") {
    const nlohmann::json large = {{"backend", "vda"}, {"model", "metric-vda-large"}, {"window", 16}};
    CHECK(!ParamsForBackendChange("depth", large, "da3").contains("model"));          // a VDA model means nothing to DA3
    CHECK(ParamsForBackendChange("depth", large, "worker:vda")["model"] == "metric-vda-large");  // the same list
    CHECK(ParamsForBackendChange("depth", large, "da3")["window"] == 16);  // kept: OnlyFor drops it from the run
    CHECK(!ParamsForBackendChange("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}, "vda").contains("model"));
    CHECK(ParamsForBackendChange("depth", {{"backend", "vda"}, {"model", ""}}, "da3")["model"] == "");
    CHECK(ParamsForBackendChange("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}, "stub")["model"] == "da3mono-large");
    CHECK(StageBackend("depth", nlohmann::json::object()) == "da3");
    CHECK(StageBackend("depth", {{"backend", "vda"}}) == "vda");
}
