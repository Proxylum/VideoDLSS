// Stage parameter schema (stage 9, MR D): every stage described, defaults mirror the stages, validation catches wrong
// types / ranges / choices, effective parameters make an explicit default and an absent key the same run.

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

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
    CHECK(FindStageSchema("flow")->Find("backend")->def == FlowStageOptions{}.backend);
    // human labels for the form, technical keys stay keys; the main form of nr is the intensity alone
    CHECK(FindStageSchema("nr")->Find("intensity")->label == "Интенсивность");
    CHECK(FindStageSchema("upscale")->Find("scale")->widget == ParamWidget::Toggle);
    CHECK(FindStageSchema("upscale")->Find("scale")->choices.size() == 3);
    int visible = 0;
    for (const auto& p : FindStageSchema("nr")->params) visible += p.advanced ? 0 : 1;
    CHECK(visible == 1);
    CHECK(FindStageSchema("nr")->guides == std::vector<std::string>{"depth_dlss", "mv_dlss"});
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
