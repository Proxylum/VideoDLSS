// Stage presets (PR-9): what a card preset and a pipeline preset hold of a stage, how they apply, how a card is reset.

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include "stages/ParamSchema.h"
#include "stages/StagePreset.h"

using namespace dlssvid;

// The rule of the card's «Сбросить» moved from StageCard to the core unchanged (the app tests of the NR and depth cards hold the
// same cases through the GUI)
TEST_CASE("ResetStageParams keeps the backend, engineer keys, unknown keys and a list key without a list", "[unit][presets][regression]") {
    const nlohmann::json nr = {{"backend", "ngx"}, {"format", "png"}, {"intensity", 1.4}, {"exposure", 1.0}, {"auto_mask", true}, {"extra_key", 3}};
    CHECK(ResetStageParams("nr", nr) == nlohmann::json{{"backend", "ngx"}, {"format", "png"}, {"extra_key", 3}});
    // aliases are folded first: an alias of a form key goes with it, the inverse no_guides too
    CHECK(ResetStageParams("nr", {{"skin-structure", 0.5}, {"no_guides", true}, {"model-scale", 0.5}}) == nlohmann::json::object());
    // depth: the backend stays; a DA3 model from the JSON has no row under DA3 and stays; a VDA model under VDA has one and goes
    CHECK(ResetStageParams("depth", {{"backend", "da3"}, {"model", "da3mono-large"}, {"stabilize", "none"}}) ==
          nlohmann::json{{"backend", "da3"}, {"model", "da3mono-large"}});
    CHECK(ResetStageParams("depth", {{"backend", "vda"}, {"model", "metric-vda-large"}, {"window", 16}, {"z_near", 0.5}}) ==
          nlohmann::json{{"backend", "vda"}, {"z_near", 0.5}});
    // the backend alone: nothing to reset
    CHECK(ResetStageParams("depth", {{"backend", "vda"}}) == nlohmann::json{{"backend", "vda"}});
    CHECK(ResetStageParams("upscale", {{"scale", 3}, {"sharpness", 0.8}, {"models_dir", "m"}}) == nlohmann::json{{"sharpness", 0.8}, {"models_dir", "m"}});
    CHECK(ResetStageParams("scenes", {{"threshold", 0.3}, {"backend", "luma"}}) == nlohmann::json{{"backend", "luma"}});
    CHECK(ResetStageParams("nr", nlohmann::json()) == nlohmann::json::object());
    CHECK(ResetStageParams("bogus", {{"a", 1}}) == nlohmann::json{{"a", 1}});  // nothing known: everything stays
    for (const StageSchema& st : StageSchemas())
        for (const ParamSpec& p : st.params) {
            INFO(st.stage << "." << p.key);
            CHECK(CardResetKey(p) == (!p.advanced && p.key != "backend"));
        }
}
