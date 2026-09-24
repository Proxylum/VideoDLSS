#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dlssvid {

// Stage parameter schema (stage 9, MR D; docs/ux-guidelines.md principle 3 «формы, не JSON»): one description of every
// stage parameter — type, range, choices, human label and hint, default — shared by the validation in ProcessRunner /
// CLI, the fingerprint (defaults filled in: an explicit default and an absent key mean the same run) and the forms of
// the GUI. The stages keep applying their own defaults; the schema mirrors them (tests check the visible ones).
enum class ParamType : uint8_t { Bool, Int, Float, Enum, String };
// How a form shows the parameter: Select (list), Toggle (a row of exclusive buttons: ×1.5 ×2 ×3), Slider (with a
// number), Spin, Check, Text.
enum class ParamWidget : uint8_t { Check, Spin, Slider, Select, Toggle, Text };

struct ParamChoice {
    std::string value;  // JSON value as text ("dlss", "2", "1.5")
    std::string label;  // human label («DLSS SR»)
    std::string hint;
};

struct ParamSpec {
    std::string key;  // JSON key in the project's stage params
    ParamType type = ParamType::String;
    ParamWidget widget = ParamWidget::Text;
    std::string label;  // «Интенсивность»
    std::string hint;
    nlohmann::json def;                // default (what the stage assumes when the key is absent)
    double min = 0, max = 0, step = 0;  // Int / Float (max <= min: no range)
    std::vector<ParamChoice> choices;  // Enum, or a fixed set of numbers (Toggle over 1.5 / 2 / 3)
    bool advanced = false;             // engineer mode only (the JSON editor); validated all the same
    // When set, the parameter belongs to these backends only: EffectiveStageParams drops it for the others, so a
    // parameter added for one backend leaves the fingerprints of the other backends' passes untouched.
    std::vector<std::string> backends;
};

struct StageSchema {
    std::string stage;     // depth | flow | upscale | nr | fg
    std::string title;     // «Глубина»
    std::string subtitle;  // «guide для DLSS, NR и FG»
    std::vector<ParamSpec> params;
    std::vector<std::string> guides;  // passes the stage reads as guides (its quality depends on them)
    const ParamSpec* Find(const std::string& key) const;
};

const std::vector<StageSchema>& StageSchemas();
const StageSchema* FindStageSchema(const std::string& stage);

// Problems with the parameters (wrong type, out of range, not a choice); unknown keys pass (backends take extra keys).
std::vector<std::string> ValidateStageParams(const std::string& stage, const nlohmann::json& params);
// The parameters with the schema defaults filled in for absent keys (unknown keys kept): what a run means.
nlohmann::json EffectiveStageParams(const std::string& stage, const nlohmann::json& params);
// A parameter's value as its form shows it: the choice label, a number, «да» / «нет».
std::string ParamValueLabel(const ParamSpec& spec, const nlohmann::json& value);

}  // namespace dlssvid
