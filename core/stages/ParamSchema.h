#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
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

// A condition on another parameter of the stage (effective value): `key` op `value`. Conditions of a list are ANDed.
struct ParamCondition {
    enum class Op : uint8_t { Eq, Ne, Gt };
    std::string key;
    Op op = Op::Eq;
    nlohmann::json value;
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
    // An engineer key of the algorithm (NIS sharpness, the DLSS preset, the RIFE model, the pass format, nr's backend): a pipeline
    // preset holds it with the form keys (PR-9, the user's answer). An engineer key without it is bound to the machine - a path
    // (models_dir, python) or a size for the GPU's memory (max_res, input_size, tile, fp32) - and no preset holds it. Off by
    // default, so a key added later goes into no preset until it is decided (a test names the machine keys).
    bool tuning = false;
    // When set, the parameter belongs to these backends only: EffectiveStageParams drops it for the others, so a
    // parameter added for one backend leaves the fingerprints of the other backends' passes untouched.
    std::vector<std::string> backends;
    // A list for the form per backend (depth.model: the VDA models for vda / worker:vda). It does not restrict the
    // values (JSON and CLI take any registry id) and the key stays for every backend; the form shows the row only for
    // a backend with a list.
    std::map<std::string, std::vector<ParamChoice>> formChoices;
    // EffectiveStageParams leaves the key out when absent or equal to the default (depth.window: 0 = the registry's), so
    // a parameter added later leaves the fingerprints of the passes made before it untouched.
    bool omitDefault = false;

    // The form (TASK-0046, docs/ux-guidelines.md): a stage card shows the non-advanced keys in groups; `more` ones sit
    // in the collapsed «Ещё настройки». Visibility and availability of a row follow the other parameters and the plan's
    // inputs (RowState); a row that cannot act is shown disabled with a note rather than hidden when the user should
    // know it exists (skin structure without the automatic mask - as in Visual Enhancer).
    std::string group;                          // «Модель», «Локальные», «Маска», «Временная стабильность» ...
    bool more = false;                          // under «Ещё настройки»
    std::vector<ParamCondition> showIf;         // hidden unless all hold
    std::vector<ParamCondition> enableIf;       // disabled with `disabledNote` unless all hold
    std::string disabledNote;
    std::vector<std::string> needsInputs;       // hidden unless one of these passes is an input of the plan
    std::vector<std::string> disabledByInputs;  // disabled with `disabledByInputsNote` when one of them is an input
    std::string disabledByInputsNote;
    // A number with a special value outside its range (nr.skin_structure: -1 = «as the local structure», else 0..2):
    // valid is `sentinel` or [rangeMin, max]; the form shows a check box «отдельно» next to the slider.
    std::optional<double> sentinel;
    std::string sentinelLabel;
    std::string sentinelFollows;  // the parameter the special value stands for (skin_structure -1: local_structure)
    double rangeMin = 0;
    std::string zeroLabel;  // shown at the minimum instead of the number («выкл»)
    std::string suffix;     // after the number (« EV»)
    int decimals = -1;      // digits a value is stored with; -1: from the step
};

struct StageSchema {
    std::string stage;     // scenes | depth | flow | upscale | nr | fg
    std::string title;     // «Глубина»
    std::string subtitle;  // «guide для DLSS, NR и FG»
    std::vector<ParamSpec> params;
    std::vector<std::string> guides;  // passes the stage reads as guides (its quality depends on them)
    bool cardPresets = false;         // the card offers presets of its own (PR-9: nr only; a pipeline preset takes every card)
    const ParamSpec* Find(const std::string& key) const;
};

// Where a key sits (TASK-0046): Basic - the card's form, More - under «Ещё настройки», Engineer - the JSON editor only.
enum class ParamLevel : uint8_t { Basic, More, Engineer };
ParamLevel LevelOf(const ParamSpec& spec);

const std::vector<StageSchema>& StageSchemas();
const StageSchema* FindStageSchema(const std::string& stage);

// The keys under the schema's spelling: the stages also take kebab-case aliases (nr.local-tone, fg.backbuffer-format)
// and nr's inverse no_guides / no-guides, which the validation and the defaults did not see - {"model-scale": 0.5} ran
// at 0.5 under a fingerprint saying 1.0 (TASK-0043). The schema key wins over an alias; the aliases are removed.
nlohmann::json NormalizeStageParams(const std::string& stage, const nlohmann::json& params);
// One problem of the parameters as data, for the GUI's Russian texts (PR-9): the kind, the key and the value; `text` is the
// English line ValidateStageParams gives. Shape: an unknown stage or parameters that are not an object (key empty).
struct ParamProblem {
    enum class Kind : uint8_t { Shape, Type, Range, Choice, Moved };
    Kind kind = Kind::Shape;
    std::string stage, key;
    nlohmann::json got;
    std::string text;
};
// Problems with the parameters (wrong type, out of range, not a choice, a moved key), after NormalizeStageParams; unknown keys
// pass (backends take extra keys).
std::vector<ParamProblem> CheckStageParams(const std::string& stage, const nlohmann::json& params);
// The texts of CheckStageParams.
std::vector<std::string> ValidateStageParams(const std::string& stage, const nlohmann::json& params);
// A key that left a stage's schema for another stage (PR-13: nr.scene_cut -> scenes.threshold), in its snake or kebab spelling:
// ValidateStageParams and ReadStageParamsFile name it instead of passing it on (unknown keys are kept for the backends, and this one
// would be hashed without an effect); the card says where it went.
struct MovedParam {
    const char* stage;
    const char* key;
    const char* to;     // "scenes.threshold"
    const char* label;  // the card label it had
};
const MovedParam* FindMovedParam(const std::string& stage, const std::string& key);
std::string MovedParamProblem(const MovedParam& m);  // "nr.scene_cut moved to scenes.threshold (PR-13): use --param scenes.threshold=..."

// The parameters normalised, with the schema defaults filled in for absent keys (unknown keys kept): what a run means.
// ProcessRunner hands exactly these to the stage and to its fingerprint (TASK-0043: the stages got the raw parameters, so
// a default that differed between the schema and the stage - nr.exposure - ran differently from what was hashed).
nlohmann::json EffectiveStageParams(const std::string& stage, const nlohmann::json& params);
// A parameter's value as its form shows it: the choice label, a number, «да» / «нет».
std::string ParamValueLabel(const ParamSpec& spec, const nlohmann::json& value);
// The same with the backend's form list first (depth.model of vda: «Large — CC-BY-NC-4.0»); a missing value of an
// omitDefault key reads «по умолчанию».
std::string ParamValueLabel(const ParamSpec& spec, const nlohmann::json& value, const std::string& backend);
// The stage's backend: params.backend, else the schema default ("" for a stage without backends).
std::string StageBackend(const std::string& stage, const nlohmann::json& params);
// The form list of the parameter for the backend, nullptr when it has none.
const std::vector<ParamChoice>* FormChoicesFor(const ParamSpec& spec, const std::string& backend);
// The parameters after a switch to `newBackend` (GUI, --depth-backend, --param depth.backend=...): a value from the old
// backend's form list that the new backend's list does not have is dropped (vda + metric-vda-large -> da3 leaves no VDA
// model), and so is a value that is not in the new backend's list; "" stays. The backend key itself is not changed.
nlohmann::json ParamsForBackendChange(const std::string& stage, const nlohmann::json& params, const std::string& newBackend);
// A file of one stage's settings: ReadStageParamsFile in stages/StagePreset.h (PR-9: a bare object or a preset).

// How the card shows one row for the stage's parameters and the plan's inputs (TASK-0046).
struct ParamRowState {
    bool visible = true;
    bool enabled = true;
    std::string note;  // why it is disabled
};
ParamRowState RowState(const ParamSpec& spec, const nlohmann::json& effectiveParams, const std::set<std::string>& inputs);
// Digits a value of the parameter is stored and shown with (decimals, else from the step: 0.05 -> 2, 0.1 -> 1).
int ParamDecimals(const ParamSpec& spec);

// The options of `dlssvid <stage>` for the parameters (the GUI's «Запустить этап»; review of TASK-0038): `--<key> <value>`
// with the schema key (the commands take the underscore spelling as an alias); a flag only when it differs from its
// default - `--<key>` when set, `--no-<key>` when a default-on flag is cleared (jitter, guides); empty strings are left
// out; upscale's target_width + target_height become `--target WxH`; a parameter of another backend (OnlyFor) and an
// omitDefault key at its default are left out (`depth --backend da3` never gets the window of a VDA setting).
std::vector<std::string> StageCliArgs(const std::string& stage, const nlohmann::json& params);

}  // namespace dlssvid
