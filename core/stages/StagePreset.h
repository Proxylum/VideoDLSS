#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "io/OutputFormat.h"
#include "stages/ParamSchema.h"
#include "util/Error.h"

namespace dlssvid {

// Presets (PR-9, docs/conventions.md «Пресеты dlssvid-preset»): a named file of settings. A card preset (scope "stage") holds one
// stage's form keys - the NR card's «Пресет ▾»; a pipeline preset (scope "pipeline") holds every stage with its switch, its form
// keys and the engineer keys of the algorithm (ParamSpec::tuning), and the «Кодирование» card. Without Qt: the GUI, the CLI and the
// tests share it. Values are checked when a preset is read: a preset with a value out of range is refused whole.
inline constexpr const char* kPresetFormat = "dlssvid-preset";
inline constexpr int kPresetSchemaVersion = 1;
inline constexpr std::uintmax_t kPresetMaxBytes = 1u << 20;
inline constexpr size_t kPresetNameMax = 120;  // code points

enum class PresetScope : uint8_t { Stage, Pipeline };  // the field `scope`: "stage" | "pipeline"
const char* PresetScopeName(PresetScope s);
inline constexpr const char* kPipelineTarget = "pipeline";  // the library's and the slots' key of pipeline presets; not a stage name

// The «Кодирование» card as a pipeline preset holds it: the codec, the bit depth and chroma, the bitrate (encode.options.b, "" =
// «Авто»). The settings note and the other encoder options have no row in the card and stay the project's.
struct PresetEncode {
    std::string codec = "h264_nvenc";
    OutputFormatRequest format;
    std::string bitrate;
    bool operator==(const PresetEncode&) const = default;
};
struct PresetStage {
    nlohmann::json params = nlohmann::json::object();  // the preset's keys of the stage, normalised, checked, canonical
    bool enabled = true;                                // pipeline presets only
};
struct Preset {
    std::string name, created, app;
    PresetScope scope = PresetScope::Stage;
    std::map<std::string, PresetStage> stages;  // Stage: exactly one; Pipeline: the stages the file names (read for one stage: that one)
    std::optional<PresetEncode> encode;       // Pipeline only; nullopt: the file has none (the project's encoder stays)
    const std::string& stage() const;           // Stage scope: its stage ("" when there is none)
};

enum class PresetIssueKind : uint8_t {
    // fatal: nothing is applied
    Unreadable, TooLarge, NotJson, NotPreset, BadSchema, NewerSchema, BadScope, BadName, BadStages, BadValue, NoStage,
    // notes: the rest is applied
    UnknownKey, EngineerKey, MovedKey, ForeignKey, UnknownStage, OtherStages, PipelineFields, MissingStage, NoEncode, EncodeField, UnknownField,
    SlotValue
};
// Strict: every preset a person makes, imports, exports or a command line reads. Slot: the «Перед …» snapshots (auto/) - a value
// problem is the note SlotValue and the value stays: a slot gives back what the project held, its errors included.
enum class PresetRead : uint8_t { Strict, Slot };
struct PresetIssue {
    PresetIssueKind kind = PresetIssueKind::NotJson;
    std::string stage, key;               // stage "encode" for the encoder; key: the field (encode: codec | bit_depth | chroma | b)
    std::string to;                       // MovedKey: where the key went; ForeignKey of a stage: the backend; EncodeField: "options" for an option
    std::string codec;                    // encode: the codec the value was checked against
    std::vector<std::string> stages;      // OtherStages: the stages left out; BadScope of a card preset: the stages it has
    std::optional<ParamProblem> problem;  // BadValue / SlotValue of a stage
    nlohmann::json got;                   // the value (BadValue of encode, an engineer / unknown key)
    int schema = 0;                       // NewerSchema
    std::string text;                     // English, as the CLI prints it
    bool fatal() const;
};
struct PresetLoad {
    Preset preset;               // what could be read: the name and the scope also of a refused file (empty when they could not be)
    bool scopeKnown = false;     // the envelope gave a scope
    std::vector<PresetIssue> issues;
    bool ok() const;             // no fatal issue
    bool Has(PresetIssueKind kind) const;
    std::string problemsText() const;  // the fatal texts joined with "; "
};
class PresetError : public Error {
public:
    PresetError(const std::string& what, std::vector<PresetIssue> issues) : Error(what), issues(std::move(issues)) {}
    std::vector<PresetIssue> issues;
};

// keys. The class of a key for a scope, whatever the backend: Stage - the form's keys (!advanced); Pipeline - also the tuning keys.
bool PresetKeyClass(const ParamSpec& spec, PresetScope scope);
// PresetKeyClass, a key of this backend (OnlyFor) and, for a key with form lists, one with a list for it (else the card has no row)
bool IsPresetKey(const ParamSpec& spec, const std::string& backend, PresetScope scope);
std::vector<std::string> PresetKeys(const std::string& stage, const std::string& backend, PresetScope scope);  // schema order
// The keys the card's «Сбросить» returns to their defaults: the form's rows, not the backend - the method has its own ↺ (TASK-0046).
bool CardResetKey(const ParamSpec& spec);
// A key of the machine: a path (models_dir, python) or a size for its GPU memory (max_res, input_size, tile, fp32) - no preset holds it.
// The other engineer keys without `tuning` (scenes.backend: one method) are left out of presets as nothing to choose.
bool IsMachineKey(const ParamSpec& spec);
inline constexpr int kPresetMaxDepth = 64;  // JSON nesting a preset or a settings file may have (a preset needs 4)

// semantics. scope: Stage - a card preset, Pipeline - a pipeline preset's entry of the stage.
// The preset's keys of the current settings with their effective values (defaults written out), canonical.
nlohmann::json PresetParams(const std::string& stage, const nlohmann::json& current, PresetScope scope);
// A preset's keys over the current settings. A backend the preset names (a key of its class) is switched first, the old backend's
// list values dropped (ParamsForBackendChange); then the preset's class of keys is replaced - a key the preset lacks returns to its
// default - and the rest stays: the engineer keys outside the class, unknown keys, a list key without a list. Keys equal to their
// default are left out (the project stays sparse), the backend never. `ignored`: preset keys that do not apply to the backend.
nlohmann::json ApplyPreset(const std::string& stage, const nlohmann::json& current, const nlohmann::json& presetParams, PresetScope scope,
                           std::vector<std::string>* ignored = nullptr);
// The card's «Сбросить»: the parameters without the form's keys. The backend, the engineer keys, keys the schema does not know and a
// list key without a list for the backend (depth.model under DA3, from the JSON: the card has no row for it) stay. Aliases folded
// first (NormalizeStageParams): an alias of a form key goes with it.
nlohmann::json ResetStageParams(const std::string& stage, const nlohmann::json& current);
struct ParamChange {
    std::string key;
    nlohmann::json was, now;  // effective values; null: the key does not act (another backend's) or an omitDefault key at its default
    std::string wasBackend, nowBackend;
};
// The keys of PresetKeyClass(scope) whose effective values differ (numbers within 1e-9), in schema order.
std::vector<ParamChange> DiffStageParams(const std::string& stage, const nlohmann::json& before, const nlohmann::json& after, PresetScope scope);

// files. forStage: read one stage of a preset of either scope (the NR menu, `--param-file nr=`, `nr --params-file`).
PresetLoad ValidatePreset(const nlohmann::json& j, const std::string& forStage = {}, PresetRead read = PresetRead::Strict);
PresetLoad LoadPreset(const std::filesystem::path& file, const std::string& forStage = {}, PresetRead read = PresetRead::Strict);
std::string PresetNameProblem(const std::string& name);  // "" = fine (after trimming the spaces); English
std::string TrimPresetName(const std::string& name);
// A card preset of the current settings: PresetParams(Stage), now, this version.
Preset MakePreset(const std::string& name, const std::string& stage, const nlohmann::json& current);
// The file's JSON in a fixed order (the envelope, the stages in run order, enabled before params, the keys in schema order, the
// encoder's fields), every value canonical.
nlohmann::ordered_json PresetToJson(const Preset& p);
// Checks PresetToJson(p) as a reader would (throws PresetError) and writes it atomically: <file>.tmp, then a rename over the file.
void SavePreset(const std::filesystem::path& file, const Preset& p, PresetRead read = PresetRead::Strict);
// The encode section: a codec of KnownCodecs(), a bit depth and chroma the codec encodes, options.b a whole 1..400 Mbit/s when the
// codec takes a bitrate. The issues found; `out` gets the settings (in Slot mode also with the values that are wrong).
std::vector<PresetIssue> CheckEncodeSection(const nlohmann::json& encode, PresetEncode* out, PresetRead read = PresetRead::Strict);

// What a preset leaves out of the current settings, for the save dialog (principle 11): keys that are set, differ from the default,
// act under the backend and are not the preset's - engineer keys (card presets), keys of the machine (pipeline presets), paths, a
// list key without a list, unknown keys; for a pipeline preset also the encoder's options other than b and a settings note turned off.
struct LeftOut {
    enum class Why : uint8_t { Engineer, Machine, Path, NoRow, Unknown, CodecOption, SettingsNote };
    Why why = Why::Unknown;
    std::string stage, key;  // stage "encode" for the encoder
    nlohmann::json value;
};
std::vector<LeftOut> StagePresetLeftOut(const std::string& stage, const nlohmann::json& current, PresetScope scope);

// One stage's settings file (`dlssvid nr --params-file`, `process --param-file <stage>=`): a bare JSON object of the stage's keys, as
// a project's stages.<stage>.params, or a dlssvid-preset of either scope (its entry of the stage).
struct StageParamsFile {
    nlohmann::json params = nlohmann::json::object();
    bool preset = false;
    PresetScope scope = PresetScope::Stage;  // preset: which kind (a pipeline preset gives its entry of the stage)
    std::string presetName;
    std::vector<std::string> dropped;  // bare object: keys the schema lacks - the caller refuses them (TASK-0047)
    std::vector<PresetIssue> notes;    // preset: what was left out
};
// A bare object: aliases normalised (NormalizeStageParams), keys the schema does not have removed and named in `dropped`; values are
// not checked here - the caller checks what runs, the file merged with the options given. A preset: read for the stage, its values
// checked. Throws (with the file's name): unreadable, larger than 1 MiB, broken JSON, not an object, a moved key of a bare object, a
// refused preset.
StageParamsFile ReadStageParamsFile(const std::string& stage, const std::filesystem::path& file);

// A path as UTF-8 text for messages (not path.string(): the ANSI code page, which throws on a character outside it).
std::string PathText(const std::filesystem::path& p);

}  // namespace dlssvid
