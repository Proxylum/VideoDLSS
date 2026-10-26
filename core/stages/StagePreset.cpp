#include "stages/StagePreset.h"

namespace dlssvid {

namespace {

// a list key without a list for the backend has no row in the card (StageCard: the form neither shows nor counts it)
bool NoRow(const ParamSpec& spec, const std::string& backend) { return !spec.formChoices.empty() && !FormChoicesFor(spec, backend); }

}  // namespace

bool CardResetKey(const ParamSpec& spec) { return !spec.advanced && spec.key != "backend"; }

nlohmann::json ResetStageParams(const std::string& stage, const nlohmann::json& current) {
    const nlohmann::json cur = current.is_object() ? NormalizeStageParams(stage, current) : nlohmann::json::object();
    const StageSchema* schema = FindStageSchema(stage);
    const std::string backend = StageBackend(stage, cur);
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [key, value] : cur.items()) {
        const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
        if (!spec || !CardResetKey(*spec) || NoRow(*spec, backend)) out[key] = value;
    }
    return out;
}

}  // namespace dlssvid
