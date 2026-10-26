#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "stages/ParamSchema.h"

namespace dlssvid {

// Stage presets (PR-9, docs/conventions.md «Пресеты»): what a card preset and a pipeline preset hold of a stage's settings, how
// they are applied and how a card is reset. Without Qt: the GUI, the CLI and the tests share it.

// The keys the card's «Сбросить» returns to their defaults: the form's rows, not the backend - the method has its own ↺ (TASK-0046).
bool CardResetKey(const ParamSpec& spec);
// The card's «Сбросить»: the parameters without the form's keys. The backend, the engineer keys, keys the schema does not know and a
// list key without a list for the backend (depth.model under DA3, from the JSON: the card has no row for it) stay. Aliases folded
// first (NormalizeStageParams): an alias of a form key goes with it.
nlohmann::json ResetStageParams(const std::string& stage, const nlohmann::json& current);

}  // namespace dlssvid
