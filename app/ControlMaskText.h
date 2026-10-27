#pragma once

#include <QString>
#include <nlohmann/json.hpp>
#include <string>

#include "pipeline/ControlMask.h"

namespace dlssvid {

// The control mask's words in the GUI (PR-8, docs/ux-guidelines.md): a refusal of the import by its kind (the core's English detail
// goes to the tooltip), why a folder is no control mask, and a version's line in the versions panel.
QString ControlMaskIssueText(const ControlMaskIssue& issue);
QString ControlMaskWhyText(const std::string& why);              // ReadControlMaskInfo's why
QString HumanControlMaskParams(const nlohmann::json& stageParams);  // «mask.png · 1920×1080 · RGB»

}  // namespace dlssvid
