#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace dlssvid {

// The settings note of a result (PR-11, spec §2g): what made the encoded colour pass, from the manifests of its chain -
// color_fg <- color_nr <- color_sr as each stage's `color_source` says - and their stageParams, written by the stage from
// the run itself (paramsCanonical of an adopted pass is the project's, not what ran). Names and numbers only: no paths.
// `finalPass`: the pass the result is encoded from ("color_fg" | "color_nr" | "color_sr"); another name gives no stages.
//   {"app": "VideoDLSS 0.4.0", "stages": {"upscale": {...}, "nr": {...}, "fg": {...}}}
nlohmann::json BuildSettingsJson(const std::filesystem::path& passesRoot, const std::string& finalPass);

// One ASCII line of at most kSettingsNoteMax bytes (the container's `comment`):
//   VideoDLSS 0.4.0 | upscale: dlss x2 | nr: ngx style=natural intensity=1.2 passes=1 ... | fg: dlssg x2
// A pass process adopted (it predates fingerprints) is marked «(adopted)».
std::string SettingsNoteText(const nlohmann::json& settings);
inline std::string BuildSettingsNote(const std::filesystem::path& passesRoot, const std::string& finalPass) {
    return SettingsNoteText(BuildSettingsJson(passesRoot, finalPass));
}
inline constexpr size_t kSettingsNoteMax = 1024;

}  // namespace dlssvid
