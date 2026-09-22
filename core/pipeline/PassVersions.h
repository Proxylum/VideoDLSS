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

struct Manifest;

// Pass versions (stage 9, MR A — docs/architecture.md «Отпечатки и версии пассов»). The current version of a pass
// lives in <root>/<pass>/ so every reader keeps working; when a run with another fingerprint replaces it, the folder
// is moved aside to <root>/<pass>.v/<YYYYMMDD-HHMMSS_<fp8>>/. Switching a version swaps the folders. `gc` keeps the
// newest N versions per pass (the current one counts) and never removes a version that any pass, current or
// historical, lists among its inputs. Folders without a fingerprint (made before MR A) are simply «current».

struct PassVersionInfo {
    std::string pass;
    std::string id;  // history folder name; empty for the current version
    bool current = false;
    std::filesystem::path dir;
    std::string fingerprint;  // empty for legacy passes
    std::string created;      // ISO-8601 UTC: the manifest's `created`, else the manifest file time
    int64_t frames = 0;
    uint64_t bytes = 0;
    nlohmann::json params = nlohmann::json::object();  // canonical parameters (empty for legacy passes)
    nlohmann::json tool = nlohmann::json::object();
    std::map<std::string, std::string> inputs;
    nlohmann::json ToJson() const;
};

std::filesystem::path PassHistoryDir(const std::filesystem::path& root, const std::string& pass);  // <root>/<pass>.v
// "<YYYYMMDD-HHMMSS>_<fp8>" of a pass folder (fp8 = "legacy" without a fingerprint).
std::string PassVersionId(const Manifest& m, const std::filesystem::path& dir);
uint64_t DirectoryBytes(const std::filesystem::path& dir);

// Names of the pass folders (with a manifest) directly under root, sorted.
std::vector<std::string> PassesUnderRoot(const std::filesystem::path& root);
std::optional<PassVersionInfo> CurrentPassVersion(const std::filesystem::path& root, const std::string& pass);
// One pass (or every pass with a current version or a history): the current version first, then history newest first.
std::vector<PassVersionInfo> ListPassVersions(const std::filesystem::path& root, const std::string& pass = {});

// Moves <root>/<pass>/ into the history; returns its id, or nothing when the folder does not exist. A folder without
// a manifest (an interrupted run) is moved as "<time>_partial".
std::optional<std::string> RetirePassVersion(const std::filesystem::path& root, const std::string& pass);
// Makes history version `id` (or its unique prefix) the current one; the previous current version goes to the
// history. Returns the id it was retired under. Throws when the version is unknown or ambiguous.
std::optional<std::string> UsePassVersion(const std::filesystem::path& root, const std::string& pass, const std::string& id);
// Fingerprints that any version of any pass under root lists among its inputs.
std::set<std::string> ReferencedFingerprints(const std::filesystem::path& root);

struct PassGcReport {
    std::vector<PassVersionInfo> removed;
    uint64_t bytes = 0;
};
// Removes history versions beyond the newest `keep - 1` per pass (the current version counts) unless they are
// referenced; keep <= 0 removes nothing. Passes are processed from the end of the pipeline so that a removed
// color_fg version releases the color_nr version it referenced in the same call.
PassGcReport GcPassVersions(const std::filesystem::path& root, int keep, const std::string& pass = {}, bool dryRun = false);

}  // namespace dlssvid
