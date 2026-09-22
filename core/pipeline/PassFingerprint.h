#pragma once

#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dlssvid {

// The «app» part of the tool in a pass fingerprint. It is NOT the release version (DLSSVID_VERSION): a release that
// changes nothing in what a stage writes must not invalidate every pass on disk. Bump it only when a stage's output
// changes for the same parameters, inputs, model and DLL (a new tonemap, a different NIS, …).
constexpr const char* kPassToolVersion = "0.1.0";

// Pass fingerprints (stage 9, MR A — docs/architecture.md «Отпечатки и версии пассов»). A pass is reused only when the
// fingerprint of the run that would produce it equals the one stored in its manifest:
//   fingerprint = sha256(canonical JSON of {source hash, stage, canonical parameters, input pass fingerprints, tool})
// so a changed parameter, input, source or tool invalidates the pass and, through the inputs, everything below it.

// Canonical JSON: object keys sorted, integral floats written as integers (2.0 == 2), no whitespace.
nlohmann::json CanonicalizeJson(const nlohmann::json& j);
std::string CanonicalJson(const nlohmann::json& j);

// What computes a pass, as far as it is known before the run: application version, backend, model id and its registry
// hash, the NVIDIA DLL the backend loads (sha256 of the file found in the search path).
struct ToolInfo {
    std::string app;        // kPassToolVersion: the pass algorithm version, not the release
    std::string backend;    // the stage's backend after defaults
    std::string model;      // registry id (depth / flow / rife); empty when the backend has no model
    std::string modelHash;  // sha256 from models/registry.json when the entry lists one
    std::string dllFile;    // nvngx_dlss.dll | nvngx_dlssnr.dll | nvngx_dlssg.dll (NGX backends only)
    std::string dll;        // "sha256:<hex>" of that DLL; empty when it is not installed
    nlohmann::json ToJson() const;
    static ToolInfo FromJson(const nlohmann::json& j);
    bool operator==(const ToolInfo&) const = default;
};
// Resolves the tool identity of a stage from its parameters (backend, model, models_dir, dll_dir) without running it.
ToolInfo ToolForStage(const std::string& stage, const nlohmann::json& params);

struct PassFingerprint {
    std::string value;                          // "sha256:<hex>"
    std::string sourceHash;                     // "sha256:<hex>" of the source video
    std::string stage;                          // depth | flow | upscale | nr | fg
    nlohmann::json params;                      // canonical stage parameters
    std::map<std::string, std::string> inputs;  // input pass -> its fingerprint (legacy passes: "legacy:<hash>:<frames>")
    ToolInfo tool;
    std::string Short() const;  // first 8 hex digits of the value
};
PassFingerprint ComputeFingerprint(const std::string& stage, const std::string& sourceHash, const nlohmann::json& params,
                                   const std::map<std::string, std::string>& inputs, const ToolInfo& tool);
// First 8 hex digits of a "sha256:<hex>" value ("legacy" when empty).
std::string ShortFingerprint(const std::string& fingerprint);

// sha256 of a file, cached per process by (path, size, mtime): NVIDIA DLLs are hashed once, not on every plan.
std::string Sha256FileCached(const std::filesystem::path& path);

// Pass folders a stage writes under the passes root, in order; the last one is the one checked for completeness.
std::vector<std::string> StageOutputPasses(const std::string& stage, const nlohmann::json& params);
// Every pass folder a stage may write (depth: depth_raw and depth_dlss), whatever the parameters.
const std::vector<std::string>& StageFamilyPasses(const std::string& stage);
// The stage that writes a pass folder ("" for masks and unknown names).
std::string StageForPass(const std::string& pass);
// Pass folders a stage reads when they exist: one group per input, the first existing member of a group is the one
// used (fg: color_nr, else color_sr) — mirrors ExistingPass() in ProcessRunner.
std::vector<std::vector<std::string>> StageInputCandidates(const std::string& stage);

// Current UTC time as ISO-8601 ("2026-09-22T14:09:31Z") — the manifest's `created`.
std::string NowIso8601();
// A file's last write time as ISO-8601 UTC (empty when the file does not exist).
std::string FileTimeIso8601(const std::filesystem::path& path);
// "YYYYMMDD-HHMMSS" from an ISO-8601 timestamp (empty when it does not parse).
std::string CompactTimestamp(const std::string& iso8601);

}  // namespace dlssvid
