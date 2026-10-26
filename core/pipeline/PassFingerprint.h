#pragma once

#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dlssvid {

struct Manifest;

// The «app» part of the tool in a pass fingerprint. It is NOT the release version (DLSSVID_VERSION): a release that
// changes nothing in what a stage writes must not invalidate every pass on disk. Bump it only when a stage's output
// changes for the same parameters, inputs, model and DLL (a new tonemap, a different NIS, …) in every stage; when one
// stage changes, give it a revision in PassToolVersionFor instead - the passes of the other stages stay valid.
constexpr const char* kPassToolVersion = "0.1.0";

// The «app» of one stage's passes: kPassToolVersion plus "+<stage>.<revision>" for a stage whose own algorithm changed
// for the same parameters:
// - depth.2 (TASK-0038, audit depth-7): temporal alignment over a centred window of ±N frames, auto = scale for metric models -
//   only when stabilize != none (without alignment the depth is what it was);
// - nr.ev (TASK-0043): exposure in stops; png.2 (TASK-0042): PNG16 colour passes scaled;
// - nr.gain / nr.i0 (PR-7): nr.intensity is the edit strength after the model (it went to NGX before) - any value but 1 is
//   another picture; 0 does not call the model at all.
std::string PassToolVersionFor(const std::string& stage, const nlohmann::json& params);
// The revisions `newApp` has and `oldApp` lacks ("0.1.0", "0.1.0+nr.gain" -> {"nr.gain"}): what the card names on tool_changed.
std::vector<std::string> AddedToolRevisions(const std::string& oldApp, const std::string& newApp);

// Pass fingerprints (stage 9, MR A — docs/architecture.md «Отпечатки и версии пассов»). A pass is reused only when the
// fingerprint of the run that would produce it equals the one stored in its manifest:
//   fingerprint = sha256(canonical JSON of {source hash, stage, canonical parameters, input pass fingerprints, tool})
// so a changed parameter, input, source or tool invalidates the pass and, through the inputs, everything below it.

// Canonical JSON: object keys sorted, integral floats written as integers (2.0 == 2), other floats rounded to 9
// significant digits (3 x 0.1 == 0.3, TASK-0045), no whitespace.
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
// Resolves the tool identity of a stage from its parameters (backend, model, models_dir) without running it. The NVIDIA
// DLL is the one the stage loads (DLSSVID_NVIDIA_DLL_DIR, <exe>/nvidia, <exe>): params.dll_dir is read by no stage of
// process, so it is not hashed (TASK-0045).
ToolInfo ToolForStage(const std::string& stage, const nlohmann::json& params);

struct PassFingerprint {
    std::string value;                          // "sha256:<hex>"
    std::string sourceHash;                     // "sha256:<hex>" of the source video
    std::string stage;                          // depth | flow | upscale | nr | fg
    nlohmann::json params;                      // canonical stage parameters
    std::map<std::string, std::string> inputs;  // input pass -> its fingerprint (legacy passes: "legacy:<hash>:<frames>[:y_up]")
    ToolInfo tool;
    std::string Short() const;  // first 8 hex digits of the value
};
// `sourceFormat` is the layout the source decodes into (FrameDesc::Name(): "yuv444p10" ...): a stage reads different
// samples from a 10-bit source than from its 8-bit 4:2:0 reduction, so it is part of the fingerprint — except the
// legacy layout "yuv420p" (and an empty string), which leaves the fingerprints of every 8-bit source as they were.
PassFingerprint ComputeFingerprint(const std::string& stage, const std::string& sourceHash, const nlohmann::json& params,
                                   const std::map<std::string, std::string>& inputs, const ToolInfo& tool, const std::string& sourceFormat = {});
// The source layout a fingerprint stands for when the manifest does not say: 8-bit 4:2:0, what every pass was made from
// before the decoder kept the source's bit depth.
constexpr const char* kLegacySourceFormat = "yuv420p";
// First 8 hex digits of a "sha256:<hex>" value ("legacy" when empty).
std::string ShortFingerprint(const std::string& fingerprint);

// sha256 of a file, cached per process by (path, size, mtime): NVIDIA DLLs are hashed once, not on every plan.
std::string Sha256FileCached(const std::filesystem::path& path);

// The stages of `process` in run order: one table for the order, StageForPass, the gc order, the stage cards and the error texts
// (PR-13: a new stage is added here once).
const std::vector<std::string>& PipelineStageNames();
std::string PipelineStageNamesText();  // "(scenes | depth | flow | upscale | nr | fg)"
// Whether the stage's output depends on the scene-cut list (PR-13): it keeps a history or interpolates across frames. The one gate
// of StageInputCandidates, the process runner (through the decision's inputs), the standalone commands and the GUI.
bool StageReadsSceneCuts(const std::string& stage, const nlohmann::json& params);
// Pass folders a stage writes under the passes root, in order; the last one is the one checked for completeness.
std::vector<std::string> StageOutputPasses(const std::string& stage, const nlohmann::json& params);
// Every pass folder a stage may write (depth: depth_raw and depth_dlss), whatever the parameters.
const std::vector<std::string>& StageFamilyPasses(const std::string& stage);
// The stage that writes a pass folder ("" for masks and unknown names).
std::string StageForPass(const std::string& pass);
// Pass folders a stage reads when they exist: one group per input, the first existing member of a group is the one
// used (fg: color_nr, else color_sr) — mirrors ExistingPass() in ProcessRunner. With the stage's (effective) parameters:
// nr with guides off reads no depth_dlss / mv_dlss, so they are no inputs of its fingerprint (TASK-0045: a new depth
// recomputed an NR pass that never read it).
std::vector<std::vector<std::string>> StageInputCandidates(const std::string& stage, const nlohmann::json& params = nlohmann::json::object());

// The fingerprint of an imported pass from its content (`dlssvid import --stamp`, TASK-0045): "sha256:" of the canonical
// {"pass", "layout": the manifest without its provenance (size, format, channels, mv.y_up, depth range), "frames": [sha256
// of each frame file]}. A pass without a fingerprint stands for its source and length only, so a mask re-imported with
// other content and the same frame count reused the NR pass made from the old one.
std::string ContentFingerprint(const std::filesystem::path& dir, const Manifest& manifest);

// Current UTC time as ISO-8601 ("2026-09-22T14:09:31Z") — the manifest's `created`.
std::string NowIso8601();
// A file's last write time as ISO-8601 UTC (empty when the file does not exist).
std::string FileTimeIso8601(const std::filesystem::path& path);
// "YYYYMMDD-HHMMSS" from an ISO-8601 timestamp (empty when it does not parse).
std::string CompactTimestamp(const std::string& iso8601);

}  // namespace dlssvid
