#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dlssvid {

struct ProcessPlan;

// The NR preview (PR-10, spec §2f phase 1): a frame or a segment of «Улучшение» with the card's settings, made by `dlssvid nr` in
// <passes root>/.preview/nr/<key>/. It is no pass and no version - no manifest at .preview or .preview/nr, no ".v" - so the plan, the
// pass versions, gc and the viewport's discovery do not see it; it has no fingerprint. Qt-free: the GUI's PreviewRunner runs the
// command line of a job and calls these around it (docs/conventions.md §6.3).

inline constexpr const char* kPreviewDir = ".preview";
inline constexpr int kNrPreviewMaxEntries = 16;                // the cache: at most this many previews...
inline constexpr uint64_t kNrPreviewMaxBytes = 4ull << 30;     // ...and this many bytes, the least recently shown go first
inline constexpr int64_t kNrPreviewHistoryFrames = 6;          // the warm-up of «с историей» and segments (benchmarks: settled by ~6)
inline constexpr uint64_t kNrPreviewDiskReserve = 2ull << 30;  // a preview leaves this much of the volume free
// A range sought through the video is checked first: the CLI demuxes the clip up to it (ScanFrameIndex). Its bytes per second in the
// estimate (an SSD) and in the GUI watchdog's allowance (an external HDD)
inline constexpr double kNrPreviewScanRate = 400e6;
inline constexpr double kNrPreviewScanRateSlow = 20e6;

std::filesystem::path NrPreviewRoot(const std::filesystem::path& passesRoot);  // <root>/.preview/nr

enum class NrPreviewMode { Frame, History, Segment };  // one frame without history | one frame after 6 | N frames after 6
std::string ToString(NrPreviewMode mode);              // frame | history | segment

// Why a preview cannot start from this plan. The inputs are only exact reuses: a stage above NR that would run, restore a version or
// wait for the scene list refuses (the preview reads what is on disk, it never computes or switches a pass).
enum class NrPreviewBlock { None, NrOff, NrError, NoDll, Upstream, Restore, Pending, ControlMask, PastEnd, Disk };
struct NrPreviewGate {
    NrPreviewBlock why = NrPreviewBlock::None;
    std::vector<std::string> stages;   // Upstream / Restore / Pending: the stages above NR to compute first, in plan order
    std::string detail;                // NrError: the plan's error; Restore: the stage's version; ControlMask: why it does not read
    std::vector<std::string> passed;   // the input passes the run gets
    std::vector<std::string> dropped;  // Frame: depth_dlss / mv_dlss / scenes left out because their stage is not reused
    bool ok() const { return why == NrPreviewBlock::None; }
};
// From the nr decision of `plan` (a plan without the force marks of the stages above: those would rerun them for nothing).
NrPreviewGate GateNrPreview(const ProcessPlan& plan, NrPreviewMode mode);

struct NrPreviewEnv {
    std::filesystem::path source, cli;  // the source video; the dlssvid.exe that runs the job (its size and time are in the key)
    bool warp = false;                  // the stub on WARP (tests)
    uint64_t freeBytes = 0;             // free space of the passes root's volume (0: not checked)
    int64_t clipFrames = -1;            // the source's frames when the plan does not know them (-1: unknown)
    double fps = 0.0;                   // the source's rate: segment seconds -> frames
    uint32_t colorW = 0, colorH = 0;    // the colour NR reads (color_sr, else the source): the disk estimate
    double msPerFrame = 0.0;            // NR wall time per frame (the plan's estimate)
    double startSeconds = 2.0;          // process start, device, NGX init and CreateFeature (E8 ≈ 1.6-2 s; stub 0.5)
};

struct NrPreviewJob {
    NrPreviewGate gate;
    NrPreviewMode mode = NrPreviewMode::Frame;
    std::string key, keyJson;          // key = 12 hex of sha256(keyJson)
    std::string controlMask;           // the control mask's fingerprint when it is an input (the run must say it applied it)
    std::string dllSha256;             // the DLL the plan hashed, hex without "sha256:" (empty: no DLL)
    std::filesystem::path root, source, cli, entryDir, workDir;
    std::string sourceHash;
    int64_t first = 0, frames = 0, warmup = 0;   // F, N', K' after the clip's start and end
    nlohmann::json params = nlohmann::json::object();  // nr schema keys only: the run's --params-file
    std::map<std::string, std::filesystem::path> dirs;  // input pass -> folder given to the run
    std::map<std::string, std::string> manifests;       // input pass -> sha256 of its manifest.json bytes when prepared
    // mask_* NR opens by name under the root (--masks-dir) that are not inputs (the plan keeps only masks of the whole clip):
    // pass -> sha256 of its manifest.json bytes
    std::map<std::string, std::string> masks;
    uint64_t estimatedBytes = 0;
    double estimatedSeconds = 0.0;   // with scanBytes / kNrPreviewScanRate
    uint64_t scanBytes = 0;          // the source's bytes the CLI demuxes before the model (colour from the video, a seek): 0 for none
    bool lengthExact = true;         // the clip's length from the plan or the colour pass; false: the caller's estimate (mkv) - the
                                     // run may end at the clip's real end, before first + frames
    bool warp = false;
    // The command line (without the program): dlssvid nr into workDir with the inputs, the params file and --json in workDir.
    std::vector<std::string> Args() const;
};
// The job of a preview at frame `first` (segment: `seconds` long). A gate that refuses comes back in job.gate with nothing else set -
// but Disk: the key is set (a preview in the cache needs no room). Writes nothing: PreviewRunner creates workDir and the params file (WriteNrPreviewParams) before it starts the command.
NrPreviewJob PrepareNrPreview(const ProcessPlan& plan, NrPreviewMode mode, int64_t first, double seconds, const NrPreviewEnv& env);
// Test seam of PublishNrPreview: called before it renames the work folder into <key> (another instance can publish first). Empty in
// the product.
struct NrPreviewTestHooks {
    std::function<void(const NrPreviewJob&)> beforeRename;
};
NrPreviewTestHooks& NrPreviewHooksForTests();

// workDir/params.json from job.params (creates workDir).
void WriteNrPreviewParams(const NrPreviewJob& job);

// A preview on disk: <key>/preview.json (the sidecar, written last), params.json, nr.json and color_nr/.
struct NrPreviewEntry {
    std::filesystem::path dir;  // the entry, or the work folder when it could not be published (not cached)
    std::string key;
    NrPreviewMode mode = NrPreviewMode::Frame;
    int64_t first = 0, last = 0, warmup = 0;
    bool cacheable = true;
    double seconds = 0.0;  // the run's wall time
    uint64_t bytes = 0;
    nlohmann::json stats = nlohmann::json::object();  // nr.json's stats
    std::filesystem::path ColorDir() const { return dir / "color_nr"; }
};

// After the command exited 0: whether the run is the job's - its range, frames written and warm-up, the control mask applied, and the
// inputs' manifests unchanged since the job was prepared (a `process` from a terminal meanwhile). `cacheable` is false when the run
// loaded another DLL than the plan hashed (the CLI and the GUI find the DLL each from its own executable): shown, not cached.
enum class NrPreviewReject { None, NoSummary, Frames, ControlMask, InputChanged };
struct NrPreviewVerdict {
    NrPreviewReject why = NrPreviewReject::None;
    std::string detail;        // English: the log's and the CLI's text; InputChanged: the pass
    int64_t written = 0;       // Frames: the frames on disk
    bool cacheable = true;
    bool ok() const { return why == NrPreviewReject::None; }
};
NrPreviewVerdict VerifyNrPreviewRun(const NrPreviewJob& job);
// The sidecar, then workDir renamed to entryDir (tried again for a moment: an antivirus holds fresh files). An entry another instance
// published first wins; a broken one is replaced. Not cacheable, or the rename failing: the entry stays in workDir, uncached.
NrPreviewEntry PublishNrPreview(const NrPreviewJob& job, double seconds, bool cacheable);
// A published preview of the job: its sidecar with the same key JSON and the whole range, frames on disk. A hit marks it as shown now.
std::optional<NrPreviewEntry> FindNrPreview(const NrPreviewJob& job);

struct NrPreviewGcReport {
    std::vector<std::filesystem::path> removed, failed;  // failed: held by another program (removed at a later gc)
    uint64_t bytes = 0;
};
// Keeps at most maxEntries previews and maxBytes, the least recently shown go first; `pinned` folders (shown, being saved, just
// published, a running work folder) stay. Work and save folders of this process that are not pinned go, other processes' after 10
// minutes; names it does not know stay.
NrPreviewGcReport GcNrPreviews(const std::filesystem::path& passesRoot, const std::vector<std::filesystem::path>& pinned, int maxEntries = kNrPreviewMaxEntries,
                               uint64_t maxBytes = kNrPreviewMaxBytes);
NrPreviewGcReport RemoveNrPreviews(const std::filesystem::path& passesRoot, const std::vector<std::filesystem::path>& pinned);  // «Очистить превью»
struct NrPreviewUsage {
    int entries = 0;
    uint64_t bytes = 0;  // from the sidecars: no walk over the frames
};
NrPreviewUsage NrPreviewUsageOf(const std::filesystem::path& passesRoot);
// «Сохранить кадр…»: frame `frame` of a preview's colour into `outFile` - .exr copies an EXR frame file and writes another format as
// EXR half, anything else is written as PNG 16-bit through the pass writer (in a save-<pid>-<n> folder under .preview/nr, then moved).
void SaveNrPreviewFrame(const NrPreviewEntry& entry, int64_t frame, const std::filesystem::path& outFile);

}  // namespace dlssvid
