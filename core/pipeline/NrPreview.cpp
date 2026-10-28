#include "pipeline/NrPreview.h"

#include <process.h>  // _getpid

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

#include "io/VideoDecoder.h"
#include "passes/ControlMaskPass.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "passes/ScenesPass.h"
#include "passes/formats/ExrIO.h"
#include "pipeline/ControlMask.h"
#include "pipeline/Estimates.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "stages/ParamSchema.h"
#include "stages/nr/NrStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid {

namespace {

namespace fs = std::filesystem;

constexpr const char* kSidecar = "preview.json";
constexpr const char* kSidecarFormat = "dlssvid-nr-preview";
constexpr const char* kParamsFile = "params.json";
constexpr const char* kRunJson = "nr.json";
constexpr auto kStaleAfter = std::chrono::minutes(10);  // a work or save folder of another process older than this is a leftover

std::string Utf8(const fs::path& p) { return ExrFileName(p); }

// typed reads of a JSON object that may be anything (a sidecar edited by hand): another type is absent
std::string Str(const nlohmann::json& j, const char* key) {
    if (!j.is_object()) return {};
    const auto it = j.find(key);
    return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}
int64_t Int(const nlohmann::json& j, const char* key, int64_t def = -1) {
    if (!j.is_object()) return def;
    const auto it = j.find(key);
    return it != j.end() && it->is_number_integer() ? it->get<int64_t>() : def;
}

nlohmann::json ReadJsonFile(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return nullptr;
    try {
        return nlohmann::json::parse(in);
    } catch (const std::exception&) {
        return nullptr;
    }
}

// The bytes of a manifest: an input rewritten in place (another `process`, an import) changes them. "" when there is none.
std::string ManifestSha256(const fs::path& dir) {
    std::error_code ec;
    const fs::path file = dir / Manifest::kFileName;
    if (!fs::is_regular_file(file, ec)) return {};
    try {
        return Sha256File(file);
    } catch (const std::exception&) {
        return {};
    }
}

const StageDecision* NrDecision(const ProcessPlan& plan) {
    for (const auto& d : plan.stages)
        if (d.name == "nr") return &d;
    return nullptr;
}

// the enabled stage above NR that writes `pass` (none: a mask, the control mask, or a pass of a disabled stage, read as it is)
const StageDecision* ProducerOf(const ProcessPlan& plan, const std::string& pass) {
    for (const auto& d : plan.stages)
        if (d.name != "nr" && d.name != "fg" && std::find(d.outputs.begin(), d.outputs.end(), pass) != d.outputs.end()) return &d;
    return nullptr;
}

bool IsKeyName(const std::string& s) { return s.size() == 12 && s.find_first_not_of("0123456789abcdef") == std::string::npos; }

// "<key>.part-<pid>-<n>" or "save-<pid>-<n>": the pid of the process that made it; -1 for any other name
int64_t WorkPid(const std::string& name) {
    size_t at = std::string::npos;
    if (name.rfind("save-", 0) == 0) at = 5;
    else if (const size_t p = name.find(".part-"); p != std::string::npos && IsKeyName(name.substr(0, p))) at = p + 6;
    if (at == std::string::npos) return -1;
    const size_t dash = name.find('-', at);
    if (dash == std::string::npos || dash == at) return -1;
    const std::string pid = name.substr(at, dash - at);
    if (pid.find_first_not_of("0123456789") != std::string::npos || dash + 1 >= name.size() ||
        name.find_first_not_of("0123456789", dash + 1) != std::string::npos)
        return -1;
    return std::stoll(pid);
}

std::string NextWorkSuffix() {
    static std::atomic<int> serial{0};
    return std::to_string(_getpid()) + "-" + std::to_string(++serial);
}

NrPreviewMode ModeOf(const std::string& s) { return s == "history" ? NrPreviewMode::History : s == "segment" ? NrPreviewMode::Segment : NrPreviewMode::Frame; }

NrPreviewEntry EntryFromSidecar(const fs::path& dir, const nlohmann::json& side) {
    NrPreviewEntry e;
    e.dir = dir;
    e.key = Str(side, "key");
    e.mode = ModeOf(Str(side, "mode"));
    e.first = Int(side, "first", 0);
    e.last = Int(side, "last", 0);
    e.warmup = Int(side, "warmup", 0);
    const auto c = side.find("cacheable");
    e.cacheable = c == side.end() || !c->is_boolean() || c->get<bool>();
    const auto s = side.find("seconds");
    e.seconds = s != side.end() && s->is_number() ? s->get<double>() : 0.0;
    e.bytes = static_cast<uint64_t>(std::max<int64_t>(Int(side, "bytes", 0), 0));
    if (const auto st = side.find("stats"); st != side.end() && st->is_object()) e.stats = *st;
    return e;
}

// whether a preview's colour pass holds first..last with every frame file on disk
bool RangeOnDisk(const fs::path& color, int64_t first, int64_t last) {
    try {
        const Manifest m = Manifest::Load(color);
        if (m.firstFrame != first || m.lastFrame != last) return false;
        std::error_code ec;
        for (int64_t f = first; f <= last; ++f)
            if (!fs::exists(color / m.FrameFileName(f), ec)) return false;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool Pinned(const fs::path& p, const std::vector<fs::path>& pinned) {
    const fs::path n = p.lexically_normal();
    for (const auto& q : pinned)
        if (q.lexically_normal() == n) return true;
    return false;
}

void Remove(const fs::path& dir, uint64_t bytes, NrPreviewGcReport& report) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (ec) {
        Log()->warn("nr preview: cannot remove {}: {} (removed at a later clean-up)", Utf8(dir), ec.message());
        report.failed.push_back(dir);
        return;
    }
    report.removed.push_back(dir);
    report.bytes += bytes;
}

}  // namespace

fs::path NrPreviewRoot(const fs::path& passesRoot) { return passesRoot / kPreviewDir / "nr"; }

std::string ToString(NrPreviewMode mode) {
    switch (mode) {
        case NrPreviewMode::Frame: return "frame";
        case NrPreviewMode::History: return "history";
        case NrPreviewMode::Segment: return "segment";
    }
    return "frame";
}

NrPreviewGate GateNrPreview(const ProcessPlan& plan, NrPreviewMode mode) {
    NrPreviewGate g;
    const StageDecision* nr = NrDecision(plan);
    if (!nr) {
        g.why = NrPreviewBlock::NrOff;
        return g;
    }
    if (!nr->error.empty()) {
        g.why = NrPreviewBlock::NrError;
        g.detail = nr->error;
        return g;
    }
    const ToolInfo tool = ToolInfo::FromJson(nr->tool);
    if (!tool.dllFile.empty() && tool.dll.empty()) {  // ngx outside the bypass and no DLL found: the run would fail at Init
        g.why = NrPreviewBlock::NoDll;
        return g;
    }
    bool run = false, pending = false;
    std::string version;
    std::vector<std::string> blocking;
    for (const auto& [pass, id] : nr->inputs) {
        (void)id;
        // a frame without history starts at a reset whatever the list says, and depth / motion vectors only steer the model there:
        // left out when their stage would not be reused (said in the viewport: «без векторов движения»)
        const bool optionalInFrame = mode == NrPreviewMode::Frame && (pass == "depth_dlss" || pass == "mv_dlss" || pass == kScenesPass);
        const StageDecision* u = ProducerOf(plan, pass);
        if (mode == NrPreviewMode::Frame && pass == kScenesPass) {
            g.dropped.push_back(pass);
            continue;
        }
        if (!u || (u->action == "reuse" && !u->pending)) {
            g.passed.push_back(pass);
            continue;
        }
        if (optionalInFrame) {
            g.dropped.push_back(pass);
            continue;
        }
        if (u->pending) pending = true;
        else if (u->action == "restore") version = u->version;
        else run = true;
        if (std::find(blocking.begin(), blocking.end(), u->name) == blocking.end()) blocking.push_back(u->name);
    }
    if (blocking.empty()) return g;
    for (const auto& d : plan.stages)
        if (std::find(blocking.begin(), blocking.end(), d.name) != blocking.end()) g.stages.push_back(d.name);
    g.why = run ? NrPreviewBlock::Upstream : pending ? NrPreviewBlock::Pending : NrPreviewBlock::Restore;
    if (g.why == NrPreviewBlock::Restore) g.detail = version;
    g.passed.clear();
    g.dropped.clear();
    return g;
}

NrPreviewJob PrepareNrPreview(const ProcessPlan& plan, NrPreviewMode mode, int64_t first, double seconds, const NrPreviewEnv& env) {
    NrPreviewJob job;
    job.mode = mode;
    job.root = plan.passesRoot;
    job.source = env.source;
    job.cli = env.cli;
    job.sourceHash = plan.sourceHash;
    job.warp = env.warp;
    job.gate = GateNrPreview(plan, mode);
    if (!job.gate.ok()) return job;
    const StageDecision& nr = *NrDecision(plan);
    for (const auto& p : job.gate.passed) job.dirs[p] = plan.passesRoot / p;

    // the range: the clip's length from the plan, else the colour pass, else the caller; the warm-up shortened at the clip's start
    int64_t clipFrames = plan.frames > 0 ? plan.frames : -1;
    uint32_t w = env.colorW, h = env.colorH;
    if (job.dirs.count("color_sr")) {
        try {
            const Manifest m = Manifest::Load(job.dirs.at("color_sr"));
            if (clipFrames <= 0) clipFrames = m.lastFrame + 1;
            w = m.width;
            h = m.height;
        } catch (const std::exception&) {
        }
    }
    if (clipFrames <= 0) clipFrames = env.clipFrames;
    if (first < 0 || (clipFrames > 0 && first >= clipFrames)) {
        job.gate.why = NrPreviewBlock::PastEnd;
        return job;
    }
    int64_t n = mode == NrPreviewMode::Segment ? std::max<int64_t>(1, std::llround(seconds * env.fps)) : 1;
    if (clipFrames > 0) n = std::min(n, clipFrames - first);
    job.first = first;
    job.frames = n;
    job.warmup = std::min<int64_t>(mode == NrPreviewMode::Frame ? 0 : kNrPreviewHistoryFrames, first);

    // the control mask, read as NR's Init will read it: a folder that is no mask stops the run there (the plan only warns at a reuse)
    if (nr.inputs.count(kControlMaskPass)) {
        std::string why;
        const auto info = ReadControlMaskInfo(plan.passesRoot / kControlMaskPass, &why);
        if (!info) {
            job.gate.why = NrPreviewBlock::ControlMask;
            job.gate.detail = why;
            return job;
        }
        job.controlMask = info->fingerprint;
    }

    // the run's settings: nr's schema keys of the decision (the parameters the fingerprint hashed); dll_dir and the like are not
    // options of dlssvid nr, and --params-file refuses a key the schema does not have
    if (const StageSchema* schema = FindStageSchema("nr"); schema && nr.params.is_object())
        for (auto it = nr.params.begin(); it != nr.params.end(); ++it)
            if (schema->Find(it.key())) job.params[it.key()] = it.value();

    job.estimatedBytes = static_cast<uint64_t>(static_cast<double>(n) * static_cast<double>(BytesPerFrame("nr", w, h, 1.0, 1)) * 1.1);
    bool overBudget = false;  // refused after the key: a preview in the cache is shown without room
    if (env.freeBytes > 0) {
        const uint64_t room = env.freeBytes > kNrPreviewDiskReserve ? env.freeBytes - kNrPreviewDiskReserve : 0;
        overBudget = job.estimatedBytes > std::min(kNrPreviewMaxBytes, room);
    } else {
        overBudget = job.estimatedBytes > kNrPreviewMaxBytes;
    }
    // through the video a range past the first frames is sought, and the CLI checks the seek first: it demuxes the clip up to the
    // range (ScanFrameIndex, NrCommands) - seconds of a long clip before the model starts. The colour pass is read by index instead
    if (!job.dirs.count("color_sr") && job.first - job.warmup > kSequentialSeekFrames && !env.source.empty()) {
        std::error_code ec;
        const uintmax_t size = fs::file_size(env.source, ec);
        if (!ec && size > 0) {
            const double part = clipFrames > 0 ? std::min(1.0, static_cast<double>(job.first + n + 64) / static_cast<double>(clipFrames)) : 1.0;
            job.scanBytes = static_cast<uint64_t>(static_cast<double>(size) * part);
        }
    }
    job.estimatedSeconds = env.startSeconds + static_cast<double>(job.warmup + n) * env.msPerFrame / 1000.0 +
                           static_cast<double>(job.scanBytes) / kNrPreviewScanRate;

    const ToolInfo tool = ToolInfo::FromJson(nr.tool);
    job.dllSha256 = tool.dll.rfind("sha256:", 0) == 0 ? tool.dll.substr(7) : tool.dll;
    nlohmann::json inputs = nlohmann::json::object();
    for (const auto& p : job.gate.passed) {
        job.manifests[p] = ManifestSha256(job.dirs.at(p));
        inputs[p] = {{"id", nr.inputs.at(p)}, {"manifest", job.manifests[p]}};
    }
    // the masks the run opens by name (NrStage::Init) though the plan left them out (not of the whole clip): their bytes in the key
    nlohmann::json masks = nlohmann::json::object();
    for (const auto& name : NrBypassParams(nr.params) ? std::vector<std::string>{} : NrStage::MaskNames()) {  // the bypass reads none
        const std::string pass = "mask_" + name;
        if (job.dirs.count(pass) || !Manifest::Exists(plan.passesRoot / pass)) continue;
        job.masks[pass] = ManifestSha256(plan.passesRoot / pass);
        masks[pass] = job.masks[pass];
    }
    // the CLI that runs it: kPassToolVersion is the same across development builds, the executable is not
    nlohmann::json cli = nlohmann::json::object();
    if (!env.cli.empty()) {
        std::error_code ec;
        const auto size = fs::file_size(env.cli, ec);
        if (!ec) cli["size"] = size;
        const auto time = fs::last_write_time(env.cli, ec);
        if (!ec) cli["mtime"] = static_cast<int64_t>(time.time_since_epoch().count());
    }
    // no absolute path: the cache moves with the passes root
    const nlohmann::json key = {{"v", 1},
                                {"mode", ToString(mode)},
                                {"first", first},
                                {"frames", n},
                                {"warmup", job.warmup},
                                {"source", plan.sourceHash},
                                {"source_format", nr.sourceFormat},
                                {"params", nr.params},
                                {"tool", nr.tool},
                                {"inputs", inputs},
                                {"masks", masks},
                                {"warp", env.warp},
                                {"cli", cli}};
    job.keyJson = CanonicalJson(key);
    job.key = Sha256Hex(job.keyJson.data(), job.keyJson.size()).substr(0, 12);
    const fs::path base = NrPreviewRoot(plan.passesRoot);
    job.entryDir = base / job.key;
    job.workDir = base / (job.key + ".part-" + NextWorkSuffix());
    if (overBudget) job.gate.why = NrPreviewBlock::Disk;
    return job;
}

std::vector<std::string> NrPreviewJob::Args() const {
    std::vector<std::string> a{"nr", "-i", Utf8(source), "-o", Utf8(workDir), "--no-auto"};
    if (!sourceHash.empty()) a.insert(a.end(), {"--source-hash", sourceHash});
    a.insert(a.end(), {"--params-file", Utf8(workDir / kParamsFile), "--start", std::to_string(first), "--frames", std::to_string(frames), "--warmup",
                       std::to_string(warmup), "--json", Utf8(workDir / kRunJson), "--masks-dir", Utf8(root)});  // the masks as process reads them
    for (const auto& [pass, flag] : {std::pair<const char*, const char*>{"color_sr", "--color-dir"}, {"depth_dlss", "--depth-dir"}, {"mv_dlss", "--mv-dir"}})
        if (const auto it = dirs.find(pass); it != dirs.end()) a.insert(a.end(), {flag, Utf8(it->second)});
    if (const auto it = dirs.find(kScenesPass); it != dirs.end()) a.insert(a.end(), {"--scenes-dir", Utf8(it->second)});
    else a.push_back("--no-scenes");
    if (warp) a.push_back("--warp");
    return a;
}

void WriteNrPreviewParams(const NrPreviewJob& job) {
    fs::create_directories(job.workDir);
    std::ofstream out(job.workDir / kParamsFile, std::ios::binary);
    out << job.params.dump(2) << "\n";
    if (!out) Throw("nr preview: cannot write " + Utf8(job.workDir / kParamsFile));
}

NrPreviewVerdict VerifyNrPreviewRun(const NrPreviewJob& job) {
    NrPreviewVerdict v;
    auto reject = [&](NrPreviewReject why, std::string detail) {
        v.why = why;
        v.detail = std::move(detail);
        return v;
    };
    const nlohmann::json run = ReadJsonFile(job.workDir / kRunJson);
    if (!run.is_object() || !run.contains("stats") || !run["stats"].is_object()) return reject(NrPreviewReject::NoSummary, "the run left no summary (nr.json)");
    const nlohmann::json& stats = run["stats"];
    const fs::path color = job.workDir / "color_nr";
    const int64_t last = job.first + job.frames - 1;
    Manifest m;
    try {
        m = Manifest::Load(color);
    } catch (const std::exception& e) {
        return reject(NrPreviewReject::Frames, std::string("the preview's colour does not read: ") + e.what());
    }
    v.written = m.frameCount;
    if (m.firstFrame != job.first || m.lastFrame != last || m.frameCount != job.frames || !RangeOnDisk(color, job.first, last) ||
        Int(stats, "frames_written") != job.frames || Int(stats, "warmup_frames") != job.warmup)
        return reject(NrPreviewReject::Frames, "written " + std::to_string(m.frameCount) + " of " + std::to_string(job.frames) + " frames (" +
                                                   std::to_string(m.firstFrame) + ".." + std::to_string(m.lastFrame) + ")");
    if (!job.controlMask.empty()) {
        const auto cm = stats.find("control_mask");
        if (cm == stats.end() || !cm->is_object() || Str(*cm, "fingerprint") != job.controlMask)
            return reject(NrPreviewReject::ControlMask, "the control mask changed during the run");
    }
    for (const auto& [pass, dir] : job.dirs)
        if (ManifestSha256(dir) != job.manifests.at(pass)) return reject(NrPreviewReject::InputChanged, pass);
    for (const auto& name : NrBypassParams(job.params) ? std::vector<std::string>{} : NrStage::MaskNames()) {  // a mask the run opened by name: written, rewritten or removed meanwhile
        const std::string pass = "mask_" + name;
        if (job.dirs.count(pass)) continue;
        const bool now = Manifest::Exists(job.root / pass);
        const auto it = job.masks.find(pass);
        if (now != (it != job.masks.end()) || (now && ManifestSha256(job.root / pass) != it->second))
            return reject(NrPreviewReject::InputChanged, pass);
    }
    // the DLL the run loaded against the one the key names: another one is shown, not cached under this key
    if (!job.dllSha256.empty())
        if (const auto d = run.find("diagnostics"); d != run.end() && d->is_object())
            if (const std::string loaded = Str(*d, "dll_sha256"); !loaded.empty() && loaded != job.dllSha256) {
                Log()->warn("nr preview: the run loaded nvngx_dlssnr.dll {} where the plan hashed {}: shown, not cached", loaded.substr(0, 8), job.dllSha256.substr(0, 8));
                v.cacheable = false;
            }
    return v;
}

NrPreviewEntry PublishNrPreview(const NrPreviewJob& job, double seconds, bool cacheable) {
    NrPreviewEntry e;
    e.dir = job.workDir;
    e.key = job.key;
    e.mode = job.mode;
    e.first = job.first;
    e.last = job.first + job.frames - 1;
    e.warmup = job.warmup;
    e.cacheable = cacheable;
    e.seconds = seconds;
    e.bytes = DirectoryBytes(job.workDir / "color_nr");
    if (const nlohmann::json run = ReadJsonFile(job.workDir / kRunJson); run.is_object() && run.contains("stats") && run["stats"].is_object()) e.stats = run["stats"];
    const nlohmann::json side = {{"format", kSidecarFormat}, {"v", 1}, {"key", job.key}, {"key_json", job.keyJson}, {"mode", ToString(job.mode)},
                                 {"first", e.first}, {"last", e.last}, {"warmup", e.warmup}, {"frames", job.frames}, {"created", NowIso8601()},
                                 {"dll_sha256", job.dllSha256}, {"cacheable", cacheable}, {"seconds", seconds}, {"bytes", e.bytes}, {"stats", e.stats}};
    {  // written last, through a temporary name: a sidecar is there whole or not at all
        const fs::path tmp = job.workDir / (std::string(kSidecar) + ".tmp");
        {
            std::ofstream out(tmp, std::ios::binary);
            out << side.dump(2) << "\n";
            if (!out) Throw("nr preview: cannot write " + Utf8(tmp));
        }
        std::error_code ec;
        fs::rename(tmp, job.workDir / kSidecar, ec);
        if (ec) Throw("nr preview: cannot write " + Utf8(job.workDir / kSidecar) + ": " + ec.message());
    }
    if (!cacheable) return e;
    std::error_code ec;
    if (fs::exists(job.entryDir, ec)) {
        if (const auto found = FindNrPreview(job)) {  // another instance over the same root published it first: that one is shown
            fs::remove_all(job.workDir, ec);
            return *found;
        }
        fs::remove_all(job.entryDir, ec);  // a broken entry under this key: replaced
    }
    for (int attempt = 0;; ++attempt) {
        ec.clear();
        fs::rename(job.workDir, job.entryDir, ec);
        if (!ec || attempt >= 2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // an antivirus or the indexer holds the fresh files a moment
    }
    if (ec) {
        Log()->warn("nr preview: cannot publish {}: {} - shown from its work folder, not cached", Utf8(job.workDir), ec.message());
        e.cacheable = false;
        return e;
    }
    e.dir = job.entryDir;
    return e;
}

std::optional<NrPreviewEntry> FindNrPreview(const NrPreviewJob& job) {
    if (job.entryDir.empty()) return std::nullopt;
    const nlohmann::json side = ReadJsonFile(job.entryDir / kSidecar);
    if (!side.is_object() || Str(side, "format") != kSidecarFormat || Str(side, "key_json") != job.keyJson) return std::nullopt;
    NrPreviewEntry e = EntryFromSidecar(job.entryDir, side);
    const int64_t last = job.first + job.frames - 1;
    if (!e.cacheable || e.first != job.first || e.last != last || !RangeOnDisk(e.ColorDir(), job.first, last)) return std::nullopt;
    std::error_code ec;
    fs::last_write_time(job.entryDir / kSidecar, fs::file_time_type::clock::now(), ec);  // shown now: the last the GC removes
    return e;
}

NrPreviewGcReport GcNrPreviews(const fs::path& passesRoot, const std::vector<fs::path>& pinned, int maxEntries, uint64_t maxBytes) {
    NrPreviewGcReport report;
    const fs::path base = NrPreviewRoot(passesRoot);
    std::error_code ec;
    if (!fs::is_directory(base, ec)) return report;
    struct Item {
        fs::path dir;
        fs::file_time_type time;
        uint64_t bytes = 0;
    };
    std::vector<Item> entries;
    const auto now = fs::file_time_type::clock::now();
    for (const auto& de : fs::directory_iterator(base, ec)) {
        std::error_code e2;
        if (!de.is_directory(e2)) continue;
        const std::string name = de.path().filename().string();
        if (IsKeyName(name)) {
            const nlohmann::json side = ReadJsonFile(de.path() / kSidecar);
            Item it{de.path(), fs::file_time_type::min(), 0};
            if (side.is_object()) {  // without a sidecar: the oldest, of the size it has
                it.time = fs::last_write_time(de.path() / kSidecar, e2);
                if (e2) it.time = fs::file_time_type::min();
                it.bytes = static_cast<uint64_t>(std::max<int64_t>(Int(side, "bytes", 0), 0));
            } else {
                it.bytes = DirectoryBytes(de.path());
            }
            entries.push_back(it);
        } else if (const int64_t pid = WorkPid(name); pid >= 0) {
            if (Pinned(de.path(), pinned)) continue;
            const auto made = fs::last_write_time(de.path(), e2);
            const bool stale = !e2 && now - made > kStaleAfter;
            if (pid == _getpid() || stale) Remove(de.path(), DirectoryBytes(de.path()), report);  // ours and not running, or a leftover
        }
        // any other name is not ours to delete
    }
    std::sort(entries.begin(), entries.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
    int64_t count = static_cast<int64_t>(entries.size());
    uint64_t bytes = 0;
    for (const auto& it : entries) bytes += it.bytes;
    for (const auto& it : entries) {
        if (count <= maxEntries && bytes <= maxBytes) break;
        if (Pinned(it.dir, pinned)) continue;
        const size_t failed = report.failed.size();
        Remove(it.dir, it.bytes, report);
        if (report.failed.size() == failed) {
            --count;
            bytes -= std::min(bytes, it.bytes);
        }
    }
    return report;
}

NrPreviewGcReport RemoveNrPreviews(const fs::path& passesRoot, const std::vector<fs::path>& pinned) { return GcNrPreviews(passesRoot, pinned, 0, 0); }

NrPreviewUsage NrPreviewUsageOf(const fs::path& passesRoot) {
    NrPreviewUsage u;
    const fs::path base = NrPreviewRoot(passesRoot);
    std::error_code ec;
    if (!fs::is_directory(base, ec)) return u;
    for (const auto& de : fs::directory_iterator(base, ec)) {
        std::error_code e2;
        if (!de.is_directory(e2) || !IsKeyName(de.path().filename().string())) continue;
        const nlohmann::json side = ReadJsonFile(de.path() / kSidecar);
        if (!side.is_object()) continue;
        ++u.entries;
        u.bytes += static_cast<uint64_t>(std::max<int64_t>(Int(side, "bytes", 0), 0));
    }
    return u;
}

void SaveNrPreviewFrame(const NrPreviewEntry& entry, int64_t frame, const fs::path& outFile) {
    const fs::path color = entry.ColorDir();
    const Manifest m = Manifest::Load(color);
    if (frame < m.firstFrame || frame > m.lastFrame) Throw("nr preview: no frame " + std::to_string(frame) + " (it has " + std::to_string(m.firstFrame) + ".." +
                                                           std::to_string(m.lastFrame) + ")");
    std::string ext = outFile.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::error_code ec;
    if (ext == ".exr" && m.format == FileFormat::Exr) {  // the frame file as it is
        fs::copy_file(color / m.FrameFileName(frame), outFile, fs::copy_options::overwrite_existing, ec);
        if (ec) Throw("nr preview: cannot write " + Utf8(outFile) + ": " + ec.message());
        return;
    }
    const fs::path tmp = entry.dir.parent_path() / ("save-" + NextWorkSuffix());
    try {
        PassReader reader = PassReader::Open(color);
        Manifest out = reader.FrameManifest();
        const bool exr = ext == ".exr";  // a pass in another format: the frame as EXR half, the name the user chose holds what it says
        out.format = exr ? FileFormat::Exr : FileFormat::Png;
        out.pixelType = exr ? PixelType::F16 : PixelType::U16;
        out.fingerprint.clear();
        out.filePattern = out.pass + "_%06d" + ExtensionFor(out.format, out.pixelType, out.channels.size());
        ExportPass(tmp, out, SourceFromReader(reader), FrameRange{frame, frame}, {});
        const fs::path written = tmp / out.FrameFileName(frame);
        fs::remove(outFile, ec);
        ec.clear();
        fs::rename(written, outFile, ec);
        if (ec) {  // another volume: a copy
            ec.clear();
            fs::copy_file(written, outFile, fs::copy_options::overwrite_existing, ec);
            if (ec) Throw("nr preview: cannot write " + Utf8(outFile) + ": " + ec.message());
        }
    } catch (...) {
        fs::remove_all(tmp, ec);
        throw;
    }
    fs::remove_all(tmp, ec);
}

}  // namespace dlssvid
