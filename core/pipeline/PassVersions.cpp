#include "pipeline/PassVersions.h"

#include <algorithm>

#include "passes/Manifest.h"
#include "pipeline/PassFingerprint.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

constexpr const char* kHistorySuffix = ".v";

PassVersionInfo InfoFor(const std::string& pass, const std::filesystem::path& dir, const std::string& id, bool current) {
    const Manifest m = Manifest::Load(dir);
    PassVersionInfo v;
    v.pass = pass;
    v.id = id;
    v.current = current;
    v.dir = dir;
    v.fingerprint = m.fingerprint;
    v.created = m.created.empty() ? FileTimeIso8601(dir / Manifest::kFileName) : m.created;
    v.frames = m.frameCount;
    v.bytes = DirectoryBytes(dir);
    v.params = m.paramsCanonical;
    v.tool = m.tool;
    v.inputs = m.inputs;
    return v;
}

bool NewerFirst(const PassVersionInfo& a, const PassVersionInfo& b) {
    if (a.created != b.created) return a.created > b.created;
    return a.id > b.id;
}

std::vector<PassVersionInfo> HistoryOf(const std::filesystem::path& root, const std::string& pass) {
    std::vector<PassVersionInfo> out;
    const auto hist = PassHistoryDir(root, pass);
    std::error_code ec;
    if (!std::filesystem::is_directory(hist, ec)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(hist, ec)) {
        if (!entry.is_directory() || !Manifest::Exists(entry.path())) continue;
        try {
            out.push_back(InfoFor(pass, entry.path(), entry.path().filename().string(), false));
        } catch (const std::exception& e) {
            Log()->warn("passes: version folder {} skipped: {}", entry.path().string(), e.what());
        }
    }
    std::sort(out.begin(), out.end(), NewerFirst);
    return out;
}

// Pass names with a history folder (<name>.v) under root.
std::vector<std::string> PassesWithHistory(const std::filesystem::path& root) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) return out;
    const std::string suffix = kHistorySuffix;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            out.push_back(name.substr(0, name.size() - suffix.size()));
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Every pass name known under root (current or history), in the order gc wants: end of the pipeline first.
std::vector<std::string> AllPasses(const std::filesystem::path& root) {
    std::set<std::string> names;
    for (const auto& n : PassesUnderRoot(root)) names.insert(n);
    for (const auto& n : PassesWithHistory(root)) names.insert(n);
    std::vector<std::string> ordered;
    for (const char* stage : {"fg", "nr", "upscale", "flow", "depth"}) {
        const auto& family = StageFamilyPasses(stage);
        for (auto it = family.rbegin(); it != family.rend(); ++it)
            if (names.erase(*it)) ordered.push_back(*it);
    }
    ordered.insert(ordered.end(), names.begin(), names.end());
    return ordered;
}

std::string ResolveVersionId(const std::filesystem::path& root, const std::string& pass, const std::string& idOrPrefix) {
    const auto history = HistoryOf(root, pass);
    std::vector<std::string> matches;
    for (const auto& v : history) {
        if (v.id == idOrPrefix) return v.id;
        if (v.id.compare(0, idOrPrefix.size(), idOrPrefix) == 0) matches.push_back(v.id);
    }
    if (matches.size() == 1) return matches[0];
    if (matches.empty()) Throw("passes: " + pass + " has no version '" + idOrPrefix + "' (see `dlssvid passes list`)");
    std::string all;
    for (const auto& m : matches) all += (all.empty() ? "" : ", ") + m;
    Throw("passes: version '" + idOrPrefix + "' of " + pass + " is ambiguous: " + all);
}

void Rename(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (ec) Throw("passes: cannot move " + from.string() + " to " + to.string() + ": " + ec.message() + " (is a viewer holding its files open?)");
}

}  // namespace

nlohmann::json PassVersionInfo::ToJson() const {
    return {{"pass", pass},       {"version", current ? "current" : id}, {"current", current}, {"dir", dir.string()}, {"fingerprint", fingerprint},
            {"created", created}, {"frames", frames},                    {"bytes", bytes},     {"params", params},    {"tool", tool},
            {"inputs", inputs}};
}

std::filesystem::path PassHistoryDir(const std::filesystem::path& root, const std::string& pass) { return root / (pass + kHistorySuffix); }

std::string PassVersionId(const Manifest& m, const std::filesystem::path& dir) {
    std::string ts = CompactTimestamp(m.created.empty() ? FileTimeIso8601(dir / Manifest::kFileName) : m.created);
    if (ts.empty()) ts = CompactTimestamp(NowIso8601());
    return ts + "_" + ShortFingerprint(m.fingerprint);
}

uint64_t DirectoryBytes(const std::filesystem::path& dir) {
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        std::error_code fec;
        if (entry.is_regular_file(fec)) total += entry.file_size(fec);
    }
    return total;
}

std::vector<std::string> PassesUnderRoot(const std::filesystem::path& root) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec))
        if (entry.is_directory() && Manifest::Exists(entry.path())) out.push_back(entry.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

std::optional<PassVersionInfo> CurrentPassVersion(const std::filesystem::path& root, const std::string& pass) {
    if (!Manifest::Exists(root / pass)) return std::nullopt;
    return InfoFor(pass, root / pass, "", true);
}

std::vector<PassVersionInfo> ListPassVersions(const std::filesystem::path& root, const std::string& pass) {
    std::vector<PassVersionInfo> out;
    std::vector<std::string> names;
    if (pass.empty()) {
        std::set<std::string> all;
        for (const auto& n : PassesUnderRoot(root)) all.insert(n);
        for (const auto& n : PassesWithHistory(root)) all.insert(n);
        names.assign(all.begin(), all.end());
    } else {
        names.push_back(pass);
    }
    for (const auto& name : names) {
        try {
            if (auto cur = CurrentPassVersion(root, name)) out.push_back(*cur);
        } catch (const std::exception& e) {
            Log()->warn("passes: {} skipped: {}", (root / name).string(), e.what());
        }
        for (auto& v : HistoryOf(root, name)) out.push_back(std::move(v));
    }
    return out;
}

std::optional<std::string> RetirePassVersion(const std::filesystem::path& root, const std::string& pass) {
    const auto dir = root / pass;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return std::nullopt;
    std::string id;
    if (Manifest::Exists(dir)) {
        try {
            id = PassVersionId(Manifest::Load(dir), dir);
        } catch (const std::exception&) {
            id = CompactTimestamp(FileTimeIso8601(dir / Manifest::kFileName)) + "_broken";
        }
    } else {
        std::string ts = CompactTimestamp(FileTimeIso8601(dir));
        if (ts.empty()) ts = CompactTimestamp(NowIso8601());
        id = ts + "_partial";
    }
    const auto hist = PassHistoryDir(root, pass);
    std::filesystem::create_directories(hist);
    std::filesystem::path target = hist / id;
    for (int n = 2; std::filesystem::exists(target); ++n) target = hist / (id + "-" + std::to_string(n));
    Rename(dir, target);
    Log()->info("passes: {} -> {}", dir.string(), target.string());
    return target.filename().string();
}

std::optional<std::string> UsePassVersion(const std::filesystem::path& root, const std::string& pass, const std::string& id) {
    const std::string full = ResolveVersionId(root, pass, id);
    const auto src = PassHistoryDir(root, pass) / full;
    const auto cur = root / pass;
    const std::optional<std::string> retired = RetirePassVersion(root, pass);
    Rename(src, cur);
    Log()->info("passes: {} is now version {}", pass, full);
    return retired;
}

namespace {
// Referenced fingerprints, ignoring the versions in `gone` (a dry run simulates its removals).
std::set<std::string> ReferencedExcept(const std::filesystem::path& root, const std::set<std::filesystem::path>& gone) {
    std::set<std::string> out;
    for (const auto& v : ListPassVersions(root)) {
        if (gone.count(v.dir)) continue;
        for (const auto& [name, fp] : v.inputs)
            if (!fp.empty()) out.insert(fp);
    }
    return out;
}
}  // namespace

std::set<std::string> ReferencedFingerprints(const std::filesystem::path& root) { return ReferencedExcept(root, {}); }

PassGcReport GcPassVersions(const std::filesystem::path& root, int keep, const std::string& pass, bool dryRun) {
    PassGcReport r;
    if (keep <= 0) return r;
    const std::vector<std::string> names = pass.empty() ? AllPasses(root) : std::vector<std::string>{pass};
    std::set<std::filesystem::path> gone;  // removed so far (or, in a dry run, what would be)
    for (const auto& name : names) {
        const auto history = HistoryOf(root, name);
        if (history.empty()) continue;
        const std::set<std::string> referenced = ReferencedExcept(root, gone);  // after the removals of the passes before
        const size_t stay = Manifest::Exists(root / name) ? static_cast<size_t>(keep - 1) : static_cast<size_t>(keep);
        for (size_t i = stay; i < history.size(); ++i) {
            const PassVersionInfo& v = history[i];
            if (!v.fingerprint.empty() && referenced.count(v.fingerprint)) continue;
            if (!dryRun) {
                std::error_code ec;
                std::filesystem::remove_all(v.dir, ec);
                if (ec) Throw("passes: cannot remove " + v.dir.string() + ": " + ec.message());
                Log()->info("passes: removed {} ({} bytes)", v.dir.string(), v.bytes);
            }
            gone.insert(v.dir);
            r.removed.push_back(v);
            r.bytes += v.bytes;
        }
        if (!dryRun) {
            std::error_code ec;
            const auto hist = PassHistoryDir(root, name);
            if (std::filesystem::is_directory(hist, ec) && std::filesystem::is_empty(hist, ec)) std::filesystem::remove(hist, ec);
        }
    }
    return r;
}

}  // namespace dlssvid
