#include "PassesCommands.h"

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ProcessRunner.h"
#include "util/Error.h"
#include "viewport/Project.h"

namespace dlssvid::cli {

namespace {

std::string HumanBytes(uint64_t b) {
    char buf[32];
    if (b >= 1ull << 30) std::snprintf(buf, sizeof buf, "%.1f GB", static_cast<double>(b) / (1ull << 30));
    else if (b >= 1ull << 20) std::snprintf(buf, sizeof buf, "%.0f MB", static_cast<double>(b) / (1ull << 20));
    else if (b >= 1ull << 10) std::snprintf(buf, sizeof buf, "%.0f KB", static_cast<double>(b) / (1ull << 10));
    else std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(b));
    return buf;
}

// The passes root from --passes or the project; the project's pass_versions_keep when it is the source.
std::filesystem::path ResolveRoot(const PassesCommands::Args& a, int* projectKeep) {
    std::filesystem::path root;
    if (!a.project.empty()) {
        const Project p = Project::Load(a.project);
        root = p.passesRoot;
        if (projectKeep) *projectKeep = p.passVersionsKeep;
    }
    if (!a.passes.empty()) root = a.passes;
    if (root.empty()) Throw("passes: --passes <root> or --project <file> is required");
    return root;
}

std::string ParamsBrief(const nlohmann::json& params, size_t width) {
    std::string s = params.is_object() && !params.empty() ? params.dump() : "";
    if (s.size() > width) s = s.substr(0, width - 3) + "...";
    return s;
}

void WriteJson(const std::string& path, const nlohmann::json& j) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) Throw("passes: cannot write " + path);
    out << j.dump(2) << "\n";
}

int CmdList(const PassesCommands::Args& a) {
    const auto root = ResolveRoot(a, nullptr);
    const std::vector<PassVersionInfo> versions = ListPassVersions(root, a.pass);
    if (versions.empty()) {
        std::printf("no passes under %s\n", root.string().c_str());
    } else {
        std::printf("%-12s %-26s %-20s %7s %10s  %-9s %s\n", "pass", "version", "created", "frames", "size", "fp", "params");
        uint64_t total = 0, history = 0;
        for (const auto& v : versions) {
            std::printf("%-12s %-26s %-20s %7lld %10s  %-9s %s\n", v.pass.c_str(), v.current ? "current" : v.id.c_str(), v.created.c_str(),
                        static_cast<long long>(v.frames), HumanBytes(v.bytes).c_str(), ShortFingerprint(v.fingerprint).c_str(), ParamsBrief(v.params, 60).c_str());
            total += v.bytes;
            if (!v.current) history += v.bytes;
        }
        std::printf("total: %s on disk, %s in previous versions (%s)\n", HumanBytes(total).c_str(), HumanBytes(history).c_str(), root.string().c_str());
    }
    nlohmann::json j = nlohmann::json::array();
    for (const auto& v : versions) j.push_back(v.ToJson());
    WriteJson(a.json, {{"passes_root", root.string()}, {"versions", j}});
    return 0;
}

int CmdUse(const PassesCommands::Args& a) {
    const auto root = ResolveRoot(a, nullptr);
    if (a.pass.empty() || a.version.empty()) Throw("passes use: <pass|stage> <version> are required");
    const std::vector<std::string>& family = StageFamilyPasses(a.pass);
    const std::vector<std::string> passes = family.empty() ? std::vector<std::string>{a.pass} : family;
    nlohmann::json report = nlohmann::json::array();
    int switched = 0;
    std::string lastError;
    for (const auto& pass : passes) {
        bool has = false;
        for (const auto& v : ListPassVersions(root, pass)) has = has || (!v.current && v.id.compare(0, a.version.size(), a.version) == 0);
        if (!has) {
            if (family.empty()) Throw("passes: " + pass + " has no version '" + a.version + "' (see `dlssvid passes list`)");
            continue;  // a stage's other pass may not have this version (depth_dlss without depth_raw)
        }
        const std::optional<std::string> retired = UsePassVersion(root, pass, a.version);
        const auto cur = CurrentPassVersion(root, pass);
        std::printf("%s: version %s is now current%s\n", pass.c_str(), cur ? ShortFingerprint(cur->fingerprint).c_str() : "?",
                    retired ? (" (previous one kept as " + *retired + ")").c_str() : "");
        report.push_back({{"pass", pass}, {"fingerprint", cur ? cur->fingerprint : ""}, {"retired", retired ? *retired : ""}});
        ++switched;
    }
    if (switched == 0) Throw("passes: no pass of stage '" + a.pass + "' has a version '" + a.version + "'");
    WriteJson(a.json, {{"passes_root", root.string()}, {"switched", report}});
    return 0;
}

int CmdGc(const PassesCommands::Args& a) {
    int projectKeep = -1;
    const auto root = ResolveRoot(a, &projectKeep);
    const int keep = a.keep >= 0 ? a.keep : (projectKeep >= 0 ? projectKeep : 2);
    const PassGcReport r = GcPassVersions(root, keep, a.pass, a.dryRun);
    for (const auto& v : r.removed) std::printf("%s %s: %s %s\n", a.dryRun ? "would remove" : "removed", v.pass.c_str(), v.id.c_str(), HumanBytes(v.bytes).c_str());
    std::printf("%s: %zu version(s), %s%s; %d kept per pass\n", a.dryRun ? "dry run" : "gc", r.removed.size(), HumanBytes(r.bytes).c_str(),
                a.dryRun ? " would be freed" : " freed", keep);
    nlohmann::json removed = nlohmann::json::array();
    for (const auto& v : r.removed) removed.push_back(v.ToJson());
    WriteJson(a.json, {{"passes_root", root.string()}, {"keep", keep}, {"dry_run", a.dryRun}, {"removed", removed}, {"bytes", r.bytes}});
    return 0;
}

}  // namespace

void PassesCommands::Register(CLI::App& app) {
    passes_ = app.add_subcommand("passes", "pass versions under a passes root: list, switch to a previous version, remove old ones (stage 9)");
    passes_->require_subcommand(1);
    auto common = [&](CLI::App* cmd) {
        cmd->add_option("--passes", a_.passes, "passes root (default: the project's)");
        cmd->add_option("--project", a_.project, "*.dlssvid.json whose passes root (and pass_versions_keep) to use")->check(CLI::ExistingFile);
        cmd->add_option("--json", a_.json, "write the result to this JSON file");
    };
    list_ = passes_->add_subcommand("list", "every pass with its current version and the previous ones kept on disk");
    common(list_);
    list_->add_option("--pass", a_.pass, "only this pass (depth_dlss, color_nr, ...)");
    use_ = passes_->add_subcommand("use", "make a previous version of a pass (or of a stage's passes) the current one; the current one is kept");
    common(use_);
    use_->add_option("pass", a_.pass, "pass folder name, or a stage (depth, flow, upscale, nr, fg) to switch all of its passes")->required();
    use_->add_option("version", a_.version, "version id from `passes list` (a unique prefix is enough)")->required();
    gc_ = passes_->add_subcommand("gc", "remove previous versions beyond the newest N per pass; versions other passes list as inputs stay");
    common(gc_);
    gc_->add_option("--keep", a_.keep, "versions kept per pass, the current one included (default: the project's pass_versions_keep or 2)")->default_val(-1);
    gc_->add_option("--pass", a_.pass, "only this pass");
    gc_->add_flag("--dry-run", a_.dryRun, "report what would be removed without removing anything");
}

int PassesCommands::Dispatch() {
    if (!passes_ || !passes_->parsed()) return -1;
    if (list_->parsed()) return CmdList(a_);
    if (use_->parsed()) return CmdUse(a_);
    if (gc_->parsed()) return CmdGc(a_);
    return -1;
}

}  // namespace dlssvid::cli
