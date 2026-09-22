#pragma once

#include <CLI/CLI.hpp>
#include <string>

namespace dlssvid::cli {

// `dlssvid passes list|use|gc` — pass versions under a passes root (stage 9, MR A): what is on disk, switch a pass
// to a previous version, remove old versions. The root comes from --passes or from a project file.
class PassesCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();

    struct Args {
        std::string passes, project;  // root, or the project whose passes root (and pass_versions_keep) to use
        std::string pass;             // list / gc: only this pass; use: the pass (or stage: depth, flow, ...) to switch
        std::string version;          // use: history id or a unique prefix
        int keep = -1;                // gc: versions kept per pass (default: the project's or 2)
        bool dryRun = false;
        std::string json;             // write the result to this JSON file
    };

private:
    CLI::App* passes_ = nullptr;
    CLI::App* list_ = nullptr;
    CLI::App* use_ = nullptr;
    CLI::App* gc_ = nullptr;
    Args a_;
};

}  // namespace dlssvid::cli
