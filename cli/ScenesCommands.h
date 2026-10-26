#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>

namespace dlssvid::cli {

// `dlssvid scenes` — the clip's scene-cut list (PR-13, spec §2d): one pass over the source's luma -> <passes>/scenes (a data pass:
// the list and the score of every frame). Stored scores are reused: another threshold is applied without decoding again.
class ScenesCommands {
public:
    void Register(CLI::App& app);
    int Dispatch();  // exit code when the subcommand was parsed, -1 otherwise

    struct Args {
        std::string input, output;
        double threshold = 0.24;
        std::string backend = "luma";
        int64_t frames = -1;
        bool reanalyze = false;  // decode again even when stored scores stand for the source
        std::string json;
    };

private:
    CLI::App* scenes_ = nullptr;
    Args a_;
};

}  // namespace dlssvid::cli
