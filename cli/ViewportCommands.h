#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <string>

namespace dlssvid::cli {

// `dlssvid project init|show` and `dlssvid render`: everything the viewport shows is reachable
// from the command line (ТЗ §6/§7) — the same ViewportState / FrameStore / ViewportRenderer
// as the GUI, rendered offscreen into a PNG.
class ViewportCommands {
public:
    void Register(CLI::App& app);
    // Exit code when one of the registered subcommands was parsed, -1 otherwise.
    int Dispatch();

    struct ProjectArgs {
        std::string action = "init";  // init | show
        std::string input;            // source video (init)
        std::string passes;           // passes root (init; default <video>_passes)
        std::string result;           // result video (init, optional)
        std::string output;           // project file (init; default <video>.dlssvid.json)
        std::string project;          // project file (show)
    };
    struct RenderArgs {
        std::string project, input, passes, result, output;
        int64_t frame = -1;
        std::string mode;      // single | overlay | grid
        std::string source;    // single: source name
        std::string layers;    // overlay: "base,name[:display[:opacity[:blend]]],..."
        std::string sources;   // grid: "a,b,c,d"
        std::string display;   // display mode of the selected layer
        int expand = -1;       // grid: expanded cell
        std::string size;      // WxH of the output (default: image size)
        float zoom = 0.f;      // 0 = fit
        float centerX = -1.f, centerY = -1.f;
        std::string wipe;      // off | v[:pos[:a:b]] | h[:pos[:a:b]]
        bool warp = false;
        int bench = 0;         // scrub N frames and print timings
        bool saveState = false;
    };

private:
    ProjectArgs pa_;
    RenderArgs ra_;
    CLI::App* project_ = nullptr;
    CLI::App* render_ = nullptr;
};

}  // namespace dlssvid::cli
