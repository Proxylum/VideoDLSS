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
        int64_t frame = -1;    // frame at the base rate
        double time = -1.0;    // seconds (wins over --frame)
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
        std::string proxy = "auto";     // auto: use a valid scrubbing proxy | off | build: make the missing ones first
        std::string benchMode = "all";  // all | cold | warm | random
        std::string benchJson;          // write the bench results as JSON
    };
    struct ProxyArgs {
        std::string project, input, passes, result;
        std::string sources = "source,result";
        bool force = false;       // rebuild a valid proxy
        bool check = false;       // print the plan and the state, build nothing
        bool background = false;  // below-normal priority (the GUI builds while the user works)
        double maxGb = 0.0;       // > 0: skip a proxy estimated above this or above a quarter of the free disk
    };

private:
    ProjectArgs pa_;
    RenderArgs ra_;
    ProxyArgs xa_;
    CLI::App* project_ = nullptr;
    CLI::App* render_ = nullptr;
    CLI::App* proxy_ = nullptr;
};

}  // namespace dlssvid::cli
