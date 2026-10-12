#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace dlssvid {

// Frame Generation of `dlssvid process` / `bench` in its own process (ТЗ §4 / §7, audit fg-2, TASK-0038): the runner
// starts `<dlssvid.exe> fg --as-stage ...` over the same pass root with the stage's parameters, so a crash of DLSS-G
// or the driver ends only that process - the run goes on and encodes color_nr. The child prints progress and its
// summary as lines on stdout (everything else it prints is its log, forwarded as is):
//   @progress <real frames done>
//   @result <one-line JSON: color_fg, backend, diagnostics, stats{frames, generated, multiplier, ms_per_frame_wall, ...},
//            seconds, disabled, disabled_reason>
inline constexpr std::string_view kFgProgressPrefix = "@progress ";
inline constexpr std::string_view kFgResultPrefix = "@result ";

struct FgProcessSpec {
    std::filesystem::path exe;  // dlssvid.exe
    std::filesystem::path input, passesRoot, colorDir, depthDir, mvDir;
    nlohmann::json params = nlohmann::json::object();  // the stage's parameters, as RunFg applies them in-process
    std::string format = "exr";
    std::string sourceHash;  // "sha256:..." of the source: the child does not hash the video again
    int64_t frames = -1;
    bool warp = false;
    bool disableUnavailable = false;
    bool verbose = false;
    // test hooks of the isolation tests (hidden flags): die after N real frames, die after printing @result, hang
    int64_t crashAfter = -1;
    bool crashOnExit = false;
    bool stall = false;
    uint32_t exitGraceMs = 60000;  // after @result: how long the child may take to exit before it is ended
};

// The child's command line (UTF-8 arguments for Subprocess::Start; paths through Utf8Arg, parameters as ASCII JSON).
std::vector<std::string> FgProcessArgs(const FgProcessSpec& spec);

// "0xC0000005 (access violation)", "3 (abort)", "105 (command line rejected; see stderr)", "1".
std::string DescribeExitCode(uint32_t code);

struct FgProcessOutcome {
    bool ok = false;        // the child reported a result: a complete color_fg, or a disabled stage
    bool disabled = false;  // the backend is unavailable (no DLL / GPU): nothing was written
    bool crashed = false;   // the child died (NTSTATUS exit code or abort)
    std::string disabledReason;
    std::string error;    // !ok: why
    std::string warning;  // ok, but the child crashed on exit after writing its result
    nlohmann::json result = nlohmann::json::object();  // the child's @result
    uint32_t exitCode = 0, pid = 0;
    int64_t framesDone = 0;  // last @progress
};

// Starts the child (in a job object: cancelling the run ends it too), forwards its log, reports progress and returns
// the outcome. Never throws for a failure of the child; the caller decides (process: --disable-unavailable).
FgProcessOutcome RunFgProcess(const FgProcessSpec& spec, const std::function<void(int64_t)>& progress);

}  // namespace dlssvid
