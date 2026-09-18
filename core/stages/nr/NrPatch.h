#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dlssvid {

// Wrapper around dlssnr-patcher (https://github.com/dev-camo/dlssnr-patcher, GPLv2 — an external tool,
// never vendored): patches the user's own nvngx_dlssnr.dll for RTX 20/30/40 and puts the result into
// bin/nvidia/ (ТЗ §4 «Пропатченная DLL», docs/dll-setup.md). `dlssvid nr-patch` and the GUI button
// «Пропатчить DLL» call RunNrPatch.
struct NrPatchOptions {
    std::filesystem::path input;    // the original nvngx_dlssnr.dll
    std::filesystem::path output;   // default: <exe dir>/nvidia/nvngx_dlssnr.dll
    std::filesystem::path patcher;  // dlssnr_patcher.py or its folder; default: DLSSNR_PATCHER_ROOT, tools/dlssnr-patcher next to / above the executable
    std::filesystem::path cudaBin;  // folder with ptxas / fatbinary / cuobjdump (CUDA 13.3); default: CUDA_PATH_V13_3/bin, else the patcher searches PATH
    std::string python;             // default: DLSSVID_PYTHON, then "python"
    std::vector<std::string> archs;  // ada | ampere | turing | blackwell (empty: the patcher's default = all four)
    bool dryRun = false;
    bool force = true;              // replace an existing output
};

struct NrPatchResult {
    std::vector<std::string> command;
    std::filesystem::path script, output;
    std::string inputSha256, outputSha256;
    uint32_t exitCode = 0;
    std::string patcherOutput;  // stdout of the patcher
    bool ok = false;
};

std::filesystem::path DefaultNrDllOutput();                              // <exe dir>/nvidia/nvngx_dlssnr.dll
std::filesystem::path FindPatcherScript(const std::filesystem::path& hint = {});  // "" when not found
std::filesystem::path DefaultCudaBin();                                  // "" when no CUDA 13 toolkit is known
std::string DefaultPatchPython();
// Pure: the patcher command line (argv) for the options; throws on an unknown architecture.
std::vector<std::string> BuildPatchCommand(const NrPatchOptions& options, const std::filesystem::path& script, const std::filesystem::path& output);
// Runs the patcher, hashes input and output, writes <output>.patch.json (input, hashes, command, date). Throws when the
// input, python or the patcher cannot be found; a failing patcher is reported in the result (ok = false).
NrPatchResult RunNrPatch(const NrPatchOptions& options);

}  // namespace dlssvid
