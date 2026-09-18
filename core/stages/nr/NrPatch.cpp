#include "stages/nr/NrPatch.h"

#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>

#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"
#include "util/Subprocess.h"

namespace dlssvid {

namespace {
std::filesystem::path ExeDir() {
    wchar_t exe[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0) return std::filesystem::path(exe).parent_path();
    return std::filesystem::current_path();
}

std::string Env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}

const char* ArchFlag(const std::string& arch) {
    if (arch == "ada") return "--ada";
    if (arch == "ampere") return "--ampere";
    if (arch == "turing") return "--turing";
    if (arch == "blackwell") return "--blackwell";
    if (arch == "all") return nullptr;
    Throw("nr-patch: --arch must be ada | ampere | turing | blackwell | all (got '" + arch + "')");
}
}  // namespace

std::filesystem::path DefaultNrDllOutput() { return ExeDir() / "nvidia" / "nvngx_dlssnr.dll"; }

std::filesystem::path FindPatcherScript(const std::filesystem::path& hint) {
    auto resolve = [](const std::filesystem::path& p) -> std::filesystem::path {
        if (p.empty()) return {};
        if (std::filesystem::is_regular_file(p)) return p;
        if (std::filesystem::is_regular_file(p / "dlssnr_patcher.py")) return p / "dlssnr_patcher.py";
        return {};
    };
    if (auto p = resolve(hint); !p.empty()) return p;
    if (!hint.empty()) return {};
    if (auto p = resolve(Env("DLSSNR_PATCHER_ROOT")); !p.empty()) return p;
    const auto exe = ExeDir();
    for (const auto& base : {exe, exe.parent_path(), exe.parent_path().parent_path(), exe.parent_path().parent_path().parent_path()})
        if (auto p = resolve(base / "tools" / "dlssnr-patcher"); !p.empty()) return p;
    return {};
}

std::filesystem::path DefaultCudaBin() {
    for (const char* var : {"CUDA_PATH_V13_3", "CUDA_PATH_V13_2", "CUDA_PATH_V13_1", "CUDA_PATH_V13_0"}) {
        const std::string v = Env(var);
        if (!v.empty() && std::filesystem::exists(std::filesystem::path(v) / "bin" / "ptxas.exe")) return std::filesystem::path(v) / "bin";
    }
    return {};
}

std::string DefaultPatchPython() {
    const std::string env = Env("DLSSVID_PYTHON");
    return env.empty() ? "python" : env;
}

std::vector<std::string> BuildPatchCommand(const NrPatchOptions& o, const std::filesystem::path& script, const std::filesystem::path& output) {
    std::vector<std::string> cmd{o.python.empty() ? DefaultPatchPython() : o.python, script.string()};
    for (const auto& a : o.archs)
        if (const char* flag = ArchFlag(a)) cmd.push_back(flag);
    if (!o.dryRun) {
        cmd.push_back("-o");
        cmd.push_back(output.string());
        if (o.force) cmd.push_back("--force");
    } else {
        cmd.push_back("--dry-run");
    }
    if (!o.cudaBin.empty()) {
        cmd.push_back("--cuda-bin");
        cmd.push_back(o.cudaBin.string());
    }
    cmd.push_back(o.input.string());
    return cmd;
}

NrPatchResult RunNrPatch(const NrPatchOptions& options) {
    NrPatchOptions o = options;
    if (o.input.empty() || !std::filesystem::is_regular_file(o.input)) Throw("nr-patch: --input must be the original nvngx_dlssnr.dll (got '" + o.input.string() + "')");
    NrPatchResult r;
    r.script = FindPatcherScript(o.patcher);
    if (r.script.empty())
        Throw("nr-patch: dlssnr_patcher.py not found — git clone https://github.com/dev-camo/dlssnr-patcher and pass --patcher <folder> or set DLSSNR_PATCHER_ROOT "
              "(docs/dll-setup.md)");
    if (o.cudaBin.empty()) o.cudaBin = DefaultCudaBin();
    if (!o.cudaBin.empty() && !std::filesystem::exists(o.cudaBin / "ptxas.exe"))
        Throw("nr-patch: " + o.cudaBin.string() + " has no ptxas.exe — --cuda-bin must point to the bin folder of CUDA Toolkit 13.3");
    if (o.python.empty()) o.python = DefaultPatchPython();
    r.output = o.output.empty() ? DefaultNrDllOutput() : o.output;
    r.inputSha256 = Sha256File(o.input);
    if (!o.dryRun) {
        std::error_code ec;
        std::filesystem::create_directories(r.output.parent_path(), ec);
    }
    r.command = BuildPatchCommand(o, r.script, r.output);
    std::string line;
    for (const auto& c : r.command) line += (line.empty() ? "" : " ") + QuoteArg(c);
    Log()->info("nr-patch: input {} (sha256 {})", o.input.string(), r.inputSha256);
    Log()->info("nr-patch: {}", line);
    if (o.cudaBin.empty()) Log()->warn("nr-patch: no --cuda-bin and no CUDA_PATH_V13_3: the patcher looks for ptxas / fatbinary / cuobjdump on PATH (CUDA Toolkit 13.3)");
    r.exitCode = Subprocess::Run(r.command, &r.patcherOutput);
    if (r.exitCode != 0) {
        Log()->error("nr-patch: the patcher exited with code {}", r.exitCode);
        return r;
    }
    if (o.dryRun) {
        r.ok = true;
        return r;
    }
    if (!std::filesystem::is_regular_file(r.output)) {
        Log()->error("nr-patch: the patcher finished but {} does not exist", r.output.string());
        return r;
    }
    r.outputSha256 = Sha256File(r.output);
    r.ok = true;
    Log()->info("nr-patch: output {} (sha256 {})", r.output.string(), r.outputSha256);
    // sidecar: which input and command produced this DLL (docs/dll-setup.md lists verified pairs)
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    nlohmann::json side = {{"input", o.input.string()},
                           {"input_sha256", r.inputSha256},
                           {"output_sha256", r.outputSha256},
                           {"command", r.command},
                           {"patcher", r.script.string()},
                           {"date", date}};
    std::ofstream(r.output.string() + ".patch.json") << side.dump(2) << "\n";
    return r;
}

}  // namespace dlssvid
