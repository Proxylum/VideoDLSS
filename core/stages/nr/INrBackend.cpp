#include "stages/nr/INrBackend.h"

#include <algorithm>
#include <cctype>

#include "stages/nr/NgxNrBackend.h"
#include "stages/nr/StubNrBackend.h"
#include "stages/upscale/IUpscaler.h"
#include "util/Error.h"

namespace dlssvid {

std::string NvidiaDriverVersion::ToString() const {
    if (!valid) return "?";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d.%02d", major, minor);
    return buf;
}

NvidiaDriverVersion NvidiaDriverFromUmd(uint64_t umdVersion) {
    NvidiaDriverVersion v;
    if (umdVersion == 0) return v;
    const uint32_t lo = static_cast<uint32_t>(umdVersion & 0xFFFFFFFFu);
    const uint32_t subversion = lo >> 16, build = lo & 0xFFFFu;
    // NVIDIA packs the driver version into the last five digits: 15.9186 -> 59186 -> 591.86
    v.major = static_cast<int>((subversion % 10) * 100 + build / 100);
    v.minor = static_cast<int>(build % 100);
    v.valid = v.major > 0;
    return v;
}

namespace {
std::string Upper(std::string_view s) {
    std::string u(s);
    for (auto& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u;
}
}  // namespace

std::string GpuArchitectureFromName(std::string_view adapterName) {
    const std::string n = Upper(adapterName);
    auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
    if (has("RTX 50") || has("RTX PRO")) return "Blackwell";
    if (has("RTX 40") || has(" ADA")) return "Ada";
    if (has("RTX 30") || has("RTX A")) return "Ampere";
    if (has("RTX 20") || has("GTX 16") || has("TITAN RTX")) return "Turing";
    return "";
}

std::string GpuArchitectureFromComputeCapability(int major, int minor) {
    if (major == 7 && minor == 5) return "Turing";
    if (major == 8 && minor == 9) return "Ada";
    if (major == 8) return "Ampere";
    if (major >= 10) return "Blackwell";
    return "";
}

nlohmann::json NrDiagnostics::ToJson() const {
    return {{"gpu", gpu},
            {"architecture", architecture},
            {"driver", driver.ToString()},
            {"driver_ok", driverOk},
            {"dll", dll.string()},
            {"dll_sha256", dllSha256},
            {"dll_size", dllSize},
            {"forwarder", forwarder.string()},
            {"init_result", initResult},
            {"init_abi", initAbi},
            {"params_block", paramsBlock},
            {"create_result", createResult},
            {"ok", ok},
            {"hint", hint}};
}

std::filesystem::path FindNrDll(const std::filesystem::path& dllDir) {
    for (const auto& dir : NvidiaDllSearchPaths(dllDir))
        if (std::filesystem::exists(dir / kNrDllName)) return dir / kNrDllName;
    return {};
}

std::filesystem::path FindNrForwarder(const std::filesystem::path& dllDir) {
    for (const auto& dir : NvidiaDllSearchPaths(dllDir))
        if (std::filesystem::exists(dir / kNrForwarderName)) return dir / kNrForwarderName;
    return {};
}

std::string NrDllInstruction() {
    return std::string(kNrDllName) +
           " not found: copy your nvngx_dlssnr.dll (from a DLSS 5 driver / game; RTX 20/30/40 need the patched copy — `dlssvid nr-patch`) into "
           "bin/nvidia/ next to the executable or set DLSSVID_NVIDIA_DLL_DIR — see docs/dll-setup.md";
}

int ParseNrStyle(const std::string& s) {
    const std::string u = Upper(s);
    if (u == "DEFAULT") return 0;
    if (u == "NATURAL") return 1;
    if (u == "CINEMATIC") return 2;
    try {
        size_t pos = 0;
        const int v = std::stoi(s, &pos);
        if (pos == s.size() && v >= 0 && v <= 15) return v;
    } catch (...) {
    }
    Throw("nr: --style must be natural | cinematic | default | 0..15 (got '" + s + "')");
}

NrAvailability NrAvailable(const std::string& backend, const std::filesystem::path& dllDir) {
    if (backend == "stub") return {true, {}};
    if (backend == "ngx") return NgxNrBackend::Available(dllDir);
    return {false, "unknown nr backend '" + backend + "' (ngx | stub)"};
}

std::unique_ptr<INrBackend> CreateNrBackend(const std::string& backend) {
    if (backend == "stub") return std::make_unique<StubNrBackend>();
    if (backend == "ngx") return std::make_unique<NgxNrBackend>();
    Throw("unknown nr backend '" + backend + "' (ngx | stub)");
}

std::vector<std::string> NrBackends() { return {"ngx", "stub"}; }

}  // namespace dlssvid
