#include "stages/fg/IFrameGenerator.h"

#include <cmath>
#include <cstring>

#include "stages/fg/BlendFrameGenerator.h"
#include "stages/fg/DlssgFrameGenerator.h"
#include "stages/fg/RifeFrameGenerator.h"
#include "stages/upscale/IUpscaler.h"
#include "util/Error.h"

namespace dlssvid {

nlohmann::json FgDiagnostics::ToJson() const {
    return {{"gpu", gpu},
            {"architecture", architecture},
            {"driver", driver.ToString()},
            {"driver_ok", driverOk},
            {"dll", dll.string()},
            {"dll_sha256", dllSha256},
            {"dll_size", dllSize},
            {"available", available},
            {"multi_frame_max", multiFrameMax},
            {"create_result", createResult},
            {"backbuffer_format", backbufferFormat},
            {"model", model},
            {"ok", ok},
            {"hint", hint}};
}

std::filesystem::path FindDlssgDll(const std::filesystem::path& dllDir) {
    for (const auto& dir : NvidiaDllSearchPaths(dllDir))
        if (std::filesystem::exists(dir / kDlssgDllName)) return dir / kDlssgDllName;
    return {};
}

std::string DlssgDllInstruction() {
    return std::string(kDlssgDllName) + " not found: copy it from <DLSS SDK>/lib/Windows_x86_64/rel/ into bin/nvidia/ next to the executable (the build does it when "
                                        "DLSS_SDK_ROOT is set) or set DLSSVID_NVIDIA_DLL_DIR — see docs/dll-setup.md";
}

FgAvailability FgAvailable(const std::string& backend, const std::filesystem::path& dllDir) {
    if (backend == "blend") return {true, {}};
    if (backend == "dlssg") return DlssgFrameGenerator::Available(dllDir);
    if (backend == "rife") return RifeFrameGenerator::Available();
    return {false, "unknown fg backend '" + backend + "' (dlssg | rife | blend)"};
}

std::unique_ptr<IFrameGenerator> CreateFrameGenerator(const std::string& backend) {
    if (backend == "blend") return std::make_unique<BlendFrameGenerator>();
    if (backend == "dlssg") return std::make_unique<DlssgFrameGenerator>();
    if (backend == "rife") return std::make_unique<RifeFrameGenerator>();
    Throw("unknown fg backend '" + backend + "' (dlssg | rife | blend)");
}

std::vector<std::string> FgBackends() { return {"dlssg", "rife", "blend"}; }

void Mul4x4(const float* a, const float* b, float* out) {
    float r[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a[i * 4 + k] * b[k * 4 + j];
            r[i * 4 + j] = s;
        }
    std::memcpy(out, r, sizeof(r));
}

FgCamera BuildFgCamera(uint32_t width, uint32_t height, float fovDegrees, float nearPlane, float farPlane) {
    FgCamera c{};
    c.nearPlane = nearPlane;
    c.farPlane = farPlane;
    c.fovRadians = fovDegrees * 3.14159265358979f / 180.f;
    c.aspect = height ? static_cast<float>(width) / static_cast<float>(height) : 1.f;
    const float f = 1.f / std::tan(c.fovRadians * 0.5f);
    const float n = nearPlane, fa = farPlane;
    // D3D-style perspective (z in [0, 1]), row-major with post-multiplication: v' = v * M
    float m[16] = {f / c.aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, fa / (fa - n), 1, 0, 0, -n * fa / (fa - n), 0};
    std::memcpy(c.viewToClip, m, sizeof(m));
    // analytic inverse of the projection above
    float inv[16] = {c.aspect / f, 0, 0, 0, 0, 1.f / f, 0, 0, 0, 0, 0, -(fa - n) / (n * fa), 0, 0, 1, 1.f / n};
    std::memcpy(c.clipToView, inv, sizeof(inv));
    float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(c.identity, id, sizeof(id));
    return c;
}

}  // namespace dlssvid
