#include "stages/upscale/IUpscaler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "stages/upscale/BicubicUpscaler.h"
#include "stages/upscale/DlssUpscaler.h"
#include "stages/upscale/NisUpscaler.h"
#include "stages/upscale/RtxVsrUpscaler.h"
#include "util/Error.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace dlssvid {

std::vector<std::string> UpscalerBackends() { return {"dlss", "nis", "bicubic", "rtxvsr"}; }

std::vector<std::filesystem::path> NvidiaDllSearchPaths(const std::filesystem::path& override) {
    std::vector<std::filesystem::path> out;
    if (!override.empty()) out.push_back(override);
    if (const char* env = std::getenv("DLSSVID_NVIDIA_DLL_DIR"); env && *env) out.push_back(env);
    wchar_t exe[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0) {
        const std::filesystem::path dir = std::filesystem::path(exe).parent_path();
        out.push_back(dir / "nvidia");
        out.push_back(dir);
    }
    return out;
}

UpscalerAvailability UpscalerAvailable(const std::string& backend, const std::filesystem::path& dllDir) {
    if (backend == "bicubic" || backend == "nis") return {true, {}};
    if (backend == "rtxvsr") return RtxVsrUpscaler::Available();
    if (backend == "dlss") return DlssUpscaler::Available(dllDir);
    return {false, "unknown upscaler '" + backend + "' (rtxvsr | dlss | nis | bicubic)"};
}

std::unique_ptr<IUpscaler> CreateUpscaler(const std::string& backend) {
    if (backend == "bicubic") return std::make_unique<BicubicUpscaler>();
    if (backend == "nis") return std::make_unique<NisUpscaler>();
    if (backend == "dlss") return std::make_unique<DlssUpscaler>();
    if (backend == "rtxvsr") return std::make_unique<RtxVsrUpscaler>();
    Throw("unknown upscaler '" + backend + "' (rtxvsr | dlss | nis | bicubic)");
}

UpscaleTarget ResolveUpscaleTarget(uint32_t inW, uint32_t inH, double scale, uint32_t targetW, uint32_t targetH, uint32_t maxW, uint32_t maxH) {
    if (!inW || !inH) Throw("ResolveUpscaleTarget: empty input");
    UpscaleTarget t;
    double w, h;
    if (targetW && targetH) {
        w = targetW;
        h = targetH;
    } else if (targetW) {
        w = targetW;
        h = static_cast<double>(targetW) * inH / inW;
    } else if (targetH) {
        h = targetH;
        w = static_cast<double>(targetH) * inW / inH;
    } else {
        if (scale <= 0.0) Throw("ResolveUpscaleTarget: scale must be positive");
        w = inW * scale;
        h = inH * scale;
    }
    if (maxW && w > maxW) {
        h *= static_cast<double>(maxW) / w;
        w = maxW;
        t.capped = true;
    }
    if (maxH && h > maxH) {
        w *= static_cast<double>(maxH) / h;
        h = maxH;
        t.capped = true;
    }
    auto even = [](double v) { return std::max(2u, static_cast<uint32_t>(std::lround(v / 2.0)) * 2u); };
    t.width = even(w);
    t.height = even(h);
    t.scale = static_cast<double>(t.width) / inW;
    return t;
}

}  // namespace dlssvid
