#include "stages/upscale/IUpscaler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "stages/upscale/BicubicUpscaler.h"
#include "stages/upscale/DlssUpscaler.h"
#include "stages/upscale/NisUpscaler.h"
#include "stages/upscale/WorkerUpscaler.h"
#ifdef DLSSVID_WITH_TENSORRT
#include "stages/upscale/TrtUpscaler.h"
#endif
#include "util/Error.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace dlssvid {

void IUpscaler::EvaluateCpu(const PassImage&, PassImage&) { Throw(std::string(Name()) + ": no CPU path"); }

void IUpscaler::EvaluateCpuWindow(const std::vector<const PassImage*>& in, uint32_t, uint32_t, std::vector<PassImage>& out) {
    out.clear();
    for (const PassImage* f : in) {
        PassImage o;
        EvaluateCpu(*f, o);
        out.push_back(std::move(o));
    }
}

std::vector<std::string> UpscalerBackends() {
#ifdef DLSSVID_WITH_TENSORRT
    return {"dlss", "nis", "bicubic", "trt", "worker"};
#else
    return {"dlss", "nis", "bicubic", "worker"};
#endif
}

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
    if (backend == "dlss") return DlssUpscaler::Available(dllDir);
    if (backend == "worker") return WorkerUpscaler::Available();
#ifdef DLSSVID_WITH_TENSORRT
    if (backend == "trt") return TrtUpscaler::Available();
#else
    if (backend == "trt") return {false, "this build has no TensorRT support (DLSSVID_WITH_TENSORRT)"};
#endif
    return {false, "unknown upscaler '" + backend + "' (dlss | nis | bicubic | trt | worker)"};
}

std::unique_ptr<IUpscaler> CreateUpscaler(const std::string& backend) {
    if (backend == "bicubic") return std::make_unique<BicubicUpscaler>();
    if (backend == "nis") return std::make_unique<NisUpscaler>();
    if (backend == "dlss") return std::make_unique<DlssUpscaler>();
    if (backend == "worker") return std::make_unique<WorkerUpscaler>();
#ifdef DLSSVID_WITH_TENSORRT
    if (backend == "trt") return std::make_unique<TrtUpscaler>();
#endif
    Throw("unknown upscaler '" + backend + "' (dlss | nis | bicubic | trt | worker)");
}

UpscaleTarget ResolveUpscaleTarget(uint32_t inW, uint32_t inH, double scale, uint32_t targetW, uint32_t targetH, uint32_t maxW, uint32_t maxH,
                                   bool capByLongSide, PixelAspect sar) {
    if (!inW || !inH) Throw("ResolveUpscaleTarget: empty input");
    UpscaleTarget t;
    // TASK-0073: the input as shown - nothing narrowed: wider pixels widen it, narrower ones make it higher; square: the input itself
    const double s = sar.Value();
    const double inDW = s > 1.0 ? inW * s : inW, inDH = s < 1.0 ? inH / s : inH;
    double w, h;
    if (targetW && targetH) {
        w = targetW;
        h = targetH;
    } else if (targetW) {
        w = targetW;
        h = static_cast<double>(targetW) * inDH / inDW;
    } else if (targetH) {
        h = targetH;
        w = static_cast<double>(targetH) * inDW / inDH;
    } else {
        if (scale <= 0.0) Throw("ResolveUpscaleTarget: scale must be positive");
        w = inDW * scale;
        h = inDH * scale;
    }
    if (capByLongSide && h > w && maxW > maxH) std::swap(maxW, maxH);  // a portrait output: the long side is its height
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
    if (sar.Square()) {
        t.width = even(w);
        t.height = even(h);
    } else {
        // the shown size back in the input's pixels: the stretched axis divided. Capped - rounded down, so that the frame as shown stays
        // within the cap (an even cap: the shown side rounds to at most it)
        auto grid = [&](double v) { return t.capped ? std::max(2u, static_cast<uint32_t>(std::floor(v / 2.0 + 1e-9)) * 2u) : even(v); };
        // by the scale alone, not capped: the stored side times the scale, as for square pixels - divided back from the shown size it
        // could land a hair under a .5 and round 2 lower (706 x1.5 at 32:27: 1058.9999999999998 -> 1058, not 1060; review of TASK-0073)
        // ... and so for a target of one side on the axis the pixels do not stretch (720x576 at 64:45, height 1060: 1326, not 1324)
        const bool scaled = !targetW && !targetH && !t.capped;
        const bool byHeight = targetH && !targetW && !t.capped, byWidth = targetW && !targetH && !t.capped;
        t.width = s > 1.0 ? (scaled ? even(inW * scale) : byHeight ? even(static_cast<double>(targetH) * inW / inH) : grid(w / s)) : even(w);
        t.height = s < 1.0 ? (scaled ? even(inH * scale) : byWidth ? even(static_cast<double>(targetW) * inH / inW) : grid(h * s)) : even(h);
    }
    t.scale = static_cast<double>(t.width) / inW;
    // the frame as shown: the target asked for (capped: within the cap), not the grid stretched back - two roundings could miss it by 2
    // or, near the cap, go past it (review of TASK-0073)
    t.displayWidth = sar.Square() ? t.width : even(w);
    t.displayHeight = sar.Square() ? t.height : even(h);
    return t;
}

}  // namespace dlssvid
