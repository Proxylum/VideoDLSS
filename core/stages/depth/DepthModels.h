#pragma once

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace dlssvid {

class ModelRegistry;

// Depth models of the registry (TASK-0039): which model a backend runs, with which window, what it needs. One place
// for the planner (ProcessRunner), the CLI (`dlssvid depth`) and both estimators (TensorRT and the PyTorch worker):
// a model of another family or an unknown id is refused before anything is decoded, exported or built.

// "vda" for vda / worker:vda, "da3" for da3 / worker / worker:da3, "" for stub, icdepth and worker:stub.
std::string DepthFamilyOf(const std::string& backend);
// The backend's model when depth.model is empty: da3metric-large, metric-vda-small. Throws for other families.
std::string DefaultDepthModel(const std::string& family);
// <modelsDir>/registry.json, or ModelRegistry::DefaultRegistryPath() when modelsDir is empty.
std::filesystem::path DepthRegistryPath(const std::string& modelsDir);

struct DepthModelChoice {
    std::string backend, family, modelId, license;
    bool nonCommercial = false;  // the licence says NC (CC-BY-NC-4.0)
    bool metric = false;
    int window = 1, overlap = 0;  // what the estimator runs
    int registryWindow = 1, registryOverlap = 0, maxWindow = 1;
    int registryInputSize = 0;  // params.input_size (0 = none)
    std::vector<std::string> warnings;
    nlohmann::json params = nlohmann::json::object();  // the registry entry's params
};

// `modelId` "" = the backend's model. `windowOverride` 0 = the registry's window (for the PyTorch path `window_torch`
// when the registry has one); for VDA 1..max_window and a multiple of 4 above 4 (the DPT head runs micro-batches of 4),
// the overlap scales with it. A window for a model without windows (DA3) is ignored with a warning - a project keeps
// depth.window after a switch from VDA. Throws with the family's models on an unknown or foreign id; backends without
// a family pass any id through unchecked.
DepthModelChoice ResolveDepthModel(const ModelRegistry& registry, const std::string& backend, const std::string& modelId, int windowOverride = 0,
                                   bool torch = false);
DepthModelChoice ResolveDepthModel(const std::string& modelsDir, const std::string& backend, const std::string& modelId, int windowOverride = 0,
                                   bool torch = false);
// The model input size the estimator uses: the registry's input_size replaces the default 518 (TrtDepthEstimator rule).
int EffectiveDepthInputSize(const DepthModelChoice& choice, int requested);

// VRAM need from the registry's measurements: params.vram (TensorRT) or params.vram_torch (worker) =
// {"points": [{"window": W, "input": [w, h], "mib": M}, ...], "measured": "..."}. Conservative: the smallest measured
// window >= `window` (linear beyond the largest two), the smallest measured input with at least as many pixels (scaled up
// by pixels beyond the largest), +10 %; fp32 doubles. nullopt without measurements.
std::optional<double> EstimateDepthVramMiB(const DepthModelChoice& choice, bool torch, uint32_t inputWidth, uint32_t inputHeight, bool fp16 = true);

struct DepthAdvice {
    std::vector<std::string> warnings;  // licence, VRAM close to the limit, ignored window
    std::string info;                   // «≈ 5.2 ГБ видеопамяти из 16»
    std::string error;                  // does not fit: refuse a run (not a reuse or a restore)
};
// `vramBytes` 0 = unknown (WARP, no device): no error.
DepthAdvice AdviseDepth(const DepthModelChoice& choice, std::optional<double> estimateMiB, uint64_t vramBytes, uint32_t inputWidth, uint32_t inputHeight);
// Before an export / engine build: an error text when the estimate exceeds the free memory, else "".
std::string CheckDepthVramFree(const DepthModelChoice& choice, double estimateMiB, uint64_t freeBytes, uint64_t totalBytes);
// A smaller VDA window to suggest: half, a multiple of 4, at least 4.
int SmallerDepthWindow(int window);

}  // namespace dlssvid
