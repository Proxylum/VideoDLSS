#include "stages/depth/TrtDepthEstimator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "ml/ModelRegistry.h"
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

TrtDepthEstimator::TrtDepthEstimator(std::string family) : family_(std::move(family)) {}
TrtDepthEstimator::~TrtDepthEstimator() = default;

void TrtDepthEstimator::Init(const DepthEstimatorConfig& config) {
    config_ = config;
    // the model first: a foreign or unknown id is refused before "needs TensorRT" could hide it (TASK-0039)
    registry_ = std::make_unique<ModelRegistry>(ModelRegistry::Load(DepthRegistryPath(config.modelsDir)));
    choice_ = ResolveDepthModel(*registry_, family_, config.modelId, config.window);
    for (const auto& w : choice_.warnings) Log()->warn("{}", w);
    std::string reason;
    if (!trt::Available(&reason)) Throw("backend '" + family_ + "' needs TensorRT: " + reason);
    modelId_ = choice_.modelId;
    const ModelEntry& e = registry_->Get(modelId_);
    license_ = e.license;
    metric_ = e.params.value("metric", false);
    disparity_ = e.params.value("output", "depth") == "disparity";
    window_ = choice_.window;
    overlap_ = choice_.overlap;
    inputName_ = e.params.value("input_name", "image");
    outputName_ = e.params.value("output_name", "depth");
    skyName_ = e.params.value("sky_output_name", "");
    skyThreshold_ = e.params.value("sky_threshold", 0.3f);
    skyQuantile_ = e.params.value("sky_quantile", 0.99f);
    if (e.params.contains("mean")) for (int i = 0; i < 3; ++i) norm_.mean[i] = e.params["mean"][i].get<float>();
    if (e.params.contains("std")) for (int i = 0; i < 3; ++i) norm_.std[i] = e.params["std"][i].get<float>();
    config_.inputSize = EffectiveDepthInputSize(choice_, config.inputSize);
    guidedRadius_ = config.extra.value("guided_radius", 8);
    Log()->info("depth backend {}: model {} ({}), window {} / overlap {}, TensorRT {}", family_, modelId_, license_, window_, overlap_, trt::LibraryVersion());
    if (choice_.nonCommercial) Log()->warn("depth: {} is licensed {} - non-commercial use only", modelId_, license_);
}

std::filesystem::path TrtDepthEstimator::OnnxFor(const ModelInputSize& size) {
    const ModelEntry& e = registry_->Get(modelId_);
    const std::string name = modelId_ + "_" + std::to_string(window_) + "x" + std::to_string(size.height) + "x" + std::to_string(size.width) + ".onnx";
    const auto path = registry_->CacheDir() / name;
    if (std::filesystem::exists(path)) return path;

    // Produce the ONNX with the export script (needs the venv; downloads weights via HF on first use).
    const auto modelsDir = registry_->CacheDir().parent_path();
    const auto venvPython = modelsDir / "export" / ".venv" / "Scripts" / "python.exe";
    const auto script = modelsDir / "export" / (choice_.family == "vda" ? "export_vda.py" : "export_da3.py");
    if (!std::filesystem::exists(venvPython) || !std::filesystem::exists(script))
        Throw("ONNX for " + modelId_ + " at " + std::to_string(size.width) + "x" + std::to_string(size.height) + " not found (" + path.string() +
              ") and the export environment is missing: create models/export/.venv (see models/export/README.md) or export the ONNX manually");
    Log()->info("exporting {} to ONNX ({}x{}) with {} — first time only", modelId_, size.width, size.height, script.filename().string());
    std::vector<std::string> args = {venvPython.string(), script.string(), "--model", modelId_, "--width", std::to_string(size.width),
                                     "--height", std::to_string(size.height), "--out", path.string(), "--cache", registry_->CacheDir().string()};
    if (choice_.family == "vda") {  // always explicit: the script must not guess the window the engine is built for
        args.push_back("--window");
        args.push_back(std::to_string(window_));
    }
    std::string out;
    const uint32_t rc = Subprocess::Run(args, &out, modelsDir / "export");
    if (rc != 0 || !std::filesystem::exists(path)) Throw("ONNX export failed (exit " + std::to_string(rc) + "):\n" + out);
    const auto data = std::filesystem::path(path.string() + ".data");
    if (std::filesystem::exists(data)) Log()->info("{} has external data {} - delete both together", path.filename().string(), data.filename().string());
    (void)e;
    return path;
}

void TrtDepthEstimator::EnsureEngine(uint32_t srcWidth, uint32_t srcHeight) {
    if (engine_ && srcW_ == srcWidth && srcH_ == srcHeight) return;
    srcW_ = srcWidth;
    srcH_ = srcHeight;
    // Frames larger than maxInputRes (shorter side) are treated as if downscaled: the model input
    // geometry only depends on the aspect ratio, so this only matters for the upsample step.
    inputSize_ = ComputeModelInputSize(srcWidth, srcHeight, config_.inputSize);
    // before an export and a build of 10-20 minutes: does the model fit the free GPU memory (registry measurements)?
    if (const auto need = EstimateDepthVramMiB(choice_, false, inputSize_.width, inputSize_.height, config_.fp16)) {
        const auto [free, total] = TrtEngine::DeviceMemoryInfo();
        const std::string err = CheckDepthVramFree(choice_, *need, free, total);
        const char* skip = std::getenv("DLSSVID_DEPTH_SKIP_VRAM_CHECK");
        if (!err.empty() && !(skip && std::string(skip) == "1")) Throw("depth: " + err);
        if (!err.empty()) Log()->warn("depth: {} (ignored: DLSSVID_DEPTH_SKIP_VRAM_CHECK=1)", err);
    }
    TrtEngine::Options opt;
    opt.fp16 = config_.fp16;
    opt.cacheDir = registry_->CacheDir();
    // the builder's scratch limit: with 2 GiB the DPT head of VDA Base / Large at 16-32 frames has no tactic at all
    // ("Could not find any implementation", the tactics ask for 2.1-3.4 GB; TASK-0039) - half the free memory, 2..8 GiB
    opt.workspaceBytes = DepthWorkspaceBytes(TrtEngine::DeviceMemoryInfo().first);
    engine_ = TrtEngine::FromOnnx(OnnxFor(inputSize_), opt);
    const auto& in = engine_->Input(inputName_);
    const size_t expected = static_cast<size_t>(window_) * 3 * inputSize_.width * inputSize_.height;
    if (in.elements != expected)
        Throw("engine input '" + inputName_ + "' has " + std::to_string(in.elements) + " elements, expected " + std::to_string(expected) +
              " (window " + std::to_string(window_) + ", " + std::to_string(inputSize_.width) + "x" + std::to_string(inputSize_.height) + ")");
    inputHost_.assign(expected, 0.f);
    outputHost_.assign(engine_->Output(outputName_).elements, 0.f);
    hasSky_ = false;
    if (!skyName_.empty()) {
        for (const auto& o : engine_->Outputs()) hasSky_ = hasSky_ || o.name == skyName_;
        if (hasSky_) skyHost_.assign(engine_->Output(skyName_).elements, 0.f);
    }
}

namespace {
// DA3 rule: pixels with sky score >= threshold take the `quantile` of the non-sky depth.
void ApplySkyRule(float* depth, const float* sky, size_t n, float threshold, float quantile) {
    std::vector<float> nonSky;
    nonSky.reserve(n);
    size_t skyCount = 0;
    for (size_t i = 0; i < n; ++i) {
        if (sky[i] < threshold) nonSky.push_back(depth[i]);
        else ++skyCount;
    }
    if (nonSky.size() <= 10 || skyCount <= 10) return;
    if (nonSky.size() > 100000) {  // subsample like DA3 (deterministic stride instead of random)
        std::vector<float> sub;
        sub.reserve(100000);
        const size_t step = nonSky.size() / 100000;
        for (size_t i = 0; i < nonSky.size() && sub.size() < 100000; i += step) sub.push_back(nonSky[i]);
        nonSky.swap(sub);
    }
    const size_t k = static_cast<size_t>(quantile * static_cast<float>(nonSky.size() - 1));
    std::nth_element(nonSky.begin(), nonSky.begin() + static_cast<ptrdiff_t>(k), nonSky.end());
    const float maxDepth = nonSky[k];
    for (size_t i = 0; i < n; ++i)
        if (sky[i] >= threshold) depth[i] = maxDepth;
}
}  // namespace

void TrtDepthEstimator::Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) {
    if (rgb.empty()) Throw("TrtDepthEstimator::Estimate: no frames");
    if (static_cast<int>(rgb.size()) > window_) Throw("TrtDepthEstimator: got " + std::to_string(rgb.size()) + " frames for a window of " + std::to_string(window_));
    EnsureEngine(rgb.front()->width, rgb.front()->height);

    // Fill the window (pad by repeating the last frame for a short tail).
    inputHost_.clear();
    for (int i = 0; i < window_; ++i) {
        const PassImage* f = rgb[std::min<size_t>(static_cast<size_t>(i), rgb.size() - 1)];
        const PassImage small = ResizeBilinear(*f, inputSize_.width, inputSize_.height);
        ToNchwNormalized(small, norm_, inputHost_);
    }
    engine_->SetInput(inputName_, inputHost_.data(), inputHost_.size());
    engine_->Execute();
    engine_->GetOutput(outputName_, outputHost_.data(), outputHost_.size());
    if (hasSky_) engine_->GetOutput(skyName_, skyHost_.data(), skyHost_.size());

    const size_t plane = static_cast<size_t>(inputSize_.width) * inputSize_.height;
    if (outputHost_.size() < plane * rgb.size()) Throw("TrtDepthEstimator: output tensor smaller than expected");
    depthOut.clear();
    for (size_t i = 0; i < rgb.size(); ++i) {
        PassImage low;
        low.Allocate(inputSize_.width, inputSize_.height, PixelType::F32, {"Z"});
        float* d = low.As<float>();
        const float* src = outputHost_.data() + i * plane;
        for (size_t k = 0; k < plane; ++k) {
            float v = src[k];
            if (disparity_) v = v > 1e-6f ? 1.f / v : 0.f;
            d[k] = std::isfinite(v) ? v : 0.f;
        }
        if (hasSky_ && skyHost_.size() >= plane * (i + 1)) ApplySkyRule(d, skyHost_.data() + i * plane, plane, skyThreshold_, skyQuantile_);
        PassImage full = UpsampleDepthGuided(low, *rgb[i], guidedRadius_);
        full.channels = {"Z"};
        depthOut.push_back(std::move(full));
    }
}

void TrtDepthEstimator::Shutdown() { engine_.reset(); }

nlohmann::json TrtDepthEstimator::Describe() const {
    return {{"backend", family_}, {"runtime", "tensorrt " + trt::LibraryVersion()}, {"model", modelId_}, {"license", license_},
            {"metric", metric_},   {"window", window_},  {"overlap", overlap_}, {"input", {inputSize_.width, inputSize_.height}}, {"fp16", config_.fp16},
            {"engine", engine_ ? engine_->EnginePath().filename().string() : ""}, {"vram_mib", engine_ ? engine_->DeviceBytes() >> 20 : 0}};
}

}  // namespace dlssvid
