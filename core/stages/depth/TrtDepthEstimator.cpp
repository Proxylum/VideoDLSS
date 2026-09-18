#include "stages/depth/TrtDepthEstimator.h"

#include <algorithm>
#include <cmath>

#include "ml/ModelRegistry.h"
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

TrtDepthEstimator::TrtDepthEstimator(std::string family) : family_(std::move(family)) {}
TrtDepthEstimator::~TrtDepthEstimator() = default;

std::string TrtDepthEstimator::DefaultModel(const std::string& family) {
    if (family == "da3") return "da3metric-large";
    if (family == "vda") return "metric-vda-small";
    Throw("unknown TensorRT depth family: " + family);
}

void TrtDepthEstimator::Init(const DepthEstimatorConfig& config) {
    config_ = config;
    std::string reason;
    if (!trt::Available(&reason)) Throw("backend '" + family_ + "' needs TensorRT: " + reason);
    const auto registryPath = config.modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(config.modelsDir) / "registry.json";
    registry_ = std::make_unique<ModelRegistry>(ModelRegistry::Load(registryPath));
    modelId_ = config.modelId.empty() ? DefaultModel(family_) : config.modelId;
    const ModelEntry& e = registry_->Get(modelId_);
    if (e.stage != "depth") Throw("model '" + modelId_ + "' is not a depth model");
    license_ = e.license;
    metric_ = e.params.value("metric", false);
    disparity_ = e.params.value("output", "depth") == "disparity";
    window_ = e.params.value("window", 1);
    overlap_ = e.params.value("overlap", 0);
    inputName_ = e.params.value("input_name", "image");
    outputName_ = e.params.value("output_name", "depth");
    skyName_ = e.params.value("sky_output_name", "");
    skyThreshold_ = e.params.value("sky_threshold", 0.3f);
    skyQuantile_ = e.params.value("sky_quantile", 0.99f);
    if (e.params.contains("mean")) for (int i = 0; i < 3; ++i) norm_.mean[i] = e.params["mean"][i].get<float>();
    if (e.params.contains("std")) for (int i = 0; i < 3; ++i) norm_.std[i] = e.params["std"][i].get<float>();
    if (config.inputSize > 0 && e.params.contains("input_size") && config.inputSize == 518) config_.inputSize = e.params["input_size"].get<int>();
    guidedRadius_ = config.extra.value("guided_radius", 8);
    Log()->info("depth backend {}: model {} ({}), TensorRT {}", family_, modelId_, license_, trt::LibraryVersion());
}

std::filesystem::path TrtDepthEstimator::OnnxFor(const ModelInputSize& size) {
    const ModelEntry& e = registry_->Get(modelId_);
    const std::string name = modelId_ + "_" + std::to_string(window_) + "x" + std::to_string(size.height) + "x" + std::to_string(size.width) + ".onnx";
    const auto path = registry_->CacheDir() / name;
    if (std::filesystem::exists(path)) return path;

    // Produce the ONNX with the export script (needs the venv; downloads weights via HF on first use).
    const auto modelsDir = registry_->CacheDir().parent_path();
    const auto venvPython = modelsDir / "export" / ".venv" / "Scripts" / "python.exe";
    const auto script = modelsDir / "export" / (family_ == "vda" ? "export_vda.py" : "export_da3.py");
    if (!std::filesystem::exists(venvPython) || !std::filesystem::exists(script))
        Throw("ONNX for " + modelId_ + " at " + std::to_string(size.width) + "x" + std::to_string(size.height) + " not found (" + path.string() +
              ") and the export environment is missing: create models/export/.venv (see models/export/README.md) or export the ONNX manually");
    Log()->info("exporting {} to ONNX ({}x{}) with {} — first time only", modelId_, size.width, size.height, script.filename().string());
    std::vector<std::string> args = {venvPython.string(), script.string(), "--model", modelId_, "--width", std::to_string(size.width),
                                     "--height", std::to_string(size.height), "--out", path.string(), "--cache", registry_->CacheDir().string()};
    if (window_ > 1) {
        args.push_back("--window");
        args.push_back(std::to_string(window_));
    }
    std::string out;
    const uint32_t rc = Subprocess::Run(args, &out, modelsDir / "export");
    if (rc != 0 || !std::filesystem::exists(path)) Throw("ONNX export failed (exit " + std::to_string(rc) + "):\n" + out);
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
    TrtEngine::Options opt;
    opt.fp16 = config_.fp16;
    opt.cacheDir = registry_->CacheDir();
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
            {"metric", metric_},   {"window", window_},  {"input", {inputSize_.width, inputSize_.height}}, {"fp16", config_.fp16},
            {"engine", engine_ ? engine_->EnginePath().filename().string() : ""}};
}

}  // namespace dlssvid
