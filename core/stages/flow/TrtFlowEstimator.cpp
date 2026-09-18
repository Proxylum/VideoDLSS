#include "stages/flow/TrtFlowEstimator.h"

#include <algorithm>
#include <cmath>

#include "convert/MvConvert.h"
#include "ml/ModelRegistry.h"
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#include "stages/depth/DepthPreprocess.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

TrtFlowEstimator::TrtFlowEstimator(std::string family) : family_(std::move(family)) {}
TrtFlowEstimator::~TrtFlowEstimator() = default;

std::string TrtFlowEstimator::DefaultModel(const std::string& family) {
    if (family == "searaft") return "sea-raft-spring-m";
    Throw("unknown TensorRT flow family: " + family);
}

void TrtFlowEstimator::Init(const FlowEstimatorConfig& config, uint32_t width, uint32_t height) {
    config_ = config;
    std::string reason;
    if (!trt::Available(&reason)) Throw("backend '" + family_ + "' needs TensorRT: " + reason);
    const auto registryPath = config.modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(config.modelsDir) / "registry.json";
    registry_ = std::make_unique<ModelRegistry>(ModelRegistry::Load(registryPath));
    modelId_ = config.modelId.empty() ? DefaultModel(family_) : config.modelId;
    const ModelEntry& e = registry_->Get(modelId_);
    if (e.stage != "flow") Throw("model '" + modelId_ + "' is not a flow model");
    license_ = e.license;
    if (e.params.contains("input_names")) inputNames_ = e.params["input_names"].get<std::vector<std::string>>();
    outputName_ = e.params.value("output_name", "flow");
    inputScale_ = e.params.value("input_scale", 255.f);
    multiple_ = e.params.value("multiple", 8);

    srcW_ = width;
    srcH_ = height;
    // Model geometry: cap the shorter side at maxRes, keep aspect, round to `multiple`.
    double scale = 1.0;
    const uint32_t shorter = std::min(width, height);
    if (config.maxRes > 0 && shorter > static_cast<uint32_t>(config.maxRes)) scale = static_cast<double>(config.maxRes) / shorter;
    modelW_ = static_cast<uint32_t>(std::max<long>(multiple_, std::lround(width * scale / multiple_) * multiple_));
    modelH_ = static_cast<uint32_t>(std::max<long>(multiple_, std::lround(height * scale / multiple_) * multiple_));

    TrtEngine::Options opt;
    opt.fp16 = config.fp16;
    opt.cacheDir = registry_->CacheDir();
    engine_ = TrtEngine::FromOnnx(OnnxFor(modelW_, modelH_), opt);
    const size_t expected = static_cast<size_t>(3) * modelW_ * modelH_;
    for (const auto& n : inputNames_)
        if (engine_->Input(n).elements != expected) Throw("flow engine input '" + n + "' has unexpected size");
    host1_.assign(expected, 0.f);
    host2_.assign(expected, 0.f);
    outHost_.assign(engine_->Output(outputName_).elements, 0.f);
    Log()->info("flow backend {}: model {} ({}), {}x{} -> {}x{}", family_, modelId_, license_, width, height, modelW_, modelH_);
}

std::filesystem::path TrtFlowEstimator::OnnxFor(uint32_t w, uint32_t h) {
    const auto path = registry_->CacheDir() / (modelId_ + "_2x" + std::to_string(h) + "x" + std::to_string(w) + ".onnx");
    if (std::filesystem::exists(path)) return path;
    const auto modelsDir = registry_->CacheDir().parent_path();
    const auto venvPython = modelsDir / "export" / ".venv" / "Scripts" / "python.exe";
    const auto script = modelsDir / "export" / "export_searaft.py";
    if (!std::filesystem::exists(venvPython) || !std::filesystem::exists(script))
        Throw("ONNX for " + modelId_ + " at " + std::to_string(w) + "x" + std::to_string(h) + " not found (" + path.string() +
              ") and models/export/.venv is missing (see models/export/README.md)");
    Log()->info("exporting {} to ONNX ({}x{}) — first time only", modelId_, w, h);
    std::string out;
    const uint32_t rc = Subprocess::Run({venvPython.string(), script.string(), "--model", modelId_, "--width", std::to_string(w), "--height", std::to_string(h),
                                         "--out", path.string(), "--cache", registry_->CacheDir().string()},
                                        &out, modelsDir / "export");
    if (rc != 0 || !std::filesystem::exists(path)) Throw("ONNX export failed (exit " + std::to_string(rc) + "):\n" + out);
    return path;
}

namespace {
void FillNchw(const PassImage& rgb, uint32_t w, uint32_t h, float scale, std::vector<float>& out) {
    const PassImage r = (rgb.width == w && rgb.height == h && rgb.type == PixelType::F32) ? rgb : ResizeBilinear(rgb, w, h);
    const size_t plane = static_cast<size_t>(w) * h;
    for (size_t c = 0; c < 3; ++c)
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) out[c * plane + static_cast<size_t>(y) * w + x] = r.Get(x, y, c) * scale;
}
}  // namespace

void TrtFlowEstimator::Estimate(const FlowInput& a, const FlowInput& b, PassImage& flowOut, PassImage* confidenceOut) {
    if (!engine_) Throw("TrtFlowEstimator: Init first");
    if (!a.rgb || !b.rgb) Throw("TrtFlowEstimator: needs CPU RGB frames");
    FillNchw(*a.rgb, modelW_, modelH_, inputScale_, host1_);
    FillNchw(*b.rgb, modelW_, modelH_, inputScale_, host2_);
    engine_->SetInput(inputNames_[0], host1_.data(), host1_.size());
    engine_->SetInput(inputNames_[1], host2_.data(), host2_.size());
    engine_->Execute();
    engine_->GetOutput(outputName_, outHost_.data(), outHost_.size());
    const size_t plane = static_cast<size_t>(modelW_) * modelH_;
    if (outHost_.size() < 2 * plane) Throw("flow output smaller than expected");
    PassImage model = MakePassImage(PassKind::MvRaw, modelW_, modelH_);
    float* m = model.As<float>();
    for (size_t i = 0; i < plane; ++i) {
        const float u = outHost_[i], v = outHost_[plane + i];
        m[2 * i] = std::isfinite(u) ? u : 0.f;
        m[2 * i + 1] = std::isfinite(v) ? v : 0.f;
    }
    flowOut = (modelW_ == srcW_ && modelH_ == srcH_) ? std::move(model) : ScaleMv(model, srcW_, srcH_);
    flowOut.channels = Spec(PassKind::MvRaw).channels;
    if (confidenceOut) {
        confidenceOut->Allocate(srcW_, srcH_, PixelType::F32, {"A"});
        float* c = confidenceOut->As<float>();
        std::fill(c, c + static_cast<size_t>(srcW_) * srcH_, 1.f);
    }
}

void TrtFlowEstimator::Shutdown() { engine_.reset(); }

nlohmann::json TrtFlowEstimator::Describe() const {
    return {{"backend", family_}, {"runtime", "tensorrt " + trt::LibraryVersion()}, {"model", modelId_}, {"license", license_},
            {"model_input", {modelW_, modelH_}}, {"fp16", config_.fp16}, {"engine", engine_ ? engine_->EnginePath().filename().string() : ""}};
}

}  // namespace dlssvid
