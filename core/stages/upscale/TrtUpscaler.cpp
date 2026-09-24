#include "stages/upscale/TrtUpscaler.h"

#include <algorithm>
#include <cmath>

#include "ml/ModelRegistry.h"
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#include "stages/upscale/Tiling.h"
#include "util/Error.h"
#include "util/Half.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

UpscalerAvailability TrtUpscaler::Available() {
    std::string reason;
    if (!trt::Available(&reason))
        return {false, "TensorRT runtime not found (" + reason + "): install the pip package tensorrt-cu12 into models/export/.venv or set DLSSVID_TENSORRT_DIR (docs/dll-setup.md)"};
    return {true, {}};
}

std::filesystem::path TrtUpscaler::OnnxFor(const ModelRegistry& registry, const std::string& modelId) {
    const ModelEntry& e = registry.Get(modelId);
    const auto path = registry.CacheDir() / (modelId + ".onnx");
    if (std::filesystem::exists(path)) return path;
    if (e.params.value("family", "") != "realesrgan")
        Throw("ONNX for '" + modelId + "' not found (" + path.string() + ") and no exporter is known for family '" + e.params.value("family", "") + "'");
    const auto modelsDir = registry.CacheDir().parent_path();
    const auto venvPython = modelsDir / "export" / ".venv" / "Scripts" / "python.exe";
    const auto script = modelsDir / "export" / "export_realesrgan.py";
    if (!std::filesystem::exists(venvPython) || !std::filesystem::exists(script))
        Throw("ONNX for " + modelId + " not found (" + path.string() + ") and the export environment is missing: create models/export/.venv (models/export/README.md) or run export_realesrgan.py --model " +
              modelId + " elsewhere and copy the file");
    Log()->info("exporting {} to ONNX with {} (downloads the weights on first use) — first time only", modelId, script.filename().string());
    std::string out;
    const uint32_t rc = Subprocess::Run({venvPython.string(), script.string(), "--model", modelId, "--out", path.string(), "--cache", registry.CacheDir().string()}, &out,
                                        modelsDir / "export");
    if (rc != 0 || !std::filesystem::exists(path)) Throw("ONNX export of " + modelId + " failed (exit " + std::to_string(rc) + "):\n" + out);
    return path;
}

void TrtUpscaler::Init(D3D12Device&, const UpscalerConfig& config) {
    Shutdown();
    config_ = config;
    const UpscalerAvailability a = Available();
    if (!a.available) Throw("trt: " + a.reason);
    const std::string modelsDir = config.extra.value("models_dir", "");
    registry_ = std::make_unique<ModelRegistry>(ModelRegistry::Load(modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(modelsDir) / "registry.json"));
    modelId_ = config.extra.value("model", "");
    if (modelId_.empty()) modelId_ = kTrtDefaultUpscaleModel;
    const ModelEntry& e = registry_->Get(modelId_);
    if (e.stage != "upscale") Throw("trt: model '" + modelId_ + "' is not an upscale model (stage " + e.stage + ")");
    license_ = e.license;
    scale_ = e.params.value("scale", 2);
    if (scale_ < 1 || scale_ > 8) Throw("trt: model '" + modelId_ + "' has an unsupported scale " + std::to_string(scale_));
    tile_ = config.extra.value("tile", 0);
    if (tile_ <= 0) tile_ = e.params.value("tile", 512);
    tile_ = std::max(16, tile_ / 8 * 8);  // multiples of 8: the models pixel-unshuffle by 2 or 4
    pad_ = e.params.value("tile_pad", 16);
    if (2 * pad_ >= tile_) pad_ = tile_ / 4;
    fp16_ = config.extra.value("fp16", true);
    inputName_ = e.params.value("input_name", "image");
    outputName_ = e.params.value("output_name", "upscaled");
    onnx_ = OnnxFor(*registry_, modelId_);
    TrtEngine::Options opt;
    opt.fp16 = fp16_;
    opt.cacheDir = registry_->CacheDir();
    opt.shapes[inputName_] = {1, 3, tile_, tile_};
    engine_ = TrtEngine::FromOnnx(onnx_, opt);
    const size_t inElems = static_cast<size_t>(3) * tile_ * tile_, outElems = inElems * static_cast<size_t>(scale_) * scale_;
    if (engine_->Input(inputName_).elements != inElems)
        Throw("trt: engine input '" + inputName_ + "' has " + std::to_string(engine_->Input(inputName_).elements) + " elements, expected " + std::to_string(inElems));
    if (engine_->Output(outputName_).elements != outElems)
        Throw("trt: engine output '" + outputName_ + "' has " + std::to_string(engine_->Output(outputName_).elements) + " elements, expected " + std::to_string(outElems) +
              " (scale " + std::to_string(scale_) + ")");
    inHost_.assign(inElems, 0.f);
    outHost_.assign(outElems, 0.f);
    Log()->info("upscale trt: model {} ({}) x{}, tile {} + {} px context, {} — engine {}", modelId_, license_, scale_, tile_, pad_, fp16_ ? "fp16" : "fp32",
                engine_->EnginePath().filename().string());
}

TrtUpscaler::TrtUpscaler() = default;
TrtUpscaler::~TrtUpscaler() { Shutdown(); }

void TrtUpscaler::Evaluate(ID3D12GraphicsCommandList*, const UpscaleInputs&, ID3D12Resource*) { Throw("trt: a CPU-side upscaler — the stage calls EvaluateCpu"); }

void TrtUpscaler::EvaluateCpu(const PassImage& rgbaIn, PassImage& rgbaOut) {
    if (!engine_) Throw("trt: not initialised");
    if (rgbaIn.type != PixelType::F16 || rgbaIn.ChannelCount() != 4) Throw("trt: expects an RGBA16F frame");
    // RGBA16F -> RGB float in [0, 1] (the models were trained on sRGB-encoded images)
    PassImage rgb;
    rgb.Allocate(rgbaIn.width, rgbaIn.height, PixelType::F32, {"R", "G", "B"});
    const uint16_t* src = rgbaIn.As<uint16_t>();
    float* dst = rgb.As<float>();
    const size_t n = static_cast<size_t>(rgbaIn.width) * rgbaIn.height;
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) dst[i * 3 + c] = std::clamp(HalfToFloat(src[i * 4 + c]), 0.f, 1.f);
    const PassImage up = UpscaleTiled(rgb, scale_, tile_, pad_, [&](const float* in, float* out) {
        engine_->SetInput(inputName_, in, inHost_.size());
        engine_->Execute();
        engine_->GetOutput(outputName_, out, outHost_.size());
    });
    rgbaOut.Allocate(up.width, up.height, PixelType::F16, {"R", "G", "B", "A"});
    const float* u = up.As<float>();
    uint16_t* o = rgbaOut.As<uint16_t>();
    const size_t m = static_cast<size_t>(up.width) * up.height;
    const uint16_t one = FloatToHalf(1.f);
    for (size_t i = 0; i < m; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = u[i * 3 + c];
            o[i * 4 + c] = FloatToHalf(std::isfinite(v) ? std::clamp(v, 0.f, 1.f) : 0.f);
        }
        o[i * 4 + 3] = one;
    }
}

nlohmann::json TrtUpscaler::Describe() const {
    return {{"backend", "trt"}, {"runtime", "tensorrt " + trt::LibraryVersion()}, {"model", modelId_}, {"license", license_}, {"model_scale", scale_},
            {"tile", tile_},    {"tile_pad", pad_},  {"fp16", fp16_},   {"engine", engine_ ? engine_->EnginePath().filename().string() : ""}};
}

void TrtUpscaler::Shutdown() {
    engine_.reset();
    inHost_.clear();
    outHost_.clear();
}

}  // namespace dlssvid
