#include "stages/fg/RifeFrameGenerator.h"

#include <algorithm>
#include <vector>

#include "stages/fg/BlendFrameGenerator.h"
#include "util/Error.h"
#include "util/Log.h"

#if defined(DLSSVID_WITH_TENSORRT)
#include "ml/ModelRegistry.h"
#include "ml/TrtEngine.h"
#include "ml/TrtLoader.h"
#endif

namespace dlssvid {

#if defined(DLSSVID_WITH_TENSORRT)

struct RifeFrameGenerator::Impl {
    std::unique_ptr<ModelRegistry> registry;
    std::unique_ptr<TrtEngine> engine;
    std::vector<std::string> inputs{"img0", "img1", "timestep"};
    std::string output = "output";
    uint32_t padW = 0, padH = 0, multiple = 32;
    std::vector<float> host0, host1, hostOut;
    int64_t frames = 0;
};

namespace {
// RGB image -> NCHW float [0, 1] padded (replicate) to padW x padH
void FillNchwPadded(const PassImage& rgb, uint32_t padW, uint32_t padH, std::vector<float>& out) {
    const PassImage f = ToRgbF32(rgb);
    out.resize(static_cast<size_t>(3) * padW * padH);
    for (size_t c = 0; c < 3; ++c)
        for (uint32_t y = 0; y < padH; ++y) {
            const uint32_t sy = std::min(y, f.height - 1);
            for (uint32_t x = 0; x < padW; ++x) out[(c * padH + y) * padW + x] = f.Get(std::min(x, f.width - 1), sy, c);
        }
}
}  // namespace

RifeFrameGenerator::RifeFrameGenerator() = default;
RifeFrameGenerator::~RifeFrameGenerator() { Shutdown(); }

FgAvailability RifeFrameGenerator::Available() {
    std::string reason;
    if (!trt::Available(&reason)) return {false, "rife needs TensorRT: " + reason};
    return {true, {}};
}

void RifeFrameGenerator::Init(D3D12Device& device, const FgConfig& config) {
    Shutdown();
    config_ = config;
    diag_ = {};
    impl_ = std::make_unique<Impl>();
    Impl& im = *impl_;
    diag_.gpu = device.AdapterName();
    diag_.architecture = GpuArchitectureFromName(device.AdapterName());
    diag_.driver = NvidiaDriverFromUmd(device.UmdDriverVersion());
    diag_.driverOk = diag_.driver.valid;
    diag_.model = config.model;
    if (config.multiplier < 2 || config.multiplier > 4) Throw("fg: --multiplier must be 2 | 3 | 4");
    std::string reason;
    if (!trt::Available(&reason)) {
        diag_.hint = "rife needs TensorRT: " + reason;
        Throw("fg: " + diag_.hint);
    }
    const auto registryPath = config.modelsDir.empty() ? ModelRegistry::DefaultRegistryPath() : std::filesystem::path(config.modelsDir) / "registry.json";
    im.registry = std::make_unique<ModelRegistry>(ModelRegistry::Load(registryPath));
    const ModelEntry& e = im.registry->Get(config.model);
    if (e.stage != "fg") Throw("fg: model '" + config.model + "' is not a frame generation model");
    if (e.params.contains("inputs")) im.inputs = e.params["inputs"].get<std::vector<std::string>>();
    if (e.params.contains("output")) im.output = e.params["output"].get<std::string>();
    im.multiple = e.params.value("multiple", 32u);
    const std::filesystem::path onnx = im.registry->Fetch(e, [](uint64_t done, uint64_t total) {
        if (total && done == total) Log()->info("fg: model downloaded ({} MiB)", total / (1024 * 1024));
    });
    im.padW = (config.width + im.multiple - 1) / im.multiple * im.multiple;
    im.padH = (config.height + im.multiple - 1) / im.multiple * im.multiple;
    TrtEngine::Options opt;
    opt.fp16 = config.fp16;
    opt.cacheDir = im.registry->CacheDir();
    opt.shapes[im.inputs[0]] = {1, 3, im.padH, im.padW};
    opt.shapes[im.inputs[1]] = {1, 3, im.padH, im.padW};
    im.engine = TrtEngine::FromOnnx(onnx, opt);
    im.hostOut.resize(im.engine->Output(im.output).elements);
    if (im.hostOut.size() != static_cast<size_t>(3) * im.padW * im.padH)
        Throw("fg: rife output has " + std::to_string(im.hostOut.size()) + " elements, expected 1x3x" + std::to_string(im.padH) + "x" + std::to_string(im.padW));
    diag_.available = true;
    diag_.multiFrameMax = 3;
    diag_.createResult = "engine " + im.engine->EnginePath().filename().string();
    diag_.ok = true;
    Log()->info("fg: rife {} {}x{} (padded {}x{}) x{} fp16={}", config.model, config.width, config.height, im.padW, im.padH, config.multiplier, config.fp16);
}

void RifeFrameGenerator::Generate(const FgInputs& in, std::vector<PassImage>& out) {
    if (!impl_ || !impl_->engine) Throw("fg: rife not initialised");
    Impl& im = *impl_;
    if (!in.prev.cpu || !in.cur.cpu) Throw("fg: rife needs CPU images of both frames");
    FillNchwPadded(*in.prev.cpu, im.padW, im.padH, im.host0);
    FillNchwPadded(*in.cur.cpu, im.padW, im.padH, im.host1);
    im.engine->SetInput(im.inputs[0], im.host0.data(), im.host0.size());
    im.engine->SetInput(im.inputs[1], im.host1.data(), im.host1.size());
    out.clear();
    for (int k = 1; k < config_.multiplier; ++k) {
        const float t = static_cast<float>(k) / static_cast<float>(config_.multiplier);
        if (im.inputs.size() > 2) im.engine->SetInput(im.inputs[2], &t, 1);
        im.engine->Execute();
        im.engine->GetOutput(im.output, im.hostOut.data(), im.hostOut.size());
        PassImage img = MakePassImage(PassKind::ColorFg, config_.width, config_.height, PixelType::F16);
        for (uint32_t y = 0; y < config_.height; ++y)
            for (uint32_t x = 0; x < config_.width; ++x)
                for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, std::clamp(im.hostOut[(c * im.padH + y) * im.padW + x], 0.f, 1.f));
        out.push_back(std::move(img));
    }
    ++im.frames;
}

nlohmann::json RifeFrameGenerator::Describe() const {
    nlohmann::json j = {{"backend", "rife"}, {"model", config_.model}, {"multiplier", config_.multiplier}, {"fp16", config_.fp16}, {"guides", false}};
    if (impl_ && impl_->engine) j["engine"] = impl_->engine->EnginePath().filename().string();
    j["diagnostics"] = diag_.ToJson();
    return j;
}

void RifeFrameGenerator::Shutdown() { impl_.reset(); }

#else  // !DLSSVID_WITH_TENSORRT

struct RifeFrameGenerator::Impl {};
RifeFrameGenerator::RifeFrameGenerator() = default;
RifeFrameGenerator::~RifeFrameGenerator() = default;
FgAvailability RifeFrameGenerator::Available() { return {false, "built without TensorRT (TENSORRT_ROOT): the rife backend is unavailable"}; }
void RifeFrameGenerator::Init(D3D12Device& device, const FgConfig& config) {
    config_ = config;
    diag_ = {};
    diag_.gpu = device.AdapterName();
    diag_.hint = Available().reason;
    Throw("fg: " + diag_.hint);
}
void RifeFrameGenerator::Generate(const FgInputs&, std::vector<PassImage>&) { Throw("fg: rife not available"); }
nlohmann::json RifeFrameGenerator::Describe() const { return {{"backend", "rife"}, {"available", false}}; }
void RifeFrameGenerator::Shutdown() {}

#endif

}  // namespace dlssvid
