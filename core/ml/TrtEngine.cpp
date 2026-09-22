#include "ml/TrtEngine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <numeric>

#include "ml/TrtLoader.h"
#include "util/Error.h"
#include "util/Half.h"
#include "util/Log.h"
#include "util/Paths.h"
#include "util/Sha256.h"

namespace dlssvid {

namespace {

void CheckCudaRt(cudaError_t err, const char* what) {
    if (err != cudaSuccess) Throw(std::string(what) + " failed: " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

class Logger final : public nvinfer1::ILogger {
public:
    explicit Logger(bool verbose) : verbose_(verbose) {}
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR: Log()->error("[TensorRT] {}", msg); break;
            case Severity::kWARNING: Log()->warn("[TensorRT] {}", msg); break;
            case Severity::kINFO:
                if (verbose_) Log()->info("[TensorRT] {}", msg);
                else Log()->debug("[TensorRT] {}", msg);
                break;
            default:
                if (verbose_) Log()->debug("[TensorRT] {}", msg);
                break;
        }
    }

private:
    bool verbose_;
};

std::string GpuKey() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "gpu";
    std::string name = prop.name;
    for (auto& c : name) c = std::isalnum(static_cast<unsigned char>(c)) ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : '_';
    return name + "_sm" + std::to_string(prop.major) + std::to_string(prop.minor);
}

size_t ElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT64: return 8;
        default: return 4;
    }
}

std::vector<char> ReadAll(const std::filesystem::path& p) {
    std::ifstream in(Win32LongPath(p), std::ios::binary);  // engine cache names are long: past MAX_PATH in deep folders
    if (!in) Throw("cannot open " + p.string());
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// TensorRT keeps one logger per process: hand every builder/runtime the same instance.
Logger& GlobalLogger(bool verbose) {
    static Logger logger(verbose);
    return logger;
}

}  // namespace

struct TrtEngine::Impl {
    Logger& logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void*> deviceBuffers;
    cudaStream_t stream = nullptr;

    explicit Impl(bool verbose) : logger(GlobalLogger(verbose)) {}
    ~Impl() {
        for (void* p : deviceBuffers)
            if (p) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
        context.reset();
        engine.reset();
        runtime.reset();
    }
};

TrtEngine::TrtEngine() = default;
TrtEngine::~TrtEngine() = default;

std::filesystem::path TrtEngine::CachePathFor(const std::filesystem::path& onnx, const Options& options) {
    const std::filesystem::path dir = options.cacheDir.empty() ? onnx.parent_path() : options.cacheDir;
    const std::string hash = Sha256File(onnx).substr(0, 16);
    std::string name = onnx.stem().string() + "." + hash + "." + GpuKey() + ".trt" + trt::LibraryVersion() + (options.fp16 ? ".fp16" : ".fp32");
    for (const auto& [input, dims] : options.shapes) {
        name += "." + input;
        for (size_t d = 0; d < dims.size(); ++d) name += (d ? "x" : "_") + std::to_string(dims[d]);
    }
    name += ".engine";
    return dir / name;
}

std::unique_ptr<TrtEngine> TrtEngine::FromOnnx(const std::filesystem::path& onnx, const Options& options) {
    std::string reason;
    if (!trt::Available(&reason)) Throw(reason);
    if (!std::filesystem::exists(onnx)) Throw("ONNX file not found: " + onnx.string());

    const std::filesystem::path cache = CachePathFor(onnx, options);
    if (std::filesystem::exists(Win32LongPath(cache))) {
        Log()->info("TensorRT engine cache hit: {}", cache.string());
        return FromEngineFile(cache);
    }

    Log()->info("building TensorRT engine from {} (fp16={}) — this can take several minutes", onnx.string(), options.fp16);
    std::unique_ptr<TrtEngine> self(new TrtEngine());
    self->impl_ = std::make_unique<Impl>(options.verbose);
    Impl& im = *self->impl_;

    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(im.logger));
    if (!builder) Throw("createInferBuilder failed (TensorRT version mismatch between headers and DLL?)");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) Throw("createNetworkV2 failed");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, im.logger));
    if (!parser) Throw("createParser failed (nvonnxparser_10.dll)");
    const std::string onnxStr = onnx.string();
    if (!parser->parseFromFile(onnxStr.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::string errs;
        for (int i = 0; i < parser->getNbErrors(); ++i) errs += std::string("\n  ") + parser->getError(i)->desc();
        Throw("ONNX parse failed for " + onnxStr + errs);
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, options.workspaceBytes);
    if (options.fp16 && builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Dynamic inputs: one optimization profile with the fixed shape the caller asked for (min = opt = max).
    bool needProfile = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor* t = network->getInput(i);
        const nvinfer1::Dims dims = t->getDimensions();
        bool dynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) dynamic = dynamic || dims.d[d] < 0;
        if (!dynamic) continue;
        needProfile = true;
        if (options.shapes.find(t->getName()) == options.shapes.end())
            Throw("input '" + std::string(t->getName()) + "' of " + onnxStr + " has a dynamic shape: pass its fixed shape in TrtEngine::Options::shapes");
    }
    if (needProfile) {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::ITensor* t = network->getInput(i);
            const auto it = options.shapes.find(t->getName());
            if (it == options.shapes.end()) continue;
            nvinfer1::Dims dims{};
            dims.nbDims = static_cast<int32_t>(it->second.size());
            for (int d = 0; d < dims.nbDims; ++d) dims.d[d] = it->second[static_cast<size_t>(d)];
            for (const auto sel : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
                if (!profile->setDimensions(t->getName(), sel, dims)) Throw("setDimensions failed for input '" + std::string(t->getName()) + "'");
        }
        if (config->addOptimizationProfile(profile) < 0) Throw("addOptimizationProfile failed");
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) Throw("buildSerializedNetwork failed for " + onnxStr);
    std::filesystem::create_directories(Win32LongPath(cache.parent_path()));
    {
        std::ofstream out(Win32LongPath(cache), std::ios::binary);
        if (!out) Throw("cannot write engine cache " + cache.string());
        out.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    }
    Log()->info("TensorRT engine cached: {} ({} MiB)", cache.string(), plan->size() / (1024 * 1024));
    return FromEngineFile(cache);
}

std::unique_ptr<TrtEngine> TrtEngine::FromEngineFile(const std::filesystem::path& path) {
    std::string reason;
    if (!trt::Available(&reason)) Throw(reason);
    std::unique_ptr<TrtEngine> self(new TrtEngine());
    self->impl_ = std::make_unique<Impl>(false);
    Impl& im = *self->impl_;
    self->enginePath_ = path;

    const std::vector<char> blob = ReadAll(path);
    im.runtime.reset(nvinfer1::createInferRuntime(im.logger));
    if (!im.runtime) Throw("createInferRuntime failed");
    im.engine.reset(im.runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!im.engine) Throw("deserializeCudaEngine failed: " + path.string() + " (delete the cache file to rebuild)");
    im.context.reset(im.engine->createExecutionContext());
    if (!im.context) Throw("createExecutionContext failed");
    CheckCudaRt(cudaStreamCreate(&im.stream), "cudaStreamCreate");

    const int n = im.engine->getNbIOTensors();
    im.deviceBuffers.assign(static_cast<size_t>(n), nullptr);
    // engines built with a profile: the inputs take the profile's (single) shape, the outputs follow from it
    for (int i = 0; i < n; ++i) {
        const char* name = im.engine->getIOTensorName(i);
        if (im.engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        const nvinfer1::Dims dims = im.engine->getTensorShape(name);
        bool dynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) dynamic = dynamic || dims.d[d] < 0;
        if (!dynamic) continue;
        const nvinfer1::Dims fixed = im.engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
        if (!im.context->setInputShape(name, fixed)) Throw(std::string("setInputShape failed for '") + name + "'");
    }
    for (int i = 0; i < n; ++i) {
        const char* name = im.engine->getIOTensorName(i);
        Binding b;
        b.name = name;
        b.isInput = im.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        const nvinfer1::Dims dims = im.context->getTensorShape(name);
        b.elements = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] < 0) Throw("dynamic shape on tensor '" + b.name + "' could not be resolved (export with fixed shapes or pass TrtEngine::Options::shapes)");
            b.shape.push_back(dims.d[d]);
            b.elements *= static_cast<size_t>(dims.d[d]);
        }
        const nvinfer1::DataType dt = im.engine->getTensorDataType(name);
        b.dataType = static_cast<int>(dt);
        b.bytes = b.elements * ElementSize(dt);
        CheckCudaRt(cudaMalloc(&b.device, b.bytes), "cudaMalloc");
        im.deviceBuffers[static_cast<size_t>(i)] = b.device;
        if (!im.context->setTensorAddress(name, b.device)) Throw("setTensorAddress failed for " + b.name);
        (b.isInput ? self->inputs_ : self->outputs_).push_back(b);
    }
    Log()->info("TensorRT engine {}: {} inputs, {} outputs", path.filename().string(), self->inputs_.size(), self->outputs_.size());
    return self;
}

const TrtEngine::Binding& TrtEngine::Input(const std::string& name) const {
    for (const auto& b : inputs_)
        if (b.name == name) return b;
    Throw("no input tensor '" + name + "'");
}

const TrtEngine::Binding& TrtEngine::Output(const std::string& name) const {
    for (const auto& b : outputs_)
        if (b.name == name) return b;
    Throw("no output tensor '" + name + "'");
}

void TrtEngine::SetInput(const std::string& name, const float* host, size_t elements) {
    const Binding& b = Input(name);
    if (elements != b.elements) Throw("input '" + name + "' expects " + std::to_string(b.elements) + " elements, got " + std::to_string(elements));
    if (b.dataType == static_cast<int>(nvinfer1::DataType::kFLOAT)) {
        CheckCudaRt(cudaMemcpy(b.device, host, b.bytes, cudaMemcpyHostToDevice), "cudaMemcpy(input)");
    } else if (b.dataType == static_cast<int>(nvinfer1::DataType::kHALF)) {
        std::vector<uint16_t> tmp(elements);
        for (size_t i = 0; i < elements; ++i) tmp[i] = FloatToHalf(host[i]);
        CheckCudaRt(cudaMemcpy(b.device, tmp.data(), b.bytes, cudaMemcpyHostToDevice), "cudaMemcpy(input half)");
    } else {
        Throw("input '" + name + "' has an unsupported data type");
    }
}

void TrtEngine::GetOutput(const std::string& name, float* host, size_t elements) const {
    const Binding& b = Output(name);
    if (elements != b.elements) Throw("output '" + name + "' has " + std::to_string(b.elements) + " elements, got buffer for " + std::to_string(elements));
    if (b.dataType == static_cast<int>(nvinfer1::DataType::kFLOAT)) {
        CheckCudaRt(cudaMemcpy(host, b.device, b.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    } else if (b.dataType == static_cast<int>(nvinfer1::DataType::kHALF)) {
        std::vector<uint16_t> tmp(elements);
        CheckCudaRt(cudaMemcpy(tmp.data(), b.device, b.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(output half)");
        for (size_t i = 0; i < elements; ++i) host[i] = HalfToFloat(tmp[i]);
    } else {
        Throw("output '" + name + "' has an unsupported data type");
    }
}

void TrtEngine::Execute() {
    Impl& im = *impl_;
    if (!im.context->enqueueV3(im.stream)) Throw("TensorRT enqueueV3 failed");
    CheckCudaRt(cudaStreamSynchronize(im.stream), "cudaStreamSynchronize");
}

}  // namespace dlssvid
