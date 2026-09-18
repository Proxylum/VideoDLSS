#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dlssvid {

// TensorRT 10 engine: built from ONNX (FP16, fixed shapes) with an on-disk cache, executed
// synchronously on CUDA device buffers owned by the engine. One instance = one model.
class TrtEngine {
public:
    struct Options {
        bool fp16 = true;
        size_t workspaceBytes = size_t{2} << 30;  // 2 GiB
        std::filesystem::path cacheDir;           // engine cache; empty = next to the ONNX file
        bool verbose = false;
    };

    struct Binding {
        std::string name;
        std::vector<int64_t> shape;  // fully specified
        size_t elements = 0;
        size_t bytes = 0;
        bool isInput = false;
        int dataType = 0;  // nvinfer1::DataType as int (0 = FLOAT, 1 = HALF, ...)
        void* device = nullptr;
    };

    // Loads a cached engine for the ONNX file (cache key = onnx hash + GPU + TensorRT version)
    // or builds and caches it. Throws dlssvid::Error.
    static std::unique_ptr<TrtEngine> FromOnnx(const std::filesystem::path& onnx, const Options& options = {});
    static std::unique_ptr<TrtEngine> FromEngineFile(const std::filesystem::path& engine);
    ~TrtEngine();

    const std::vector<Binding>& Inputs() const { return inputs_; }
    const std::vector<Binding>& Outputs() const { return outputs_; }
    const Binding& Input(const std::string& name) const;
    const Binding& Output(const std::string& name) const;
    const std::filesystem::path& EnginePath() const { return enginePath_; }

    // Host <-> device copies (blocking) and execution (blocking).
    void SetInput(const std::string& name, const float* host, size_t elements);
    void GetOutput(const std::string& name, float* host, size_t elements) const;
    void Execute();

    // Expected cache path for an ONNX file with these options (without building anything).
    static std::filesystem::path CachePathFor(const std::filesystem::path& onnx, const Options& options);

private:
    struct Impl;
    TrtEngine();
    std::unique_ptr<Impl> impl_;
    std::vector<Binding> inputs_, outputs_;
    std::filesystem::path enginePath_;
};

}  // namespace dlssvid
