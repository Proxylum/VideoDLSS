#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "passes/PassImage.h"
#include "stages/upscale/IUpscaler.h"

namespace dlssvid {

class ModelRegistry;
class TrtEngine;

// Open-source super-resolution models through TensorRT — Real-ESRGAN by default (models/registry.json, family
// `realesrgan`): one ONNX with dynamic height / width per model (models/export/export_realesrgan.py, exported on first
// use), one cached engine per tile size, tiles with padding (Tiling.h) fed through TrtEngine's host copies. A CPU
// backend of the stage (ProcessesOnCpu): the frame comes and goes as RGBA16F images, the stage resamples the model's
// native scale to the target when they differ.
class TrtUpscaler final : public IUpscaler {
public:
    static UpscalerAvailability Available();
    TrtUpscaler();
    ~TrtUpscaler() override;

    std::string_view Name() const override { return "trt"; }
    void Init(D3D12Device& device, const UpscalerConfig& config) override;
    void Evaluate(ID3D12GraphicsCommandList* cl, const UpscaleInputs& in, ID3D12Resource* output) override;
    bool ProcessesOnCpu() const override { return true; }
    int NativeScale() const override { return scale_; }
    void EvaluateCpu(const PassImage& rgbaIn, PassImage& rgbaOut) override;
    nlohmann::json Describe() const override;
    void Shutdown() override;

    // The model's ONNX: <cache>/<id>.onnx, exported by the venv script when missing (throws with instructions when the
    // export environment is missing too).
    static std::filesystem::path OnnxFor(const ModelRegistry& registry, const std::string& modelId);
    const std::string& ModelId() const { return modelId_; }
    int Tile() const { return tile_; }
    int Pad() const { return pad_; }

private:
    UpscalerConfig config_;
    std::unique_ptr<ModelRegistry> registry_;
    std::unique_ptr<TrtEngine> engine_;
    std::string modelId_, license_, inputName_, outputName_;
    std::filesystem::path onnx_;
    int scale_ = 2, tile_ = 512, pad_ = 16;
    bool fp16_ = true;
    std::vector<float> inHost_, outHost_;
};

}  // namespace dlssvid
