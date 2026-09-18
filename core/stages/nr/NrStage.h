#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "passes/PassSequence.h"
#include "pipeline/IStage.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrCompose.h"
#include "stages/tonemap/Tonemapper.h"
#include "stages/upscale/BicubicUpscaler.h"

namespace dlssvid {

// Pipeline stage "Neural Rendering" (ТЗ §4): colour (color_sr, the cache slot or the decoded frame) ->
// tonemap -> [downsample to the model resolution] -> N passes of the model (depth_dlss / mv_dlss as
// guides) -> resolve with masks and the optional temporal filter -> `color_nr` (EXR half / PNG16), the
// cache slot and an optional preview video. Missing DLL / driver / GPU: the CLI fails with the
// instruction; in a pipeline `disableWhenUnavailable` turns the stage into a no-op with a message.
struct NrStageOptions {
    std::string backend = "ngx";        // ngx | stub
    // model (docs/plans/06-nr.md)
    float intensity = 1.f;
    std::string style = "natural";      // natural | cinematic | default | 0..15
    int preset = 3;                     // 0..3
    float localTone = 1.f, localStructure = 1.f, skinStructure = -1.f;
    bool autoMask = false;
    int passes = 1;                     // 1..2: the model applied again to its own output (own history per pass)
    double modelScale = 1.0;            // model resolution relative to the colour (0.25..2)
    float transfer = 1.f;               // ratio transfer strength when modelScale != 1
    float maxRatio = 4.f;
    bool useGuides = true;              // depth_dlss / mv_dlss to the model (A/B: --no-guides)
    float temporal = 0.f;               // temporal filter weight (0 = off)
    float temporalThreshold = 0.1f;
    float skinBlend = 1.f;              // edit strength inside face / skin masks
    TonemapOptions tonemap;
    std::string paramsBlock = "capability";
    bool requireDriver = true;
    std::filesystem::path dllDir;
    // inputs
    std::filesystem::path colorDir;     // color_sr pass folder (empty: slot color_sr when present, else the decoded frame)
    std::filesystem::path depthDir, mvDir;   // depth_dlss / mv_dlss pass folders (optional)
    std::filesystem::path masksDir;     // root with mask_ui / mask_ignore / mask_face / mask_skin pass folders (optional)
    bool useCachePasses = true;         // take color_sr / depth_dlss / mv_dlss from the cache slot when present
    // outputs
    std::filesystem::path outputDir;    // pass root: <outputDir>/color_nr (empty: no files)
    FileFormat format = FileFormat::Exr;
    bool uploadToGpu = true;            // keep color_nr in the cache slot
    std::filesystem::path videoOut;
    std::string videoCodec = "h264_nvenc";
    bool disableWhenUnavailable = false;
    ColorInfo color;
    Rational fps;
    std::string sourceFile, sourceHash;
    std::function<void(int64_t frame, double ms)> onFrame;
};

class NrStage final : public IStage {
public:
    explicit NrStage(NrStageOptions options, std::vector<std::unique_ptr<INrBackend>> backends = {});
    ~NrStage() override;

    std::string_view Name() const override { return "nr"; }
    void Init(const StageConfig& config, D3D12Device& device) override;
    void Process(FrameContext& ctx) override;
    void Finish() override;
    void Shutdown() override;

    struct Stats {
        int64_t frames = 0;
        double msSum = 0.0;  // GPU submit + wait per frame
        double MeanMs() const { return frames ? msSum / static_cast<double>(frames) : 0.0; }
        uint32_t width = 0, height = 0, workWidth = 0, workHeight = 0;
        std::string backend, colorSource;  // colorSource: video | color_sr | slot
        int passes = 0;
        bool depthUsed = false, mvUsed = false;
        std::vector<std::string> masks;
        bool disabled = false;
        std::string disabledReason;
    };
    const Stats& GetStats() const { return stats_; }
    const INrBackend* Backend(size_t pass = 0) const { return pass < backends_.size() ? backends_[pass].get() : nullptr; }
    const NrDiagnostics* Diagnostics() const { return diag_ ? &*diag_ : nullptr; }
    std::filesystem::path OutDir() const { return options_.outputDir / "color_nr"; }
    static const std::vector<std::string>& MaskNames();  // ui, ignore, face, skin

private:
    void ApplyParams(const nlohmann::json& params);
    void Setup(uint32_t w, uint32_t h);
    void LoadMasks(int64_t frame);
    void Disable(const std::string& reason);

    NrStageOptions options_;
    std::vector<std::unique_ptr<INrBackend>> injected_, backends_;
    std::optional<NrDiagnostics> diag_;
    D3D12Device* device_ = nullptr;
    std::unique_ptr<Tonemapper> tonemapper_;
    std::unique_ptr<Resampler> resampler_;
    std::unique_ptr<NrCompose> compose_;
    ComPtr<ID3D12Resource> input_, proxy_, work_, output_, prev_, protect_, skin_;
    std::vector<ComPtr<ID3D12Resource>> modelOut_;
    uint32_t inW_ = 0, inH_ = 0, workW_ = 0, workH_ = 0, maskW_ = 0, maskH_ = 0;
    bool reduced_ = false, first_ = true, hasPrev_ = false, setup_ = false, disabled_ = false, warnedDisabled_ = false;
    std::optional<PassReader> colorReader_, depthReader_, mvReader_;
    std::map<std::string, PassReader> maskReaders_;
    bool protectFrame_ = false, skinFrame_ = false;
    std::unique_ptr<PassWriter> writer_;
    std::unique_ptr<VideoEncoder> encoder_;
    Stats stats_;
    bool initialized_ = false;
};

struct NrRunResult {
    NrStage::Stats stats;
    std::filesystem::path outDir;
    nlohmann::json backend;      // INrBackend::Describe of the first pass
    nlohmann::json diagnostics;  // NrDiagnostics (ngx)
};
NrRunResult RunNr(VideoDecoder& decoder, D3D12Device& device, NrStageOptions options, int64_t maxFrames = -1, const std::function<void(int64_t)>& progress = {});

// U8 / U16 / float mask image -> F32 [0, 1] single channel (masks are 8-bit PNG, ТЗ §5).
PassImage MaskToFloat(const PassImage& mask);
// Per-pixel max of two float masks (same size).
PassImage MaskMax(const PassImage& a, const PassImage& b);

}  // namespace dlssvid
