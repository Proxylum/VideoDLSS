#include "UpscaleCommands.h"

#include "ScenesArg.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
#include "stages/nr/NrLowpass.h"
#include "pipeline/PassFingerprint.h"
#include "stages/upscale/UpscaleStage.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid::cli {

namespace {

void Progress(int64_t done, int64_t total) {
    if (done % 10 == 0 || done == total) {
        if (total > 0) std::fprintf(stderr, "\r%lld/%lld frames", static_cast<long long>(done), static_cast<long long>(total));
        else std::fprintf(stderr, "\r%lld frames", static_cast<long long>(done));
    }
}

void ParseSize(const std::string& s, uint32_t& w, uint32_t& h) {
    if (s.empty()) return;
    const size_t x = s.find('x');
    if (x == std::string::npos) Throw("--target expects WxH (got '" + s + "')");
    w = static_cast<uint32_t>(std::stoul(s.substr(0, x)));
    h = static_cast<uint32_t>(std::stoul(s.substr(x + 1)));
}

int CmdUpscale(const UpscaleCommands::UpscaleArgs& a) {
    VideoDecoder decoder(a.input);
    const auto& info = decoder.Info();
    UpscaleStageOptions o;
    o.backend = a.backend;
    o.allowFallback = !a.noFallback;
    o.scale = a.scale;
    ParseSize(a.target, o.targetWidth, o.targetHeight);
    o.depthDir = a.depthDir;
    o.mvDir = a.mvDir;
    o.sharpness = a.sharpness;
    o.preset = a.preset;
    o.artifactReductionOnly = a.artifactReductionOnly;
    o.useJitter = !a.noJitter;
    o.jitterSign = a.jitterSign;
    o.dllDir = a.dllDir;
    o.model = a.model;
    o.tile = a.tile;
    o.modelsDir = a.modelsDir;
    o.window = a.window;
    o.overlap = a.overlap;
    o.outputDir = a.output;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt) Throw("--format must be exr | png16 | tiff | npz | raw");  // what the schema offers (TASK-0044: exr | png16 only)
    o.format = *fmt;
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
    // PR-13: the clip's scene-cut list, <output>/scenes unless told otherwise - for DLSS SR and the worker, which keep a history
    o.sceneCuts = ResolveScenes(a.scenesDir, a.noScenes, false, a.output, o.sourceHash, StageReadsSceneCuts("upscale", {{"backend", a.backend}}), "upscale",
                                a.frames > 0 ? a.frames : 0);
    const int64_t total = a.frames > 0 ? a.frames : info.frameCount;
    o.onFrame = [](int64_t frame, double ms) {
        if (frame % 10 == 0) std::fprintf(stderr, "  frame %lld %.1f ms\n", static_cast<long long>(frame), ms);
    };
    D3D12Device device({a.warp, false});
    const auto t0 = std::chrono::steady_clock::now();
    const UpscaleRunResult r = RunUpscale(decoder, device, o, a.frames, [total](int64_t n) { Progress(n, total); });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "\n");
    Log()->info("upscale done: {} frames in {:.1f} s ({:.1f} ms/frame wall, {:.1f} ms/frame GPU), backend {}", r.stats.frames, sec,
                r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0, r.stats.MeanMs(), r.upscaler.dump());
    std::printf("color_sr:    %s\n", r.outDir.string().c_str());
    std::printf("backend:     %s%s\n", r.stats.backend.c_str(), r.stats.fellBack ? "  (fallback)" : "");
    std::printf("size:        %ux%u -> %ux%u\n", r.stats.inputWidth, r.stats.inputHeight, r.stats.outputWidth, r.stats.outputHeight);
    std::printf("frames:      %lld\nms/frame:    %.1f (GPU %.1f)\n", static_cast<long long>(r.stats.frames), r.stats.frames ? 1000.0 * sec / r.stats.frames : 0.0,
                r.stats.MeanMs());
    if (r.stats.jitterPhases) std::printf("jitter:      %d phases\n", r.stats.jitterPhases);
    if (!a.video.empty()) std::printf("video:       %s\n", a.video.c_str());
    return 0;
}

// A video file or a pass folder, read frame by frame as display-referred RGB.
class FrameSource {
public:
    explicit FrameSource(const std::filesystem::path& path) {
        if (std::filesystem::is_directory(path)) {
            reader_.emplace(PassReader::Open(path));
            first_ = reader_->Man().firstFrame;
            count_ = reader_->Man().frameCount;
            width_ = reader_->Man().width;
            height_ = reader_->Man().height;
        } else {
            decoder_ = std::make_unique<VideoDecoder>(path);
            color_ = ColorInfoFromStream(decoder_->Info());
            count_ = decoder_->Info().frameCount;
            width_ = decoder_->Info().width;
            height_ = decoder_->Info().height;
        }
    }
    bool IsVideo() const { return decoder_ != nullptr; }
    const ColorInfo& Color() const { return color_; }
    uint32_t Width() const { return width_; }
    uint32_t Height() const { return height_; }
    int64_t Count() const { return count_; }
    int64_t First() const { return first_; }  // the number of frame 0: a pass's first frame, 0 for a video
    // Sequential access: frame i of the source (pass frames by index, video frames in order).
    bool Next(int64_t index, PassImage& out) {
        if (reader_) {
            if (!reader_->HasFrame(first_ + index)) return false;
            out = reader_->ReadFrame(first_ + index);
            return true;
        }
        CpuFrame f;
        while (decoded_ <= index) {
            if (!decoder_->NextFrame(f)) return false;
            ++decoded_;
        }
        out = YuvToRgb(f, color_, PixelType::F32);
        return true;
    }

private:
    std::optional<PassReader> reader_;
    std::unique_ptr<VideoDecoder> decoder_;
    ColorInfo color_;
    int64_t first_ = 0, count_ = 0, decoded_ = 0;
    uint32_t width_ = 0, height_ = 0;
};

std::vector<uint32_t> ParseBands(const std::string& s) {
    std::vector<uint32_t> d;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t used = 0;
        unsigned long v = 0;
        try {
            v = std::stoul(item, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (item.empty() || used != item.size() || v < 2 || v > 4096) Throw("compare: --bands expects cell sizes 2..4096 separated by commas (got '" + s + "')");
        d.push_back(static_cast<uint32_t>(v));
    }
    if (d.empty()) Throw("compare: --bands expects cell sizes 2..4096 separated by commas (got '" + s + "')");
    return d;
}

// The colour of a frame as F32 RGB in [0, 1] (integers full range), the frame's own values for floats.
PassImage Rgb32(const PassImage& img, const char* what) {
    if (img.channels.size() < 3) Throw(std::string("compare: --bands needs colour frames (") + what + " has " + std::to_string(img.channels.size()) + " channel(s))");
    const float s = img.type == PixelType::U8 ? 1.f / 255.f : img.type == PixelType::U16 ? 1.f / 65535.f : 1.f;
    PassImage out;
    out.Allocate(img.width, img.height, PixelType::F32, {"R", "G", "B"});
    float* o = out.As<float>();
    for (uint32_t y = 0; y < img.height; ++y)
        for (uint32_t x = 0; x < img.width; ++x)
            for (size_t c = 0; c < 3; ++c) o[(static_cast<size_t>(y) * img.width + x) * 3 + c] = img.Get(x, y, c) * s;
    return out;
}

// PSNR (peak 1) of two luma planes of one size; infinity when they are equal.
double PsnrOfLuma(const PassImage& a, const PassImage& b) {
    const float *pa = a.As<float>(), *pb = b.As<float>();
    const size_t n = static_cast<size_t>(a.width) * a.height;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) mse += (static_cast<double>(pa[i]) - pb[i]) * (static_cast<double>(pa[i]) - pb[i]);
    mse /= static_cast<double>(n);
    return mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : std::numeric_limits<double>::infinity();
}

// One frame at every d: the low-band share of D = test - ref, and with I the PSNR of saturate(I + LP(test - I)) (the low part of the
// edit alone) and of saturate(test - LP(test - I)) (the high part alone).
nlohmann::json FrameBands(const PassImage& ref, const PassImage& test, const PassImage* input, const std::vector<uint32_t>& divs) {
    const PassImage r = Rgb32(ref, "--ref"), t = Rgb32(test, "--test");
    PassImage diff = t;
    for (size_t i = 0; i < diff.data.size() / sizeof(float); ++i) diff.As<float>()[i] -= r.As<float>()[i];
    const PassImage diffY = LumaOf(diff), refY = LumaOf(r);
    std::optional<PassImage> in, edit;
    if (input) {
        in = Rgb32(*input, "--edit-from");
        edit = t;
        for (size_t i = 0; i < edit->data.size() / sizeof(float); ++i) edit->As<float>()[i] -= in->As<float>()[i];
    }
    nlohmann::json out = nlohmann::json::array();
    for (const uint32_t d : divs) {
        nlohmann::json b = {{"d", d}, {"low_energy_rgb", NrLowShare(diff, d)}, {"low_energy_y", NrLowShare(diffY, d)}};
        if (in) {
            const PassImage lp = NrLowpassCpu(*edit, d);
            PassImage low = *in, high = t;
            const size_t n = low.data.size() / sizeof(float);
            for (size_t i = 0; i < n; ++i) {
                low.As<float>()[i] = std::clamp(low.As<float>()[i] + lp.As<float>()[i], 0.f, 1.f);
                high.As<float>()[i] = std::clamp(high.As<float>()[i] - lp.As<float>()[i], 0.f, 1.f);
            }
            b["psnr_y_low"] = PsnrOfLuma(refY, LumaOf(low));
            b["psnr_y_high"] = PsnrOfLuma(refY, LumaOf(high));
        }
        out.push_back(std::move(b));
    }
    return out;
}

int CmdCompare(const UpscaleCommands::CompareArgs& a) {
    FrameSource ref(a.ref), test(a.test);
    if (ref.Width() != test.Width() || ref.Height() != test.Height())
        Throw("compare: sizes differ (" + std::to_string(ref.Width()) + "x" + std::to_string(ref.Height()) + " vs " + std::to_string(test.Width()) + "x" +
              std::to_string(test.Height()) + ") — compare at one resolution (upscale the reference or downscale the test)");
    if (ref.IsVideo() && test.IsVideo() && (ref.Color().matrix != test.Color().matrix || ref.Color().range != test.Color().range))
        Log()->warn("compare: the videos decode with different colour matrices/ranges ({} {} vs {} {}) — tag both with the same colorspace, otherwise PSNR measures the mismatch",
                    ToString(ref.Color().matrix), ToString(ref.Color().range), ToString(test.Color().matrix), ToString(test.Color().range));
    const std::vector<uint32_t> divs = a.bands.empty() ? std::vector<uint32_t>{} : ParseBands(a.bands);
    std::optional<PassReader> input;
    if (!a.editFrom.empty()) {
        if (divs.empty()) Throw("compare: --edit-from needs --bands");
        if (!std::filesystem::is_directory(a.editFrom)) Throw("compare: --edit-from must be a pass folder (the input of the edit, color_sr)");
        input.emplace(PassReader::Open(a.editFrom));
        if (input->Man().width != ref.Width() || input->Man().height != ref.Height())
            Throw("compare: --edit-from is " + std::to_string(input->Man().width) + "x" + std::to_string(input->Man().height) + ", the reference " +
                  std::to_string(ref.Width()) + "x" + std::to_string(ref.Height()));
    }
    std::vector<nlohmann::json> bandSum(divs.size(), nlohmann::json{{"low_energy_rgb", 0.0}, {"low_energy_y", 0.0}, {"psnr_y_low", 0.0}, {"psnr_y_high", 0.0}});
    nlohmann::json frames = nlohmann::json::array();
    double psnrY = 0, psnrRgb = 0, ssim = 0, mae = 0, maeY = 0, minPsnr = std::numeric_limits<double>::infinity(), minSsim = 1.0;
    int64_t n = 0;
    if (a.step < 1) Throw("compare: --step must be >= 1");
    for (int64_t i = 0; a.frames < 0 || i < a.frames; ++i) {
        PassImage r, t;
        const int64_t ri = a.start + i * a.step;
        if (!ref.Next(ri, r)) break;
        if (!test.Next(ri + a.offset, t)) break;
        const ImageMetrics m = CompareImages(r, t);
        if (!a.quiet) std::printf("frame %6lld  PSNR Y %7.3f dB  PSNR RGB %7.3f dB  SSIM %.4f\n", static_cast<long long>(ri), m.psnrY, m.psnrRgb, m.ssimY);
        frames.push_back({{"frame", ri}, {"psnr_y", m.psnrY}, {"psnr_rgb", m.psnrRgb}, {"ssim_y", m.ssimY}, {"mae_rgb", m.maeRgb}});
        if (!divs.empty()) {
            // I by the reference's frame number (ref and I need not start at the same frame); a missing one is an error, not the end
            const int64_t number = ref.First() + ri;
            std::optional<PassImage> in;
            if (input) {
                if (!input->HasFrame(number)) Throw("compare: --edit-from has no frame " + std::to_string(number));
                in = input->ReadFrame(number);
            }
            nlohmann::json bands = FrameBands(r, t, in ? &*in : nullptr, divs);
            for (size_t k = 0; k < divs.size(); ++k)
                for (auto& [key, v] : bands[k].items())
                    if (key != "d") bandSum[k][key] = bandSum[k][key].get<double>() + (std::isfinite(v.get<double>()) ? v.get<double>() : 100.0);
            frames.back()["bands"] = std::move(bands);
        }
        mae += m.maeRgb;
        maeY += m.maeY;
        psnrY += std::isfinite(m.psnrY) ? m.psnrY : 100.0;
        psnrRgb += std::isfinite(m.psnrRgb) ? m.psnrRgb : 100.0;
        ssim += m.ssimY;
        minPsnr = std::min(minPsnr, m.psnrY);
        minSsim = std::min(minSsim, m.ssimY);
        ++n;
    }
    if (n == 0) Throw("compare: no frames compared");
    nlohmann::json summary = {{"frames", n},
                              {"psnr_y_mean", psnrY / n},
                              {"psnr_rgb_mean", psnrRgb / n},
                              {"ssim_y_mean", ssim / n},
                              {"psnr_y_min", minPsnr},
                              {"ssim_y_min", minSsim},
                              {"mae_rgb_mean", mae / n},  // mean |reference - test|: the size of an edit (NR A/B), not a quality score
                              {"mae_y_mean", maeY / n},
                              {"size", {ref.Width(), ref.Height()}}};
    if (!divs.empty()) {
        nlohmann::json bands = nlohmann::json::array();
        for (size_t k = 0; k < divs.size(); ++k) {
            nlohmann::json b = {{"d", divs[k]}};
            for (const char* key : {"low_energy_rgb", "low_energy_y", "psnr_y_low", "psnr_y_high"})
                if (input || std::string(key).rfind("low_energy", 0) == 0) b[key] = bandSum[k][key].get<double>() / static_cast<double>(n);
            bands.push_back(std::move(b));
        }
        summary["bands"] = bands;
    }
    std::printf("frames:      %lld\nPSNR Y:      %.3f dB (min %.3f)\nPSNR RGB:    %.3f dB\nSSIM Y:      %.4f (min %.4f)\nMAE RGB:     %.5f (Y %.5f)\n",
                static_cast<long long>(n), psnrY / n, minPsnr, psnrRgb / n, ssim / n, minSsim, mae / n, maeY / n);
    if (summary.contains("bands"))
        for (const auto& b : summary["bands"]) {
            std::printf("bands:       d=%u low %.3f (Y %.3f)", b["d"].get<uint32_t>(), b["low_energy_rgb"].get<double>(), b["low_energy_y"].get<double>());
            if (b.contains("psnr_y_low")) std::printf("  PSNR Y low part %.3f dB, high part %.3f dB", b["psnr_y_low"].get<double>(), b["psnr_y_high"].get<double>());
            std::printf("\n");
        }
    if (!a.json.empty()) {
        std::ofstream out(a.json);
        out << nlohmann::json{{"reference", a.ref}, {"test", a.test}, {"summary", summary}, {"frames", frames}}.dump(2) << "\n";
    }
    return 0;
}

}  // namespace

void UpscaleCommands::Register(CLI::App& app) {
    upscale_ = app.add_subcommand("upscale", "upscale a video into the color_sr pass (dlss | nis | bicubic | trt | worker)");
    upscale_->add_option("-i,--input", ua_.input, "input video file")->required()->check(CLI::ExistingFile);
    upscale_->add_option("-o,--output", ua_.output, "pass root folder (writes color_sr/)")->required();
    upscale_->add_option("--backend", ua_.backend, "dlss (default; NGX, jitter emulation) | nis | bicubic | trt (Real-ESRGAN and other ONNX models through TensorRT) | worker (RealBasicVSR in PyTorch, sr_worker)")->default_val("dlss");
    upscale_->add_option("--model", ua_.model, "trt: model id from models/registry.json (default realesrgan-x2plus)");
    upscale_->add_option("--tile", ua_.tile, "trt: tile size in px (0 = the model's registry default, 512)")->default_val(0);
    upscale_->add_option("--models-dir,--models_dir", ua_.modelsDir, "folder with registry.json (default: auto)");
    upscale_->add_option("--window", ua_.window, "worker: frames per pass of the video model (0 = registry default)")->default_val(0);
    upscale_->add_option("--overlap", ua_.overlap, "worker: frames the next window re-estimates and blends (0 = registry default)")->default_val(0);
    upscale_->add_option("--scale", ua_.scale, "scale factor: 1.5 | 2 | 3 (output capped at 3840x2160)")->default_val(2.0);
    upscale_->add_option("--target", ua_.target, "explicit output size WxH (instead of --scale)");
    upscale_->add_option("--depth-dir", ua_.depthDir, "depth_dlss pass folder (dlss)");
    upscale_->add_option("--mv-dir", ua_.mvDir, "mv_dlss pass folder (dlss)");
    upscale_->add_option("--scenes-dir,--scenes_dir", ua_.scenesDir,
                         "dlss / worker: the clip's scene-cut list (the pass of `dlssvid scenes`; default: <output>/scenes when present): "
                         "the history starts again at its cuts");
    upscale_->add_flag("--no-scenes,--no_scenes", ua_.noScenes, "dlss / worker: no scene-cut list");
    upscale_->add_option("--sharpness", ua_.sharpness, "nis sharpness 0..1")->default_val(0.5f);
    upscale_->add_option("--preset", ua_.preset, "dlss render preset: default | J | K | L | M")->default_val("default");
    upscale_->add_flag("--no-scaling,--artifact-reduction-only,--artifact_reduction_only", ua_.artifactReductionOnly,
                       "no scaling (x1): dlss = DLAA anti-aliasing, nis = NVSharpen; does not remove compression artifacts "
                       "(--artifact-reduction-only is the old name)");
    upscale_->add_flag("--no-jitter", ua_.noJitter, "dlss: feed frames without the jitter emulation");
    upscale_->add_option("--jitter-sign,--jitter_sign", ua_.jitterSign, "dlss: sign of the reported jitter offset (+1 | -1)")->default_val(1.f);
    upscale_->add_flag("--no-fallback,--no_fallback", ua_.noFallback, "fail instead of falling back to nis when the backend is unavailable");
    upscale_->add_option("--format", ua_.format, "exr | png16 | tiff | npz | raw")->default_val("exr");
    upscale_->add_option("--video", ua_.video, "also encode the result into this video file (no audio)");
    upscale_->add_option("--codec", ua_.codec, "encoder for --video: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    upscale_->add_option("--frames", ua_.frames, "process at most N frames")->default_val(-1);
    upscale_->add_flag("--warp", ua_.warp, "use the WARP software adapter (nis / bicubic)");
    upscale_->add_option("--dll-dir", ua_.dllDir, "folder with nvngx_dlss.dll (default: bin/nvidia next to the executable)");

    compare_ = app.add_subcommand("compare", "PSNR / SSIM / MAE between two videos or pass folders of the same size");
    compare_->add_option("--ref", ca_.ref, "reference video or pass folder")->required();
    compare_->add_option("--test", ca_.test, "video or pass folder under test")->required();
    compare_->add_option("--frames", ca_.frames, "compare at most N frames")->default_val(-1);
    compare_->add_option("--offset", ca_.offset, "test frame index offset")->default_val(0);
    compare_->add_option("--start", ca_.start, "first reference frame index")->default_val(0);
    compare_->add_option("--step", ca_.step, "frame stride (e.g. --start 1 --step 2 = only the generated frames of an x2 color_fg)")->default_val(1);
    compare_->add_option("--json", ca_.json, "write per-frame metrics to this JSON file");
    compare_->add_flag("-q,--quiet", ca_.quiet, "summary only");
    compare_->add_option("--bands", ca_.bands,
                         "d1,d2,...: the share of test - ref in the low band of the NR control mask's split at each cell size d (pixels), "
                         "per frame and in the summary");
    compare_->add_option("--edit-from,--edit_from", ca_.editFrom,
                         "with --bands: the pass folder of the edit's input I (color_sr; frames by the reference's frame numbers): the PSNR Y "
                         "of I + low(test - I) and of test - low(test - I) against the reference");
}

int UpscaleCommands::Dispatch() {
    if (upscale_ && upscale_->parsed()) return CmdUpscale(ua_);
    if (compare_ && compare_->parsed()) return CmdCompare(ca_);
    return -1;
}

}  // namespace dlssvid::cli
