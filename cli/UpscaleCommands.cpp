#include "UpscaleCommands.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
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
    o.outputDir = a.output;
    const auto fmt = ParseFileFormat(a.format);
    if (!fmt || (*fmt != FileFormat::Exr && *fmt != FileFormat::Png)) Throw("--format must be exr|png16");
    o.format = *fmt;
    o.videoOut = a.video;
    o.videoCodec = a.codec;
    o.sourceFile = std::filesystem::path(a.input).filename().string();
    o.sourceHash = "sha256:" + Sha256File(a.input);
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
        out = Yuv420pToRgb(f, color_, PixelType::F32);
        return true;
    }

private:
    std::optional<PassReader> reader_;
    std::unique_ptr<VideoDecoder> decoder_;
    ColorInfo color_;
    int64_t first_ = 0, count_ = 0, decoded_ = 0;
    uint32_t width_ = 0, height_ = 0;
};

int CmdCompare(const UpscaleCommands::CompareArgs& a) {
    FrameSource ref(a.ref), test(a.test);
    if (ref.Width() != test.Width() || ref.Height() != test.Height())
        Throw("compare: sizes differ (" + std::to_string(ref.Width()) + "x" + std::to_string(ref.Height()) + " vs " + std::to_string(test.Width()) + "x" +
              std::to_string(test.Height()) + ") — compare at one resolution (upscale the reference or downscale the test)");
    if (ref.IsVideo() && test.IsVideo() && (ref.Color().matrix != test.Color().matrix || ref.Color().range != test.Color().range))
        Log()->warn("compare: the videos decode with different colour matrices/ranges ({} {} vs {} {}) — tag both with the same colorspace, otherwise PSNR measures the mismatch",
                    ToString(ref.Color().matrix), ToString(ref.Color().range), ToString(test.Color().matrix), ToString(test.Color().range));
    nlohmann::json frames = nlohmann::json::array();
    double psnrY = 0, psnrRgb = 0, ssim = 0, minPsnr = std::numeric_limits<double>::infinity(), minSsim = 1.0;
    int64_t n = 0;
    if (a.step < 1) Throw("compare: --step must be >= 1");
    for (int64_t i = 0; a.frames < 0 || i < a.frames; ++i) {
        PassImage r, t;
        const int64_t ri = a.start + i * a.step;
        if (!ref.Next(ri, r)) break;
        if (!test.Next(ri + a.offset, t)) break;
        const ImageMetrics m = CompareImages(r, t);
        if (!a.quiet) std::printf("frame %6lld  PSNR Y %7.3f dB  PSNR RGB %7.3f dB  SSIM %.4f\n", static_cast<long long>(ri), m.psnrY, m.psnrRgb, m.ssimY);
        frames.push_back({{"frame", ri}, {"psnr_y", m.psnrY}, {"psnr_rgb", m.psnrRgb}, {"ssim_y", m.ssimY}});
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
                              {"size", {ref.Width(), ref.Height()}}};
    std::printf("frames:      %lld\nPSNR Y:      %.3f dB (min %.3f)\nPSNR RGB:    %.3f dB\nSSIM Y:      %.4f (min %.4f)\n", static_cast<long long>(n), psnrY / n, minPsnr,
                psnrRgb / n, ssim / n, minSsim);
    if (!a.json.empty()) {
        std::ofstream out(a.json);
        out << nlohmann::json{{"reference", a.ref}, {"test", a.test}, {"summary", summary}, {"frames", frames}}.dump(2) << "\n";
    }
    return 0;
}

}  // namespace

void UpscaleCommands::Register(CLI::App& app) {
    upscale_ = app.add_subcommand("upscale", "upscale a video into the color_sr pass (rtxvsr | dlss | nis | bicubic)");
    upscale_->add_option("-i,--input", ua_.input, "input video file")->required()->check(CLI::ExistingFile);
    upscale_->add_option("-o,--output", ua_.output, "pass root folder (writes color_sr/)")->required();
    upscale_->add_option("--backend", ua_.backend, "rtxvsr (default; RTX Video SDK) | dlss (NGX, jitter emulation) | nis | bicubic")->default_val("rtxvsr");
    upscale_->add_option("--scale", ua_.scale, "scale factor: 1.5 | 2 | 3 (output capped at 3840x2160)")->default_val(2.0);
    upscale_->add_option("--target", ua_.target, "explicit output size WxH (instead of --scale)");
    upscale_->add_option("--depth-dir", ua_.depthDir, "depth_dlss pass folder (dlss)");
    upscale_->add_option("--mv-dir", ua_.mvDir, "mv_dlss pass folder (dlss)");
    upscale_->add_option("--sharpness", ua_.sharpness, "nis sharpness 0..1")->default_val(0.5f);
    upscale_->add_option("--preset", ua_.preset, "dlss render preset: default | J | K | L | M")->default_val("default");
    upscale_->add_flag("--artifact-reduction-only", ua_.artifactReductionOnly, "no scaling: rtxvsr artifact reduction / nis sharpen");
    upscale_->add_flag("--no-jitter", ua_.noJitter, "dlss: feed frames without the jitter emulation");
    upscale_->add_option("--jitter-sign", ua_.jitterSign, "dlss: sign of the reported jitter offset (+1 | -1)")->default_val(1.f);
    upscale_->add_flag("--no-fallback", ua_.noFallback, "fail instead of falling back to nis when the backend is unavailable");
    upscale_->add_option("--format", ua_.format, "exr | png16")->default_val("exr");
    upscale_->add_option("--video", ua_.video, "also encode the result into this video file (no audio)");
    upscale_->add_option("--codec", ua_.codec, "encoder for --video: h264_nvenc | hevc_nvenc | av1_nvenc | ffv1")->default_val("h264_nvenc");
    upscale_->add_option("--frames", ua_.frames, "process at most N frames")->default_val(-1);
    upscale_->add_flag("--warp", ua_.warp, "use the WARP software adapter (nis / bicubic)");
    upscale_->add_option("--dll-dir", ua_.dllDir, "folder with nvngx_dlss.dll (default: bin/nvidia next to the executable)");

    compare_ = app.add_subcommand("compare", "PSNR / SSIM between two videos or pass folders of the same size");
    compare_->add_option("--ref", ca_.ref, "reference video or pass folder")->required();
    compare_->add_option("--test", ca_.test, "video or pass folder under test")->required();
    compare_->add_option("--frames", ca_.frames, "compare at most N frames")->default_val(-1);
    compare_->add_option("--offset", ca_.offset, "test frame index offset")->default_val(0);
    compare_->add_option("--start", ca_.start, "first reference frame index")->default_val(0);
    compare_->add_option("--step", ca_.step, "frame stride (e.g. --start 1 --step 2 = only the generated frames of an x2 color_fg)")->default_val(1);
    compare_->add_option("--json", ca_.json, "write per-frame metrics to this JSON file");
    compare_->add_flag("-q,--quiet", ca_.quiet, "summary only");
}

int UpscaleCommands::Dispatch() {
    if (upscale_ && upscale_->parsed()) return CmdUpscale(ua_);
    if (compare_ && compare_->parsed()) return CmdCompare(ca_);
    return -1;
}

}  // namespace dlssvid::cli
