// UpscaleStage end to end: color_sr pass folder, cache slot texture, preview video, fallback,
// and DLSS SR (NVIDIA GPU + nvngx_dlss.dll, skipped otherwise) with the jitter emulation checked
// against a downscale/upscale reference.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
#include "gpu/GpuFrameCache.h"
#include "pipeline/Pipeline.h"
#include "stages/upscale/UpscaleStage.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "upscale" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// Continuous test scene sampled at pixel centres (resolution independent): gradients, soft discs,
// fine stripes and a moving disc, so a low-resolution clip is a true downscale of the reference.
float Scene(double u, double v, int64_t t, int c) {
    const double cx = 0.35 + 0.02 * t, cy = 0.45 + 0.01 * t;
    const double d = std::sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy));
    const double disc = 1.0 / (1.0 + std::exp((d - 0.12) * 200.0));
    const double stripes = 0.5 + 0.5 * std::sin(u * 220.0) * std::sin(v * 180.0);
    const double base = c == 0 ? u : c == 1 ? v : 0.5 * (1.0 - u);
    return static_cast<float>(std::clamp(0.15 + 0.5 * base + 0.25 * disc * (c == 2 ? 1.0 : 0.3) + 0.15 * stripes, 0.0, 1.0));
}

PassImage RenderScene(int64_t t, uint32_t w, uint32_t h, int ss) {
    PassImage img;
    img.Allocate(w, h, PixelType::F32, {"R", "G", "B"});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (int sy = 0; sy < ss; ++sy)
                    for (int sx = 0; sx < ss; ++sx)
                        acc += Scene((x + (sx + 0.5) / ss) / w, (y + (sy + 0.5) / ss) / h, t, c);
                img.Set(x, y, static_cast<size_t>(c), static_cast<float>(acc / (ss * ss)));
            }
    return img;
}

PassImage BoxDown2(const PassImage& src) {
    PassImage out;
    out.Allocate(src.width / 2, src.height / 2, PixelType::F32, src.channels);
    for (uint32_t y = 0; y < out.height; ++y)
        for (uint32_t x = 0; x < out.width; ++x)
            for (size_t c = 0; c < src.channels.size(); ++c)
                out.Set(x, y, c, 0.25f * (src.Get(2 * x, 2 * y, c) + src.Get(2 * x + 1, 2 * y, c) + src.Get(2 * x, 2 * y + 1, c) + src.Get(2 * x + 1, 2 * y + 1, c)));
    return out;
}

struct RefPair {
    std::filesystem::path reference, input;
    uint32_t w, h;
    int frames;
};
// reference clip (w x h) and its exact 2x box downscale as the pipeline input
RefPair WriteReferencePair(const std::filesystem::path& dir, uint32_t w, uint32_t h, int frames) {
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    const ColorInfo color{ColorMatrix::Bt709, ColorRange::Limited};
    std::vector<PassImage> refs;
    for (int i = 0; i < frames; ++i) refs.push_back(RenderScene(i, w, h, 2));
    RefPair p;
    p.w = w;
    p.h = h;
    p.frames = frames;
    p.reference = WriteClipWith(dir / "ref.mkv", spec, [&](int64_t i) {
        CpuFrame f;
        RgbToYuv420p(refs[static_cast<size_t>(i)], color, f);
        f.index = i;
        f.pts = i;
        return f;
    });
    ClipSpec smallSpec = spec;
    smallSpec.width = w / 2;
    smallSpec.height = h / 2;
    p.input = WriteClipWith(dir / "in.mkv", smallSpec, [&](int64_t i) {
        CpuFrame f;
        RgbToYuv420p(BoxDown2(refs[static_cast<size_t>(i)]), color, f);
        f.index = i;
        f.pts = i;
        return f;
    });
    return p;
}

double MeanPsnr(const std::filesystem::path& reference, const std::filesystem::path& colorSrDir, int frames) {
    const std::vector<CpuFrame> ref = DecodeAll(reference);
    VideoDecoder dec(reference);
    const ColorInfo ci = ColorInfoFromStream(dec.Info());
    const PassReader sr = PassReader::Open(colorSrDir);
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < frames && i < static_cast<int>(ref.size()); ++i) {
        if (!sr.HasFrame(i)) break;
        const ImageMetrics m = CompareImages(Yuv420pToRgb(ref[static_cast<size_t>(i)], ci, PixelType::F32), sr.ReadFrame(i));
        sum += m.psnrY;
        ++n;
    }
    return n ? sum / n : 0.0;
}

}  // namespace

TEST_CASE("UpscaleStage writes color_sr with nis on WARP, uploads it to the slot and encodes a preview", "[integration][upscale]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto dir = Dir("nis");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    VideoDecoder dec(clip);
    D3D12Device dev({true, false});
    UpscaleStageOptions o;
    o.backend = "nis";
    o.scale = 2.0;
    o.outputDir = dir / "passes";
    o.videoOut = dir / "sr.mkv";
    o.videoCodec = "ffv1";
    o.sourceFile = "clip.mkv";
    int frames = 0;
    o.onFrame = [&](int64_t, double ms) {
        ++frames;
        CHECK(ms >= 0.0);
    };
    const UpscaleRunResult r = RunUpscale(dec, dev, o);
    CHECK(r.stats.frames == 4);
    CHECK(frames == 4);
    CHECK(r.stats.backend == "nis");
    CHECK(!r.stats.fellBack);
    CHECK(r.stats.outputWidth == 96);
    CHECK(r.stats.outputHeight == 64);
    CHECK(r.stats.jitterPhases == 0);
    const PassReader sr = PassReader::Open(r.outDir);
    sr.Validate({96, 64, 4, PassKind::ColorSr});
    CHECK(sr.Man().colorspace == "srgb");
    CHECK(sr.Man().stageParams["backend"] == "nis");
    CHECK(sr.Man().stageParams.contains("ms_per_frame"));
    const PassImage f0 = sr.ReadFrame(0);
    CHECK(f0.type == PixelType::F16);
    CHECK(f0.channels.size() == 3);
    // the preview video has the target size and matches the pass within 8-bit quantisation
    const std::vector<CpuFrame> video = DecodeAll(dir / "sr.mkv");
    REQUIRE(video.size() == 4);
    CHECK(video[0].desc.width == 96);
    CHECK(video[0].desc.height == 64);
    VideoDecoder previewDec(dir / "sr.mkv");  // the preview is encoded with the input's colour matrix (BT.601 for a 32-line clip)
    const ImageMetrics m = CompareImages(f0, Yuv420pToRgb(video[0], ColorInfoFromStream(previewDec.Info()), PixelType::F32));
    CHECK(m.psnrY > 38.0);    // luma survives the 8-bit round trip
    CHECK(m.psnrRgb > 26.0);  // chroma is 4:2:0 subsampled on a checkered synthetic pattern
}

TEST_CASE("UpscaleStage: rtxvsr falls back to nis, bicubic baseline, png16 output, 4K cap and slot upload", "[integration][upscale]") {
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 40;
    spec.height = 24;
    const auto dir = Dir("fallback");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    D3D12Device dev({true, false});
    {
        VideoDecoder dec(clip);
        UpscaleStageOptions o;
        o.backend = "rtxvsr";
        o.outputDir = dir / "vsr";
        const UpscaleRunResult r = RunUpscale(dec, dev, o);
        CHECK(r.stats.fellBack);
        CHECK(r.stats.backend == "nis");
        CHECK(PassReader::Open(r.outDir).Man().stageParams["fallback"] == true);
    }
    {
        VideoDecoder dec(clip);
        UpscaleStageOptions o;
        o.backend = "rtxvsr";
        o.allowFallback = false;
        CHECK_THROWS(RunUpscale(dec, dev, o));
    }
    {
        VideoDecoder dec(clip);
        UpscaleStageOptions o;
        o.backend = "bicubic";
        o.scale = 1.5;
        o.format = FileFormat::Png;
        o.outputDir = dir / "bicubic";
        const UpscaleRunResult r = RunUpscale(dec, dev, o);
        CHECK(r.stats.outputWidth == 60);
        CHECK(r.stats.outputHeight == 36);
        const PassReader sr = PassReader::Open(r.outDir);
        sr.Validate({60, 36, 2, PassKind::ColorSr});
        CHECK(sr.Man().format == FileFormat::Png);
        CHECK(sr.Man().pixelType == PixelType::U16);
    }
    {
        // explicit target above the cap is clamped, aspect kept
        VideoDecoder dec(clip);
        UpscaleStageOptions o;
        o.backend = "bicubic";
        o.targetWidth = 8000;
        o.maxWidth = 200;
        o.maxHeight = 200;
        const UpscaleRunResult r = RunUpscale(dec, dev, o, 1);
        CHECK(r.stats.outputWidth == 200);
        CHECK(r.stats.outputHeight == 120);
    }
    {
        // the stage as part of a pipeline: color_sr lands in the cache slot
        VideoDecoder dec(clip);
        Pipeline pipeline(dev, 2);
        UpscaleStageOptions o;
        o.backend = "bicubic";
        o.color = ColorInfoFromStream(dec.Info());
        pipeline.AddStage(std::make_unique<UpscaleStage>(o));
        pipeline.Init();
        CpuFrame f;
        REQUIRE(dec.NextFrame(f));
        pipeline.Cache().Acquire(f.index, f.desc);
        pipeline.ProcessFrame(f);
        GpuFrameCache::Slot* slot = pipeline.Cache().Find(0);
        REQUIRE(slot != nullptr);
        REQUIRE(slot->passes.count("color_sr") == 1);
        CHECK(slot->passes["color_sr"].width == 80);
        const PassImage down = pipeline.Cache().DownloadPass(*slot, "color_sr");
        CHECK(down.width == 80);
        CHECK(down.channels.size() == 3);
        pipeline.Finish();
        pipeline.Shutdown();
    }
}

TEST_CASE("DLSS SR upscales a downscaled clip back to the reference (jitter emulation, sign chosen by PSNR)", "[integration][upscale][dlss][gpu]") {
    D3D12Device dev;
    if (!dev.IsNvidia()) SKIP("DLSS needs an NVIDIA GPU");
    const UpscalerAvailability avail = UpscalerAvailable("dlss");
    if (!avail.available) SKIP(avail.reason);
    const auto dir = Dir("dlss");
    const RefPair pair = WriteReferencePair(dir, 960, 544, 12);

    auto run = [&](const std::string& backend, const char* name, bool jitter, float sign) {
        VideoDecoder dec(pair.input);
        UpscaleStageOptions o;
        o.backend = backend;
        o.allowFallback = false;
        o.scale = 2.0;
        o.useJitter = jitter;
        o.jitterSign = sign;
        o.outputDir = dir / name;
        const UpscaleRunResult r = RunUpscale(dec, dev, o);
        REQUIRE(r.stats.frames == pair.frames);
        REQUIRE(r.stats.outputWidth == pair.w);
        REQUIRE(r.stats.outputHeight == pair.h);
        PassReader::Open(r.outDir).Validate({pair.w, pair.h, pair.frames, PassKind::ColorSr});
        // steady state: skip the first frames while DLSS accumulates history
        const std::vector<CpuFrame> ref = DecodeAll(pair.reference);
        VideoDecoder rdec(pair.reference);
        const ColorInfo ci = ColorInfoFromStream(rdec.Info());
        const PassReader sr = PassReader::Open(r.outDir);
        double sum = 0.0;
        int n = 0;
        for (int i = 4; i < pair.frames; ++i) {
            sum += CompareImages(Yuv420pToRgb(ref[static_cast<size_t>(i)], ci, PixelType::F32), sr.ReadFrame(i)).psnrY;
            ++n;
        }
        const double psnr = sum / n;
        std::printf("%-22s PSNR Y %.2f dB (frames 4..%d)\n", name, psnr, pair.frames - 1);
        return psnr;
    };
    const double bicubic = run("bicubic", "bicubic", false, 1.f);
    const double nis = run("nis", "nis", false, 1.f);
    const double dlssNoJitter = run("dlss", "dlss_nojitter", false, 1.f);
    const double dlssPlus = run("dlss", "dlss_jitter_plus", true, 1.f);
    const double dlssMinus = run("dlss", "dlss_jitter_minus", true, -1.f);
    const double dlssBest = std::max(dlssPlus, dlssMinus);
    std::printf("bicubic %.2f  nis %.2f  dlss no-jitter %.2f  dlss +j %.2f  dlss -j %.2f  -> jitter sign %s\n", bicubic, nis, dlssNoJitter, dlssPlus, dlssMinus,
                dlssPlus >= dlssMinus ? "+1" : "-1");
    CHECK(dlssBest > 20.0);
    CHECK(dlssBest > bicubic - 10.0);  // the smooth synthetic scene favours bicubic; DLSS must stay in the same league
    CHECK(dlssPlus > dlssMinus);       // the jitter sign convention of docs/architecture.md (content shifted by +j, +j reported)
    CHECK(dlssPlus > dlssNoJitter);    // the emulation adds information
    CHECK(nis > 20.0);
}
