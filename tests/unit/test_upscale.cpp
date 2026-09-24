// Stage 5 units: jitter sequence, target resolution, RGB <-> YUV, PSNR/SSIM, the resampler and the
// NIS / bicubic upscalers on WARP.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <set>
#include <vector>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "passes/ImageMetrics.h"
#include "stages/upscale/BicubicUpscaler.h"
#include "stages/upscale/IUpscaler.h"
#include "stages/upscale/Jitter.h"
#include "stages/upscale/NisUpscaler.h"
#include "stages/upscale/Tiling.h"
#include "util/Half.h"

using namespace dlssvid;

namespace {

PassImage MakeRgb(uint32_t w, uint32_t h, const std::function<std::array<float, 3>(uint32_t, uint32_t)>& f, PixelType type = PixelType::F32) {
    PassImage img;
    img.Allocate(w, h, type, {"R", "G", "B"});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const auto c = f(x, y);
            for (size_t ch = 0; ch < 3; ++ch) img.Set(x, y, ch, c[ch]);
        }
    return img;
}

// RGBA16F texture from an RGB image, and the reverse.
ComPtr<ID3D12Resource> UploadRgba16f(D3D12Device& dev, const PassImage& rgb, bool uav = false) {
    const PassImage rgba = ToRgba16f(rgb);
    auto tex = dev.CreateTexture2D(rgb.width, rgb.height, DXGI_FORMAT_R16G16B16A16_FLOAT, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
    dev.UploadTexture2D(tex.Get(), rgba.data.data(), rgba.RowBytes());
    return tex;
}
PassImage DownloadRgb(D3D12Device& dev, ID3D12Resource* tex, uint32_t w, uint32_t h) {
    size_t pitch = 0;
    const std::vector<uint8_t> bytes = dev.ReadbackTexture2D(tex, pitch);
    PassImage rgb;
    rgb.Allocate(w, h, PixelType::F32, {"R", "G", "B"});
    const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < 3; ++c) rgb.Set(x, y, c, HalfToFloat(src[(static_cast<size_t>(y) * w + x) * 4 + c]));
    return rgb;
}

}  // namespace

TEST_CASE("Halton jitter sequence", "[upscale][jitter]") {
    CHECK(Halton(1, 2) == 0.5f);
    CHECK(Halton(2, 2) == 0.25f);
    CHECK(Halton(3, 2) == 0.75f);
    CHECK(std::fabs(Halton(1, 3) - 1.f / 3.f) < 1e-6f);
    CHECK(JitterPhaseCount(1920, 1080, 3840, 2160) == 32);
    CHECK(JitterPhaseCount(1920, 1080, 2880, 1620) == 18);
    CHECK(JitterPhaseCount(1280, 720, 3840, 2160) == 72);
    CHECK(JitterPhaseCount(1920, 1080, 1920, 1080) == 8);
    std::set<std::pair<float, float>> seen;
    for (int64_t f = 0; f < 32; ++f) {
        const JitterOffset j = HaltonJitter(f, 32);
        CHECK(j.x >= -0.5f);
        CHECK(j.x < 0.5f);
        CHECK(j.y >= -0.5f);
        CHECK(j.y < 0.5f);
        seen.insert({j.x, j.y});
    }
    CHECK(seen.size() == 32);  // distinct phases
    CHECK(HaltonJitter(0, 32).x == HaltonJitter(32, 32).x);  // periodic
}

TEST_CASE("ResolveUpscaleTarget keeps the aspect ratio, even sizes and the 4K cap", "[upscale]") {
    UpscaleTarget t = ResolveUpscaleTarget(1920, 1080, 2.0);
    CHECK(t.width == 3840);
    CHECK(t.height == 2160);
    CHECK(!t.capped);
    CHECK(t.scale == 2.0);
    t = ResolveUpscaleTarget(1920, 1080, 3.0);
    CHECK(t.width == 3840);
    CHECK(t.height == 2160);
    CHECK(t.capped);
    t = ResolveUpscaleTarget(1280, 720, 3.0);
    CHECK(t.width == 3840);
    CHECK(t.height == 2160);
    CHECK(!t.capped);
    t = ResolveUpscaleTarget(1920, 1080, 1.5);
    CHECK(t.width == 2880);
    CHECK(t.height == 1620);
    t = ResolveUpscaleTarget(1001, 501, 2.0);
    CHECK(t.width == 2002);
    CHECK(t.height == 1002);
    t = ResolveUpscaleTarget(1920, 1080, 0.0, 1280, 0);
    CHECK(t.width == 1280);
    CHECK(t.height == 720);
    t = ResolveUpscaleTarget(1920, 800, 3.0);
    CHECK(t.width == 3840);
    CHECK(t.height == 1600);
    CHECK(t.capped);
    t = ResolveUpscaleTarget(1920, 1080, 1.0);
    CHECK(t.width == 1920);
    CHECK_THROWS(ResolveUpscaleTarget(0, 0, 2.0));
    CHECK_THROWS(ResolveUpscaleTarget(100, 100, -1.0));
}

TEST_CASE("Upscaler factory and availability", "[upscale]") {
#ifdef DLSSVID_WITH_TENSORRT
    CHECK(UpscalerBackends().size() == 4);  // dlss, nis, bicubic, trt
#else
    CHECK(UpscalerBackends().size() == 3);
#endif
    CHECK(UpscalerAvailable("nis").available);
    CHECK(UpscalerAvailable("bicubic").available);
    CHECK(!UpscalerAvailable("nope").available);
    CHECK(!UpscalerAvailable("rtxvsr").available);  // the stub was removed (TASK-0021): unknown like any other name
    CHECK(CreateUpscaler("nis")->Name() == "nis");
    CHECK(CreateUpscaler("bicubic")->Name() == "bicubic");
    CHECK(CreateUpscaler("dlss")->WantsDepthAndMv());
    CHECK_THROWS(CreateUpscaler("nope"));
}

TEST_CASE("RgbToYuv420p inverts Yuv420pToRgb on smooth images", "[upscale][convert]") {
    const PassImage rgb = MakeRgb(64, 48, [](uint32_t x, uint32_t y) { return std::array<float, 3>{x / 63.f, y / 47.f, 0.5f - 0.3f * (x / 63.f)}; });
    for (const ColorInfo info : {ColorInfo{ColorMatrix::Bt709, ColorRange::Limited}, ColorInfo{ColorMatrix::Bt601, ColorRange::Full}}) {
        CpuFrame yuv;
        RgbToYuv420p(rgb, info, yuv);
        CHECK(yuv.desc.width == 64);
        CHECK(yuv.desc.height == 48);
        const PassImage back = Yuv420pToRgb(yuv, info, PixelType::F32);
        double maxDiff = 0.0;
        for (uint32_t y = 1; y < 47; ++y)
            for (uint32_t x = 1; x < 63; ++x)
                for (size_t c = 0; c < 3; ++c) maxDiff = std::max(maxDiff, std::fabs(static_cast<double>(rgb.Get(x, y, c)) - back.Get(x, y, c)));
        CHECK(maxDiff < 0.03);  // 8-bit quantisation + 2x2 chroma averaging on a gradient
    }
    // anchor values (limited range): black -> 16/128/128, white -> 235/128/128
    CpuFrame black, white;
    RgbToYuv420p(MakeRgb(4, 4, [](uint32_t, uint32_t) { return std::array<float, 3>{0, 0, 0}; }), ColorInfo{}, black);
    RgbToYuv420p(MakeRgb(4, 4, [](uint32_t, uint32_t) { return std::array<float, 3>{1, 1, 1}; }), ColorInfo{}, white);
    CHECK(black.Plane(0)[0] == 16);
    CHECK(black.Plane(1)[0] == 128);
    CHECK(white.Plane(0)[0] == 235);
    CHECK(white.Plane(2)[0] == 128);
}

TEST_CASE("PSNR / SSIM metrics", "[upscale][metrics]") {
    const PassImage a = MakeRgb(96, 64, [](uint32_t x, uint32_t y) { return std::array<float, 3>{x / 95.f, y / 63.f, 0.25f + 0.5f * ((x / 8 + y / 8) % 2)}; });
    ImageMetrics same = CompareImages(a, a);
    CHECK(std::isinf(same.psnrY));
    CHECK(std::isinf(same.psnrRgb));
    CHECK(std::fabs(same.ssimY - 1.0) < 1e-6);
    // uniform noise of amplitude 0.05: MSE = a^2/3 -> PSNR ~ 30.8 dB
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    PassImage b = a;
    for (uint32_t y = 0; y < b.height; ++y)
        for (uint32_t x = 0; x < b.width; ++x)
            for (size_t c = 0; c < 3; ++c) b.Set(x, y, c, std::clamp(a.Get(x, y, c) + dist(rng), 0.f, 1.f));
    const ImageMetrics noisy = CompareImages(a, b);
    CHECK(noisy.psnrRgb > 29.0);
    CHECK(noisy.psnrRgb < 33.0);
    CHECK(noisy.ssimY < 1.0);
    CHECK(noisy.ssimY > 0.3);
    // a blurred copy keeps structure better than an inverted one
    PassImage inverted = a;
    for (uint32_t y = 0; y < a.height; ++y)
        for (uint32_t x = 0; x < a.width; ++x)
            for (size_t c = 0; c < 3; ++c) inverted.Set(x, y, c, 1.f - a.Get(x, y, c));
    CHECK(CompareImages(a, inverted).ssimY < noisy.ssimY);
    // integer inputs are scaled to [0, 1]
    const PassImage u8 = MakeRgb(96, 64, [](uint32_t x, uint32_t y) { return std::array<float, 3>{std::round(x / 95.f * 255.f), std::round(y / 63.f * 255.f), 64.f}; }, PixelType::U8);
    const PassImage f = MakeRgb(96, 64, [](uint32_t x, uint32_t y) { return std::array<float, 3>{std::round(x / 95.f * 255.f) / 255.f, std::round(y / 63.f * 255.f) / 255.f, 64.f / 255.f}; });
    CHECK(CompareImages(u8, f).psnrRgb > 60.0);
    PassImage other;
    other.Allocate(10, 10, PixelType::F32, {"R", "G", "B"});
    CHECK_THROWS(CompareImages(a, other));
}

TEST_CASE("Resampler: integer shifts and identity are exact, upscale of a constant is constant", "[upscale][gpu]") {
    D3D12Device dev({true, false});
    Resampler rs(dev);
    const uint32_t w = 8, h = 8;
    const PassImage src = MakeRgb(w, h, [](uint32_t x, uint32_t y) { return std::array<float, 3>{x / 8.f, y / 8.f, ((x + y) % 2) ? 1.f : 0.f}; });
    auto in = UploadRgba16f(dev, src);
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { rs.Run(cl, in.Get(), w, h, out.Get(), w, h, 0.f, 0.f, Resampler::Filter::CatmullRom); });
    PassImage back = DownloadRgb(dev, out.Get(), w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < 3; ++c) CHECK(std::fabs(back.Get(x, y, c) - src.Get(x, y, c)) < 2e-3f);

    dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { rs.Run(cl, in.Get(), w, h, out.Get(), w, h, 1.f, 0.f, Resampler::Filter::Bilinear); });
    back = DownloadRgb(dev, out.Get(), w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x + 1 < w; ++x) CHECK(std::fabs(back.Get(x, y, 0) - src.Get(x + 1, y, 0)) < 2e-3f);  // shifted by one input pixel
    CHECK(std::fabs(back.Get(7, 3, 0) - src.Get(7, 3, 0)) < 2e-3f);  // clamped at the edge

    const PassImage flat = MakeRgb(w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.25f, 0.5f, 0.75f}; });
    auto flatTex = UploadRgba16f(dev, flat);
    auto big = dev.CreateTexture2D(2 * w, 2 * h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { rs.Run(cl, flatTex.Get(), w, h, big.Get(), 2 * w, 2 * h, 0.f, 0.f, Resampler::Filter::CatmullRom); });
    back = DownloadRgb(dev, big.Get(), 2 * w, 2 * h);
    for (uint32_t y = 0; y < 2 * h; ++y)
        for (uint32_t x = 0; x < 2 * w; ++x) {
            CHECK(std::fabs(back.Get(x, y, 0) - 0.25f) < 2e-3f);
            CHECK(std::fabs(back.Get(x, y, 2) - 0.75f) < 2e-3f);
        }
}

TEST_CASE("NIS and bicubic upscalers on WARP", "[upscale][nis][gpu]") {
    D3D12Device dev({true, false});
    const uint32_t w = 32, h = 24;
    const PassImage src = MakeRgb(w, h, [](uint32_t x, uint32_t y) {
        const float r = std::sqrt(static_cast<float>((x - 16.f) * (x - 16.f) + (y - 12.f) * (y - 12.f)));
        return std::array<float, 3>{x / 31.f, y / 23.f, r < 7.f ? 0.9f : 0.2f};
    });
    auto in = UploadRgba16f(dev, src);

    auto run = [&](const std::string& backend, uint32_t outW, uint32_t outH, bool arOnly = false) {
        UpscalerConfig cfg;
        cfg.backend = backend;
        cfg.inputWidth = w;
        cfg.inputHeight = h;
        cfg.outputWidth = outW;
        cfg.outputHeight = outH;
        cfg.sharpness = 0.5f;
        cfg.artifactReductionOnly = arOnly;
        auto up = CreateUpscaler(backend);
        up->Init(dev, cfg);
        auto out = dev.CreateTexture2D(outW, outH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        UpscaleInputs inputs;
        inputs.color = in.Get();
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { up->Evaluate(cl, inputs, out.Get()); });
        const PassImage img = DownloadRgb(dev, out.Get(), outW, outH);
        CHECK(up->Describe()["backend"] == backend);
        up->Shutdown();
        return img;
    };

    const PassImage bicubic = run("bicubic", 2 * w, 2 * h);
    const PassImage nis = run("nis", 2 * w, 2 * h);
    CHECK(nis.width == 2 * w);
    CHECK(nis.height == 2 * h);
    const PassImage nis2 = run("nis", 2 * w, 2 * h);
    CHECK(PassImage::MaxAbsDiff(nis, nis2) == 0.0);  // deterministic
    const ImageMetrics m = CompareImages(bicubic, nis);
    CHECK(m.psnrRgb > 20.0);  // same picture, different filter / sharpening
    CHECK(m.ssimY > 0.6);
    // values stay in range
    for (uint32_t y = 0; y < nis.height; ++y)
        for (uint32_t x = 0; x < nis.width; ++x)
            for (size_t c = 0; c < 3; ++c) {
                CHECK(nis.Get(x, y, c) >= 0.f);
                CHECK(nis.Get(x, y, c) <= 1.f);
            }
    // x3 goes through two NIS passes, x1.5 through one; sharpen-only keeps the size
    const PassImage nis3 = run("nis", 3 * w, 3 * h);
    CHECK(nis3.width == 3 * w);
    const PassImage nis15 = run("nis", 48, 36);
    CHECK(nis15.width == 48);
    const PassImage sharp = run("nis", w, h, true);
    CHECK(sharp.width == w);
    CHECK(CompareImages(src, sharp).psnrRgb > 20.0);
    // flat input stays flat through NIS
    const PassImage flat = MakeRgb(w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.3f, 0.6f, 0.9f}; });
    in = UploadRgba16f(dev, flat);
    const PassImage flatUp = run("nis", 2 * w, 2 * h);
    for (uint32_t y = 2; y + 2 < flatUp.height; ++y)
        for (uint32_t x = 2; x + 2 < flatUp.width; ++x) CHECK(std::fabs(flatUp.Get(x, y, 1) - 0.6f) < 0.01f);
}

// TASK-0022: the tiling behind the TensorRT upscaler — a translation-invariant "model" (nearest x2) must give the same
// frame whatever the tile size: overlapping windows, shifted last windows, frames smaller than a window (reflection).
TEST_CASE("UpscaleTiled reassembles a frame exactly from padded windows", "[upscale]") {
    const auto nearest2 = [](int T) {
        return [T](const float* in, float* out) {
            const int TS = T * 2;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < TS; ++y)
                    for (int x = 0; x < TS; ++x) out[(static_cast<size_t>(c) * TS + y) * TS + x] = in[(static_cast<size_t>(c) * T + y / 2) * T + x / 2];
        };
    };
    const auto pattern = [](uint32_t x, uint32_t y) { return std::array<float, 3>{static_cast<float>(x * 7 % 13) / 13.f, static_cast<float>(y * 5 % 11) / 11.f, static_cast<float>((x + y) % 17) / 17.f}; };
    struct Case { uint32_t w, h; int tile, pad; };
    for (const Case c : {Case{37, 23, 16, 2}, Case{10, 7, 16, 4}, Case{64, 64, 32, 8}, Case{33, 17, 8, 1}, Case{1, 1, 16, 2}}) {
        const PassImage rgb = MakeRgb(c.w, c.h, pattern);
        const PassImage up = UpscaleTiled(rgb, 2, c.tile, c.pad, nearest2(c.tile));
        REQUIRE(up.width == c.w * 2);
        REQUIRE(up.height == c.h * 2);
        size_t bad = 0;
        for (uint32_t y = 0; y < up.height; ++y)
            for (uint32_t x = 0; x < up.width; ++x)
                for (size_t ch = 0; ch < 3; ++ch)
                    if (std::abs(up.Get(x, y, ch) - rgb.Get(x / 2, y / 2, ch)) > 1e-6f) ++bad;
        INFO("frame " << c.w << "x" << c.h << " tile " << c.tile << " pad " << c.pad);
        CHECK(bad == 0);
    }
    const PassImage rgb = MakeRgb(8, 8, pattern);
    CHECK_THROWS(UpscaleTiled(rgb, 2, 8, 4, nearest2(8)));  // the padding leaves nothing to own
}
