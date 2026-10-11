// Bit depth through the NVIDIA passes (TASK-0035). A shallow 10-bit ramp — 64 limited-range codes across the width, so
// every code is a plateau of 20 source px — goes through the upscale stage (NIS on WARP as the reference of the method,
// DLSS SR), Neural Rendering and Frame Generation. The plateaus must come out as 10-bit steps: an 8-bit path somewhere
// would merge four codes into one (a quarter of the distinct codes, plateaus four times as long, steps of four codes).
// DLSS Programming Guide §3.1.2 says LDR mode quantises to 8 bits; measured 2026-10-11 with RGBA16F inputs it does not
// (docs/benchmarks.md «Разрядность по пассам»). These checks fail the day a DLL or preset brings 8-bit steps back.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/PassSequence.h"
#include "stages/fg/FgStage.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrStage.h"
#include "stages/upscale/IUpscaler.h"
#include "stages/upscale/UpscaleStage.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

constexpr uint32_t kW = 1280, kH = 720;
constexpr int kBase = 448, kLevels = 64;  // 10-bit limited-range codes 448..511 (mid grey): a plateau of kW / kLevels = 20 px per code
constexpr int kFrames = 6;

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "precision" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// 10-bit 4:2:0 frame: a luma ramp of kLevels codes across the width, neutral chroma; the same picture in every frame
// (DLSS accumulates a static scene over its history).
CpuFrame RampFrame(int64_t index) {
    CpuFrame f;
    f.Allocate(FrameDesc{kW, kH, PixelFormat::Yuv420p16, 10});
    f.index = index;
    f.pts = index;
    for (uint32_t y = 0; y < kH; ++y)
        for (uint32_t x = 0; x < kW; ++x) f.Plane16(0)[static_cast<size_t>(y) * kW + x] = static_cast<uint16_t>(kBase + (x * (kLevels - 1)) / (kW - 1));
    std::fill(f.Plane16(1), f.Plane16(1) + f.desc.PlaneCols(1) * f.desc.PlaneHeight(1), static_cast<uint16_t>(512));
    std::fill(f.Plane16(2), f.Plane16(2) + f.desc.PlaneCols(2) * f.desc.PlaneHeight(2), static_cast<uint16_t>(512));
    return f;
}

std::filesystem::path RampClip(const std::filesystem::path& dir) {
    ClipSpec spec;
    spec.width = kW;
    spec.height = kH;
    spec.frames = kFrames;
    spec.pixFmt = "yuv420p10le";
    return WriteClipWith(dir / "ramp.mkv", spec, RampFrame);
}

// The middle row of a colour image on the 10-bit grid: distinct codes, the plateaus (runs of one code at least 8 px
// long), their median length and the median code step between neighbouring plateaus.
struct Plateaus {
    int distinctCodes = 0;
    int distinct8 = 0;  // on the 8-bit grid
    int plateaus = 0;
    double medianLength = 0.0;
    double medianStep = 0.0;
};

Plateaus Analyse(const PassImage& img, uint32_t margin) {
    const uint32_t y = img.height / 2;
    std::vector<int> codes;
    std::vector<int> codes8;
    for (uint32_t x = margin; x + margin < img.width; ++x) {
        const float v = std::clamp(img.Get(x, y, 0), 0.f, 1.f);
        codes.push_back(static_cast<int>(std::lround(v * 1023.f)));
        codes8.push_back(static_cast<int>(std::lround(v * 255.f)));
    }
    Plateaus p;
    auto distinct = [](std::vector<int> v) {
        std::sort(v.begin(), v.end());
        return static_cast<int>(std::unique(v.begin(), v.end()) - v.begin());
    };
    p.distinctCodes = distinct(codes);
    p.distinct8 = distinct(codes8);
    std::vector<int> lengths, values;
    for (size_t i = 0; i < codes.size();) {
        size_t j = i;
        while (j < codes.size() && codes[j] == codes[i]) ++j;
        if (j - i >= 8) {
            lengths.push_back(static_cast<int>(j - i));
            values.push_back(codes[i]);
        }
        i = j;
    }
    p.plateaus = static_cast<int>(lengths.size());
    auto median = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    p.medianLength = median(std::vector<double>(lengths.begin(), lengths.end()));
    std::vector<double> steps;
    for (size_t i = 1; i < values.size(); ++i) steps.push_back(std::fabs(static_cast<double>(values[i] - values[i - 1])));
    p.medianStep = median(steps);
    std::printf("precision: %ux%u row %u: %d distinct 10-bit codes (%d on the 8-bit grid), %d plateaus, median length %.0f px, median step %.1f codes\n", img.width,
                img.height, y, p.distinctCodes, p.distinct8, p.plateaus, p.medianLength, p.medianStep);
    return p;
}

// The expectations of a ramp that kept its 10-bit steps at output scale `scale`: all 64 codes (a few more at the
// transitions), plateaus about 20 * scale px long, steps of one code. An 8-bit path gives ~16 codes, plateaus of
// 80 * scale px and steps of four.
void CheckTenBitRamp(const Plateaus& p, double scale) {
    CHECK(p.distinctCodes >= 50);
    CHECK(p.distinctCodes <= 110);
    CHECK(p.plateaus >= 45);
    CHECK(p.medianLength <= 2.0 * 20.0 * scale);
    CHECK(p.medianStep <= 1.5);
}

PassImage SrFrame(const std::filesystem::path& passDir, int64_t index) { return PassReader::Open(passDir).ReadFrame(index); }

}  // namespace

TEST_CASE("A shallow 10-bit ramp keeps its steps through decoding and the upscale stage (NIS on WARP: the method)", "[integration][precision]") {
    const auto dir = Dir("nis");
    const auto clip = RampClip(dir);
    // the source as the stages see it: every code of the ramp is there
    VideoDecoder probe(clip);
    CpuFrame f;
    REQUIRE(probe.NextFrame(f));
    REQUIRE(f.desc == FrameDesc{kW, kH, PixelFormat::Yuv420p16, 10});
    const Plateaus src = Analyse(YuvToRgb(f, ColorInfoFromStream(probe.Info()), PixelType::F32), 32);
    CHECK(src.distinctCodes >= kLevels - 4);  // the 32 px margin cuts into the first and last plateaus
    CHECK(src.distinctCodes <= kLevels);
    CHECK(src.plateaus >= kLevels - 4);
    CHECK(src.medianLength == 20.0);
    CHECK(src.medianStep == 1.0);
    CHECK(src.distinct8 <= 20);  // on the 8-bit grid the ramp has a quarter of the steps: that is what an 8-bit path would leave

    D3D12Device dev({true, false});
    VideoDecoder dec(clip);
    UpscaleStageOptions o;
    o.backend = "nis";
    o.scale = 2.0;
    o.outputDir = dir / "passes";
    o.sourceFile = "ramp.mkv";
    const UpscaleRunResult r = RunUpscale(dec, dev, o);
    REQUIRE(r.stats.frames == kFrames);
    CheckTenBitRamp(Analyse(SrFrame(r.outDir, kFrames - 1), 64), 2.0);
}

TEST_CASE("DLSS SR keeps the 10-bit steps of a shallow ramp (no 8-bit quantisation in LDR mode)", "[integration][precision][dlss][gpu]") {
    D3D12Device dev;
    if (!dev.IsNvidia()) SKIP("DLSS needs an NVIDIA GPU");
    const UpscalerAvailability avail = UpscalerAvailable("dlss");
    if (!avail.available) SKIP(avail.reason);
    const auto dir = Dir("dlss");
    const auto clip = RampClip(dir);
    VideoDecoder dec(clip);
    UpscaleStageOptions o;
    o.backend = "dlss";
    o.allowFallback = false;
    o.scale = 2.0;
    o.outputDir = dir / "passes";
    o.sourceFile = "ramp.mkv";
    const UpscaleRunResult r = RunUpscale(dec, dev, o);
    REQUIRE(r.stats.frames == kFrames);
    REQUIRE(r.stats.backend == "dlss");
    CheckTenBitRamp(Analyse(SrFrame(r.outDir, kFrames - 1), 64), 2.0);  // after the history settled
}

TEST_CASE("Neural Rendering keeps the 10-bit steps of a shallow ramp", "[integration][precision][nr][gpu][ngx]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("Neural Rendering needs an NVIDIA GPU");
    const NrAvailability avail = NrAvailable("ngx");
    if (!avail.available) SKIP(avail.reason);
    const NvidiaDriverVersion drv = NvidiaDriverFromUmd(dev.UmdDriverVersion());
    if (drv.valid && !drv.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor)) SKIP("driver " + drv.ToString() + " is older than 616.56 (TASK-0011)");
    const auto dir = Dir("nr");
    const auto clip = RampClip(dir);
    {  // the colour input of NR: the ramp at 2x through NIS (a pure filter, keeps the steps)
        VideoDecoder dec(clip);
        UpscaleStageOptions u;
        u.backend = "nis";
        u.scale = 2.0;
        u.outputDir = dir / "passes";
        u.sourceFile = "ramp.mkv";
        REQUIRE(RunUpscale(dec, dev, u).stats.frames == kFrames);
    }
    VideoDecoder dec(clip);
    NrStageOptions o;
    o.backend = "ngx";
    o.colorDir = dir / "passes" / "color_sr";
    o.outputDir = dir / "passes";
    o.sourceFile = "ramp.mkv";
    NrRunResult r;
    try {
        r = RunNr(dec, dev, o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("FAIL_FeatureNotSupported") != std::string::npos) SKIP("the DLL refuses this GPU — patch it with `dlssvid nr-patch` (" + what + ")");
        throw;
    }
    REQUIRE(r.stats.frames == kFrames);
    // the model adds texture and re-tones the ramp, so plateaus are short and the tone differs; the code set must stay
    // 10-bit: an 8-bit path leaves at most ~20 distinct codes
    const Plateaus p = Analyse(SrFrame(r.outDir, kFrames - 1), 64);
    CHECK(p.distinctCodes >= 45);
    CHECK(p.distinctCodes <= 140);
}

TEST_CASE("DLSS Frame Generation keeps the 10-bit steps of a shallow ramp in the generated frames", "[integration][precision][fg][gpu][dlssg]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("DLSS Frame Generation needs an NVIDIA GPU");
    const FgAvailability avail = FgAvailable("dlssg");
    if (!avail.available) SKIP(avail.reason);
    const auto dir = Dir("fg");
    const auto clip = RampClip(dir);
    {
        VideoDecoder dec(clip);
        UpscaleStageOptions u;
        u.backend = "nis";
        u.scale = 2.0;
        u.outputDir = dir / "passes";
        u.sourceFile = "ramp.mkv";
        REQUIRE(RunUpscale(dec, dev, u).stats.frames == kFrames);
    }
    VideoDecoder dec(clip);
    FgStageOptions o;
    o.backend = "dlssg";
    o.multiplier = 2;
    o.colorDir = dir / "passes" / "color_sr";
    o.outputDir = dir / "passes";
    o.sourceFile = "ramp.mkv";
    FgRunResult r;
    try {
        r = RunFg(dec, dev, o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("not available on this GPU") != std::string::npos) SKIP(what);
        throw;
    }
    REQUIRE(r.stats.frames == kFrames);
    REQUIRE(r.stats.generated == kFrames - 1);
    const PassReader rd = PassReader::Open(r.outDir);
    // a generated frame late in the clip (between the last two real frames) and the real frame before it
    const Plateaus gen = Analyse(rd.ReadFrame(FgInterpIndex(kFrames - 2, 1, 2)), 64);
    CHECK(gen.distinctCodes >= 45);
    CHECK(gen.distinctCodes <= 140);
    CheckTenBitRamp(Analyse(rd.ReadFrame(FgRealIndex(kFrames - 1, 2)), 64), 2.0);  // real frames pass through untouched
}
