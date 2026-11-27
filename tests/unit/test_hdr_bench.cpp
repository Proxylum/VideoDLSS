// TASK-0080 (HDR step TC-1), bench B1 (tc-plan §15.10, the user's decision 2026-11-27: the AVX2 row in TC): what the HDR -> SDR transfer
// costs a guide stage (depth and flow read F32 frames, FG and export F16) at 1080p and 4K, 10-bit 4:2:0, by content and kernel, and the
// build of the gamut tables. Hidden ([.]): run `dlssvid_unit_tests "[bench]"` on an idle machine (no CI job); the result goes to
// docs/benchmarks.md. The thresholds hold the AVX2 row at the auto thread count when it is 8 or more; below that the numbers are printed.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

#include "convert/ColorConvert.h"
#include "convert/HdrTransfer.h"

using namespace dlssvid;

namespace {

uint64_t Mix(uint64_t x) {  // splitmix64
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

enum class Content { Typical, Grey, Worst };
const char* Name(Content c) { return c == Content::Typical ? "typical" : c == Content::Grey ? "grey" : "worst"; }

// 2x2 blocks of one colour (PQ codes, BT.2020 NCL limited): typical - greys 0.25..0.68 with a chroma amplitude of 0.0176 (about 90 % of
// the pixels inside the knee), grey - Cb = Cr = 512, worst - saturated BT.2020 (every pixel compressed).
CpuFrame MakeFrame(uint32_t w, uint32_t h, Content c, uint64_t seed) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p16, 10});
    uint16_t* Y = f.Plane16(0);
    uint16_t* U = f.Plane16(1);
    uint16_t* V = f.Plane16(2);
    const uint32_t cw = static_cast<uint32_t>(f.desc.PlaneCols(1));
    uint64_t s = seed;
    const auto uni = [&] {
        s = Mix(s);
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    };
    const double kr = 0.2627, kb = 0.0593, kg = 1 - kr - kb;
    for (uint32_t by = 0; by < h / 2; ++by)
        for (uint32_t bx = 0; bx < w / 2; ++bx) {
            double rgb[3];
            if (c == Content::Typical) {
                const double g = 0.25 + 0.43 * uni(), phi = 6.283185307179586 * uni(), a = 0.0176 * std::sqrt(uni());
                const double d[3] = {std::cos(phi), std::cos(phi - 2.0943951023931953), std::cos(phi + 2.0943951023931953)};
                for (int k = 0; k < 3; ++k) rgb[k] = std::clamp(g + a * d[k], 0.0, 1.0);
            } else if (c == Content::Worst) {
                int perm[3] = {0, 1, 2};
                for (int i = 2; i > 0; --i) std::swap(perm[i], perm[static_cast<int>(uni() * (i + 1)) % (i + 1)]);
                rgb[perm[0]] = 0.55 + 0.2 * uni();
                rgb[perm[1]] = 0.08 + 0.45 * uni();
                rgb[perm[2]] = 0.06 + 0.04 * uni();
            } else {
                rgb[0] = rgb[1] = rgb[2] = 0.5;
            }
            const double yy = kr * rgb[0] + kg * rgb[1] + kb * rgb[2];
            const double cb = (rgb[2] - yy) / (2 * (1 - kb)), cr = (rgb[0] - yy) / (2 * (1 - kr));
            U[static_cast<size_t>(by) * cw + bx] = c == Content::Grey ? 512 : static_cast<uint16_t>(std::clamp(std::lround(512 + 896 * cb), 64L, 960L));
            V[static_cast<size_t>(by) * cw + bx] = c == Content::Grey ? 512 : static_cast<uint16_t>(std::clamp(std::lround(512 + 896 * cr), 64L, 960L));
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const double y10 = c == Content::Grey ? 64 + 876 * uni() : 64 + 876 * yy + 4 * (uni() - 0.5);
                    Y[static_cast<size_t>(by * 2 + dy) * w + bx * 2 + dx] = static_cast<uint16_t>(std::clamp(std::lround(y10), 64L, 940L));
                }
        }
    return f;
}

double MedianMs(int reps, const std::function<void()>& fn) {
    fn();  // warm: the tables, the caches
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

ColorInfo Hdr(Curve curve, Gamut gamut) {
    ColorInfo ci{ColorMatrix::Bt2020, ColorRange::Limited};
    ci.convert = Convert::HdrToSdr1;
    ci.curve = curve;
    ci.gamut = gamut;
    return ci;
}

}  // namespace

TEST_CASE("HDR to SDR at 1080p and 4K 10-bit: cost by content and kernel, and the table build", "[.][bench][hdr]") {
    const int autoThreads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 8);
    const bool avx2 = hdrrow::HasAvx2();
    std::printf("B1: %u logical CPUs, auto = %d threads, AVX2 %s\n", std::thread::hardware_concurrency(), autoThreads, avx2 ? "yes" : "no");

    // the gamut tables: built once per process
    for (int t : {autoThreads, 4, 1}) {
        std::vector<double> ms;
        for (int i = 0; i < 5; ++i) ms.push_back(hdr::BuildSdr1GamutTables(t).buildMs);
        std::sort(ms.begin(), ms.end());
        std::printf("B1 tables: %.1f ms on %d threads (median of 5)\n", ms[2], t);
        if (t == autoThreads && autoThreads >= 8) CHECK(ms[2] <= 50.0);
    }
    (void)HdrGamut::Get();

    struct Path {
        const char* name;
        Curve curve;
        Gamut gamut;
    };
    const Path paths[] = {{"PQ 709", Curve::Pq, Gamut::Bt709}, {"PQ 2020", Curve::Pq, Gamut::Bt2020}, {"HLG 2020", Curve::Hlg, Gamut::Bt2020}};
    struct Size {
        const char* name;
        uint32_t w, h;
        int reps;
    };
    const Size sizes[] = {{"1080p", 1920, 1080, 11}, {"4K", 3840, 2160, 7}};
    for (const Size& sz : sizes)
        for (Content c : {Content::Typical, Content::Grey, Content::Worst}) {
            const CpuFrame f = MakeFrame(sz.w, sz.h, c, 1000 + static_cast<uint64_t>(c));
            for (PixelType type : {PixelType::F32, PixelType::F16}) {
                const char* tn = type == PixelType::F32 ? "F32" : "F16";
                const ColorInfo sdr{ColorMatrix::Bt2020, ColorRange::Limited};
                const double base = MedianMs(sz.reps, [&] { (void)YuvToRgb(f, sdr, type, 1); });
                const double baseAuto = MedianMs(sz.reps, [&] { (void)YuvToRgb(f, sdr, type, autoThreads); });
                std::printf("B1 %s %s %s: base YuvToRgb(None) %.1f ms on 1 thread, %.1f ms on %d\n", sz.name, Name(c), tn, base, baseAuto, autoThreads);
                for (const Path& p : paths)
                    for (HdrKernel kernel : {HdrKernel::Portable, HdrKernel::Auto}) {
                        if (kernel == HdrKernel::Auto && !avx2) continue;
                        SetHdrKernel(kernel);
                        const ColorInfo ci = Hdr(p.curve, p.gamut);
                        double ms[3];
                        const int threads[3] = {1, 4, autoThreads};
                        for (int i = 0; i < 3; ++i) ms[i] = MedianMs(sz.reps, [&] { (void)YuvToRgb(f, ci, type, threads[i]); });
                        std::printf("B1 %s %s %s %-8s %-8s: %6.1f / %6.1f / %6.1f ms on 1 / 4 / %d threads = %.2f / %.2f / %.2f x base\n", sz.name, Name(c),
                                    tn, p.name, kernel == HdrKernel::Auto ? "AVX2" : "portable", ms[0], ms[1], ms[2], autoThreads, ms[0] / base, ms[1] / base,
                                    ms[2] / base);
                        // the guard of regressions (tc-plan §15.10, §16): the AVX2 row, F32, at the auto count when it is 8 or more - not
                        // slower than the single-thread base on typical and grey frames and at BT.709; the synthetic worst frame (every
                        // pixel compressed) within 1.5
                        if (kernel == HdrKernel::Auto && type == PixelType::F32 && autoThreads >= 8) {
                            const double r = ms[2] / base;
                            if (p.gamut == Gamut::Bt709 || c != Content::Worst) CHECK(r <= 1.0);
                            else CHECK(r <= 1.5);
                        }
                    }
                SetHdrKernel(HdrKernel::Auto);
            }
        }
}
