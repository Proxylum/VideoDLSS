#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <limits>

#include "stages/depth/DepthPostProcess.h"

using namespace dlssvid;

namespace {
PassImage Ramp(uint32_t w, uint32_t h, float scale, float shift) {
    PassImage d = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) d.Set(x, y, 0, (1.f + x * 0.1f + y * 0.05f) * scale + shift);
    return d;
}
}  // namespace

TEST_CASE("SolveScaleShift recovers a known affine transform", "[depth][postprocess]") {
    const PassImage ref = Ramp(30, 20, 1.f, 0.f);
    const PassImage d = Ramp(30, 20, 0.5f, -0.2f);  // ref = 2 * d + 0.4
    const ScaleShift ss = SolveScaleShift(d, ref, true);
    CHECK(ss.samples == 600);
    CHECK(std::fabs(ss.scale - 2.f) < 1e-4f);
    CHECK(std::fabs(ss.shift - 0.4f) < 1e-4f);
    PassImage aligned = d;
    ApplyScaleShift(aligned, ss);
    CHECK(PassImage::MaxAbsDiff(aligned, ref) < 1e-4);
    // scale-only
    const PassImage d2 = Ramp(30, 20, 0.25f, 0.f);
    const ScaleShift so = SolveScaleShift(d2, ref, false);
    CHECK(std::fabs(so.scale - 4.f) < 1e-4f);
    CHECK(so.shift == 0.f);
    // invalid samples are ignored, degenerate input yields identity
    PassImage withNan = d;
    withNan.Set(0, 0, 0, std::numeric_limits<float>::quiet_NaN());
    withNan.Set(1, 0, 0, -3.f);
    const ScaleShift s3 = SolveScaleShift(withNan, ref, true);
    CHECK(s3.samples == 598);
    CHECK(std::fabs(s3.scale - 2.f) < 1e-3f);
    PassImage one = MakePassImage(PassKind::DepthRaw, 1, 1);
    one.Set(0, 0, 0, 1.f);
    CHECK(SolveScaleShift(one, one, true).scale == 1.f);
}

TEST_CASE("FillInvalidDepth propagates nearest valid values", "[depth][postprocess]") {
    PassImage d = Ramp(10, 6, 1.f, 0.f);
    d.Set(5, 3, 0, std::numeric_limits<float>::quiet_NaN());
    d.Set(6, 3, 0, 0.f);
    d.Set(0, 0, 0, -std::numeric_limits<float>::infinity());
    const size_t filled = FillInvalidDepth(d);
    CHECK(filled == 3);
    for (uint32_t y = 0; y < 6; ++y)
        for (uint32_t x = 0; x < 10; ++x) CHECK(std::isfinite(d.Get(x, y, 0)));
    CHECK(d.Get(5, 3, 0) > 0.f);
    CHECK(std::fabs(d.Get(5, 3, 0) - Ramp(10, 6, 1.f, 0.f).Get(4, 3, 0)) < 0.2f);  // neighbour value
    CHECK(FillInvalidDepth(d) == 0);
    // all invalid -> constant
    PassImage bad = MakePassImage(PassKind::DepthRaw, 3, 3);
    for (uint32_t i = 0; i < 9; ++i) bad.As<float>()[i] = 0.f;
    CHECK(FillInvalidDepth(bad) == 9);
    CHECK(bad.Get(1, 1, 0) == 1.f);
}

TEST_CASE("TemporalAlignmentError is relative and zero for identical frames", "[depth][postprocess]") {
    const PassImage a = Ramp(16, 8, 1.f, 0.f);
    CHECK(TemporalAlignmentError(a, a) == 0.0);
    const PassImage b = Ramp(16, 8, 1.1f, 0.f);  // 10 % larger everywhere
    CHECK(std::fabs(TemporalAlignmentError(a, b) - 0.1) < 1e-4);
    const PassImage c = Ramp(32, 8, 1.f, 0.f);
    CHECK_THROWS(TemporalAlignmentError(a, c));
}

// ---- TemporalStabilizer: centred window ±N, no recursion (TASK-0038, audit depth-7) -----------------------------------

namespace {
struct Out {
    int64_t index;
    PassImage depth;
    ScaleShift applied;
};

std::vector<Out> RunAll(TemporalStabilizer& st, const std::vector<PassImage>& frames, int64_t first = 0) {
    std::vector<Out> out;
    Out o;
    for (size_t i = 0; i < frames.size(); ++i) {
        st.Push(first + static_cast<int64_t>(i), frames[i]);
        while (st.Pop(o.index, o.depth, &o.applied)) out.push_back(o);
    }
    st.Close();
    while (st.Pop(o.index, o.depth, &o.applied)) out.push_back(o);
    return out;
}

double Median(const PassImage& d) {
    std::vector<double> v;
    for (uint32_t y = 0; y < d.height; ++y)
        for (uint32_t x = 0; x < d.width; ++x)
            if (std::isfinite(d.Get(x, y, 0))) v.push_back(d.Get(x, y, 0));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>((v.size() - 1) / 2), v.end());
    return v[(v.size() - 1) / 2];
}

double MedianAbsDev(const PassImage& d) {
    const double m = Median(d);
    std::vector<double> v;
    for (uint32_t y = 0; y < d.height; ++y)
        for (uint32_t x = 0; x < d.width; ++x)
            if (std::isfinite(d.Get(x, y, 0))) v.push_back(std::fabs(d.Get(x, y, 0) - m));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>((v.size() - 1) / 2), v.end());
    return v[(v.size() - 1) / 2];
}

// deterministic flicker of at most ±12 %: neighbouring frames differ by < 1.5 (no segment break)
float Flicker(int k) {
    static const float f[] = {1.12f, 0.9f, 1.08f, 0.88f, 1.1f, 0.92f, 1.05f, 0.89f, 1.11f, 0.95f};
    return f[k % 10];
}
}  // namespace

TEST_CASE("TemporalStabilizer (scale) removes per-frame scale flicker over a centred window", "[depth][postprocess][stabilize]") {
    const int n = 40, radius = 8;
    std::vector<PassImage> in;
    for (int k = 0; k < n; ++k) in.push_back(Ramp(32, 16, Flicker(k), 0.f));
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, radius);
    const std::vector<Out> out = RunAll(st, in);
    REQUIRE(out.size() == static_cast<size_t>(n));
    double maxIn = 0.0, maxOut = 0.0;
    for (int k = 0; k < n; ++k) {
        CHECK(out[static_cast<size_t>(k)].index == k);
        const PassImage& o = out[static_cast<size_t>(k)].depth;
        // only a scale: the ratio out/in is one number per frame
        const double r0 = o.Get(0, 0, 0) / in[static_cast<size_t>(k)].Get(0, 0, 0), r1 = o.Get(31, 15, 0) / in[static_cast<size_t>(k)].Get(31, 15, 0);
        CHECK(std::fabs(r0 - r1) < 1e-5);
        CHECK(out[static_cast<size_t>(k)].applied.shift == 0.f);
        if (k >= radius && k <= n - 1 - radius) {  // full windows
            if (k > radius) {
                maxIn = std::max(maxIn, std::fabs(std::log(Median(in[static_cast<size_t>(k)]) / Median(in[static_cast<size_t>(k - 1)]))));
                maxOut = std::max(maxOut, std::fabs(std::log(Median(o) / Median(out[static_cast<size_t>(k - 1)].depth))));
            }
            double window = 0.0;  // the frame takes the geometric mean level of its window: no drift, no recursion
            for (int j = k - radius; j <= k + radius; ++j) window += std::log(Flicker(j)) / (2 * radius + 1);
            CHECK(std::fabs(std::log(Median(o) / Median(Ramp(32, 16, 1.f, 0.f))) - window) < 1e-4);
        }
    }
    CHECK(maxOut <= 0.25 * maxIn);
}

TEST_CASE("TemporalStabilizer keeps the scale of moving content (no drift, regression TASK-0038 depth-7)", "[depth][postprocess][stabilize][regression]") {
    // a periodic depth texture moving 5 px per frame: the content is the same every frame, only shifted - the old
    // recursive least squares towards the blurred mean of aligned frames shrank it frame after frame
    const uint32_t w = 64, h = 8;
    std::vector<PassImage> in;
    for (int k = 0; k < 60; ++k) {
        PassImage d = MakePassImage(PassKind::DepthRaw, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) d.Set(x, y, 0, 2.f + std::sin(2.f * 3.14159265f * static_cast<float>(x + 5 * k) / 16.f));
        in.push_back(d);
    }
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, 8);
    const std::vector<Out> out = RunAll(st, in);
    REQUIRE(out.size() == 60);
    for (const Out& o : out) CHECK(std::fabs(o.applied.scale - 1.f) < 1e-5f);
    CHECK(std::fabs(Median(out.back().depth) - Median(out.front().depth)) < 1e-5);
}

TEST_CASE("TemporalStabilizer passes a depth trend through and does not average across a cut", "[depth][postprocess][stabilize]") {
    SECTION("a dolly: depth growing 2 % per frame stays as it is") {
        std::vector<PassImage> in;
        for (int k = 0; k < 20; ++k) in.push_back(Ramp(24, 12, std::pow(1.02f, static_cast<float>(k)), 0.f));
        TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, 8);
        for (const Out& o : RunAll(st, in)) CHECK(std::fabs(o.applied.scale - 1.f) < 1e-5f);  // the centre and the shrunk ends
    }
    SECTION("a cut: the two shots are aligned apart") {
        std::vector<PassImage> in;
        for (int k = 0; k < 20; ++k) in.push_back(Ramp(24, 12, (k < 10 ? 1.f : 3.f) * Flicker(k), 0.f));
        TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, 8);
        const std::vector<Out> out = RunAll(st, in);
        REQUIRE(out.size() == 20);
        const double base = Median(Ramp(24, 12, 1.f, 0.f));
        CHECK(std::fabs(Median(out[5].depth) / base - 1.0) < 0.03);  // r = min(5, 4): the first shot's level
        for (const int k : {9, 10}) {  // the last frame of a shot and the first of the next: no one-sided pull
            CHECK(out[static_cast<size_t>(k)].applied.scale == 1.f);
            CHECK(out[static_cast<size_t>(k)].applied.shift == 0.f);
            CHECK(out[static_cast<size_t>(k)].depth.data == in[static_cast<size_t>(k)].data);
        }
        for (int k = 0; k < 10; ++k) CHECK(out[static_cast<size_t>(k)].depth.Get(3, 3, 0) / in[static_cast<size_t>(k)].Get(3, 3, 0) <= 1.2f);  // never towards 3x
    }
}

TEST_CASE("TemporalStabilizer delays by the radius and flushes the tail on Close; None passes frames through", "[depth][postprocess][stabilize]") {
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, 3);
    CHECK(st.Delay() == 3);
    int64_t index = -1;
    PassImage d;
    for (int k = 0; k < 3; ++k) {
        st.Push(10 + k, Ramp(16, 8, Flicker(k), 0.f));
        CHECK_FALSE(st.Pop(index, d));
    }
    st.Push(13, Ramp(16, 8, Flicker(3), 0.f));
    REQUIRE(st.Pop(index, d));
    CHECK(index == 10);
    CHECK_FALSE(st.Pop(index, d));
    CHECK_THROWS(st.Push(15, Ramp(16, 8, 1.f, 0.f)));  // frames must come in order
    st.Push(14, Ramp(16, 8, Flicker(4), 0.f));
    st.Close();
    std::vector<int64_t> rest;
    while (st.Pop(index, d)) rest.push_back(index);
    CHECK(rest == std::vector<int64_t>{11, 12, 13, 14});
    CHECK(st.Buffered() == 0);

    TemporalStabilizer none(TemporalStabilizer::Mode::None, 8);
    CHECK(none.Delay() == 0);
    const PassImage c = Ramp(8, 4, 3.f, 0.f);
    none.Push(0, c);
    ScaleShift applied;
    REQUIRE(none.Pop(index, d, &applied));
    CHECK(d.data == c.data);
    CHECK(applied.scale == 1.f);

    SECTION("a frame without valid depth leaves unchanged and splits the window") {
        std::vector<PassImage> in;
        for (int k = 0; k < 11; ++k) {
            PassImage f = Ramp(16, 8, Flicker(k), 0.f);
            if (k == 5)
                for (uint32_t y = 0; y < 8; ++y)
                    for (uint32_t x = 0; x < 16; ++x) f.Set(x, y, 0, std::numeric_limits<float>::quiet_NaN());
            in.push_back(f);
        }
        TemporalStabilizer s(TemporalStabilizer::Mode::ScaleOnly, 3);
        const std::vector<Out> out = RunAll(s, in);
        REQUIRE(out.size() == 11);
        CHECK(out[5].depth.data == in[5].data);
        for (const int k : {4, 6}) CHECK(out[static_cast<size_t>(k)].applied.scale == 1.f);  // not averaged with the gap
    }
}

TEST_CASE("TemporalStabilizer restarts on a resolution change without throwing", "[depth][postprocess][stabilize]") {
    std::vector<PassImage> in;
    for (int k = 0; k < 5; ++k) in.push_back(Ramp(20, 10, Flicker(k), 0.f));
    for (int k = 5; k < 10; ++k) in.push_back(Ramp(16, 8, Flicker(k), 0.f));
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleOnly, 3);
    const std::vector<Out> out = RunAll(st, in);
    REQUIRE(out.size() == 10);
    for (int k = 0; k < 10; ++k) CHECK(out[static_cast<size_t>(k)].index == k);
    CHECK(out[5].depth.width == 16);
    CHECK(out[5].applied.scale == 1.f);  // a new segment
    CHECK(out[5].applied.shift == 0.f);
}

TEST_CASE("TemporalStabilizer (scale_shift) aligns location and spread of relative depth", "[depth][postprocess][stabilize]") {
    const int n = 30, radius = 6;
    std::vector<PassImage> in;
    for (int k = 0; k < n; ++k) in.push_back(Ramp(32, 16, 1.f + 0.1f * (Flicker(k) - 1.f), 0.2f * (Flicker(k + 3) - 1.f)));
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleShift, radius);
    const std::vector<Out> out = RunAll(st, in);
    REQUIRE(out.size() == static_cast<size_t>(n));
    const double loc0 = Median(out[radius].depth), spread0 = MedianAbsDev(out[radius].depth);
    for (int k = radius; k <= n - 1 - radius; ++k) {
        CHECK(std::fabs(Median(out[static_cast<size_t>(k)].depth) / loc0 - 1.0) < 0.02);
        CHECK(std::fabs(MedianAbsDev(out[static_cast<size_t>(k)].depth) / spread0 - 1.0) < 0.02);
    }
    for (const Out& o : out)
        for (uint32_t y = 0; y < 16; ++y)
            for (uint32_t x = 0; x < 32; ++x) CHECK(o.depth.Get(x, y, 0) > 0.f);
}

TEST_CASE("TemporalStabilizer (scale_shift) is robust to far outliers", "[depth][postprocess][stabilize]") {
    // 5 % of the pixels far away (sky of a disparity model), their depth jumping 2x between frames: the median statistics
    // ignore them, a mean would follow them
    const int n = 24, radius = 4;
    std::vector<PassImage> in;
    for (int k = 0; k < n; ++k) {
        PassImage d = Ramp(40, 20, 1.f + 0.1f * (Flicker(k) - 1.f), 0.1f * (Flicker(k + 5) - 1.f));
        for (uint32_t i = 0; i < 40; ++i) d.Set(i, 0, 0, k % 2 ? 1e6f : 2e6f);  // 40 of 800 pixels
        in.push_back(d);
    }
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleShift, radius);
    const std::vector<Out> out = RunAll(st, in);
    const double loc0 = Median(out[radius].depth), spread0 = MedianAbsDev(out[radius].depth);
    for (int k = radius; k <= n - 1 - radius; ++k) {
        CHECK(std::fabs(Median(out[static_cast<size_t>(k)].depth) / loc0 - 1.0) < 0.02);
        CHECK(std::fabs(MedianAbsDev(out[static_cast<size_t>(k)].depth) / spread0 - 1.0) < 0.02);
    }
}
