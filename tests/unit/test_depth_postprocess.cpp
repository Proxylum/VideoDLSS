#include <catch2/catch_test_macros.hpp>

#include <cmath>
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

TEST_CASE("TemporalStabilizer removes per-frame scale flicker", "[depth][postprocess]") {
    TemporalStabilizer st(TemporalStabilizer::Mode::ScaleShift, 8);
    PassImage first = Ramp(20, 10, 1.f, 0.f);
    st.Stabilize(first);  // reference
    const PassImage reference = first;
    const float scales[] = {1.3f, 0.8f, 1.1f, 0.95f, 1.25f};
    for (float s : scales) {
        PassImage f = Ramp(20, 10, s, 0.05f * s);
        const ScaleShift ss = st.Stabilize(f);
        CHECK(std::fabs(ss.scale * s - 1.f) < 0.02f);
        CHECK(TemporalAlignmentError(reference, f) < 0.01);
    }
    // ScaleOnly keeps the shift at zero; None does nothing
    TemporalStabilizer so(TemporalStabilizer::Mode::ScaleOnly, 4);
    PassImage a = Ramp(8, 4, 1.f, 0.f);
    so.Stabilize(a);
    PassImage b = Ramp(8, 4, 2.f, 0.f);
    const ScaleShift ss = so.Stabilize(b);
    CHECK(std::fabs(ss.scale - 0.5f) < 1e-4f);
    CHECK(ss.shift == 0.f);
    TemporalStabilizer none(TemporalStabilizer::Mode::None, 4);
    PassImage c = Ramp(8, 4, 3.f, 0.f);
    const PassImage before = c;
    none.Stabilize(c);
    CHECK(c.data == before.data);
    // resolution change resets the window instead of throwing
    PassImage big = Ramp(16, 8, 1.f, 0.f);
    CHECK(so.Stabilize(big).scale == 1.f);
}
