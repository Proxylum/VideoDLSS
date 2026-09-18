#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "convert/MvConvert.h"

using namespace dlssvid;

namespace {

PassImage ConstantFlow(uint32_t w, uint32_t h, float u, float v) {
    PassImage f = MakePassImage(PassKind::MvRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            f.Set(x, y, 0, u);
            f.Set(x, y, 1, v);
        }
    return f;
}

PassImage ConstantDepth(uint32_t w, uint32_t h, float z) {
    PassImage d = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) d.Set(x, y, 0, z);
    return d;
}

}  // namespace

TEST_CASE("InvertFlow: constant translation gives the negated vector everywhere", "[convert][mv]") {
    const PassImage flow = ConstantFlow(20, 12, 3.f, -2.f);
    const PassImage back = InvertFlow(flow, nullptr, /*fillHoles*/ true);
    REQUIRE(back.SameLayout(MakePassImage(PassKind::MvDlss, 20, 12)));
    for (uint32_t y = 0; y < 12; ++y)
        for (uint32_t x = 0; x < 20; ++x) {
            CHECK(back.Get(x, y, 0) == -3.f);
            CHECK(back.Get(x, y, 1) == 2.f);
        }
    // without hole filling the uncovered border stays zero
    const PassImage noFill = InvertFlow(flow, nullptr, false);
    CHECK(noFill.Get(0, 5, 0) == 0.f);   // x < 3 nobody maps here
    CHECK(noFill.Get(10, 5, 0) == -3.f);
}

TEST_CASE("InvertFlow: closer surface wins collisions when depth is given", "[convert][mv]") {
    // Left half moves +4 px, right half moves -4 px: they collide around the middle.
    const uint32_t w = 16, h = 4;
    PassImage flow = MakePassImage(PassKind::MvRaw, w, h);
    PassImage depth = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            flow.Set(x, y, 0, x < 8 ? 4.f : -4.f);
            flow.Set(x, y, 1, 0.f);
            depth.Set(x, y, 0, x < 8 ? 10.f : 2.f);  // right half is closer
        }
    // Left pixels land on 4..11 (far), right pixels land on 4..11 too (near): the near
    // surface wins every collision, so the overlap carries the right half's inverted vector.
    const PassImage back = InvertFlow(flow, &depth, true);
    for (uint32_t x = 4; x <= 11; ++x) CHECK(back.Get(x, 1, 0) == 4.f);
    // 0..3 and 12..15 are holes: filled from the nearest covered pixel
    CHECK(back.Get(0, 1, 0) == 4.f);
    CHECK(back.Get(15, 1, 0) == 4.f);
    // Without depth the last writer (the right half, scanned later) wins as well, but a
    // reversed depth flips the outcome: make the left half closer and check it wins.
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) depth.Set(x, y, 0, x < 8 ? 2.f : 10.f);
    const PassImage back2 = InvertFlow(flow, &depth, true);
    for (uint32_t x = 4; x <= 11; ++x) CHECK(back2.Get(x, 1, 0) == -4.f);
}

TEST_CASE("ForwardFlowToBackwardMv scales to the target resolution", "[convert][mv]") {
    const PassImage flow = ConstantFlow(10, 6, 1.f, 0.5f);
    MvConvertOptions opt;
    opt.targetWidth = 20;
    opt.targetHeight = 12;
    opt.dilateRadius = 0;
    const PassImage mv = ForwardFlowToBackwardMv(flow, nullptr, opt);
    CHECK(mv.width == 20);
    CHECK(mv.height == 12);
    CHECK(mv.Get(7, 7, 0) == -2.f);
    CHECK(mv.Get(7, 7, 1) == -1.f);
    CHECK(mv.channels == std::vector<std::string>{"u", "v"});
}

TEST_CASE("BackwardMvToForwardFlow inverts ForwardFlowToBackwardMv for translations", "[convert][mv]") {
    const PassImage flow = ConstantFlow(24, 16, -5.f, 3.f);
    MvConvertOptions opt;
    opt.dilateRadius = 0;
    const PassImage mv = ForwardFlowToBackwardMv(flow, nullptr, opt);
    const PassImage again = BackwardMvToForwardFlow(mv, nullptr, opt);
    CHECK(again.channels == Spec(PassKind::MvRaw).channels);
    for (uint32_t y = 0; y < 16; ++y)
        for (uint32_t x = 0; x < 24; ++x) {
            CHECK(again.Get(x, y, 0) == -5.f);
            CHECK(again.Get(x, y, 1) == 3.f);
        }
}

TEST_CASE("DilateMvByDepth pushes the nearer object's vector over the edge", "[convert][mv]") {
    const uint32_t w = 12, h = 3;
    PassImage mv = MakePassImage(PassKind::MvDlss, w, h);
    PassImage depth = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            mv.Set(x, y, 0, x < 6 ? 1.f : 9.f);
            depth.Set(x, y, 0, x < 6 ? 1.f : 20.f);  // left object is much closer
        }
    const PassImage d1 = DilateMvByDepth(mv, depth, 1, 0.02f);
    CHECK(d1.Get(6, 1, 0) == 1.f);  // background pixel next to the edge takes the foreground vector
    CHECK(d1.Get(7, 1, 0) == 9.f);  // one pixel further is untouched with radius 1
    CHECK(d1.Get(5, 1, 0) == 1.f);  // foreground unchanged
    const PassImage d2 = DilateMvByDepth(mv, depth, 2, 0.02f);
    CHECK(d2.Get(7, 1, 0) == 1.f);
    CHECK(d2.Get(8, 1, 0) == 9.f);
    CHECK(DilateMvByDepth(mv, depth, 0, 0.02f).data == mv.data);
}

TEST_CASE("ScaleMv, FlipMvY and ZeroMv", "[convert][mv]") {
    const PassImage flow = ConstantFlow(8, 4, 2.f, 1.f);
    const PassImage half = ScaleMv(flow, 4, 2);
    CHECK(half.Get(3, 1, 0) == 1.f);
    CHECK(half.Get(3, 1, 1) == 0.5f);
    const PassImage flipped = FlipMvY(flow);
    CHECK(flipped.Get(0, 0, 0) == 2.f);
    CHECK(flipped.Get(0, 0, 1) == -1.f);
    const PassImage zero = ZeroMv(3, 3);
    CHECK(zero.Get(2, 2, 0) == 0.f);
    CHECK(zero.channels.size() == 2);
    CHECK_THROWS(ScaleMv(flow, 0, 4));
    CHECK_THROWS(InvertFlow(MakePassImage(PassKind::DepthRaw, 2, 2), nullptr, true));
}
