#include <catch2/catch_test_macros.hpp>

#include "gpu/D3D12Device.h"
#include "gpu/GpuFrameCache.h"

using namespace dlssvid;

namespace {
CpuFrame MakeFrame(int64_t index, uint32_t w, uint32_t h, uint8_t seed) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    f.index = index;
    uint8_t x = seed;
    for (auto& b : f.data) {
        b = x;
        x = static_cast<uint8_t>(x * 13 + 1);
    }
    return f;
}
}  // namespace

TEST_CASE("GpuFrameCache round trips frames and rings over slots", "[gpu][cache]") {
    D3D12Device dev;
    GpuFrameCache cache(dev, 3);
    REQUIRE(cache.SlotCount() == 3);

    for (int64_t i = 0; i < 7; ++i) {
        CpuFrame in = MakeFrame(i, 64, 48, static_cast<uint8_t>(i + 1));
        auto& slot = cache.Acquire(i, in.desc);
        cache.Upload(slot, in);
        CpuFrame out;
        cache.Download(slot, out);
        REQUIRE(out.desc == in.desc);
        REQUIRE(out.data == in.data);
    }
    // Slot ring: frames 4,5,6 are resident, 3 was evicted by 6.
    CHECK(cache.Find(6) != nullptr);
    CHECK(cache.Find(4) != nullptr);
    CHECK(cache.Find(3) == nullptr);
    CHECK(cache.Find(-1) == nullptr);
}

TEST_CASE("GpuFrameCache reallocates when the frame size changes", "[gpu][cache]") {
    D3D12Device dev;
    GpuFrameCache cache(dev, 2);
    CpuFrame a = MakeFrame(0, 32, 32, 1);
    auto& s0 = cache.Acquire(0, a.desc);
    cache.Upload(s0, a);
    CpuFrame b = MakeFrame(2, 128, 64, 2);  // same slot (2 % 2 == 0), bigger frame
    auto& s2 = cache.Acquire(2, b.desc);
    CHECK(&s0 == &s2);
    cache.Upload(s2, b);
    CpuFrame out;
    cache.Download(s2, out);
    CHECK(out.data == b.data);
    CHECK_THROWS(cache.Upload(s2, a));  // format mismatch
}
