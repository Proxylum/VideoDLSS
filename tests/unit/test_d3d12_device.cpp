#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <numeric>
#include <vector>

#include "gpu/D3D12Device.h"

using namespace dlssvid;

namespace {
std::vector<uint8_t> Pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    uint8_t x = seed;
    for (auto& b : v) {
        b = x;
        x = static_cast<uint8_t>(x * 31 + 7);
    }
    return v;
}
}  // namespace

TEST_CASE("D3D12Device creates on hardware or WARP", "[gpu]") {
    D3D12Device dev;
    REQUIRE(dev.Get() != nullptr);
    REQUIRE(dev.Queue() != nullptr);
    CHECK(!dev.AdapterName().empty());
}

TEST_CASE("D3D12Device WARP adapter is always available", "[gpu]") {
    D3D12Device dev({true, false});
    CHECK(dev.IsWarp());
}

TEST_CASE("D3D12Device buffer upload/readback is bit-exact", "[gpu]") {
    D3D12Device dev;
    const size_t size = 1 << 20;  // 1 MiB
    auto buf = dev.CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    const auto in = Pattern(size, 3);
    dev.UploadBuffer(buf.Get(), in.data(), size);
    const auto out = dev.ReadbackBuffer(buf.Get(), size);
    REQUIRE(out == in);
}

TEST_CASE("D3D12Device texture upload/readback handles row pitch alignment", "[gpu]") {
    D3D12Device dev;
    // Width not a multiple of 256: footprint row pitch != source row pitch.
    const uint32_t w = 333, h = 77;
    auto tex = dev.CreateTexture2D(w, h, DXGI_FORMAT_R8_UNORM);
    const auto in = Pattern(static_cast<size_t>(w) * h, 9);
    dev.UploadTexture2D(tex.Get(), in.data(), w);
    size_t pitch = 0;
    const auto out = dev.ReadbackTexture2D(tex.Get(), pitch);
    REQUIRE(pitch == w);
    REQUIRE(out == in);
}

TEST_CASE("D3D12Device RGBA16F texture round trip", "[gpu]") {
    D3D12Device dev;
    const uint32_t w = 640, h = 360;
    auto tex = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    const size_t rowBytes = static_cast<size_t>(w) * 8;
    const auto in = Pattern(rowBytes * h, 42);
    dev.UploadTexture2D(tex.Get(), in.data(), rowBytes);
    size_t pitch = 0;
    const auto out = dev.ReadbackTexture2D(tex.Get(), pitch);
    REQUIRE(pitch == rowBytes);
    REQUIRE(out == in);
}

// TASK-0048: this process's video memory on the adapter (DXGI QueryVideoMemoryInfo) - what `nr --check` reports for the NR
// feature; per process, so the GPU the CI runner shares with other work does not skew it.
TEST_CASE("D3D12Device reports the local video memory of this process and its growth", "[gpu]") {
    const VideoMemoryInfo a{100, 1000}, b{50, 1000}, c{150, 1000};
    CHECK(VideoMemoryGrowth(std::nullopt, a) == -1);
    CHECK(VideoMemoryGrowth(a, std::nullopt) == -1);
    CHECK(VideoMemoryGrowth(a, b) == 0);  // a release in between reads as no growth
    CHECK(VideoMemoryGrowth(b, c) == 100);
    D3D12Device dev({false, false});
    if (dev.IsWarp()) SKIP("no hardware adapter: WARP may not report video memory");
    const auto before = dev.LocalVideoMemory();
    REQUIRE(before);
    CHECK(before->budget > 0);
    const auto tex = dev.CreateTexture2D(4096, 4096, DXGI_FORMAT_R8G8B8A8_UNORM);  // 64 MiB; the driver may sub-allocate or page lazily
    CHECK(VideoMemoryGrowth(before, dev.LocalVideoMemory()) >= (32ll << 20));
}
