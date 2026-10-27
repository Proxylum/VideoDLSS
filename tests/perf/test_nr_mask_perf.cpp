// The cost of NR's control mask (PR-8, docs/benchmarks.md «Маска управления»): NrCompose's resolve at 2560x1440 timed with GPU
// timestamps around each Resolve - without a mask, grey (R16), colour without the split, colour with the split (d 8) and with the
// split and the flicker filter, in modes 0 and 1 - and on the CPU the resize of the mask to the frame and the import of a PNG8 and of
// a 4K RGB mask. NGX is not involved: the model's own time varies by ~10 ms between processes (docs/benchmarks.md), the mask's share
// is a fraction of a millisecond. Needs an NVIDIA GPU; run serially (ctest label perf). The soft bound - under 1 ms added - catches a
// low band at full resolution, not a few percent.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include "GpuTextures.h"
#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "passes/ControlMaskPass.h"
#include "passes/formats/PngIO.h"
#include "pipeline/ControlMask.h"
#include "stages/nr/NrCompose.h"
#include "stages/nr/NrLowpass.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

float Noise(uint32_t x, uint32_t y, uint32_t salt) {
    uint32_t h = x * 73856093u ^ y * 19349663u ^ (salt + 1) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return static_cast<float>(h & 0xFFFF) / 65535.f;
}

std::vector<float> NoiseRgb(uint32_t w, uint32_t h, uint32_t salt) {
    std::vector<float> v(static_cast<size_t>(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (uint32_t c = 0; c < 3; ++c) v[(static_cast<size_t>(y) * w + x) * 3 + c] = 0.2f + 0.6f * Noise(x, y, c + salt);
    return v;
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

double Ms(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

}  // namespace

TEST_CASE("NrCompose control mask cost at 2560x1440 (GPU timestamps)", "[perf][nr][mask]") {
    D3D12Device dev;
    if (dev.IsWarp() || !dev.IsNvidia()) SKIP("no NVIDIA GPU: the cost is measured on the reference hardware");
    const uint32_t w = 2560, h = 1440, ww = 1280, wh = 720;
    const auto orig = UploadRgba16f(dev, w, h, NoiseRgb(w, h, 0));
    const auto modelFull = UploadRgba16f(dev, w, h, NoiseRgb(w, h, 10));
    const auto proxy = UploadRgba16f(dev, ww, wh, NoiseRgb(ww, wh, 20));
    const auto modelWork = UploadRgba16f(dev, ww, wh, NoiseRgb(ww, wh, 30));
    const auto prev = UploadRgba16f(dev, w, h, NoiseRgb(w, h, 40));
    const auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    std::vector<uint16_t> grey(static_cast<size_t>(w) * h), even(static_cast<size_t>(w) * h * 4), split(even.size());
    for (size_t i = 0; i < grey.size(); ++i) {
        grey[i] = static_cast<uint16_t>(65535.f * Noise(static_cast<uint32_t>(i), 1, 50));
        even[i * 4] = split[i * 4] = grey[i];
        even[i * 4 + 1] = even[i * 4 + 2] = split[i * 4 + 1] = 40000;
        split[i * 4 + 2] = 10000;  // G != B: the low band runs
        even[i * 4 + 3] = split[i * 4 + 3] = 65535;
    }
    const auto greyTex = UploadR16Unorm(dev, w, h, grey), evenTex = UploadRgba16Unorm(dev, w, h, even), splitTex = UploadRgba16Unorm(dev, w, h, split);

    constexpr uint32_t kPerList = 16;  // the ring of ComputeKernel: 16 records per command list
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kPerList * 2;
    ComPtr<ID3D12QueryHeap> heap;
    REQUIRE(SUCCEEDED(dev.Get()->CreateQueryHeap(&qd, IID_PPV_ARGS(&heap))));
    const auto readback = dev.CreateBuffer(sizeof(uint64_t) * qd.Count, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    uint64_t freq = 0;
    REQUIRE(SUCCEEDED(dev.Queue()->GetTimestampFrequency(&freq)));

    struct Config {
        const char* name;
        ID3D12Resource* control;
        bool rgb, split, temporal;
    };
    const std::vector<Config> configs{{"no mask", nullptr, false, false, false},
                                      {"grey (R16)", greyTex.Get(), false, false, false},
                                      {"RGB, G = B", evenTex.Get(), true, false, false},
                                      {"RGB + split d 8", splitTex.Get(), true, true, false},
                                      {"no mask, temporal 0.5", nullptr, false, false, true},  // the baseline of the next one
                                      {"RGB + split, temporal 0.5", splitTex.Get(), true, true, true}};
    NrCompose compose(dev);
    std::printf("NrCompose resolve at %ux%u on %s (d %u, low band grid %ux%u):\n", w, h, dev.AdapterName().c_str(), NrLowpassDivisor(w, h),
                NrLowpassGrid(w, h, NrLowpassDivisor(w, h)).width, NrLowpassGrid(w, h, NrLowpassDivisor(w, h)).height);
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> baseline(2, 0.0);
        for (const Config& c : configs) {
            NrResolveParams p;
            p.width = w;
            p.height = h;
            p.workWidth = mode ? ww : w;
            p.workHeight = mode ? wh : h;
            p.temporal = c.temporal ? 0.5f : 0.f;
            p.mvWidth = ww;
            p.mvHeight = wh;
            NrResolveInputs in;
            in.original = orig.Get();
            in.model = mode ? modelWork.Get() : modelFull.Get();
            in.proxy = mode ? proxy.Get() : nullptr;
            in.prev = c.temporal ? prev.Get() : nullptr;
            in.control = c.control;
            in.controlRgb = c.rgb;
            in.controlSplit = c.split;
            for (int warm = 0; warm < 2; ++warm)  // 2 x 10 idle resolves: kernels made, clocks up
                dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
                    for (int i = 0; i < 10; ++i) compose.Resolve(cl, in, out.Get(), p);
                });
            std::vector<double> ms;
            for (int submit = 0; submit < 10; ++submit) {
                dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
                    for (uint32_t i = 0; i < kPerList; ++i) {
                        cl->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2);
                        compose.Resolve(cl, in, out.Get(), p);
                        cl->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2 + 1);
                    }
                    cl->ResolveQueryData(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, qd.Count, readback.Get(), 0);
                });
                uint64_t* t = nullptr;
                const D3D12_RANGE range{0, sizeof(uint64_t) * qd.Count};
                REQUIRE(SUCCEEDED(readback->Map(0, &range, reinterpret_cast<void**>(&t))));
                for (uint32_t i = 0; i < kPerList; ++i) ms.push_back(static_cast<double>(t[i * 2 + 1] - t[i * 2]) * 1000.0 / static_cast<double>(freq));
                const D3D12_RANGE none{0, 0};
                readback->Unmap(0, &none);
            }
            const double med = Median(ms);
            const int slot = c.temporal ? 1 : 0;
            if (!c.control) baseline[slot] = med;
            const double added = c.control ? med - baseline[slot] : 0.0;
            std::printf("  mode %d  %-28s %7.3f ms  %+7.3f ms\n", mode, c.name, med, added);
            if (c.control) CHECK(added < 1.0);
        }
    }

    // CPU: the resize at Setup (once per run) and the import (the GUI's thread, the CLI)
    const auto dir = TempDir() / "perf_nr_mask";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto png = [&](const char* name, uint32_t mw, uint32_t mh, PixelType type, size_t channels) {
        PassImage img;
        img.Allocate(mw, mh, type, channels == 3 ? std::vector<std::string>{"R", "G", "B"} : std::vector<std::string>{"A"});
        const float top = type == PixelType::U8 ? 255.f : 65535.f;
        for (uint32_t y = 0; y < mh; ++y)
            for (uint32_t x = 0; x < mw; ++x)
                for (size_t c = 0; c < channels; ++c) img.Set(x, y, c, std::floor(top * Noise(x / 7, y / 7, static_cast<uint32_t>(c))));
        WritePng(dir / name, img);
        return dir / name;
    };
    for (const auto& [name, mw, mh, type, ch] : {std::make_tuple("rgb8_1440p.png", 2560u, 1440u, PixelType::U8, size_t{3}),
                                                 std::make_tuple("rgb16_4k.png", 3840u, 2160u, PixelType::U16, size_t{3}),
                                                 std::make_tuple("rgb16_1080p.png", 1920u, 1080u, PixelType::U16, size_t{3})}) {
        const auto file = png(name, mw, mh, type, ch);
        const auto root = dir / (std::string(name) + ".passes");
        auto t0 = std::chrono::steady_clock::now();
        const ControlMaskImport r = ImportControlMask(file, root);
        const double importMs = Ms(t0);
        t0 = std::chrono::steady_clock::now();
        const ControlMaskTexels tex = LoadControlMaskForFrame(root / kControlMaskPass, w, h, r.mask.fingerprint);
        const double resizeMs = Ms(t0);
        t0 = std::chrono::steady_clock::now();
        const auto gpu = UploadRgba16Unorm(dev, w, h, tex.data);
        const double uploadMs = Ms(t0);
        std::printf("  %-16s %ux%u: import %7.1f ms, resize to %ux%u %6.1f ms, upload %5.1f ms\n", name, mw, mh, importMs, w, h, resizeMs, uploadMs);
    }
}
