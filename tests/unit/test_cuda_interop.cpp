#include <catch2/catch_test_macros.hpp>

#include "gpu/D3D12Device.h"

#ifdef DLSSVID_WITH_CUDA
#include "gpu/CudaInterop.h"
#endif

using namespace dlssvid;

#ifdef DLSSVID_WITH_CUDA

TEST_CASE("CUDA writes into a shared D3D12 buffer are visible to D3D12", "[gpu][cuda]") {
    D3D12Device dev;
    std::string reason;
    if (dev.IsWarp() || !dev.IsNvidia() || !CudaInterop::Available(&reason)) {
        SKIP("no NVIDIA GPU / CUDA runtime: " << reason);
    }
    CudaInterop cuda(dev);

    const size_t size = 4 << 20;  // 4 MiB
    auto buf = dev.CreateBuffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_HEAP_FLAG_SHARED);
    auto imported = cuda.ImportBuffer(buf.Get(), size);
    REQUIRE(imported);

    std::vector<uint8_t> in(size);
    for (size_t i = 0; i < size; ++i) in[i] = static_cast<uint8_t>((i * 7) ^ (i >> 8));
    cuda.CopyToDevice(imported.devicePtr, in.data(), size);
    cuda.Synchronize();

    const auto out = dev.ReadbackBuffer(buf.Get(), size);
    REQUIRE(out == in);

    // And the other direction: D3D12 upload, CUDA read.
    std::vector<uint8_t> in2(size, 0xA5);
    dev.UploadBuffer(buf.Get(), in2.data(), size);
    std::vector<uint8_t> out2(size, 0);
    cuda.CopyToHost(out2.data(), imported.devicePtr, size);
    REQUIRE(out2 == in2);

    cuda.Release(imported);
    CHECK_FALSE(imported);
}

#else

TEST_CASE("CUDA interop not built", "[gpu][cuda]") { SUCCEED("built without DLSSVID_WITH_CUDA"); }

#endif
