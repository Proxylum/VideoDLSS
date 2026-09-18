#pragma once

// Small RGBA16F / R32F / RG32F texture helpers for the compute-kernel tests (WARP).

#include <cstdint>
#include <vector>

#include "gpu/D3D12Device.h"
#include "util/Half.h"

namespace dlssvid::test {

// rgb: width * height * 3 floats -> RGBA16F texture (alpha 1)
inline ComPtr<ID3D12Resource> UploadRgba16f(D3D12Device& dev, uint32_t w, uint32_t h, const std::vector<float>& rgb, bool uav = false) {
    std::vector<uint16_t> half(static_cast<size_t>(w) * h * 4);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        half[i * 4] = FloatToHalf(rgb[i * 3]);
        half[i * 4 + 1] = FloatToHalf(rgb[i * 3 + 1]);
        half[i * 4 + 2] = FloatToHalf(rgb[i * 3 + 2]);
        half[i * 4 + 3] = FloatToHalf(1.f);
    }
    auto tex = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
    dev.UploadTexture2D(tex.Get(), reinterpret_cast<const uint8_t*>(half.data()), static_cast<size_t>(w) * 8);
    return tex;
}

inline ComPtr<ID3D12Resource> UploadRgba16fConst(D3D12Device& dev, uint32_t w, uint32_t h, float r, float g, float b, bool uav = false) {
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return UploadRgba16f(dev, w, h, rgb, uav);
}

// RGBA16F texture -> width * height * 3 floats
inline std::vector<float> ReadbackRgb(D3D12Device& dev, ID3D12Resource* tex, uint32_t w, uint32_t h) {
    size_t pitch = 0;
    const std::vector<uint8_t> bytes = dev.ReadbackTexture2D(tex, pitch);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(bytes.data());
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgb[i * 3] = HalfToFloat(src[i * 4]);
        rgb[i * 3 + 1] = HalfToFloat(src[i * 4 + 1]);
        rgb[i * 3 + 2] = HalfToFloat(src[i * 4 + 2]);
    }
    return rgb;
}

inline ComPtr<ID3D12Resource> UploadR32f(D3D12Device& dev, uint32_t w, uint32_t h, const std::vector<float>& v) {
    auto tex = dev.CreateTexture2D(w, h, DXGI_FORMAT_R32_FLOAT);
    dev.UploadTexture2D(tex.Get(), reinterpret_cast<const uint8_t*>(v.data()), static_cast<size_t>(w) * 4);
    return tex;
}

inline ComPtr<ID3D12Resource> UploadRg32f(D3D12Device& dev, uint32_t w, uint32_t h, const std::vector<float>& uv) {
    auto tex = dev.CreateTexture2D(w, h, DXGI_FORMAT_R32G32_FLOAT);
    dev.UploadTexture2D(tex.Get(), reinterpret_cast<const uint8_t*>(uv.data()), static_cast<size_t>(w) * 8);
    return tex;
}

}  // namespace dlssvid::test
