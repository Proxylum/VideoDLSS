#include "stages/upscale/Tiling.h"

#include <algorithm>
#include <string>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

namespace {

// Window origins along one axis: every `stride` pixels, the last one shifted back to end at the edge; a single window
// at 0 when the axis is shorter than the window (the window is then filled by reflection).
std::vector<int> TileOrigins(int size, int tile, int stride) {
    if (size <= tile) return {0};
    std::vector<int> out;
    for (int x = 0;; x += stride) {
        const int x0 = std::min(x, size - tile);
        out.push_back(x0);
        if (x0 + tile >= size) break;
    }
    return out;
}

int Reflect(int x, int size) {
    if (size <= 1) return 0;
    if (x < 0) x = -x;
    if (x >= size) x = 2 * size - x - 2;
    return std::clamp(x, 0, size - 1);
}

}  // namespace

PassImage UpscaleTiled(const PassImage& rgb, int scale, int tile, int pad, const TileRunFn& run) {
    if (rgb.type != PixelType::F32 || rgb.ChannelCount() != 3) Throw("UpscaleTiled: expects RGB F32");
    if (scale < 1 || tile < 4 || pad < 0 || 2 * pad >= tile)
        Throw("UpscaleTiled: bad tiling (scale " + std::to_string(scale) + ", tile " + std::to_string(tile) + ", pad " + std::to_string(pad) + ")");
    const int W = static_cast<int>(rgb.width), H = static_cast<int>(rgb.height), T = tile, S = scale;
    const int stride = T - 2 * pad;
    PassImage out;
    out.Allocate(rgb.width * static_cast<uint32_t>(S), rgb.height * static_cast<uint32_t>(S), PixelType::F32, {"R", "G", "B"});
    const float* src = rgb.As<float>();
    float* dst = out.As<float>();
    std::vector<float> in(static_cast<size_t>(3) * T * T), tileOut(static_cast<size_t>(3) * T * S * T * S);
    const int outW = W * S, TS = T * S;
    for (const int y0 : TileOrigins(H, T, stride))
        for (const int x0 : TileOrigins(W, T, stride)) {
            // the window, reflected where it leaves the frame; NCHW
            for (int c = 0; c < 3; ++c)
                for (int ty = 0; ty < T; ++ty) {
                    const int sy = Reflect(y0 + ty, H);
                    float* row = in.data() + (static_cast<size_t>(c) * T + ty) * T;
                    for (int tx = 0; tx < T; ++tx) row[tx] = src[(static_cast<size_t>(sy) * W + Reflect(x0 + tx, W)) * 3 + c];
                }
            run(in.data(), tileOut.data());
            // what this window owns: past the padding, except at the frame's edges
            const int ax = x0 == 0 ? 0 : x0 + pad, bx = x0 + T >= W ? W : x0 + T - pad;
            const int ay = y0 == 0 ? 0 : y0 + pad, by = y0 + T >= H ? H : y0 + T - pad;
            for (int y = ay * S; y < by * S; ++y) {
                const int ty = y - y0 * S;
                for (int x = ax * S; x < bx * S; ++x) {
                    const int tx = x - x0 * S;
                    float* px = dst + (static_cast<size_t>(y) * outW + x) * 3;
                    for (int c = 0; c < 3; ++c) px[c] = tileOut[(static_cast<size_t>(c) * TS + ty) * TS + tx];
                }
            }
        }
    return out;
}

}  // namespace dlssvid
