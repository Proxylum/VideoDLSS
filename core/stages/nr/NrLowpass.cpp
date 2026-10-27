#include "stages/nr/NrLowpass.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "util/Error.h"

namespace dlssvid {

uint32_t NrLowpassDivisor(uint32_t w, uint32_t h, uint32_t override) {
    if (override) return override;
    const long d = std::lround(static_cast<double>(std::min(w, h)) / kNrLowpassCells);
    return static_cast<uint32_t>(std::max(2L, d));
}

NrLowGrid NrLowpassGrid(uint32_t w, uint32_t h, uint32_t div) {
    if (div == 0) Throw("NrLowpassGrid: d must be at least 1");
    return {(w + div - 1) / div, (h + div - 1) / div, div};
}

PassImage NrLowpassCpu(const PassImage& e, uint32_t div) {
    if (e.Empty()) Throw("NrLowpassCpu: empty image");
    const uint32_t w = e.width, h = e.height;
    const size_t nc = e.channels.size();
    const NrLowGrid g = NrLowpassGrid(w, h, div);
    auto at = [nc](std::vector<float>& v, uint32_t gw, int x, int y) { return v.data() + (static_cast<size_t>(y) * gw + x) * nc; };

    // CSEditBox: the mean over the pixels of the cell that exist, row by row as the kernel sums
    std::vector<float> box(static_cast<size_t>(g.width) * g.height * nc), low(box.size());
    std::vector<float> sum(nc), row(nc);
    for (uint32_t cy = 0; cy < g.height; ++cy)
        for (uint32_t cx = 0; cx < g.width; ++cx) {
            const uint32_t x0 = cx * div, y0 = cy * div, x1 = std::min(x0 + div, w), y1 = std::min(y0 + div, h);
            std::fill(sum.begin(), sum.end(), 0.f);
            for (uint32_t y = y0; y < y1; ++y) {
                std::fill(row.begin(), row.end(), 0.f);
                for (uint32_t x = x0; x < x1; ++x)
                    for (size_t c = 0; c < nc; ++c) {
                        const float v = e.Get(x, y, c);
                        row[c] += std::isfinite(v) ? v : 0.f;
                    }
                for (size_t c = 0; c < nc; ++c) sum[c] += row[c];
            }
            const float n = static_cast<float>((x1 - x0) * (y1 - y0));
            float* b = at(box, g.width, cx, cy);
            for (size_t c = 0; c < nc; ++c) b[c] = sum[c] / n;
        }

    // CSEditGauss: [1 2 1] x [1 2 1] / 16, edges clamped
    const int hx = static_cast<int>(g.width) - 1, hy = static_cast<int>(g.height) - 1;
    auto B = [&](int x, int y, size_t c) { return at(box, g.width, std::clamp(x, 0, hx), std::clamp(y, 0, hy))[c]; };
    for (int cy = 0; cy <= hy; ++cy)
        for (int cx = 0; cx <= hx; ++cx) {
            float* l = at(low, g.width, cx, cy);
            for (size_t c = 0; c < nc; ++c) {
                const float r0 = B(cx - 1, cy - 1, c) + 2.f * B(cx, cy - 1, c) + B(cx + 1, cy - 1, c);
                const float r1 = B(cx - 1, cy, c) + 2.f * B(cx, cy, c) + B(cx + 1, cy, c);
                const float r2 = B(cx - 1, cy + 1, c) + 2.f * B(cx, cy + 1, c) + B(cx + 1, cy + 1, c);
                l[c] = (r0 + 2.f * r1 + r2) / 16.f;
            }
        }

    // LowUp: bilinear in fp32, cell centres at (i + 0.5) d
    PassImage out;
    out.Allocate(w, h, PixelType::F32, e.channels);
    float* o = out.As<float>();
    const float fd = static_cast<float>(div);
    auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
    for (uint32_t y = 0; y < h; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) / fd - 0.5f, fy0 = std::floor(sy), fy = sy - fy0;
        const int ay = std::clamp(static_cast<int>(fy0), 0, hy), by = std::clamp(static_cast<int>(fy0) + 1, 0, hy);
        for (uint32_t x = 0; x < w; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) / fd - 0.5f, fx0 = std::floor(sx), fx = sx - fx0;
            const int ax = std::clamp(static_cast<int>(fx0), 0, hx), bx = std::clamp(static_cast<int>(fx0) + 1, 0, hx);
            const float *a00 = at(low, g.width, ax, ay), *a10 = at(low, g.width, bx, ay), *a01 = at(low, g.width, ax, by), *a11 = at(low, g.width, bx, by);
            float* d = o + (static_cast<size_t>(y) * w + x) * nc;
            for (size_t c = 0; c < nc; ++c) d[c] = lerp(lerp(a00[c], a10[c], fx), lerp(a01[c], a11[c], fx), fy);
        }
    }
    return out;
}

double NrLowShare(const PassImage& diff, uint32_t div) {
    const PassImage lowBand = NrLowpassCpu(diff, div);
    const float* l = lowBand.As<float>();
    double low = 0.0, all = 0.0;
    const size_t nc = diff.channels.size();
    for (uint32_t y = 0; y < diff.height; ++y)
        for (uint32_t x = 0; x < diff.width; ++x)
            for (size_t c = 0; c < nc; ++c) {
                const double v = diff.Get(x, y, c), lv = l[(static_cast<size_t>(y) * diff.width + x) * nc + c];
                if (!std::isfinite(v)) continue;
                all += v * v;
                low += lv * lv;
            }
    return all > 0.0 ? low / all : 0.0;
}

}  // namespace dlssvid
