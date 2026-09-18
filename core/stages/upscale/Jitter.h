#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dlssvid {

// Halton low-discrepancy sequence, index >= 1.
inline float Halton(uint32_t index, uint32_t base) {
    float f = 1.f, r = 0.f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

struct JitterOffset {
    float x = 0.f, y = 0.f;
};

// Number of jitter phases recommended by the DLSS guide: 8 * (target / render)^2.
inline int JitterPhaseCount(uint32_t inW, uint32_t inH, uint32_t outW, uint32_t outH) {
    if (!inW || !inH) return 8;
    const double ratio = (static_cast<double>(outW) / inW) * (static_cast<double>(outH) / inH);
    return std::max(8, static_cast<int>(std::ceil(8.0 * ratio)));
}

// Halton(2,3) jitter for a frame, in [-0.5, 0.5) input pixels (the sequence repeats every `phases`).
inline JitterOffset HaltonJitter(int64_t frame, int phases) {
    phases = std::max(1, phases);
    const uint32_t i = static_cast<uint32_t>(frame % phases) + 1;
    return {Halton(i, 2) - 0.5f, Halton(i, 3) - 0.5f};
}

}  // namespace dlssvid
