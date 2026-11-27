#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

// The bands of rows the colour converters split a frame into (ColorConvert.cpp; here for its test).
namespace dlssvid::detail {

// rows(y0, y1) over bands of [0, h) on several threads (0 = auto: half the cores, at most 8, from 256 rows - TASK-0040:
// 8-12 ms single-threaded at 1080p, a third of the scrubbing budget). Every pixel is computed alone, so the result is the
// same bit for bit whatever the count. An exception in a band (the frame is checked before, so only a failed allocation of
// the HDR rows' buffers) reaches the caller after every band is joined (TASK-0080).
template <typename Rows>
void ForEachBand(uint32_t h, int threads, const Rows& rows) {
    int n = threads > 0 ? threads : (h >= 256 ? std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 1, 8) : 1);
    n = std::clamp(n, 1, static_cast<int>(std::max(1u, h)));
    if (n == 1) {
        rows(0u, h);
        return;
    }
    std::exception_ptr error;
    std::mutex errorMutex;
    const auto guarded = [&](uint32_t y0, uint32_t y1) {
        try {
            rows(y0, y1);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(errorMutex);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n - 1));
    const auto band = [&](int i) { return static_cast<uint32_t>(static_cast<uint64_t>(h) * static_cast<uint64_t>(i) / static_cast<uint64_t>(n)); };
    for (int i = 1; i < n; ++i) pool.emplace_back(guarded, band(i), band(i + 1));
    guarded(0u, band(1));
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace dlssvid::detail
