#pragma once

#include <cstdint>

#include "io/VideoDecoder.h"  // Rational

namespace dlssvid {

// Paces viewport playback at the exact rational frame rate (audit play-fg-rate, TASK-0038). Frame k of a run that
// starts at `startTime` is due at origin + k·den/num seconds of a monotonic clock in nanoseconds; the caller passes the
// clock, so tests drive it without sleeping. At most one frame is shown per tick. A frame shown later than its due
// time but within one period keeps the grid (no drift), a later one restarts the count from now (no racing to catch
// up). A frame not ready in time is waited for - a slow decode plays slower than real time instead of skipping - and
// skipped after kStallNs, so playback never freezes. Before, a QTimer of floor(1000 / fps) ms advanced one frame per
// tick: 48 fps played at 50 Hz (+4 %), 24 fps at 24.39 Hz.
class PlaybackClock {
public:
    static constexpr int64_t kStallNs = 2'000'000'000;  // a frame not decoded within this is skipped
    static constexpr int64_t kPollNs = 2'000'000;       // how often to look again while waiting for a frame

    enum class Action { Wait, Show, Skip };

    void Start(int64_t nowNs, double startTime, Rational rate);
    // a jump while playing (scrubbing, a loop): continue from `startTime`, keeping the grid if not late by a period
    void Restart(int64_t nowNs, double startTime);

    double NextTime() const;  // the time of the next frame
    int64_t DueNs() const;    // when it is due
    int64_t PeriodNs() const;
    Rational Rate() const { return rate_; }

    Action Decide(int64_t nowNs, bool nextReady) const;
    void Advance(int64_t nowNs);          // the next frame was shown (or skipped)
    int64_t WakeNs(int64_t nowNs) const;  // when to tick again

private:
    int64_t DueOf(int64_t k) const;

    Rational rate_{24, 1};
    int64_t originNs_ = 0;
    int64_t k_ = 0;  // frames shown since the origin
    double startTime_ = 0.0;
};

}  // namespace dlssvid
