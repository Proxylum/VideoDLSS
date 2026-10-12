#include "PlaybackClock.h"

#include <cmath>

namespace dlssvid {

namespace {
// a start on the grid of the rate: from a time between frames (a seek at another timeline rate) the rounding of the
// next frame times would otherwise fall on a frame boundary and skip a frame
double Snap(double t, Rational r) { return std::round(t * r.num / r.den) * r.den / r.num; }
}  // namespace

void PlaybackClock::Start(int64_t nowNs, double startTime, Rational rate) {
    rate_ = rate.num > 0 && rate.den > 0 ? rate : Rational{1, 1};
    originNs_ = nowNs;
    k_ = 0;
    startTime_ = Snap(startTime, rate_);
}

void PlaybackClock::Restart(int64_t nowNs, double startTime) {
    const int64_t late = nowNs - DueNs();
    originNs_ = late >= 0 && late < PeriodNs() ? DueNs() : nowNs;  // on time (a loop): keep the grid; else count from now
    k_ = 0;
    startTime_ = Snap(startTime, rate_);
}

int64_t PlaybackClock::DueOf(int64_t k) const {
    return originNs_ + std::llround(static_cast<double>(k) * 1e9 * rate_.den / rate_.num);
}

double PlaybackClock::NextTime() const { return startTime_ + static_cast<double>(k_ + 1) * rate_.den / rate_.num; }

int64_t PlaybackClock::DueNs() const { return DueOf(k_ + 1); }

int64_t PlaybackClock::PeriodNs() const { return std::llround(1e9 * rate_.den / rate_.num); }

PlaybackClock::Action PlaybackClock::Decide(int64_t nowNs, bool nextReady) const {
    const int64_t due = DueNs();
    if (nowNs < due) return Action::Wait;
    if (nextReady) return Action::Show;
    return nowNs - due >= kStallNs ? Action::Skip : Action::Wait;
}

void PlaybackClock::Advance(int64_t nowNs) {
    if (nowNs - DueNs() >= PeriodNs()) {
        // a period or more late (a slow decode, a busy GUI thread): restart the count instead of racing through frames
        startTime_ = NextTime();
        originNs_ = nowNs;
        k_ = 0;
    } else {
        ++k_;
    }
}

int64_t PlaybackClock::WakeNs(int64_t nowNs) const {
    const int64_t due = DueNs();
    return nowNs < due ? due : nowNs + kPollNs;
}

}  // namespace dlssvid
