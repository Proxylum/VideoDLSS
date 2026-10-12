#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "pipeline/Frame.h"

namespace dlssvid {

// Scrubbing proxy of a video source (TASK-0040, ТЗ §9(6): random jumps at ≥ 30 fps on 1080p). A long-GOP video (one
// keyframe in 240-250 frames: ToS, NVENC results) costs 80-350 ms per jump - the decoder walks from the keyframe. The
// proxy is the same frames in a lossless intra codec with a cheap decode (ProxyCodecFor: MagicYUV for 8 bits - 3.5 ms
// a 1080p frame on one core; FFVHuff for 10-16 bits - 9 ms; FFV1 only as a fallback, its 37 ms of CPU a frame ate
// the scrubbing budget when the neighbours of a jump decoded at the same time): bit for bit what the original decodes to,
// so the viewport mixes them freely (fingerprints, golden frames and PNG exports do not change). It lives next to the
// passes in `<passes>/.proxy/` (a dot folder: not a pass, not a pass version) with a sidecar that ties it to the source
// file (size, modification time, hashes of its first and last MiB); a source changed since makes it invalid.
inline constexpr int kProxyFormat = 2;  // 2: the codec by bit depth (format 1 was FFV1 always: rebuilt)
inline constexpr int64_t kProxyMaxGop = 8;  // keyframe intervals up to this decode fast enough without a proxy

struct ProxyPlan {
    bool needed = false;          // the stream has long keyframe intervals
    int64_t frames = 0;
    int64_t maxKeyframeInterval = 0;
    uint64_t estimatedBytes = 0;  // of the proxy (FFV1 of camera footage: ~45 % of the raw frames)
    std::string reason;           // human-readable
};
ProxyPlan PlanProxy(const std::filesystem::path& video);
// The encoder of the proxy for a decoded layout: magicyuv (8-bit planar), ffvhuff (9-16 bits), ffv1 when those are
// missing from the FFmpeg build or do not take the layout exactly.
std::string ProxyCodecFor(const FrameDesc& decoded);

std::filesystem::path ProxyDir(const std::filesystem::path& passesRoot);
std::filesystem::path ProxyFile(const std::filesystem::path& passesRoot, const std::string& name);  // .proxy/<name>.mkv

// The proxy of `video` under `passesRoot` when it exists and still matches the file; nullopt otherwise.
std::optional<std::filesystem::path> ValidProxy(const std::filesystem::path& video, const std::filesystem::path& passesRoot, const std::string& name);

struct ProxyBuildOptions {
    const std::atomic<bool>* cancel = nullptr;                      // set: stop, the partial file is removed
    std::function<void(int64_t done, int64_t total)> progress;      // frames
};
// Decodes `video` from its first frame and writes the proxy. Refuses (throws) when the frame indices are not 0, 1, 2 ...
// (a variable-rate or oddly timed stream: the viewport then keeps decoding the original) and when the source changed
// while it was read. Returns the proxy path; false-free: a cancel throws too (the message says so).
std::filesystem::path BuildProxy(const std::filesystem::path& video, const std::filesystem::path& passesRoot, const std::string& name,
                                 const ProxyBuildOptions& options = {});
// Removes the proxy and its sidecar (Вид → «Удалить прокси»).
void RemoveProxy(const std::filesystem::path& passesRoot, const std::string& name);

}  // namespace dlssvid
