#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "pipeline/Frame.h"

namespace dlssvid::test {

std::filesystem::path TempDir();  // per-run scratch directory under the build tree

// Deterministic synthetic frame: gradients + moving block, distinct per index.
CpuFrame SyntheticFrame(int64_t index, uint32_t width, uint32_t height);

struct ClipSpec {
    uint32_t width = 320;
    uint32_t height = 180;
    int frames = 24;
    int fpsNum = 24;
    int fpsDen = 1;
    bool audio = false;  // add a pcm_s16le sine tone track
    std::string codec = "ffv1";
};

// Writes a lossless (ffv1 + optional pcm audio) clip with SyntheticFrame(i) as frame i.
std::filesystem::path WriteClip(const std::filesystem::path& path, const ClipSpec& spec);

// Decodes every frame of a file with plain FFmpeg (software) into memory.
std::vector<CpuFrame> DecodeAll(const std::filesystem::path& path);

struct StreamSummary {
    int videoStreams = 0;
    int audioStreams = 0;
    std::string audioCodec;
    int64_t audioPackets = 0;
    int64_t audioSamples = 0;  // sum of packet durations in samples, 0 if unknown
};
StreamSummary Summarize(const std::filesystem::path& path);

}  // namespace dlssvid::test
