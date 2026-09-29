#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
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
    // Encoder input layout the YUV420P frames are converted to: "yuv420p" (as is), "p010le" (10-bit 4:2:0) or
    // "yuv444p16le" (16-bit 4:4:4, chroma replicated) — for sources that NVDEC does not decode into NV12.
    std::string pixFmt = "yuv420p";
};

// Writes a lossless (ffv1 + optional pcm audio) clip with SyntheticFrame(i) as frame i.
std::filesystem::path WriteClip(const std::filesystem::path& path, const ClipSpec& spec);
// Same with caller-supplied frames (YUV420P, spec.width x spec.height).
std::filesystem::path WriteClipWith(const std::filesystem::path& path, const ClipSpec& spec, const std::function<CpuFrame(int64_t)>& frame);

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
