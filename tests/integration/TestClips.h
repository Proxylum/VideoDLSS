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
// The same picture in another layout (8-bit 4:2:2 / 4:4:4 with the chroma replicated, 10 / 12 / 16-bit with the
// 8-bit samples scaled up by the depth difference plus a per-pixel ramp in the low bits so that every bit matters).
CpuFrame SyntheticFrame(int64_t index, const FrameDesc& desc);

struct ClipSpec {
    uint32_t width = 320;
    uint32_t height = 180;
    int frames = 24;
    int fpsNum = 24;
    int fpsDen = 1;
    bool audio = false;  // add a pcm_s16le sine tone track
    std::string codec = "ffv1";
    // Encoder input layout. "yuv420p" takes the frames as they are; "p010le" and "yuv444p16le" take 8-bit frames and
    // put their samples into the top bits of 16-bit words (sources NVDEC does not decode into NV12); "yuv420p10le",
    // "yuv422p10le", "yuv444p10le", "yuv444p12le" and the 8-bit "yuv422p" / "yuv444p" take frames of exactly that
    // layout (SyntheticFrame(index, desc) or WriteClipWith) and write them as they are — ffv1 and prores_ks keep them.
    std::string pixFmt = "yuv420p";
};
// The CpuFrame layout `WriteClip` feeds the encoder for a ClipSpec::pixFmt.
FrameDesc FrameDescFor(const ClipSpec& spec);

// Writes a lossless (ffv1 + optional pcm audio) clip with SyntheticFrame(i) as frame i (in the layout of spec.pixFmt).
std::filesystem::path WriteClip(const std::filesystem::path& path, const ClipSpec& spec);
// Same with caller-supplied frames (the layout FrameDescFor(spec) expects; 8-bit 4:2:0 for p010le / yuv444p16le).
std::filesystem::path WriteClipWith(const std::filesystem::path& path, const ClipSpec& spec, const std::function<CpuFrame(int64_t)>& frame);

// Decodes every frame of a file with plain FFmpeg (software) into memory.
std::vector<CpuFrame> DecodeAll(const std::filesystem::path& path);

struct StreamSummary {
    int videoStreams = 0;
    int audioStreams = 0;
    std::string audioCodec;
    int64_t audioPackets = 0;
    int64_t audioSamples = 0;  // sum of packet durations in samples, 0 if unknown
    std::string videoCodec;
    std::string pixelFormat;  // the video stream's pixel format name
    int profile = -1;         // the video stream's profile (codec-specific; HEVC: 1 Main, 2 Main 10, 4 Rext)
    int colorRange = 0, colorSpace = 2, colorPrimaries = 2, colorTrc = 2;  // AVCOL_* tags of the video stream
};
StreamSummary Summarize(const std::filesystem::path& path);

}  // namespace dlssvid::test
