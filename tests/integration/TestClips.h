#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
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
    int gop = 0;  // keyframe interval of the encoder (0 = its default: 12 for ffv1); 1 = intra only (TASK-0040)
    int bframes = -1;  // TASK-0076: the encoder's B-frames between references (max_b_frames); -1 = its default
    // The pts of frame i in 1/fps units (empty: i). A gap makes a variable-frame-rate clip (PR-10: the preview refuses to seek into it).
    std::function<int64_t(int64_t)> pts;
    // TASK-0072: how players show the clip - a display matrix on the video stream (av_display_rotation_set clockwise, then
    // av_display_matrix_flip left-right: FFmpeg's command line applies exactly this orientation), the frames written as they are (stored)
    DisplayOrientation orientation;
    std::string rotateTag;  // non-empty: the stream's old 'rotate' tag (degrees clockwise), written as well as any matrix
    // TASK-0073: the pixels' proportions, where they are said: "all" - the codec (FFV1 then at level 3: its slices carry it), the stream
    // (MKV DisplayWidth, MP4 / MOV pasp, AVI vprp) and the frames; "container" - the stream only; "codec" - the codec and the frames only
    // (with a container without the field - `format` "m4v", a raw MPEG-4 stream - the bitstream alone says it)
    PixelAspect sar;
    std::string sarIn = "all";
    // The muxer by name (empty: by the file's extension - FFmpeg takes its MP4 'ipod' muxer for .m4v, not the raw one)
    std::string format;
    // Non-square: the container says these pixels while the codec keeps `sar` (an MKV DisplayWidth correcting a stream's VUI) - the
    // codec parameters' SAR is cleared before the header (the muxer refuses two non-zero ones that differ)
    PixelAspect containerSar;
};
// The CpuFrame layout `WriteClip` feeds the encoder for a ClipSpec::pixFmt.
FrameDesc FrameDescFor(const ClipSpec& spec);

// Writes a lossless (ffv1 + optional pcm audio) clip with SyntheticFrame(i) as frame i (in the layout of spec.pixFmt).
std::filesystem::path WriteClip(const std::filesystem::path& path, const ClipSpec& spec);
// Same with caller-supplied frames (the layout FrameDescFor(spec) expects; 8-bit 4:2:0 for p010le / yuv444p16le).
std::filesystem::path WriteClipWith(const std::filesystem::path& path, const ClipSpec& spec, const std::function<CpuFrame(int64_t)>& frame);

// Decodes every frame of a file in software into memory - with the product's VideoDecoder: a clip shown turned comes turned (TASK-0072).
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
    // TASK-0073: the video stream's size and pixels' proportions - the container's (stream), the codec parameters', and the one players take
    // (av_guess_sample_aspect_ratio: the container's, else the codec's)
    uint32_t width = 0, height = 0;
    PixelAspect streamSar, codecSar, sar;
    std::string comment;  // the container's `comment` tag (the settings note, PR-11)
};
StreamSummary Summarize(const std::filesystem::path& path);

// FFmpeg's log while it lives (still written to stderr as before): what an encoder said when it refused. One at a time.
class FfmpegLogCapture {
public:
    FfmpegLogCapture();
    ~FfmpegLogCapture();
    FfmpegLogCapture(const FfmpegLogCapture&) = delete;
    FfmpegLogCapture& operator=(const FfmpegLogCapture&) = delete;
    std::string Text() const;

private:
    std::string text_;
};
// NVENC saying this GPU lacks the encoder, its depth or 4:4:4: «Codec not supported», «10 bit encode not supported», «YUV444P not
// supported» (libavcodec/nvenc.c).
// Nothing else: «No capable devices found» (also a session not opened now), «unsupported param», an invalid spec, a test's own
// error are failures.
bool SaysUnsupported(const std::string& text);
// WriteClip; nullopt (`why` says what the encoder said) when the encoder refuses the spec as not supported here (a GPU tests skip). Any
// other failure throws: a test fails on it rather than skipping (a broken spec or a regression is not «unsupported»).
std::optional<std::filesystem::path> WriteClipIfSupported(const std::filesystem::path& path, const ClipSpec& spec, std::string* why = nullptr);

// TASK-0076: AV1 streams NVENC does not write, made of ones it does (an IVF file holds a temporal unit per frame):
// `a` then `b` one after the other in one IVF file (b's timestamps going on from a's); `partB` may write b's units anew
std::filesystem::path ConcatIvf(const std::filesystem::path& a, const std::filesystem::path& b, const std::filesystem::path& out,
                                const std::function<std::string(const std::string&)>& partB = {});
// an IVF file `copies` times one after the other, the timestamps going on; `each(index, unit)` writes each unit anew
std::filesystem::path RepeatIvf(const std::filesystem::path& src, int copies, const std::filesystem::path& out,
                                const std::function<std::string(int, const std::string&)>& each);
// a temporal unit whose sequence headers (8-bit 4:2:0, profile 0) say 12-bit 4:2:0 (profile 2): av1dec asks for yuv420p12 - which
// FFmpeg offers to no NVDEC - before it decodes the frames after it
std::string TwelveBitTemporalUnit(const std::string& unit);
// a temporal unit without its sequence header OBUs
std::string WithoutSequenceHeaders(const std::string& unit);
// a temporal unit whose sequence headers (av1_nvenc's: no timing info, no initial display delay) say an initial display delay - other
// bytes, the same decoder (size, depth, chroma)
std::string InitialDisplayDelayTemporalUnit(const std::string& unit);
// a temporal unit whose sequence headers say a frame `rows` rows shorter (max_frame_height_minus_1 less by it, in the same bits): an odd
// height made of av1_nvenc's even one - the frames' headers (no frame size override in them) stay valid while the 8-row block rows
// (MiRows) stay the same
std::string ShorterFrameTemporalUnit(const std::string& unit, int rows);
// the same for the width (max_frame_width_minus_1): an odd width made of av1_nvenc's even one - the 8-pixel block columns (MiCols) the same
std::string NarrowerFrameTemporalUnit(const std::string& unit, int columns);
// the frame size an IVF file's header says (what the demuxer reports before any frame)
void SetIvfFrameSize(const std::filesystem::path& ivf, uint16_t width, uint16_t height);

}  // namespace dlssvid::test
