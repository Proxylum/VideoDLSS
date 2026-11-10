#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>

#include "gpu/GpuFrame.h"
#include "pipeline/Frame.h"

struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVCodecParameters;
struct AVBufferRef;
struct SwsContext;

namespace dlssvid {

struct Rational {
    int num = 0;
    int den = 1;
    double ToDouble() const { return den ? static_cast<double>(num) / den : 0.0; }
};

enum class HwAccel { None, Cuda };

struct VideoStreamInfo {
    // The frame as shown (TASK-0072): the stored frame turned by `orientation`, what NextFrame yields.
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t codedWidth = 0;   // the stored frame
    uint32_t codedHeight = 0;
    DisplayOrientation orientation;  // how players show the stream: its display matrix, else the 'rotate' tag
    std::string orientationFrom;     // "display matrix" | "rotate tag"; "" when not turned
    Rational frameRate;    // av_guess_frame_rate
    Rational timeBase;     // stream time base (pts units)
    int64_t frameCount = 0;  // from container; 0 if unknown
    int64_t durationUs = 0;  // container duration in microseconds; 0 if unknown
    std::string codecName;
    std::string pixelFormat;  // source pixel format name (FFmpeg: "yuv420p", "yuv444p10le", ...)
    int bitDepth = 8;         // bits per sample of the source pixel format
    std::string chroma = "420";  // chroma subsampling of the source: "420" | "422" | "444" | "other"
    FrameDesc decoded;        // the CpuFrame layout NextFrame() yields for this stream (width/height set): the native
                              // planar YUV layout at the source's bit depth, or 8-bit 4:2:0 for formats that go through swscale;
                              // turned by `orientation` (OrientedDesc: 4:4:4 where the chroma grid does not map onto itself)
    int colorSpace = 2;       // AVColorSpace (2 = unspecified)
    int colorRange = 0;       // AVColorRange (0 = unspecified, 1 = limited/MPEG, 2 = full/JPEG)
    int colorPrimaries = 2;   // AVColorPrimaries (2 = unspecified)
    int colorTrc = 2;         // AVColorTransferCharacteristic (2 = unspecified)
    bool hasAudio = false;
    std::string audioCodecName;
};

// Keyframes of a video stream from its packets (demux only, no decoding; TASK-0040): the longest run of frames from
// a keyframe to the next one or to the end - what a seek into the stream may have to decode.
struct KeyframeScan {
    int64_t frames = 0;           // video packets
    int64_t keyframes = 0;
    int64_t maxInterval = 0;      // frames from a keyframe up to the next keyframe (or the end), the longest
    bool intraOnly() const { return frames > 0 && keyframes == frames; }
};
KeyframeScan ScanKeyframes(const std::filesystem::path& path);

// The index of the frame at `pts` (in `timeBase` units) at `rate`, counted from the stream's `startTime`: what a seek and the frame
// indices after it assume - a constant frame rate.
int64_t PtsToIndex(int64_t pts, int64_t startTime, Rational timeBase, Rational rate);

// Whether the indices a seek derives from pts are the frames' own up to frame `upTo` (PR-10: dlssvid nr --start on the video path).
// Demux only, no decoding: the pts of the video packets up to `upTo` and a margin for reordering, sorted into presentation order;
// frame k must carry the time of frame k. A variable frame rate (or a gap in the timestamps) does not, and SeekToFrame would land on
// another frame. `indexRate` as VideoDecoder::Options::indexRate ({0, 1} = the stream's guessed rate).
struct FrameIndexScan {
    int64_t frames = 0;     // video packets read
    int64_t firstBad = -1;  // the first frame whose pts names another index (or that has no pts nor dts); -1: none
    int64_t badIndex = -1;  // the index that frame's pts names (-1 when it has no pts nor dts)
    bool noRate = false;    // the stream has no frame rate: no index can be taken from pts (firstBad 0)
    bool Consistent() const { return firstBad < 0; }
    std::string Why() const;  // «frame 61 carries the time of frame 66», «frame 12 has no timestamp», «the stream has no frame rate»
};
FrameIndexScan ScanFrameIndex(const std::filesystem::path& path, int64_t upTo, Rational indexRate = {0, 1});
// SeekToFrame decodes forward instead of seeking when the target is at most this many frames ahead: the indices are counted then, not
// taken from pts.
inline constexpr int64_t kSequentialSeekFrames = 48;

// The CpuFrame layout a stream's pixel format decodes into (VideoStreamInfo::decoded for a format name): planar YUV
// 4:2:0 / 4:2:2 / 4:4:4 at 8, 10, 12 or 16 bits stays as it is, everything else (RGB, 4:1:1, gray ...) becomes 8-bit
// 4:2:0 through swscale. `width` / `height` are left at 0.
FrameDesc DecodedFrameDescFor(const std::string& pixelFormat);

// The source's own layout, as shown: DecodedFrameDescFor(pixelFormat) at the shown size - what the encoder's «auto» follows (a 4:2:2
// stream turned by 90 decodes as 4:4:4, its result stays 4:2:2; TASK-0072).
FrameDesc SourceLayout(const VideoStreamInfo& info);

// The orientation of a display matrix (AV_PKT_DATA_DISPLAYMATRIX: 3x3 int32, 16.16 / 2.30 fixed point) as FFmpeg's command line
// applies it (fftools: transpose clock / cclock_flip, hflip / vflip, cclock / clock_flip). Not a multiple of 90 deg or degenerate: the
// identity, `why` says so. The matrix's scale is not looked at (non-square pixels are another matter).
DisplayOrientation OrientationFromDisplayMatrix(const int32_t matrix[9], std::string* why = nullptr);
// The 'rotate' tag of older muxers: degrees clockwise ("90", "-90", "450"); not a whole multiple of 90: the identity, `why` says so.
DisplayOrientation OrientationFromRotateTag(std::string_view tag, std::string* why = nullptr);

// FFmpeg demuxer + video decoder producing tightly packed planar YUV CpuFrames in the source's own layout: 8-bit
// 4:2:0 sources as before (bit-exact with FFmpeg), 10 / 12 / 16-bit and 4:2:2 / 4:4:4 sources in the matching 16-bit
// or 8-bit planar format of Frame.h (MSB-aligned decoder outputs such as P010 are shifted to LSB alignment, so NVDEC
// and software decoding agree bit for bit). With HwAccel::Cuda the stream is decoded by NVDEC; an NV12 frame stays in
// CUDA memory (primary context) and is exposed as a GpuFrame, and the CPU copy is made only when asked for (bit-exact
// NV12 -> YUV420P). Audio packets are handed to the sink untouched for stream copy. A stream shown turned (a display matrix, the
// 'rotate' tag: a phone video) is turned on the CPU after the copy (OrientFrame) and has no GpuFrame (TASK-0072).
// The container's global metadata (tags) of a media file, keys lower-cased (Matroska stores COMMENT): `comment`, `dlssvid_settings`
// (PR-11). Throws when the file cannot be opened.
std::map<std::string, std::string> ReadContainerTags(const std::filesystem::path& path);

class VideoDecoder {
public:
    struct Options {
        HwAccel hwaccel = HwAccel::None;
        bool probe = false;  // opened only to read Info(): logged at debug level, not as a second "open" of the file
        // Slice threads only (an intra-only stream such as the scrubbing proxy): frame threads hold the first frame after
        // a seek back by a few decodes, which is the whole cost of a jump in an intra stream (TASK-0040).
        bool sliceThreadsOnly = false;
        // Frame indices from pts at this rate instead of the stream's (a proxy written at the rate of its original:
        // the container's guess of the rate may differ, the indices must not). {0, 1} = the stream's.
        Rational indexRate{0, 1};
    };
    static Options Probe() {
        Options o;
        o.probe = true;
        return o;
    }

    explicit VideoDecoder(const std::filesystem::path& path, const Options& options = {});
    ~VideoDecoder();

    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    const VideoStreamInfo& Info() const { return info_; }
    bool HasAudio() const { return audioStream_ >= 0; }
    const AVCodecParameters* AudioCodecParameters() const;
    Rational AudioTimeBase() const;
    bool UsingHwAccel() const { return hwDevice_ != nullptr; }

    using AudioPacketSink = std::function<void(AVPacket*)>;
    void SetAudioPacketSink(AudioPacketSink sink) { audioSink_ = std::move(sink); }

    // Next decoded frame in presentation order. Returns false at end of stream.
    // `gpu` (optional): filled with the on-device NV12 frame when decoding with NVDEC; it is
    // valid until the next NextFrame() call. `cpu = false` skips the device->host copy (the
    // CpuFrame then only carries index/pts/desc) — only allowed when gpu is requested. When NVDEC
    // yields a layout other than NV12 (10/12-bit, 4:4:4), or the stream is shown turned, `gpu` stays invalid and the CpuFrame is
    // always filled (in the source's own bit depth and chroma layout, turned), whatever `cpu` says.
    bool NextFrame(CpuFrame& out, GpuFrame* gpu = nullptr, bool cpu = true);

    // Random access: positions the decoder so that the next NextFrame() returns `index`
    // (assumes a constant frame rate for the pts <-> index mapping). Decoding forward from
    // the current position is used when the target is close ahead. Returns false when the
    // frame does not exist.
    bool SeekToFrame(int64_t index);
    // Whether SeekToFrame(index) decodes forward from here (indices counted, any frame rate) rather than seeking by pts.
    bool SeekIsSequential(int64_t index) const { return !drained_ && index >= nextIndex_ && index - nextIndex_ <= kSequentialSeekFrames; }
    int64_t NextIndex() const { return nextIndex_; }
    bool Drained() const { return drained_; }
    // Positions the demuxer at the keyframe at or before frame `index` and resets the decoder;
    // the caller walks forward with NextFrame() (frames carry their pts-derived index).
    // Returns false when the frame does not exist.
    bool SeekToKeyframeBefore(int64_t index);
    // false from a seek until the first frame after it: NextIndex() is a placeholder then (TASK-0040).
    bool PositionKnown() const { return !seeked_ || decodedSinceSeek_; }
    int64_t FramesDecoded() const { return framesDecoded_; }  // frames received from the decoder, all of them
    int64_t Seeks() const { return seeks_; }

private:
    Rational IndexRate() const { return indexRate_.num > 0 && indexRate_.den > 0 ? indexRate_ : info_.frameRate; }
    bool ReceiveFrame(CpuFrame& out, GpuFrame* gpu, bool cpu);
    int64_t IndexFromPts(int64_t pts) const;
    bool DiscardFrames(int64_t upTo);  // decode and drop frames until nextIndex_ == upTo
    void ConvertFrame(AVFrame* frame, CpuFrame& out);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hwDevice_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* swFrame_ = nullptr;
    AVFrame* hwKeep_ = nullptr;  // last CUDA frame kept alive for GpuFrame
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ = nullptr;
    int videoStream_ = -1;
    int audioStream_ = -1;
    int64_t nextIndex_ = 0;
    bool eofSent_ = false;
    bool drained_ = false;
    bool warnedConversion_ = false;
    bool warnedNonNv12_ = false;  // NVDEC layout other than NV12: the GPU fast path is off for this stream
    bool warnedTurnedGpu_ = false;  // a turned stream under NVDEC: no GpuFrame
    bool checkedFrameMatrix_ = false;  // the first frame's own display matrix (an orientation SEI) compared with the stream's
    CpuFrame unturned_;  // the frame as stored, before OrientFrame
    bool seeked_ = false;  // indices come from pts after a seek
    bool decodedSinceSeek_ = false;
    int64_t framesDecoded_ = 0, seeks_ = 0;
    Rational indexRate_{0, 1};
    VideoStreamInfo info_;
    AudioPacketSink audioSink_;
};

}  // namespace dlssvid
