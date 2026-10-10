#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

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
    uint32_t width = 0;
    uint32_t height = 0;
    Rational frameRate;    // av_guess_frame_rate
    Rational timeBase;     // stream time base (pts units)
    int64_t frameCount = 0;  // from container; 0 if unknown
    int64_t durationUs = 0;  // container duration in microseconds; 0 if unknown
    std::string codecName;
    std::string pixelFormat;  // source pixel format name (FFmpeg: "yuv420p", "yuv444p10le", ...)
    int bitDepth = 8;         // bits per sample of the source pixel format
    std::string chroma = "420";  // chroma subsampling of the source: "420" | "422" | "444" | "other"
    FrameDesc decoded;        // the CpuFrame layout NextFrame() yields for this stream (width/height set): the native
                              // planar YUV layout at the source's bit depth, or 8-bit 4:2:0 for formats that go through swscale
    int colorSpace = 2;       // AVColorSpace (2 = unspecified)
    int colorRange = 0;       // AVColorRange (0 = unspecified, 1 = limited/MPEG, 2 = full/JPEG)
    int colorPrimaries = 2;   // AVColorPrimaries (2 = unspecified)
    int colorTrc = 2;         // AVColorTransferCharacteristic (2 = unspecified)
    bool hasAudio = false;
    std::string audioCodecName;
};

// The CpuFrame layout a stream's pixel format decodes into (VideoStreamInfo::decoded for a format name): planar YUV
// 4:2:0 / 4:2:2 / 4:4:4 at 8, 10, 12 or 16 bits stays as it is, everything else (RGB, 4:1:1, gray ...) becomes 8-bit
// 4:2:0 through swscale. `width` / `height` are left at 0.
FrameDesc DecodedFrameDescFor(const std::string& pixelFormat);

// FFmpeg demuxer + video decoder producing tightly packed planar YUV CpuFrames in the source's own layout: 8-bit
// 4:2:0 sources as before (bit-exact with FFmpeg), 10 / 12 / 16-bit and 4:2:2 / 4:4:4 sources in the matching 16-bit
// or 8-bit planar format of Frame.h (MSB-aligned decoder outputs such as P010 are shifted to LSB alignment, so NVDEC
// and software decoding agree bit for bit). With HwAccel::Cuda the stream is decoded by NVDEC; an NV12 frame stays in
// CUDA memory (primary context) and is exposed as a GpuFrame, and the CPU copy is made only when asked for (bit-exact
// NV12 -> YUV420P). Audio packets are handed to the sink untouched for stream copy.
class VideoDecoder {
public:
    struct Options {
        HwAccel hwaccel = HwAccel::None;
        bool probe = false;  // opened only to read Info(): logged at debug level, not as a second "open" of the file
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
    // yields a layout other than NV12 (10/12-bit, 4:4:4), `gpu` stays invalid and the CpuFrame is
    // always filled (in the source's own bit depth and chroma layout), whatever `cpu` says.
    bool NextFrame(CpuFrame& out, GpuFrame* gpu = nullptr, bool cpu = true);

    // Random access: positions the decoder so that the next NextFrame() returns `index`
    // (assumes a constant frame rate for the pts <-> index mapping). Decoding forward from
    // the current position is used when the target is close ahead. Returns false when the
    // frame does not exist.
    bool SeekToFrame(int64_t index);
    int64_t NextIndex() const { return nextIndex_; }
    bool Drained() const { return drained_; }
    // Positions the demuxer at the keyframe at or before frame `index` and resets the decoder;
    // the caller walks forward with NextFrame() (frames carry their pts-derived index).
    // Returns false when the frame does not exist.
    bool SeekToKeyframeBefore(int64_t index);

private:
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
    bool seeked_ = false;  // indices come from pts after a seek
    VideoStreamInfo info_;
    AudioPacketSink audioSink_;
};

}  // namespace dlssvid
