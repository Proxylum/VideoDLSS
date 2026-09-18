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
    std::string pixelFormat;  // source pixel format name
    int colorSpace = 2;       // AVColorSpace (2 = unspecified)
    int colorRange = 0;       // AVColorRange (0 = unspecified, 1 = limited/MPEG, 2 = full/JPEG)
    bool hasAudio = false;
    std::string audioCodecName;
};

// FFmpeg demuxer + video decoder producing tightly packed YUV420P CpuFrames.
// With HwAccel::Cuda the stream is decoded by NVDEC; the NV12 frame stays in CUDA memory
// (primary context) and is exposed as a GpuFrame, and the CPU copy is made only when asked
// for (bit-exact NV12 -> YUV420P). Audio packets are handed to the sink untouched for stream copy.
class VideoDecoder {
public:
    struct Options {
        HwAccel hwaccel = HwAccel::None;
    };

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
    // CpuFrame then only carries index/pts/desc) — only allowed when gpu is requested.
    bool NextFrame(CpuFrame& out, GpuFrame* gpu = nullptr, bool cpu = true);

private:
    bool ReceiveFrame(CpuFrame& out, GpuFrame* gpu, bool cpu);
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
    VideoStreamInfo info_;
    AudioPacketSink audioSink_;
};

}  // namespace dlssvid
