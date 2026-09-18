#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

#include "io/VideoDecoder.h"
#include "pipeline/Frame.h"

struct AVFormatContext;
struct AVCodecContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct AVCodecParameters;

namespace dlssvid {

// FFmpeg muxer + video encoder (NVENC by default) with audio stream copy.
// The container is chosen from the output extension (.mp4 / .mov / .mkv ...).
class VideoEncoder {
public:
    struct Options {
        std::string codec = "h264_nvenc";  // h264_nvenc | hevc_nvenc | av1_nvenc | ffv1 (lossless, tests)
        std::map<std::string, std::string> codecOptions;  // passed to the encoder private options
        Rational frameRate;                               // required
        Rational timeBase;                                // pts units for WriteFrame; defaults to 1/frameRate
    };

    VideoEncoder(const std::filesystem::path& path, const FrameDesc& frameDesc, const Options& options);
    ~VideoEncoder();

    VideoEncoder(const VideoEncoder&) = delete;
    VideoEncoder& operator=(const VideoEncoder&) = delete;

    // Optional, before Open(): copy an audio stream (codec parameters from the source).
    void AddAudioStreamCopy(const AVCodecParameters* params, Rational sourceTimeBase);

    void Open();
    void WriteFrame(const CpuFrame& frame);
    void WriteAudioPacket(AVPacket* packet);  // packet in the source audio time base
    void Close();

    bool IsOpen() const { return opened_; }
    int64_t FramesWritten() const { return framesWritten_; }
    int64_t AudioPacketsWritten() const { return audioPacketsWritten_; }
    const std::string& CodecName() const { return codecName_; }

    static bool EncoderAvailable(const std::string& name);

private:
    void Drain(bool flush);

    std::filesystem::path path_;
    FrameDesc frameDesc_;
    Options options_;
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVStream* videoStream_ = nullptr;
    AVStream* audioStream_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    Rational audioSourceTimeBase_;
    std::string codecName_;
    int64_t framesWritten_ = 0;
    int64_t audioPacketsWritten_ = 0;
    bool opened_ = false;
    bool closed_ = false;
};

}  // namespace dlssvid
