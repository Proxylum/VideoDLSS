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

// Colour tags written into the output stream (FFmpeg's AVCOL_* values; 0 / 2 = unspecified): the matrix and range the
// RGB -> YUV conversion used, the primaries and transfer of the source passed through.
struct ColorTags {
    int range = 0;      // AVColorRange: 1 limited, 2 full
    int matrix = 2;     // AVColorSpace: 1 BT.709, 5 / 6 BT.601
    int primaries = 2;  // AVColorPrimaries
    int transfer = 2;   // AVColorTransferCharacteristic
};

// FFmpeg muxer + video encoder (NVENC by default) with audio stream copy. The container is chosen from the output
// extension (.mp4 / .mov / .mkv ...). Frames come in any CpuFrame layout (8-bit or 10 / 12 / 16-bit, 4:2:0 / 4:2:2 /
// 4:4:4); the encoder's pixel format is the first one it supports that holds that layout exactly — yuv420p10le for
// FFV1 and ProRes, p010le / yuv444p16le (MSB-aligned words) for NVENC — and WriteFrame repacks into it losslessly.
class VideoEncoder {
public:
    struct Options {
        std::string codec = "h264_nvenc";  // h264_nvenc | hevc_nvenc | av1_nvenc | prores_ks | ffv1 (lossless, tests)
        std::map<std::string, std::string> codecOptions;  // passed to the encoder private options
        Rational frameRate;                               // required
        Rational timeBase;                                // pts units for WriteFrame; defaults to 1/frameRate
        ColorTags color;                                  // stream colour tags (none by default)
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
    const FrameDesc& Frame() const { return frameDesc_; }
    const std::string& PixelFormatName() const { return pixelFormatName_; }  // the encoder's input format ("p010le", "yuv422p10le" ...)

    static bool EncoderAvailable(const std::string& name);
    // The encoder's input pixel format for a frame layout (empty when the encoder takes none that holds it).
    static std::string PixelFormatFor(const std::string& codec, const FrameDesc& frameDesc);

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
    std::string pixelFormatName_;
    int pixelFormat_ = -1;   // AVPixelFormat of the encoder input
    bool semiPlanar_ = false;  // encoder input has interleaved chroma (nv12, p010 ...)
    int shiftLeft_ = 0;        // 16-bit encoder inputs: bits the LSB-aligned samples move up (p010: 6)
    int64_t framesWritten_ = 0;
    int64_t audioPacketsWritten_ = 0;
    bool opened_ = false;
    bool closed_ = false;
};

}  // namespace dlssvid
