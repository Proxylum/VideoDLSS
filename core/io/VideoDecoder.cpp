#include "io/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

AVPixelFormat GetHwFormat(AVCodecContext*, const AVPixelFormat* formats) {
    for (const AVPixelFormat* p = formats; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
    }
    Log()->warn("NVDEC pixel format not offered by decoder, falling back to software");
    return formats[0];
}

}  // namespace

VideoDecoder::VideoDecoder(const std::filesystem::path& path, const Options& options) {
    const std::string url = path.string();
    CheckAv(avformat_open_input(&fmt_, url.c_str(), nullptr, nullptr), "avformat_open_input");
    CheckAv(avformat_find_stream_info(fmt_, nullptr), "avformat_find_stream_info");

    const AVCodec* decoder = nullptr;
    videoStream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (videoStream_ < 0) Throw("no video stream in " + url);
    audioStream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStream_ < 0) audioStream_ = -1;

    AVStream* vs = fmt_->streams[videoStream_];
    codec_ = avcodec_alloc_context3(decoder);
    if (!codec_) Throw("avcodec_alloc_context3 failed");
    CheckAv(avcodec_parameters_to_context(codec_, vs->codecpar), "avcodec_parameters_to_context");
    codec_->pkt_timebase = vs->time_base;

    if (options.hwaccel == HwAccel::Cuda) {
        const int err = av_hwdevice_ctx_create(&hwDevice_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
        if (err < 0) {
            Log()->warn("CUDA hwaccel unavailable ({}), decoding in software", AvErrorToString(err));
            hwDevice_ = nullptr;
        } else {
            codec_->hw_device_ctx = av_buffer_ref(hwDevice_);
            codec_->get_format = GetHwFormat;
        }
    }

    CheckAv(avcodec_open2(codec_, decoder, nullptr), "avcodec_open2");

    frame_ = av_frame_alloc();
    swFrame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !swFrame_ || !packet_) Throw("av_frame_alloc/av_packet_alloc failed");

    info_.width = static_cast<uint32_t>(codec_->width);
    info_.height = static_cast<uint32_t>(codec_->height);
    const AVRational fr = av_guess_frame_rate(fmt_, vs, nullptr);
    info_.frameRate = Rational{fr.num, fr.den};
    info_.timeBase = Rational{vs->time_base.num, vs->time_base.den};
    info_.frameCount = vs->nb_frames;
    info_.durationUs = fmt_->duration > 0 ? fmt_->duration : 0;
    info_.codecName = decoder->name;
    const char* pixName = av_get_pix_fmt_name(static_cast<AVPixelFormat>(vs->codecpar->format));
    info_.pixelFormat = pixName ? pixName : "unknown";
    info_.colorSpace = static_cast<int>(vs->codecpar->color_space);
    info_.colorRange = static_cast<int>(vs->codecpar->color_range);
    info_.hasAudio = audioStream_ >= 0;
    if (info_.hasAudio) {
        const AVCodecDescriptor* d = avcodec_descriptor_get(fmt_->streams[audioStream_]->codecpar->codec_id);
        info_.audioCodecName = d ? d->name : "unknown";
    }

    Log()->info("open {}: {}x{} {} {}/{} fps, {} frames{}{}", url, info_.width, info_.height, info_.codecName,
                fr.num, fr.den, info_.frameCount, info_.hasAudio ? ", audio " + info_.audioCodecName : "",
                hwDevice_ ? ", NVDEC" : "");
}

VideoDecoder::~VideoDecoder() {
    if (sws_) sws_freeContext(sws_);
    av_packet_free(&packet_);
    av_frame_free(&swFrame_);
    av_frame_free(&frame_);
    avcodec_free_context(&codec_);
    av_buffer_unref(&hwDevice_);
    avformat_close_input(&fmt_);
}

const AVCodecParameters* VideoDecoder::AudioCodecParameters() const {
    return audioStream_ >= 0 ? fmt_->streams[audioStream_]->codecpar : nullptr;
}

Rational VideoDecoder::AudioTimeBase() const {
    if (audioStream_ < 0) return {};
    const AVRational tb = fmt_->streams[audioStream_]->time_base;
    return Rational{tb.num, tb.den};
}

bool VideoDecoder::NextFrame(CpuFrame& out) {
    if (drained_) return false;
    for (;;) {
        if (ReceiveFrame(out)) return true;
        if (drained_) return false;
        if (eofSent_) {
            drained_ = true;
            return false;
        }
        // Feed the next video packet; route audio packets to the sink.
        const int err = av_read_frame(fmt_, packet_);
        if (err == AVERROR_EOF) {
            CheckAv(avcodec_send_packet(codec_, nullptr), "avcodec_send_packet(flush)");
            eofSent_ = true;
            continue;
        }
        CheckAv(err, "av_read_frame");
        if (packet_->stream_index == videoStream_) {
            CheckAv(avcodec_send_packet(codec_, packet_), "avcodec_send_packet");
        } else if (packet_->stream_index == audioStream_ && audioSink_) {
            audioSink_(packet_);
        }
        av_packet_unref(packet_);
    }
}

bool VideoDecoder::ReceiveFrame(CpuFrame& out) {
    const int err = avcodec_receive_frame(codec_, frame_);
    if (err == AVERROR(EAGAIN)) return false;
    if (err == AVERROR_EOF) {
        drained_ = true;
        return false;
    }
    CheckAv(err, "avcodec_receive_frame");

    AVFrame* src = frame_;
    if (frame_->format == AV_PIX_FMT_CUDA) {
        av_frame_unref(swFrame_);
        CheckAv(av_hwframe_transfer_data(swFrame_, frame_, 0), "av_hwframe_transfer_data");
        swFrame_->best_effort_timestamp = frame_->best_effort_timestamp;
        swFrame_->pts = frame_->pts;
        src = swFrame_;
    }
    ConvertFrame(src, out);
    out.index = nextIndex_++;
    out.pts = src->best_effort_timestamp != AV_NOPTS_VALUE ? src->best_effort_timestamp : src->pts;
    av_frame_unref(frame_);
    return true;
}

void VideoDecoder::ConvertFrame(AVFrame* f, CpuFrame& out) {
    const auto w = static_cast<uint32_t>(f->width);
    const auto h = static_cast<uint32_t>(f->height);
    const auto fmt = static_cast<AVPixelFormat>(f->format);

    if (fmt == AV_PIX_FMT_YUV420P || fmt == AV_PIX_FMT_YUVJ420P) {
        const uint8_t* const planes[3] = {f->data[0], f->data[1], f->data[2]};
        const int pitch[3] = {f->linesize[0], f->linesize[1], f->linesize[2]};
        CopyPlanar420(planes, pitch, w, h, out);
        return;
    }
    if (fmt == AV_PIX_FMT_NV12) {
        Nv12ToYuv420p(f->data[0], f->linesize[0], f->data[1], f->linesize[1], w, h, out);
        return;
    }

    // Anything else (10-bit, 4:2:2, RGB ...) is converted with swscale. Not bit-exact:
    // stage 2+ will keep high bit depth on the GPU; stage 0 documents the limitation.
    if (!warnedConversion_) {
        Log()->warn("source pixel format {} converted to yuv420p with swscale (lossy for >8-bit sources)",
                    av_get_pix_fmt_name(fmt) ? av_get_pix_fmt_name(fmt) : "?");
        warnedConversion_ = true;
    }
    sws_ = sws_getCachedContext(sws_, f->width, f->height, fmt, f->width, f->height, AV_PIX_FMT_YUV420P,
                                SWS_POINT, nullptr, nullptr, nullptr);
    if (!sws_) Throw("sws_getCachedContext failed");
    out.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    uint8_t* dst[3] = {out.Plane(0), out.Plane(1), out.Plane(2)};
    const int dstPitch[3] = {static_cast<int>(out.desc.PlaneWidth(0)), static_cast<int>(out.desc.PlaneWidth(1)),
                             static_cast<int>(out.desc.PlaneWidth(2))};
    sws_scale(sws_, f->data, f->linesize, 0, f->height, dst, dstPitch);
}

}  // namespace dlssvid
