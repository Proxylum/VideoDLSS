#include "io/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <cmath>

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
        // Primary CUDA context: device pointers are then usable from cudart, TensorRT and the
        // Optical Flow API in this process without context juggling.
        const int err = av_hwdevice_ctx_create(&hwDevice_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
        if (err < 0) {
            Log()->warn("CUDA hwaccel unavailable ({}), decoding in software", AvErrorToString(err));
            hwDevice_ = nullptr;
        } else {
            codec_->hw_device_ctx = av_buffer_ref(hwDevice_);
            codec_->get_format = GetHwFormat;
        }
    }

    if (!hwDevice_) {
        // software decoding: frame + slice threads, count chosen by libavcodec (viewport scrubbing at 1080p)
        codec_->thread_count = 0;
        codec_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    }
    CheckAv(avcodec_open2(codec_, decoder, nullptr), "avcodec_open2");

    frame_ = av_frame_alloc();
    swFrame_ = av_frame_alloc();
    hwKeep_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !swFrame_ || !hwKeep_ || !packet_) Throw("av_frame_alloc/av_packet_alloc failed");

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

    Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info, "open {}: {}x{} {} {}/{} fps, {} frames{}{}", url, info_.width, info_.height,
               info_.codecName, fr.num, fr.den, info_.frameCount, info_.hasAudio ? ", audio " + info_.audioCodecName : "", hwDevice_ ? ", NVDEC" : "");
}

VideoDecoder::~VideoDecoder() {
    if (sws_) sws_freeContext(sws_);
    av_packet_free(&packet_);
    av_frame_free(&swFrame_);
    av_frame_free(&hwKeep_);
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

bool VideoDecoder::NextFrame(CpuFrame& out, GpuFrame* gpu, bool cpu) {
    if (gpu) *gpu = GpuFrame{};
    if (drained_) return false;
    for (;;) {
        if (ReceiveFrame(out, gpu, cpu)) return true;
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

bool VideoDecoder::ReceiveFrame(CpuFrame& out, GpuFrame* gpu, bool cpu) {
    const int err = avcodec_receive_frame(codec_, frame_);
    if (err == AVERROR(EAGAIN)) return false;
    if (err == AVERROR_EOF) {
        drained_ = true;
        return false;
    }
    CheckAv(err, "avcodec_receive_frame");

    const int64_t pts = frame_->best_effort_timestamp != AV_NOPTS_VALUE ? frame_->best_effort_timestamp : frame_->pts;
    int64_t index = nextIndex_;
    if (seeked_ && pts != AV_NOPTS_VALUE) index = IndexFromPts(pts);
    nextIndex_ = index + 1;
    AVFrame* src = frame_;
    if (frame_->format == AV_PIX_FMT_CUDA) {
        if (gpu) {
            // Keep a reference so the device memory stays valid until the next frame.
            av_frame_unref(hwKeep_);
            CheckAv(av_frame_ref(hwKeep_, frame_), "av_frame_ref");
            const auto* hwFrames = reinterpret_cast<const AVHWFramesContext*>(hwKeep_->hw_frames_ctx->data);
            if (hwFrames->sw_format != AV_PIX_FMT_NV12) Throw("NVDEC frame is not NV12 (" + std::string(av_get_pix_fmt_name(hwFrames->sw_format)) + ")");
            gpu->width = static_cast<uint32_t>(hwKeep_->width);
            gpu->height = static_cast<uint32_t>(hwKeep_->height);
            gpu->y = reinterpret_cast<uintptr_t>(hwKeep_->data[0]);
            gpu->uv = reinterpret_cast<uintptr_t>(hwKeep_->data[1]);
            gpu->pitch = static_cast<size_t>(hwKeep_->linesize[0]);
            gpu->index = index;
        }
        if (!cpu) {
            out.desc = FrameDesc{static_cast<uint32_t>(frame_->width), static_cast<uint32_t>(frame_->height), PixelFormat::Yuv420p};
            out.data.clear();
            out.index = index;
            out.pts = pts;
            av_frame_unref(frame_);
            return true;
        }
        av_frame_unref(swFrame_);
        CheckAv(av_hwframe_transfer_data(swFrame_, frame_, 0), "av_hwframe_transfer_data");
        src = swFrame_;
    }
    if (!cpu) {  // software frame skipped by the caller (seeking): no conversion
        out.desc = FrameDesc{static_cast<uint32_t>(src->width), static_cast<uint32_t>(src->height), PixelFormat::Yuv420p};
        out.data.clear();
        out.index = index;
        out.pts = pts;
        av_frame_unref(frame_);
        return true;
    }
    ConvertFrame(src, out);
    out.index = index;
    out.pts = pts;
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

int64_t VideoDecoder::IndexFromPts(int64_t pts) const {
    if (info_.frameRate.num <= 0 || info_.timeBase.num <= 0) return nextIndex_;
    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    const double seconds = static_cast<double>(pts - start) * info_.timeBase.num / info_.timeBase.den;
    return static_cast<int64_t>(std::llround(seconds * info_.frameRate.num / info_.frameRate.den));
}

bool VideoDecoder::DiscardFrames(int64_t upTo) {
    CpuFrame tmp;
    while (nextIndex_ < upTo) {
        if (!NextFrame(tmp, nullptr, false)) return false;
    }
    return nextIndex_ == upTo;
}

bool VideoDecoder::SeekToKeyframeBefore(int64_t index) {
    if (index < 0) return false;
    if (info_.frameCount > 0 && index >= info_.frameCount) return false;
    if (info_.frameRate.num <= 0 || info_.timeBase.num <= 0) return false;
    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    const AVRational fps{info_.frameRate.num, info_.frameRate.den};
    const AVRational tb{info_.timeBase.num, info_.timeBase.den};
    const int64_t target = start + av_rescale_q(index, av_inv_q(fps), tb);
    if (av_seek_frame(fmt_, videoStream_, target, AVSEEK_FLAG_BACKWARD) < 0) return false;
    avcodec_flush_buffers(codec_);
    eofSent_ = false;
    drained_ = false;
    seeked_ = true;
    nextIndex_ = 0;  // replaced by the pts-derived index of the first decoded frame
    return true;
}

bool VideoDecoder::SeekToFrame(int64_t index) {
    if (index < 0) return false;
    if (info_.frameCount > 0 && index >= info_.frameCount) return false;
    if (!drained_ && index == nextIndex_) return true;
    if (!drained_ && index > nextIndex_ && index - nextIndex_ <= 48) return DiscardFrames(index);

    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    const AVRational fps{info_.frameRate.num, info_.frameRate.den};
    const AVRational tb{info_.timeBase.num, info_.timeBase.den};
    auto seekTo = [&](int64_t frameIndex) {
        const int64_t target = start + av_rescale_q(std::max<int64_t>(frameIndex, 0), av_inv_q(fps), tb);
        if (av_seek_frame(fmt_, videoStream_, target, AVSEEK_FLAG_BACKWARD) < 0) return false;
        avcodec_flush_buffers(codec_);
        eofSent_ = false;
        drained_ = false;
        seeked_ = true;
        nextIndex_ = 0;  // replaced by the pts-derived index of the first decoded frame
        return true;
    };
    // Land on the keyframe at or before the frame preceding `index`, then walk forward.
    if (!seekTo(index > 0 ? index - 1 : 0)) return false;
    if (index == 0) return true;
    CpuFrame tmp;
    for (;;) {
        if (!NextFrame(tmp, nullptr, false)) return false;
        if (tmp.index + 1 == index) return true;  // next NextFrame() yields `index`
        if (tmp.index >= index) {                  // the demuxer landed late: walk from the start
            if (av_seek_frame(fmt_, videoStream_, start, AVSEEK_FLAG_BACKWARD) < 0) return false;
            avcodec_flush_buffers(codec_);
            eofSent_ = drained_ = false;
            nextIndex_ = 0;
            return DiscardFrames(index);
        }
    }
}

}  // namespace dlssvid
