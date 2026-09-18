#include "io/VideoEncoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

#include <cstring>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

bool VideoEncoder::EncoderAvailable(const std::string& name) { return avcodec_find_encoder_by_name(name.c_str()) != nullptr; }

VideoEncoder::VideoEncoder(const std::filesystem::path& path, const FrameDesc& frameDesc, const Options& options)
    : path_(path), frameDesc_(frameDesc), options_(options) {
    if (frameDesc_.format != PixelFormat::Yuv420p) Throw("VideoEncoder: only yuv420p frames are supported in stage 0");
    if (options_.frameRate.num <= 0 || options_.frameRate.den <= 0) Throw("VideoEncoder: frameRate is required");
    if (options_.timeBase.num <= 0 || options_.timeBase.den <= 0) {
        options_.timeBase = Rational{options_.frameRate.den, options_.frameRate.num};
    }

    const std::string url = path_.string();
    CheckAv(avformat_alloc_output_context2(&fmt_, nullptr, nullptr, url.c_str()), "avformat_alloc_output_context2");

    const AVCodec* enc = avcodec_find_encoder_by_name(options_.codec.c_str());
    if (!enc) Throw("encoder not found: " + options_.codec + " (try --codec ffv1 or hevc_nvenc)");
    codecName_ = enc->name;

    codec_ = avcodec_alloc_context3(enc);
    if (!codec_) Throw("avcodec_alloc_context3 failed");
    codec_->width = static_cast<int>(frameDesc_.width);
    codec_->height = static_cast<int>(frameDesc_.height);
    codec_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_->time_base = AVRational{options_.timeBase.num, options_.timeBase.den};
    codec_->framerate = AVRational{options_.frameRate.num, options_.frameRate.den};
    if (fmt_->oformat->flags & AVFMT_GLOBALHEADER) codec_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    videoStream_ = avformat_new_stream(fmt_, nullptr);
    if (!videoStream_) Throw("avformat_new_stream(video) failed");
    videoStream_->time_base = codec_->time_base;
    videoStream_->avg_frame_rate = codec_->framerate;

    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !packet_) Throw("av_frame_alloc/av_packet_alloc failed");
}

VideoEncoder::~VideoEncoder() {
    try {
        if (opened_ && !closed_) Close();
    } catch (const std::exception& e) {
        Log()->error("VideoEncoder::Close in destructor: {}", e.what());
    }
    av_packet_free(&packet_);
    av_frame_free(&frame_);
    avcodec_free_context(&codec_);
    if (fmt_) {
        if (fmt_->pb && !(fmt_->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_->pb);
        avformat_free_context(fmt_);
    }
}

void VideoEncoder::AddAudioStreamCopy(const AVCodecParameters* params, Rational sourceTimeBase) {
    if (opened_) Throw("AddAudioStreamCopy after Open");
    if (!params) return;
    audioStream_ = avformat_new_stream(fmt_, nullptr);
    if (!audioStream_) Throw("avformat_new_stream(audio) failed");
    CheckAv(avcodec_parameters_copy(audioStream_->codecpar, params), "avcodec_parameters_copy");
    audioStream_->codecpar->codec_tag = 0;
    audioStream_->time_base = AVRational{sourceTimeBase.num, sourceTimeBase.den};
    audioSourceTimeBase_ = sourceTimeBase;
}

void VideoEncoder::Open() {
    if (opened_) return;
    AVDictionary* opts = nullptr;
    for (const auto& [k, v] : options_.codecOptions) av_dict_set(&opts, k.c_str(), v.c_str(), 0);
    const int err = avcodec_open2(codec_, nullptr, &opts);
    if (opts) {
        const AVDictionaryEntry* e = nullptr;
        while ((e = av_dict_iterate(opts, e))) Log()->warn("encoder option not consumed: {}={}", e->key, e->value);
        av_dict_free(&opts);
    }
    CheckAv(err, ("avcodec_open2(" + codecName_ + ")").c_str());
    CheckAv(avcodec_parameters_from_context(videoStream_->codecpar, codec_), "avcodec_parameters_from_context");

    const std::string url = path_.string();
    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
        CheckAv(avio_open(&fmt_->pb, url.c_str(), AVIO_FLAG_WRITE), "avio_open");
    }
    CheckAv(avformat_write_header(fmt_, nullptr), "avformat_write_header");
    opened_ = true;
    Log()->info("encode {}: {} {}x{} {}/{} fps{}", url, codecName_, frameDesc_.width, frameDesc_.height,
                options_.frameRate.num, options_.frameRate.den, audioStream_ ? ", audio copy" : "");
}

void VideoEncoder::WriteFrame(const CpuFrame& in) {
    if (!opened_) Throw("VideoEncoder::WriteFrame before Open");
    if (in.desc != frameDesc_) Throw("VideoEncoder::WriteFrame: frame format mismatch");

    av_frame_unref(frame_);
    frame_->format = AV_PIX_FMT_YUV420P;
    frame_->width = codec_->width;
    frame_->height = codec_->height;
    CheckAv(av_frame_get_buffer(frame_, 0), "av_frame_get_buffer");
    CheckAv(av_frame_make_writable(frame_), "av_frame_make_writable");
    for (size_t p = 0; p < 3; ++p) {
        const size_t rowBytes = in.desc.PlaneWidth(p);
        const size_t rows = in.desc.PlaneHeight(p);
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(frame_->data[p] + static_cast<ptrdiff_t>(r) * frame_->linesize[p], in.Plane(p) + r * rowBytes, rowBytes);
        }
    }
    frame_->pts = in.pts;
    CheckAv(avcodec_send_frame(codec_, frame_), "avcodec_send_frame");
    Drain(false);
    ++framesWritten_;
}

void VideoEncoder::Drain(bool flush) {
    if (flush) CheckAv(avcodec_send_frame(codec_, nullptr), "avcodec_send_frame(flush)");
    for (;;) {
        const int err = avcodec_receive_packet(codec_, packet_);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
        CheckAv(err, "avcodec_receive_packet");
        av_packet_rescale_ts(packet_, codec_->time_base, videoStream_->time_base);
        packet_->stream_index = videoStream_->index;
        CheckAv(av_interleaved_write_frame(fmt_, packet_), "av_interleaved_write_frame(video)");
        av_packet_unref(packet_);
    }
}

void VideoEncoder::WriteAudioPacket(AVPacket* packet) {
    if (!audioStream_) return;
    if (!opened_) Throw("VideoEncoder::WriteAudioPacket before Open");
    AVPacket* copy = av_packet_clone(packet);
    if (!copy) Throw("av_packet_clone failed");
    av_packet_rescale_ts(copy, AVRational{audioSourceTimeBase_.num, audioSourceTimeBase_.den}, audioStream_->time_base);
    copy->stream_index = audioStream_->index;
    copy->pos = -1;
    const int err = av_interleaved_write_frame(fmt_, copy);
    av_packet_free(&copy);
    CheckAv(err, "av_interleaved_write_frame(audio)");
    ++audioPacketsWritten_;
}

void VideoEncoder::Close() {
    if (!opened_ || closed_) return;
    Drain(true);
    CheckAv(av_write_trailer(fmt_), "av_write_trailer");
    if (fmt_->pb && !(fmt_->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_->pb);
    closed_ = true;
    Log()->info("encoded {} frames{} -> {}", framesWritten_,
                audioStream_ ? " + " + std::to_string(audioPacketsWritten_) + " audio packets" : "", path_.string());
}

}  // namespace dlssvid
