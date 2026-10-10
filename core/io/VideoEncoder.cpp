#include "io/VideoEncoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <cstring>
#include <vector>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

// Encoder input formats that hold a CpuFrame layout exactly, best first: the planar format of the same depth (FFV1 and
// ProRes keep every bit), then the MSB-aligned 16-bit words NVENC takes (p010le / yuv444p16le), then the rest.
std::vector<AVPixelFormat> CandidatesFor(const FrameDesc& d) {
    const std::string_view chroma = ChromaName(d.format);
    if (!d.HighBitDepth()) {
        if (chroma == "444") return {AV_PIX_FMT_YUV444P, AV_PIX_FMT_NV24};
        if (chroma == "422") return {AV_PIX_FMT_YUV422P, AV_PIX_FMT_NV16};
        return {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12};
    }
    if (chroma == "444") {
        if (d.bitDepth == 10) return {AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P16LE, AV_PIX_FMT_P410LE, AV_PIX_FMT_P416LE};
        if (d.bitDepth == 12) return {AV_PIX_FMT_YUV444P12LE, AV_PIX_FMT_YUV444P16LE, AV_PIX_FMT_P416LE};
        return {AV_PIX_FMT_YUV444P16LE, AV_PIX_FMT_P416LE};
    }
    if (chroma == "422") {
        if (d.bitDepth == 10) return {AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_P210LE, AV_PIX_FMT_YUV422P16LE, AV_PIX_FMT_P216LE};
        if (d.bitDepth == 12) return {AV_PIX_FMT_YUV422P12LE, AV_PIX_FMT_YUV422P16LE, AV_PIX_FMT_P216LE};
        return {AV_PIX_FMT_YUV422P16LE, AV_PIX_FMT_P216LE};
    }
    if (d.bitDepth == 10) return {AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_P010LE, AV_PIX_FMT_YUV420P16LE, AV_PIX_FMT_P016LE};
    if (d.bitDepth == 12) return {AV_PIX_FMT_YUV420P12LE, AV_PIX_FMT_P016LE, AV_PIX_FMT_YUV420P16LE};
    return {AV_PIX_FMT_YUV420P16LE, AV_PIX_FMT_P016LE};
}

// The pixel formats an encoder accepts (empty = unrestricted).
std::vector<AVPixelFormat> SupportedFormats(const AVCodec* enc) {
    std::vector<AVPixelFormat> out;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
    const AVPixelFormat* list = nullptr;
    int n = 0;
    if (avcodec_get_supported_config(nullptr, enc, AV_CODEC_CONFIG_PIX_FORMAT, 0, reinterpret_cast<const void**>(&list), &n) >= 0 && list)
        for (int i = 0; i < n; ++i) out.push_back(list[i]);
#else
    if (enc->pix_fmts)
        for (const AVPixelFormat* p = enc->pix_fmts; *p != AV_PIX_FMT_NONE; ++p) out.push_back(*p);
#endif
    return out;
}

AVPixelFormat ChooseFormat(const AVCodec* enc, const FrameDesc& d) {
    const std::vector<AVPixelFormat> supported = SupportedFormats(enc);
    for (const AVPixelFormat f : CandidatesFor(d)) {
        if (supported.empty()) return f;
        for (const AVPixelFormat s : supported)
            if (s == f) return f;
    }
    return AV_PIX_FMT_NONE;
}

std::string TagsText(const ColorTags& t) {
    std::string s;
    if (t.matrix == 1) s += "bt709";
    else if (t.matrix == 5 || t.matrix == 6) s += "bt601";
    else if (t.matrix == 9 || t.matrix == 10) s += "bt2020";
    if (t.range == 1) s += (s.empty() ? "" : " ") + std::string("limited");
    else if (t.range == 2) s += (s.empty() ? "" : " ") + std::string("full");
    return s;
}

}  // namespace

bool VideoEncoder::EncoderAvailable(const std::string& name) { return avcodec_find_encoder_by_name(name.c_str()) != nullptr; }

std::string VideoEncoder::PixelFormatFor(const std::string& codec, const FrameDesc& frameDesc) {
    const AVCodec* enc = avcodec_find_encoder_by_name(codec.c_str());
    if (!enc) return {};
    const AVPixelFormat f = ChooseFormat(enc, frameDesc);
    const char* name = f == AV_PIX_FMT_NONE ? nullptr : av_get_pix_fmt_name(f);
    return name ? name : "";
}

VideoEncoder::VideoEncoder(const std::filesystem::path& path, const FrameDesc& frameDesc, const Options& options)
    : path_(path), frameDesc_(frameDesc), options_(options) {
    if (options_.frameRate.num <= 0 || options_.frameRate.den <= 0) Throw("VideoEncoder: frameRate is required");
    if (options_.timeBase.num <= 0 || options_.timeBase.den <= 0) {
        options_.timeBase = Rational{options_.frameRate.den, options_.frameRate.num};
    }
    if (frameDesc_.HighBitDepth() ? (frameDesc_.bitDepth != 10 && frameDesc_.bitDepth != 12 && frameDesc_.bitDepth != 16) : frameDesc_.bitDepth != 8)
        Throw("VideoEncoder: bit depth " + std::to_string(frameDesc_.bitDepth) + " does not fit " + std::string(ToString(frameDesc_.format)));

    const std::string url = path_.string();
    CheckAv(avformat_alloc_output_context2(&fmt_, nullptr, nullptr, url.c_str()), "avformat_alloc_output_context2");

    const AVCodec* enc = avcodec_find_encoder_by_name(options_.codec.c_str());
    if (!enc) Throw("encoder not found: " + options_.codec + " (try --codec ffv1 or hevc_nvenc)");
    codecName_ = enc->name;

    const AVPixelFormat pix = ChooseFormat(enc, frameDesc_);
    if (pix == AV_PIX_FMT_NONE)
        Throw("encoder " + codecName_ + " does not take " + frameDesc_.Describe() + " frames" +
              (codecName_ == "h264_nvenc" && frameDesc_.HighBitDepth() ? " (H.264 NVENC encodes 8-bit only: use hevc_nvenc or av1_nvenc)" :
               codecName_ == "prores_ks" && !frameDesc_.HighBitDepth() ? " (ProRes is 10-bit: encode 10-bit frames)" : ""));
    pixelFormat_ = pix;
    pixelFormatName_ = av_get_pix_fmt_name(pix);
    const AVPixFmtDescriptor* pd = av_pix_fmt_desc_get(pix);
    semiPlanar_ = pd->comp[1].plane == 1 && pd->comp[2].plane == 1;
    shiftLeft_ = frameDesc_.HighBitDepth() ? (pd->comp[0].depth + pd->comp[0].shift) - frameDesc_.bitDepth : 0;

    codec_ = avcodec_alloc_context3(enc);
    if (!codec_) Throw("avcodec_alloc_context3 failed");
    codec_->width = static_cast<int>(frameDesc_.width);
    codec_->height = static_cast<int>(frameDesc_.height);
    codec_->pix_fmt = pix;
    codec_->time_base = AVRational{options_.timeBase.num, options_.timeBase.den};
    codec_->framerate = AVRational{options_.frameRate.num, options_.frameRate.den};
    codec_->color_range = static_cast<AVColorRange>(options_.color.range);
    codec_->colorspace = static_cast<AVColorSpace>(options_.color.matrix);
    codec_->color_primaries = static_cast<AVColorPrimaries>(options_.color.primaries);
    codec_->color_trc = static_cast<AVColorTransferCharacteristic>(options_.color.transfer);
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
    if (err < 0 && frameDesc_.HighBitDepth())
        Throw("avcodec_open2(" + codecName_ + ") failed for " + frameDesc_.Describe() + " (" + pixelFormatName_ + "): " + AvErrorToString(err) +
              " — this encoder / GPU may not support that bit depth or chroma layout (or the frame size is outside its limits)");
    CheckAv(err, ("avcodec_open2(" + codecName_ + ")").c_str());
    CheckAv(avcodec_parameters_from_context(videoStream_->codecpar, codec_), "avcodec_parameters_from_context");
    // HEVC in MP4 / MOV: the `hvc1` sample entry (parameter sets in the header) is what QuickTime, Apple and Premiere expect;
    // FFmpeg's default `hev1` plays nowhere on macOS.
    const std::string container = fmt_->oformat->name ? fmt_->oformat->name : "";
    if (codec_->codec_id == AV_CODEC_ID_HEVC && (container == "mp4" || container == "mov")) videoStream_->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');

    const std::string url = path_.string();
    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
        CheckAv(avio_open(&fmt_->pb, url.c_str(), AVIO_FLAG_WRITE), "avio_open");
    }
    CheckAv(avformat_write_header(fmt_, nullptr), "avformat_write_header");
    opened_ = true;
    const std::string tags = TagsText(options_.color);
    Log()->info("encode {}: {} {}x{} {} ({}) {}/{} fps{}{}", url, codecName_, frameDesc_.width, frameDesc_.height, pixelFormatName_, frameDesc_.Describe(),
                options_.frameRate.num, options_.frameRate.den, tags.empty() ? "" : ", " + tags, audioStream_ ? ", audio copy" : "");
}

void VideoEncoder::WriteFrame(const CpuFrame& in) {
    if (!opened_) Throw("VideoEncoder::WriteFrame before Open");
    if (in.desc != frameDesc_) Throw("VideoEncoder::WriteFrame: frame format mismatch (" + in.desc.Name() + " into an encoder opened for " + frameDesc_.Name() + ")");
    if (in.data.size() < in.desc.ByteSize()) Throw("VideoEncoder::WriteFrame: frame has no data");

    av_frame_unref(frame_);
    frame_->format = pixelFormat_;
    frame_->width = codec_->width;
    frame_->height = codec_->height;
    frame_->color_range = codec_->color_range;
    frame_->colorspace = codec_->colorspace;
    frame_->color_primaries = codec_->color_primaries;
    frame_->color_trc = codec_->color_trc;
    CheckAv(av_frame_get_buffer(frame_, 0), "av_frame_get_buffer");
    CheckAv(av_frame_make_writable(frame_), "av_frame_make_writable");
    const bool wide = in.desc.HighBitDepth();
    if (!semiPlanar_ && shiftLeft_ == 0) {  // same layout: rows copied as they are
        for (size_t p = 0; p < 3; ++p) {
            const size_t rowBytes = in.desc.PlaneWidth(p);
            const size_t rows = in.desc.PlaneHeight(p);
            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(frame_->data[p] + static_cast<ptrdiff_t>(r) * frame_->linesize[p], in.Plane(p) + r * rowBytes, rowBytes);
            }
        }
    } else if (!semiPlanar_) {  // planar 16-bit words with the samples moved up (yuv444p16le for a 10-bit frame)
        for (size_t p = 0; p < 3; ++p) {
            const size_t cols = in.desc.PlaneCols(p), rows = in.desc.PlaneHeight(p);
            const uint16_t* src = in.Plane16(p);
            for (size_t r = 0; r < rows; ++r) {
                uint16_t* dst = reinterpret_cast<uint16_t*>(frame_->data[p] + static_cast<ptrdiff_t>(r) * frame_->linesize[p]);
                for (size_t c = 0; c < cols; ++c) dst[c] = static_cast<uint16_t>(src[r * cols + c] << shiftLeft_);
            }
        }
    } else {  // interleaved chroma: nv12 / nv16 / nv24 (8-bit) and p010 / p016 / p210 / p216 / p410 / p416 (16-bit words)
        const size_t rowBytes = in.desc.PlaneWidth(0);
        for (size_t r = 0; r < in.desc.height; ++r) {
            if (wide) {
                const uint16_t* src = in.Plane16(0) + r * in.desc.width;
                uint16_t* dst = reinterpret_cast<uint16_t*>(frame_->data[0] + static_cast<ptrdiff_t>(r) * frame_->linesize[0]);
                for (size_t c = 0; c < in.desc.width; ++c) dst[c] = static_cast<uint16_t>(src[c] << shiftLeft_);
            } else {
                std::memcpy(frame_->data[0] + static_cast<ptrdiff_t>(r) * frame_->linesize[0], in.Plane(0) + r * rowBytes, rowBytes);
            }
        }
        const size_t cw = in.desc.PlaneCols(1), ch = in.desc.PlaneHeight(1);
        for (size_t r = 0; r < ch; ++r) {
            if (wide) {
                const uint16_t* u = in.Plane16(1) + r * cw;
                const uint16_t* v = in.Plane16(2) + r * cw;
                uint16_t* dst = reinterpret_cast<uint16_t*>(frame_->data[1] + static_cast<ptrdiff_t>(r) * frame_->linesize[1]);
                for (size_t c = 0; c < cw; ++c) {
                    dst[2 * c] = static_cast<uint16_t>(u[c] << shiftLeft_);
                    dst[2 * c + 1] = static_cast<uint16_t>(v[c] << shiftLeft_);
                }
            } else {
                const uint8_t* u = in.Plane(1) + r * cw;
                const uint8_t* v = in.Plane(2) + r * cw;
                uint8_t* dst = frame_->data[1] + static_cast<ptrdiff_t>(r) * frame_->linesize[1];
                for (size_t c = 0; c < cw; ++c) {
                    dst[2 * c] = u[c];
                    dst[2 * c + 1] = v[c];
                }
            }
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
