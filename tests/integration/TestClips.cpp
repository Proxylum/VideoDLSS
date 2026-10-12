#include "TestClips.h"

#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <cmath>
#include <cstring>

#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "util/Error.h"

namespace dlssvid::test {

std::filesystem::path TempDir() {
    static const std::filesystem::path dir = [] {
        std::filesystem::path d = DLSSVID_TEST_TMP;
        std::filesystem::create_directories(d);
        return d;
    }();
    return dir;
}

CpuFrame SyntheticFrame(int64_t index, uint32_t width, uint32_t height) {
    CpuFrame f;
    f.Allocate(FrameDesc{width, height, PixelFormat::Yuv420p});
    f.index = index;
    f.pts = index;
    uint8_t* y = f.Plane(0);
    for (uint32_t r = 0; r < height; ++r)
        for (uint32_t c = 0; c < width; ++c) y[r * width + c] = static_cast<uint8_t>((c + r + index * 3) & 0xFF);
    // moving bright block
    const uint32_t bx = static_cast<uint32_t>((index * 5) % (width > 16 ? width - 16 : 1));
    for (uint32_t r = 8; r < std::min<uint32_t>(24, height); ++r)
        for (uint32_t c = bx; c < std::min<uint32_t>(bx + 16, width); ++c) y[r * width + c] = 235;
    const size_t cw = f.desc.PlaneWidth(1), ch = f.desc.PlaneHeight(1);
    uint8_t* u = f.Plane(1);
    uint8_t* v = f.Plane(2);
    for (size_t r = 0; r < ch; ++r)
        for (size_t c = 0; c < cw; ++c) {
            u[r * cw + c] = static_cast<uint8_t>(64 + (c * 128 / cw) + index);
            v[r * cw + c] = static_cast<uint8_t>(64 + (r * 128 / ch) - index);
        }
    return f;
}

CpuFrame SyntheticFrame(int64_t index, const FrameDesc& desc) {
    const CpuFrame base = SyntheticFrame(index, desc.width, desc.height);
    if (desc == base.desc) return base;
    CpuFrame f;
    f.Allocate(desc);
    f.index = index;
    f.pts = index;
    const int extra = desc.HighBitDepth() ? desc.bitDepth - 8 : 0;
    const uint32_t lowMask = extra > 0 ? (1u << extra) - 1u : 0u;
    const int sx = ChromaShiftX(desc.format), sy = ChromaShiftY(desc.format);
    const size_t srcCw = base.desc.PlaneCols(1);
    auto store = [&](size_t plane, size_t col, size_t row, uint32_t v8, uint32_t ramp) {
        const uint32_t v = (v8 << extra) | (ramp & lowMask);
        if (desc.HighBitDepth()) f.Plane16(plane)[row * desc.PlaneCols(plane) + col] = static_cast<uint16_t>(v);
        else f.Plane(plane)[row * desc.PlaneCols(plane) + col] = static_cast<uint8_t>(v);
    };
    for (uint32_t r = 0; r < desc.height; ++r)
        for (uint32_t c = 0; c < desc.width; ++c) store(0, c, r, base.Plane(0)[static_cast<size_t>(r) * desc.width + c], c + r + static_cast<uint32_t>(index));
    const size_t cw = desc.PlaneCols(1), ch = desc.PlaneHeight(1);
    for (size_t r = 0; r < ch; ++r)
        for (size_t c = 0; c < cw; ++c) {
            const size_t sc = (c << sx) >> 1, sr = (r << sy) >> 1;  // the 4:2:0 chroma sample under this one
            store(1, c, r, base.Plane(1)[sr * srcCw + sc], static_cast<uint32_t>(c * 3 + r));
            store(2, c, r, base.Plane(2)[sr * srcCw + sc], static_cast<uint32_t>(c + r * 5));
        }
    return f;
}

namespace {

AVPixelFormat EncoderFormatFor(const std::string& pixFmt) {
    const AVPixelFormat f = av_get_pix_fmt(pixFmt.c_str());
    if (f == AV_PIX_FMT_NONE) Throw("test clip pixel format not supported: " + pixFmt);
    return f;
}

}  // namespace

FrameDesc FrameDescFor(const ClipSpec& spec) {
    if (spec.pixFmt == "yuv420p" || spec.pixFmt == "p010le" || spec.pixFmt == "yuv444p16le") return FrameDesc{spec.width, spec.height, PixelFormat::Yuv420p, 8};
    FrameDesc out = DecodedFrameDescFor(spec.pixFmt);
    out.width = spec.width;
    out.height = spec.height;
    return out;
}

std::filesystem::path WriteClip(const std::filesystem::path& path, const ClipSpec& spec) {
    const FrameDesc desc = FrameDescFor(spec);
    return WriteClipWith(path, spec, [&, desc](int64_t i) { return SyntheticFrame(i, desc); });
}

std::filesystem::path WriteClipWith(const std::filesystem::path& path, const ClipSpec& spec, const std::function<CpuFrame(int64_t)>& frameFn) {
    std::filesystem::create_directories(path.parent_path());
    const std::string url = path.string();

    AVFormatContext* fmt = nullptr;
    CheckAv(avformat_alloc_output_context2(&fmt, nullptr, nullptr, url.c_str()), "alloc output");

    // --- video (ffv1, lossless, unless the spec says otherwise) ---
    const AVCodec* venc = avcodec_find_encoder_by_name(spec.codec.c_str());
    if (!venc) Throw("test clip encoder not found: " + spec.codec);
    AVCodecContext* vctx = avcodec_alloc_context3(venc);
    vctx->width = static_cast<int>(spec.width);
    vctx->height = static_cast<int>(spec.height);
    const AVPixelFormat encFmt = EncoderFormatFor(spec.pixFmt);
    const bool legacyWide = encFmt == AV_PIX_FMT_P010LE || encFmt == AV_PIX_FMT_YUV444P16LE;  // 8-bit frames into the top bits
    const FrameDesc expected = FrameDescFor(spec);
    vctx->pix_fmt = encFmt;
    vctx->time_base = AVRational{spec.fpsDen, spec.fpsNum};
    vctx->framerate = AVRational{spec.fpsNum, spec.fpsDen};
    if (spec.gop > 0) vctx->gop_size = spec.gop;
    if (fmt->oformat->flags & AVFMT_GLOBALHEADER) vctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    CheckAv(avcodec_open2(vctx, venc, nullptr), ("open " + spec.codec + " (" + spec.pixFmt + ")").c_str());
    AVStream* vs = avformat_new_stream(fmt, nullptr);
    vs->time_base = vctx->time_base;
    vs->avg_frame_rate = vctx->framerate;
    CheckAv(avcodec_parameters_from_context(vs->codecpar, vctx), "video params");

    // --- audio (pcm_s16le sine) ---
    AVCodecContext* actx = nullptr;
    AVStream* as = nullptr;
    if (spec.audio) {
        const AVCodec* aenc = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
        actx = avcodec_alloc_context3(aenc);
        actx->sample_rate = 48000;
        actx->sample_fmt = AV_SAMPLE_FMT_S16;
        av_channel_layout_default(&actx->ch_layout, 1);
        actx->time_base = AVRational{1, actx->sample_rate};
        if (fmt->oformat->flags & AVFMT_GLOBALHEADER) actx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        CheckAv(avcodec_open2(actx, aenc, nullptr), "open pcm");
        as = avformat_new_stream(fmt, nullptr);
        as->time_base = actx->time_base;
        CheckAv(avcodec_parameters_from_context(as->codecpar, actx), "audio params");
    }

    if (!(fmt->oformat->flags & AVFMT_NOFILE)) CheckAv(avio_open(&fmt->pb, url.c_str(), AVIO_FLAG_WRITE), "avio_open");
    CheckAv(avformat_write_header(fmt, nullptr), "write header");

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();

    auto drain = [&](AVCodecContext* ctx, AVStream* st, bool flush) {
        if (flush) CheckAv(avcodec_send_frame(ctx, nullptr), "flush");
        for (;;) {
            const int err = avcodec_receive_packet(ctx, pkt);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
            CheckAv(err, "receive packet");
            av_packet_rescale_ts(pkt, ctx->time_base, st->time_base);
            pkt->stream_index = st->index;
            CheckAv(av_interleaved_write_frame(fmt, pkt), "write packet");
            av_packet_unref(pkt);
        }
    };

    for (int i = 0; i < spec.frames; ++i) {
        const CpuFrame src = frameFn(i);
        if (src.desc != expected) Throw("test clip: frame " + std::to_string(i) + " is " + src.desc.Name() + ", the clip takes " + expected.Name());
        av_frame_unref(frame);
        frame->format = encFmt;
        frame->width = vctx->width;
        frame->height = vctx->height;
        CheckAv(av_frame_get_buffer(frame, 0), "frame buffer");
        const size_t W = spec.width, H = spec.height, cw = src.desc.PlaneWidth(1);
        auto row16 = [&](int plane, size_t r) { return reinterpret_cast<uint16_t*>(frame->data[plane] + static_cast<ptrdiff_t>(r) * frame->linesize[plane]); };
        if (!legacyWide) {  // the frame's own layout: rows copied as they are (8-bit or LSB-aligned 16-bit planes)
            for (size_t p = 0; p < 3; ++p) {
                const size_t rowBytes = src.desc.PlaneWidth(p), rows = src.desc.PlaneHeight(p);
                for (size_t r = 0; r < rows; ++r)
                    std::memcpy(frame->data[p] + static_cast<ptrdiff_t>(r) * frame->linesize[p], src.Plane(p) + r * rowBytes, rowBytes);
            }
        } else {
            // 8-bit samples into the top bits of 16-bit words (exact for the decoder round trip: value >> 8 gives them back)
            for (size_t r = 0; r < H; ++r)
                for (size_t x = 0; x < W; ++x) row16(0, r)[x] = static_cast<uint16_t>(src.Plane(0)[r * W + x] << 8);
            if (encFmt == AV_PIX_FMT_P010LE) {
                for (size_t r = 0; r < H / 2; ++r)
                    for (size_t x = 0; x < W / 2; ++x) {
                        row16(1, r)[2 * x] = static_cast<uint16_t>(src.Plane(1)[r * cw + x] << 8);
                        row16(1, r)[2 * x + 1] = static_cast<uint16_t>(src.Plane(2)[r * cw + x] << 8);
                    }
            } else {  // yuv444p16le: chroma replicated to full resolution
                for (int p = 1; p <= 2; ++p)
                    for (size_t r = 0; r < H; ++r)
                        for (size_t x = 0; x < W; ++x) row16(p, r)[x] = static_cast<uint16_t>(src.Plane(static_cast<size_t>(p))[(r / 2) * cw + x / 2] << 8);
            }
        }
        frame->pts = i;
        CheckAv(avcodec_send_frame(vctx, frame), "send video frame");
        drain(vctx, vs, false);
    }
    drain(vctx, vs, true);

    if (actx) {
        const int frameSize = 1024;
        const int64_t totalSamples = static_cast<int64_t>(spec.frames) * actx->sample_rate * spec.fpsDen / spec.fpsNum;
        int64_t written = 0;
        while (written < totalSamples) {
            av_frame_unref(frame);
            frame->format = actx->sample_fmt;
            frame->sample_rate = actx->sample_rate;
            frame->nb_samples = static_cast<int>(std::min<int64_t>(frameSize, totalSamples - written));
            av_channel_layout_copy(&frame->ch_layout, &actx->ch_layout);
            CheckAv(av_frame_get_buffer(frame, 0), "audio buffer");
            auto* s = reinterpret_cast<int16_t*>(frame->data[0]);
            for (int n = 0; n < frame->nb_samples; ++n)
                s[n] = static_cast<int16_t>(8000.0 * std::sin(2.0 * 3.14159265 * 440.0 * static_cast<double>(written + n) / actx->sample_rate));
            frame->pts = written;
            written += frame->nb_samples;
            CheckAv(avcodec_send_frame(actx, frame), "send audio frame");
            drain(actx, as, false);
        }
        drain(actx, as, true);
    }

    CheckAv(av_write_trailer(fmt), "trailer");
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&vctx);
    if (actx) avcodec_free_context(&actx);
    if (fmt->pb && !(fmt->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt->pb);
    avformat_free_context(fmt);
    return path;
}

std::vector<CpuFrame> DecodeAll(const std::filesystem::path& path) {
    VideoDecoder dec(path);
    std::vector<CpuFrame> frames;
    CpuFrame f;
    while (dec.NextFrame(f)) frames.push_back(f);
    return frames;
}

StreamSummary Summarize(const std::filesystem::path& path) {
    StreamSummary s;
    AVFormatContext* fmt = nullptr;
    const std::string url = path.string();
    CheckAv(avformat_open_input(&fmt, url.c_str(), nullptr, nullptr), "open");
    CheckAv(avformat_find_stream_info(fmt, nullptr), "stream info");
    int audioIndex = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        const AVCodecParameters* p = fmt->streams[i]->codecpar;
        if (p->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (s.videoStreams == 0) {
                const AVCodecDescriptor* d = avcodec_descriptor_get(p->codec_id);
                s.videoCodec = d ? d->name : "?";
                const char* pix = av_get_pix_fmt_name(static_cast<AVPixelFormat>(p->format));
                s.pixelFormat = pix ? pix : "?";
                s.profile = p->profile;
                s.colorRange = static_cast<int>(p->color_range);
                s.colorSpace = static_cast<int>(p->color_space);
                s.colorPrimaries = static_cast<int>(p->color_primaries);
                s.colorTrc = static_cast<int>(p->color_trc);
            }
            ++s.videoStreams;
        }
        if (p->codec_type == AVMEDIA_TYPE_AUDIO) {
            ++s.audioStreams;
            audioIndex = static_cast<int>(i);
            const AVCodecDescriptor* d = avcodec_descriptor_get(p->codec_id);
            s.audioCodec = d ? d->name : "?";
        }
    }
    AVPacket* pkt = av_packet_alloc();
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == audioIndex) {
            ++s.audioPackets;
            const AVRational tb = fmt->streams[audioIndex]->time_base;
            const int sr = fmt->streams[audioIndex]->codecpar->sample_rate;
            if (pkt->duration > 0 && sr > 0) s.audioSamples += av_rescale_q(pkt->duration, tb, AVRational{1, sr});
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    avformat_close_input(&fmt);
    return s;
}

}  // namespace dlssvid::test
