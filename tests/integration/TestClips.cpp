#include "TestClips.h"

#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/display.h>
#include <libavutil/ambient_viewing_environment.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/hdr_dynamic_metadata.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <cctype>
#include <stdexcept>
#include <iterator>
#include <fstream>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <mutex>

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

std::string ReadAllBytes(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + file.string());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

AVPixelFormat EncoderFormatFor(const std::string& pixFmt) {
    const AVPixelFormat f = av_get_pix_fmt(pixFmt.c_str());
    if (f == AV_PIX_FMT_NONE) Throw("test clip pixel format unknown to FFmpeg: " + pixFmt);  // a test's mistake: never read as «unsupported»
    return f;
}

AVRational R(const Ratio& q) { return AVRational{q.num, q.den}; }

// TASK-0078: the HDR records as the stream's coded side data (the muxers write them from there)
void AddHdrSideData(AVCodecParameters* par, const ClipHdr& hdr) {
    auto add = [&](AVPacketSideDataType type, size_t size) {
        AVPacketSideData* sd = av_packet_side_data_new(&par->coded_side_data, &par->nb_coded_side_data, type, size, 0);
        if (!sd) Throw("test clip: no room for HDR side data");
        std::memset(sd->data, 0, size);
        return sd->data;
    };
    if (hdr.mastering) {
        auto* m = reinterpret_cast<AVMasteringDisplayMetadata*>(add(AV_PKT_DATA_MASTERING_DISPLAY_METADATA, sizeof(AVMasteringDisplayMetadata)));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j) m->display_primaries[i][j] = R(hdr.mastering->primaries[i][j]);
        m->white_point[0] = R(hdr.mastering->white[0]);
        m->white_point[1] = R(hdr.mastering->white[1]);
        m->min_luminance = R(hdr.mastering->minLum);
        m->max_luminance = R(hdr.mastering->maxLum);
        m->has_primaries = hdr.mastering->hasPrimaries;
        m->has_luminance = hdr.mastering->hasLuminance;
    }
    if (hdr.light) {
        auto* c = reinterpret_cast<AVContentLightMetadata*>(add(AV_PKT_DATA_CONTENT_LIGHT_LEVEL, sizeof(AVContentLightMetadata)));
        c->MaxCLL = hdr.light->maxCll;
        c->MaxFALL = hdr.light->maxFall;
    }
    if (hdr.ambient) {
        auto* a = reinterpret_cast<AVAmbientViewingEnvironment*>(add(AV_PKT_DATA_AMBIENT_VIEWING_ENVIRONMENT, sizeof(AVAmbientViewingEnvironment)));
        a->ambient_illuminance = R(hdr.ambient->illuminance);
        a->ambient_light_x = R(hdr.ambient->lightX);
        a->ambient_light_y = R(hdr.ambient->lightY);
    }
    if (hdr.dovi) {
        size_t size = 0;
        AVDOVIDecoderConfigurationRecord* d = av_dovi_alloc(&size);
        if (!d) Throw("test clip: no memory for a Dolby Vision record");
        d->dv_version_major = static_cast<uint8_t>(hdr.dovi->major);
        d->dv_version_minor = static_cast<uint8_t>(hdr.dovi->minor);
        d->dv_profile = static_cast<uint8_t>(hdr.dovi->profile);
        d->dv_level = static_cast<uint8_t>(hdr.dovi->level);
        d->rpu_present_flag = hdr.dovi->rpu;
        d->el_present_flag = hdr.dovi->el;
        d->bl_present_flag = hdr.dovi->bl;
        d->dv_bl_signal_compatibility_id = static_cast<uint8_t>(hdr.dovi->blCompat);
        if (!av_packet_side_data_add(&par->coded_side_data, &par->nb_coded_side_data, AV_PKT_DATA_DOVI_CONF, d, size, 0)) {
            av_free(d);
            Throw("test clip: no room for the Dolby Vision record");
        }
    }
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
    CheckAv(avformat_alloc_output_context2(&fmt, nullptr, spec.format.empty() ? nullptr : spec.format.c_str(), url.c_str()), "alloc output");

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
    if (spec.bframes >= 0) vctx->max_b_frames = spec.bframes;
    const AVRational sar = spec.sar.Square() ? AVRational{0, 1} : AVRational{static_cast<int>(spec.sar.num), static_cast<int>(spec.sar.den)};
    if (spec.sarIn != "container") {
        vctx->sample_aspect_ratio = sar;
        if (!spec.sar.Square() && spec.codec == "ffv1") vctx->level = 3;  // the slices carry the frame's proportions from version 3
    }
    if (fmt->oformat->flags & AVFMT_GLOBALHEADER) vctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    // TASK-0078: the encoder's colour (an NVENC VUI says it; all unspecified by default, as before)
    vctx->colorspace = static_cast<AVColorSpace>(spec.color.matrix);
    vctx->color_primaries = static_cast<AVColorPrimaries>(spec.color.primaries);
    vctx->color_trc = static_cast<AVColorTransferCharacteristic>(spec.color.trc);
    vctx->color_range = static_cast<AVColorRange>(spec.color.range);
    vctx->chroma_sample_location = static_cast<AVChromaLocation>(spec.color.chromaLoc);
    CheckAv(avcodec_open2(vctx, venc, nullptr), ("open " + spec.codec + " (" + spec.pixFmt + ")").c_str());
    AVStream* vs = avformat_new_stream(fmt, nullptr);
    vs->time_base = vctx->time_base;
    vs->avg_frame_rate = vctx->framerate;
    CheckAv(avcodec_parameters_from_context(vs->codecpar, vctx), "video params");
    if (spec.sarIn != "codec") vs->sample_aspect_ratio = sar;
    if (!spec.containerSar.Square()) {
        vs->codecpar->sample_aspect_ratio = AVRational{0, 1};
        vs->sample_aspect_ratio = AVRational{static_cast<int>(spec.containerSar.num), static_cast<int>(spec.containerSar.den)};
    }
    if (!spec.orientation.Identity()) {  // TASK-0072: a phone video's display matrix
        AVPacketSideData* sd = av_packet_side_data_new(&vs->codecpar->coded_side_data, &vs->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX,
                                                       9 * sizeof(int32_t), 0);
        if (!sd) Throw("test clip: no room for the display matrix");
        auto* m = reinterpret_cast<int32_t*>(sd->data);
        av_display_rotation_set(m, spec.orientation.rotate);
        if (spec.orientation.hflip) av_display_matrix_flip(m, 1, 0);
    }
    if (!spec.rotateTag.empty()) av_dict_set(&vs->metadata, "rotate", spec.rotateTag.c_str(), 0);
    if (spec.containerColor) {  // TASK-0078: the container says other tags than the encoder
        vs->codecpar->color_space = static_cast<AVColorSpace>(spec.containerColor->matrix);
        vs->codecpar->color_primaries = static_cast<AVColorPrimaries>(spec.containerColor->primaries);
        vs->codecpar->color_trc = static_cast<AVColorTransferCharacteristic>(spec.containerColor->trc);
        vs->codecpar->color_range = static_cast<AVColorRange>(spec.containerColor->range);
    }
    AddHdrSideData(vs->codecpar, spec.hdr);
    if (spec.extradata) {
        const std::string bytes = spec.extradata(std::string(reinterpret_cast<const char*>(vs->codecpar->extradata),
                                                             static_cast<size_t>(std::max(vs->codecpar->extradata_size, 0))));
        auto* data = static_cast<uint8_t*>(av_mallocz(bytes.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!data) Throw("test clip: no memory for extradata");
        std::memcpy(data, bytes.data(), bytes.size());
        av_freep(&vs->codecpar->extradata);
        vs->codecpar->extradata = data;
        vs->codecpar->extradata_size = static_cast<int>(bytes.size());
    }
    if (spec.strict != INT32_MIN) fmt->strict_std_compliance = spec.strict;

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

    int64_t videoPackets = 0;
    auto drain = [&](AVCodecContext* ctx, AVStream* st, bool flush) {
        if (flush) CheckAv(avcodec_send_frame(ctx, nullptr), "flush");
        for (;;) {
            const int err = avcodec_receive_packet(ctx, pkt);
            if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
            CheckAv(err, "receive packet");
            if (st == vs) {  // TASK-0078: a packet written anew, side data added
                const int64_t index = videoPackets++;
                if (spec.videoPacket) {
                    const std::string bytes = spec.videoPacket(index, std::string(reinterpret_cast<const char*>(pkt->data), static_cast<size_t>(pkt->size)));
                    AVBufferRef* buf = av_buffer_alloc(bytes.size() + AV_INPUT_BUFFER_PADDING_SIZE);
                    if (!buf) Throw("test clip: no memory for a packet");
                    std::memcpy(buf->data, bytes.data(), bytes.size());
                    std::memset(buf->data + bytes.size(), 0, AV_INPUT_BUFFER_PADDING_SIZE);
                    av_buffer_unref(&pkt->buf);
                    pkt->buf = buf;
                    pkt->data = buf->data;
                    pkt->size = static_cast<int>(bytes.size());
                }
                if (spec.videoPacketSideData)
                    for (const auto& [type, bytes] : spec.videoPacketSideData(index)) {
                        uint8_t* d = av_packet_new_side_data(pkt, static_cast<AVPacketSideDataType>(type), bytes.size());
                        if (!d) Throw("test clip: no room for packet side data");
                        std::memcpy(d, bytes.data(), bytes.size());
                    }
            }
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
        frame->sample_aspect_ratio = spec.sarIn != "container" ? sar : AVRational{0, 1};
        {  // TASK-0078: the frame's tags (prores_ks writes them into its header)
            const RawTags fc = spec.frameColor ? spec.frameColor(i) : spec.color;
            frame->colorspace = static_cast<AVColorSpace>(fc.matrix);
            frame->color_primaries = static_cast<AVColorPrimaries>(fc.primaries);
            frame->color_trc = static_cast<AVColorTransferCharacteristic>(fc.trc);
            frame->color_range = static_cast<AVColorRange>(fc.range);
        }
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
        frame->pts = spec.pts ? spec.pts(i) : i;
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
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {  // TASK-0078: the tags as the demuxer read them, before the decoder's turn
        const AVCodecParameters* p = fmt->streams[i]->codecpar;
        if (p->codec_type == AVMEDIA_TYPE_VIDEO) {
            s.container = RawTags{static_cast<int>(p->color_space), static_cast<int>(p->color_primaries), static_cast<int>(p->color_trc),
                                  static_cast<int>(p->color_range), static_cast<int>(p->chroma_location)};
            break;
        }
    }
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
                s.width = static_cast<uint32_t>(p->width);
                s.height = static_cast<uint32_t>(p->height);
                const AVRational st = fmt->streams[i]->sample_aspect_ratio, guess = av_guess_sample_aspect_ratio(fmt, fmt->streams[i], nullptr);
                s.streamSar = PixelAspectFrom(st.num, st.den);
                s.codecSar = PixelAspectFrom(p->sample_aspect_ratio.num, p->sample_aspect_ratio.den);
                s.sar = PixelAspectFrom(guess.num, guess.den);
                ReadCodedSideData(p->coded_side_data, p->nb_coded_side_data, s.hdr, &s.dovi);
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
    if (const AVDictionaryEntry* c = av_dict_get(fmt->metadata, "comment", nullptr, 0)) s.comment = c->value;
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

std::vector<RawTags> FrameColors(const std::filesystem::path& path, int maxFrames, bool bitstreamOnly) {
    std::vector<RawTags> out;
    AVFormatContext* fmt = nullptr;
    AVCodecContext* ctx = nullptr;
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    const std::string url = path.string();
    try {
        CheckAv(avformat_open_input(&fmt, url.c_str(), nullptr, nullptr), "open");
        CheckAv(avformat_find_stream_info(fmt, nullptr), "stream info");
        const AVCodec* dec = nullptr;
        const int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
        if (vs < 0 || !dec) Throw("FrameColors: no video stream or decoder");
        if (dec->id == AV_CODEC_ID_AV1) Throw("FrameColors: AV1 has no software decoder in this build");
        ctx = avcodec_alloc_context3(dec);
        CheckAv(avcodec_parameters_to_context(ctx, fmt->streams[vs]->codecpar), "params");
        if (bitstreamOnly) {
            ctx->color_range = AVCOL_RANGE_UNSPECIFIED;
            ctx->colorspace = AVCOL_SPC_UNSPECIFIED;
            ctx->color_primaries = AVCOL_PRI_UNSPECIFIED;
            ctx->color_trc = AVCOL_TRC_UNSPECIFIED;
        }
        ctx->thread_type = FF_THREAD_SLICE;
        CheckAv(avcodec_open2(ctx, dec, nullptr), "open decoder");
        auto receive = [&] {
            for (;;) {
                const int err = avcodec_receive_frame(ctx, frame);
                if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return;
                CheckAv(err, "receive frame");
                if (static_cast<int>(out.size()) < maxFrames) out.push_back(FrameTagsOf(frame));
                av_frame_unref(frame);
            }
        };
        while (static_cast<int>(out.size()) < maxFrames && av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index == vs) {
                CheckAv(avcodec_send_packet(ctx, pkt), "send packet");
                receive();
            }
            av_packet_unref(pkt);
        }
        if (static_cast<int>(out.size()) < maxFrames) {
            CheckAv(avcodec_send_packet(ctx, nullptr), "flush");
            receive();
        }
    } catch (...) {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        throw;
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);
    return out;
}

std::string Hdr10PlusSideData() {
    size_t size = 0;
    AVDynamicHDRPlus* h = av_dynamic_hdr_plus_alloc(&size);
    if (!h) Throw("Hdr10PlusSideData: no memory");
    h->itu_t_t35_country_code = 0xB5;
    h->application_version = 1;
    h->num_windows = 1;
    h->targeted_system_display_maximum_luminance = AVRational{400, 1};
    AVHDRPlusColorTransformParams& p = h->params[0];
    for (AVRational& m : p.maxscl) m = AVRational{50000, 100000};
    p.average_maxrgb = AVRational{10000, 100000};
    const uint8_t percentages[9] = {1, 5, 10, 25, 50, 75, 90, 95, 99};
    p.num_distribution_maxrgb_percentiles = 9;
    for (int i = 0; i < 9; ++i) p.distribution_maxrgb[i] = AVHDRPlusPercentile{percentages[i], AVRational{1000 * (i + 1), 100000}};
    p.fraction_bright_pixels = AVRational{0, 1000};
    std::string bytes(reinterpret_cast<const char*>(h), size);
    av_free(h);
    return bytes;
}

std::string HevcPrefixSei(int payloadType, const std::string& payload) {
    std::string rbsp;  // the SEI message
    for (int t = payloadType;; t -= 255) {
        if (t < 255) {
            rbsp += static_cast<char>(t);
            break;
        }
        rbsp += static_cast<char>(0xFF);
    }
    for (size_t n = payload.size();; n -= 255) {
        if (n < 255) {
            rbsp += static_cast<char>(n);
            break;
        }
        rbsp += static_cast<char>(0xFF);
    }
    rbsp += payload;
    rbsp += static_cast<char>(0x80);  // rbsp trailing bits
    // start code, NAL header: type 39 (prefix SEI), layer 0, nuh_temporal_id_plus1 1; then the payload with emulation prevention (no
    // 00 00 0x with x <= 3 inside a NAL unit: a 03 goes before the third byte)
    std::string nal("\x00\x00\x00\x01\x4E\x01", 6);
    int zeros = 0;
    for (char ch : rbsp) {
        const auto b = static_cast<unsigned char>(ch);
        if (zeros >= 2 && b <= 3) {
            nal += '\x03';
            zeros = 0;
        }
        nal += ch;
        zeros = b == 0 ? zeros + 1 : 0;
    }
    return nal;
}

std::string AlternativeTransferSei(int trc) { return HevcPrefixSei(147, std::string(1, static_cast<char>(trc))); }

std::string WithNalBeforeSlice(const std::string& au, const std::string& nal) {
    // Annex B: the first NAL unit of a picture (VCL: types 0-31) - the SEI goes before its start code
    for (size_t i = 0; i + 3 < au.size(); ++i) {
        if (au[i] != 0 || au[i + 1] != 0 || au[i + 2] != 1) continue;
        const size_t start = i + 3;
        const int type = (static_cast<unsigned char>(au[start]) >> 1) & 0x3F;
        if (type < 32) {
            const size_t at = i > 0 && au[i - 1] == 0 ? i - 1 : i;  // a 4-byte start code begins one before
            return au.substr(0, at) + nal + au.substr(at);
        }
        i = start;
    }
    throw std::runtime_error("WithNalBeforeSlice: no slice in the access unit");
}

std::string H264Sei(int payloadType, const std::string& payload) {
    // the same SEI message and emulation prevention as HEVC's, under H.264's one-byte NAL header (nal_ref_idc 0, type 6)
    return std::string("\x00\x00\x00\x01\x06", 5) + HevcPrefixSei(payloadType, payload).substr(6);
}

std::string H264AlternativeTransferSei(int trc) { return H264Sei(147, std::string(1, static_cast<char>(trc))); }

std::string WithNalBeforeH264Slice(const std::string& au, const std::string& nal) {
    for (size_t i = 0; i + 3 < au.size(); ++i) {
        if (au[i] != 0 || au[i + 1] != 0 || au[i + 2] != 1) continue;
        const size_t start = i + 3;
        const int type = static_cast<unsigned char>(au[start]) & 0x1F;
        if (type >= 1 && type <= 5) {
            const size_t at = i > 0 && au[i - 1] == 0 ? i - 1 : i;
            return au.substr(0, at) + nal + au.substr(at);
        }
        i = start;
    }
    throw std::runtime_error("WithNalBeforeH264Slice: no slice in the access unit");
}

std::filesystem::path ConcatFiles(const std::filesystem::path& a, const std::filesystem::path& b, const std::filesystem::path& out) {
    const std::string both = ReadAllBytes(a) + ReadAllBytes(b);
    std::ofstream o(out, std::ios::binary);
    o.write(both.data(), static_cast<std::streamsize>(both.size()));
    if (!o) throw std::runtime_error("ConcatFiles: cannot write " + out.string());
    return out;
}

namespace {
std::mutex g_logMutex;
std::string* g_logSink = nullptr;  // the capture in force (one at a time), under g_logMutex

void CaptureLog(void* avcl, int level, const char* fmt, va_list vl) {
    va_list copy;
    va_copy(copy, vl);
    av_log_default_callback(avcl, level, fmt, vl);  // stderr as before
    if (level <= av_log_get_level()) {
        char line[1024];
        int prefix = 1;
        av_log_format_line2(avcl, level, fmt, copy, line, sizeof line, &prefix);
        std::lock_guard<std::mutex> lock(g_logMutex);
        if (g_logSink) *g_logSink += line;
    }
    va_end(copy);
}
}  // namespace

FfmpegLogCapture::FfmpegLogCapture() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_logSink = &text_;
    av_log_set_callback(CaptureLog);
}

FfmpegLogCapture::~FfmpegLogCapture() {
    av_log_set_callback(av_log_default_callback);
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_logSink = nullptr;
}

std::string FfmpegLogCapture::Text() const {
    std::lock_guard<std::mutex> lock(g_logMutex);
    return text_;
}

bool SaysUnsupported(const std::string& text) {
    // NVENC's own words (libavcodec/nvenc.c, FFmpeg 7.1) for a GPU without the encoder, its depth or 4:4:4 - nothing else: a refused parameter
    // («unsupported param»), an invalid spec, a test's own error and «No capable devices found» (said for a session that failed to open
    // now - out of memory, a busy GPU - as well) are failures, not a missing capability; no NVIDIA GPU at all is skipped before (HaveNvidia)
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* words : {"codec not supported", "10 bit encode not supported", "yuv444p not supported"})
        if (lower.find(words) != std::string::npos) return true;
    return false;
}

std::optional<std::filesystem::path> WriteClipIfSupported(const std::filesystem::path& path, const ClipSpec& spec, std::string* why) {
    FfmpegLogCapture log;
    try {
        return WriteClip(path, spec);
    } catch (const std::exception& e) {
        const std::string said = std::string(e.what()) + "\n" + log.Text();
        if (!SaysUnsupported(said)) throw;
        if (why) *why = said;
        return std::nullopt;
    }
}

namespace {
// A check of the AV1 / IVF test streams below: a failure throws (the test fails with it)
void Need(bool ok, int line) {
    if (!ok) throw std::runtime_error("AV1 / IVF test stream: check failed at TestClips.cpp:" + std::to_string(line));
}

std::string ReadAll(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
}  // namespace

// An IVF file of two AV1 streams one after the other - `a`, then `b` (other parameters: a part of the stream av1dec meets later), b's
// timestamps going on from a's. av1_nvenc writes the sequence header into every keyframe of an IVF file: b's starts b's part.
std::filesystem::path ConcatIvf(const std::filesystem::path& a, const std::filesystem::path& b, const std::filesystem::path& out,
                                const std::function<std::string(const std::string&)>& partB) {
    auto u16 = [](const std::string& s, size_t at) { return static_cast<uint32_t>(static_cast<uint8_t>(s[at])) | static_cast<uint32_t>(static_cast<uint8_t>(s[at + 1])) << 8; };
    auto u32 = [&](const std::string& s, size_t at) { return u16(s, at) | u16(s, at + 2) << 16; };
    auto u64 = [&](const std::string& s, size_t at) { return static_cast<uint64_t>(u32(s, at)) | static_cast<uint64_t>(u32(s, at + 4)) << 32; };
    auto put = [](std::string& s, size_t at, uint64_t v, int bytes) {
        for (int i = 0; i < bytes; ++i) s[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    };
    struct Frame {
        uint64_t pts;
        std::string data;
    };
    auto frames = [&](const std::string& f) {
        Need(f.size() >= 32, __LINE__);
        Need(f.compare(0, 4, "DKIF") == 0, __LINE__);
        std::vector<Frame> out;
        for (size_t at = u16(f, 6); at + 12 <= f.size();) {
            const uint32_t size = u32(f, at);
            Need(at + 12 + size <= f.size(), __LINE__);
            out.push_back({u64(f, at + 4), f.substr(at + 12, size)});
            at += 12 + size;
        }
        return out;
    };
    const std::string fa = ReadAll(a), fb = ReadAll(b);
    Need(fa.compare(16, 8, fb, 16, 8) == 0, __LINE__);  // the same time base
    const std::vector<Frame> ka = frames(fa), kb = frames(fb);
    Need(ka.size() >= 2, __LINE__);
    Need(!(kb.empty()), __LINE__);
    const uint64_t step = ka[1].pts - ka[0].pts;
    std::string ivf = fa.substr(0, u16(fa, 6));
    put(ivf, 24, ka.size() + kb.size(), 4);
    auto append = [&](uint64_t pts, const std::string& data) {
        std::string head(12, '\0');
        put(head, 0, data.size(), 4);
        put(head, 4, pts, 8);
        ivf += head + data;
    };
    for (const Frame& f : ka) append(f.pts, f.data);
    for (const Frame& f : kb) append(ka.back().pts + step + (f.pts - kb.front().pts), partB ? partB(f.data) : f.data);
    std::ofstream(out, std::ios::binary) << ivf;
    return out;
}

// Bits of an AV1 bitstream, most significant first (AV1 spec 4.10: f(n), uvlc())
struct BitReader {
    const std::string& s;
    size_t pos = 0;  // in bits
    uint32_t f(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i, ++pos) {
            Need(pos / 8 < s.size(), __LINE__);
            v = v << 1 | ((static_cast<uint8_t>(s[pos / 8]) >> (7 - pos % 8)) & 1u);
        }
        return v;
    }
    uint32_t uvlc() {
        int zeros = 0;
        while (!f(1)) ++zeros;
        return zeros >= 32 ? UINT32_MAX : f(zeros) + ((1u << zeros) - 1);
    }
};
struct BitWriter {
    std::string out;
    size_t pos = 0;
    void put(uint32_t v, int n) {
        for (int i = n - 1; i >= 0; --i, ++pos) {
            if (pos % 8 == 0) out.push_back(0);
            if ((v >> i) & 1u) out.back() = static_cast<char>(static_cast<uint8_t>(out.back()) | (0x80u >> (pos % 8)));
        }
    }
};

std::string Leb128(uint64_t v) {
    std::string out;
    do {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (v) byte |= 0x80;
        out.push_back(static_cast<char>(byte));
    } while (v);
    return out;
}

// Where a sequence header says its frame size (AV1 spec 5.5.1): the bit of max_frame_width_minus_1 and the two fields' widths
struct SequenceFrameSize {
    size_t at = 0;
    int widthBits = 0, heightBits = 0;
};
SequenceFrameSize FrameSizeOf(const std::string& payload) {
    BitReader b{payload};
    b.f(3);  // seq_profile
    b.f(1);  // still_picture
    Need(b.f(1) == 0, __LINE__);  // reduced_still_picture_header: a video
    bool decoderModel = false;
    int bufferDelayBits = 0;
    if (b.f(1)) {  // timing_info_present_flag
        b.f(32);
        b.f(32);
        if (b.f(1)) b.uvlc();  // equal_picture_interval: num_ticks_per_picture_minus_1
        decoderModel = b.f(1);
        if (decoderModel) {
            bufferDelayBits = static_cast<int>(b.f(5)) + 1;
            b.f(32);
            b.f(5);
            b.f(5);
        }
    }
    const bool initialDisplayDelay = b.f(1);
    const int operatingPoints = static_cast<int>(b.f(5)) + 1;
    for (int i = 0; i < operatingPoints; ++i) {
        b.f(12);                      // operating_point_idc
        if (b.f(5) > 7) b.f(1);       // seq_level_idx, seq_tier
        if (decoderModel && b.f(1)) {  // decoder_model_present_for_this_op
            b.f(bufferDelayBits);
            b.f(bufferDelayBits);
            b.f(1);
        }
        if (initialDisplayDelay && b.f(1)) b.f(4);
    }
    const int widthBits = static_cast<int>(b.f(4)) + 1, heightBits = static_cast<int>(b.f(4)) + 1;
    return {b.pos, widthBits, heightBits};
}

// The payload of an 8-bit 4:2:0 sequence header (profile 0, AV1 spec 5.5) said as 12-bit 4:2:0 (profile 2) - the same up to its
// color_config. The frames after it are no 12-bit pictures, but av1dec asks for the pixel format (yuv420p12, which FFmpeg offers to no
// NVDEC) before it decodes any of them: a part of the stream NVDEC does not take, as one written so would be.
std::string TwelveBitSequenceHeader(const std::string& payload) {
    BitReader b{payload};
    Need(b.f(3) == 0, __LINE__);  // seq_profile: the 8-bit 4:2:0 profile
    const SequenceFrameSize size = FrameSizeOf(payload);
    b.pos = size.at + static_cast<size_t>(size.widthBits + size.heightBits);
    if (b.f(1)) {  // frame_id_numbers_present_flag
        b.f(4);
        b.f(3);
    }
    for (int i = 0; i < 7; ++i) b.f(1);  // 128x128 superblocks, filter intra, intra edge, interintra, masked, warped motion, dual filter
    const bool orderHint = b.f(1);
    if (orderHint) {
        b.f(1);
        b.f(1);
    }
    const uint32_t screenContent = b.f(1) ? 2 : b.f(1);  // seq_choose_screen_content_tools: SELECT, else seq_force_screen_content_tools
    if (screenContent > 0 && !b.f(1)) b.f(1);           // seq_choose_integer_mv, else seq_force_integer_mv
    if (orderHint) b.f(3);
    b.f(1);  // enable_superres
    b.f(1);  // enable_cdef
    b.f(1);  // enable_restoration
    const size_t colorConfig = b.pos;
    Need(b.f(1) == 0, __LINE__);  // high_bitdepth: an 8-bit stream
    Need(b.f(1) == 0, __LINE__);  // mono_chrome
    const uint32_t description = b.f(1);
    uint32_t primaries = 2, transfer = 2, matrix = 2;
    if (description) {
        primaries = b.f(8);
        transfer = b.f(8);
        matrix = b.f(8);
    }
    Need(!((primaries == 1 && transfer == 13 && matrix == 0)), __LINE__);  // not the sRGB 4:4:4 shorthand
    const uint32_t range = b.f(1);
    const uint32_t samplePosition = b.f(2);
    const uint32_t separateUv = b.f(1);
    const uint32_t filmGrain = b.f(1);
    BitWriter w;
    w.put(2, 3);  // seq_profile 2
    BitReader copy{payload};
    copy.pos = 3;
    while (copy.pos < colorConfig) w.put(copy.f(1), 1);
    w.put(1, 1);  // high_bitdepth
    w.put(1, 1);  // twelve_bit
    w.put(0, 1);  // mono_chrome
    w.put(description, 1);
    if (description) {
        w.put(primaries, 8);
        w.put(transfer, 8);
        w.put(matrix, 8);
    }
    w.put(range, 1);
    w.put(1, 1);  // subsampling_x
    w.put(1, 1);  // subsampling_y: 4:2:0
    w.put(samplePosition, 2);
    w.put(separateUv, 1);
    w.put(filmGrain, 1);
    w.put(1, 1);  // trailing bits
    while (w.pos % 8) w.put(0, 1);
    return w.out;
}

// A temporal unit (IVF frame) with its sequence header OBUs (OBU_SEQUENCE_HEADER) written by `header`; the other OBUs as they are
std::string WithSequenceHeadersWritten(const std::string& unit, const std::function<std::string(const std::string&)>& header) {
    std::string out;
    for (size_t at = 0; at < unit.size();) {
        const uint8_t head = static_cast<uint8_t>(unit[at]);
        const int type = (head >> 3) & 15;
        Need((head & 2), __LINE__);  // obu_has_size_field
        const size_t headSize = (head & 4) ? 2 : 1;
        uint64_t size = 0;
        size_t sizeBytes = 0;
        for (;; ++sizeBytes) {
            Need(at + headSize + sizeBytes < unit.size(), __LINE__);
            const uint8_t byte = static_cast<uint8_t>(unit[at + headSize + sizeBytes]);
            size |= static_cast<uint64_t>(byte & 0x7f) << (7 * sizeBytes);
            if (!(byte & 0x80)) {
                ++sizeBytes;
                break;
            }
        }
        const std::string payload = unit.substr(at + headSize + sizeBytes, static_cast<size_t>(size));
        const std::string written = type == 1 ? header(payload) : payload;
        out += unit.substr(at, headSize) + Leb128(written.size()) + written;
        at += headSize + sizeBytes + static_cast<size_t>(size);
    }
    return out;
}

// A temporal unit (IVF frame) with its sequence header OBUs said as 12-bit (TwelveBitSequenceHeader); the other OBUs as they are
std::string TwelveBitTemporalUnit(const std::string& unit) { return WithSequenceHeadersWritten(unit, TwelveBitSequenceHeader); }

// A sequence header of av1_nvenc's (no timing info, no decoder model, no initial display delay) with an initial display delay of 10
// frames for every operating point (AV1 5.5: initial_display_delay_present_flag and _minus_1) - everything else as it was: a header with
// other bytes and the same decoder (size, depth, chroma)
std::string InitialDisplayDelaySequenceHeader(const std::string& payload) {
    BitReader b{payload};
    const uint32_t profile = b.f(3), still = b.f(1);
    Need(b.f(1) == 0, __LINE__);  // reduced_still_picture_header
    Need(b.f(1) == 0, __LINE__);  // timing_info_present_flag
    Need(b.f(1) == 0, __LINE__);  // initial_display_delay_present_flag
    const uint32_t points = b.f(5) + 1;
    struct Point {
        uint32_t idc, level, tier;
    };
    std::vector<Point> ops;
    for (uint32_t i = 0; i < points; ++i) {
        Point p{b.f(12), b.f(5), 0};
        if (p.level > 7) p.tier = b.f(1);
        ops.push_back(p);
    }
    // the trailing one bit: the last bit set
    size_t trailing = payload.size() * 8;
    while (trailing > 0 && !((static_cast<uint8_t>(payload[(trailing - 1) / 8]) >> (7 - (trailing - 1) % 8)) & 1u)) --trailing;
    Need(trailing > b.pos, __LINE__);
    --trailing;
    BitWriter w;
    w.put(profile, 3);
    w.put(still, 1);
    w.put(0, 1);  // reduced_still_picture_header
    w.put(0, 1);  // timing_info_present_flag
    w.put(1, 1);  // initial_display_delay_present_flag
    w.put(points - 1, 5);
    for (const Point& p : ops) {
        w.put(p.idc, 12);
        w.put(p.level, 5);
        if (p.level > 7) w.put(p.tier, 1);
        w.put(1, 1);  // initial_display_delay_present_for_this_op
        w.put(9, 4);  // initial_display_delay_minus_1
    }
    BitReader rest{payload};
    rest.pos = b.pos;
    while (rest.pos < trailing) w.put(rest.f(1), 1);
    w.put(1, 1);  // trailing bits
    while (w.pos % 8) w.put(0, 1);
    return w.out;
}

std::string InitialDisplayDelayTemporalUnit(const std::string& unit) { return WithSequenceHeadersWritten(unit, InitialDisplayDelaySequenceHeader); }

// A sequence header whose max_frame_height_minus_1 is `rows` less (in the same bits): everything else as it was
// A sequence header whose max_frame_width_minus_1 / max_frame_height_minus_1 are `columns` / `rows` less (in the same bits): everything else
// as it was
std::string SmallerFrameSequenceHeader(const std::string& payload, int columns, int rows) {
    const SequenceFrameSize size = FrameSizeOf(payload);
    std::string out = payload;
    const auto lessBy = [&](size_t at, int bits, int by) {
        if (by == 0) return;
        BitReader b{payload};
        b.pos = at;
        const uint32_t value = b.f(bits);  // max_frame_{width,height}_minus_1
        Need(by > 0 && value >= static_cast<uint32_t>(by), __LINE__);
        const uint32_t less = value - static_cast<uint32_t>(by);
        for (int i = 0; i < bits; ++i) {
            const size_t bit = at + static_cast<size_t>(i);
            const auto mask = static_cast<uint8_t>(0x80u >> (bit % 8));
            const bool one = (less >> (bits - 1 - i)) & 1u;
            const auto byte = static_cast<uint8_t>(out[bit / 8]);
            out[bit / 8] = static_cast<char>(one ? byte | mask : byte & static_cast<uint8_t>(~mask));
        }
    };
    Need(columns > 0 || rows > 0, __LINE__);
    lessBy(size.at, size.widthBits, columns);
    lessBy(size.at + static_cast<size_t>(size.widthBits), size.heightBits, rows);
    return out;
}

std::string ShorterFrameTemporalUnit(const std::string& unit, int rows) {
    return WithSequenceHeadersWritten(unit, [rows](const std::string& payload) { return SmallerFrameSequenceHeader(payload, 0, rows); });
}

std::string NarrowerFrameTemporalUnit(const std::string& unit, int columns) {
    return WithSequenceHeadersWritten(unit, [columns](const std::string& payload) { return SmallerFrameSequenceHeader(payload, columns, 0); });
}

void SetIvfFrameSize(const std::filesystem::path& ivf, uint16_t width, uint16_t height) {
    std::string f = ReadAll(ivf);
    Need(f.size() >= 32, __LINE__);
    Need(f.compare(0, 4, "DKIF") == 0, __LINE__);
    f[12] = static_cast<char>(width & 0xff);
    f[13] = static_cast<char>(width >> 8);
    f[14] = static_cast<char>(height & 0xff);
    f[15] = static_cast<char>(height >> 8);
    std::ofstream(ivf, std::ios::binary | std::ios::trunc) << f;
}

// The temporal unit without its sequence header OBUs (the frames after it need one met before)
std::string WithoutSequenceHeaders(const std::string& unit) {
    std::string out;
    for (size_t at = 0; at < unit.size();) {
        const uint8_t head = static_cast<uint8_t>(unit[at]);
        Need((head & 2), __LINE__);  // obu_has_size_field
        const size_t headSize = (head & 4) ? 2 : 1;
        uint64_t size = 0;
        size_t sizeBytes = 0;
        for (;; ++sizeBytes) {
            Need(at + headSize + sizeBytes < unit.size(), __LINE__);
            const uint8_t byte = static_cast<uint8_t>(unit[at + headSize + sizeBytes]);
            size |= static_cast<uint64_t>(byte & 0x7f) << (7 * sizeBytes);
            if (!(byte & 0x80)) {
                ++sizeBytes;
                break;
            }
        }
        const size_t whole = headSize + sizeBytes + static_cast<size_t>(size);
        if (((head >> 3) & 15) != 1) out += unit.substr(at, whole);  // OBU_SEQUENCE_HEADER dropped
        at += whole;
    }
    return out;
}

// An IVF file `copies` times one after the other in `out`, the timestamps going on; `each(index, unit)` writes each temporal unit anew
std::filesystem::path RepeatIvf(const std::filesystem::path& src, int copies, const std::filesystem::path& out,
                                const std::function<std::string(int, const std::string&)>& each) {
    const std::string f = ReadAll(src);
    Need(f.size() >= 32, __LINE__);
    Need(f.compare(0, 4, "DKIF") == 0, __LINE__);
    auto le = [&](size_t at, int bytes) {
        uint64_t v = 0;
        for (int i = bytes - 1; i >= 0; --i) v = v << 8 | static_cast<uint8_t>(f[at + i]);
        return v;
    };
    std::vector<std::pair<uint64_t, std::string>> frames;
    for (size_t at = le(6, 2); at + 12 <= f.size();) {
        const size_t size = static_cast<size_t>(le(at, 4));
        Need(at + 12 + size <= f.size(), __LINE__);
        frames.emplace_back(le(at + 4, 8), f.substr(at + 12, size));
        at += 12 + size;
    }
    Need(frames.size() >= 2, __LINE__);
    const uint64_t step = frames[1].first - frames[0].first;
    std::string ivf = f.substr(0, static_cast<size_t>(le(6, 2)));
    auto put = [](std::string& s, size_t at, uint64_t v, int bytes) {
        for (int i = 0; i < bytes; ++i) s[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    };
    int index = 0;
    for (int c = 0; c < copies; ++c)
        for (const auto& [pts, unit] : frames) {
            const std::string data = each(index, unit);
            std::string head(12, '\0');
            put(head, 0, data.size(), 4);
            put(head, 4, static_cast<uint64_t>(index) * step, 8);
            ivf += head + data;
            ++index;
        }
    put(ivf, 24, static_cast<uint64_t>(index), 4);
    std::ofstream(out, std::ios::binary) << ivf;
    return out;
}

}  // namespace dlssvid::test
