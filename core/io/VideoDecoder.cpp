#include "io/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/rational.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cctype>
#include <system_error>
#include <charconv>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#ifdef DLSSVID_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "io/NvdecFormat.h"
#include "io/NvdecSupport.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

// AVCodecContext::opaque of a hardware-only decoder (TASK-0076): get_format must not fall back to a software format it does not have
int kHardwareOnly = 1;

// TASK-0076: a hardware-only decoder refused in get_format - av1dec's own words that follow («Your platform doesn't support hardware
// accelerated AV1 decoding», «Failed to get pixel format») go to debug: the refusal is reported with its reason (CheckDecode), and they
// would contradict a start failure. nvdec.c's words before it (cuvidCreateDecoder failing) stay; the context is not fed again (refused_)
AVPixelFormat RefuseQuietly(AVCodecContext* ctx) {
    ctx->log_level_offset = AV_LOG_DEBUG - AV_LOG_ERROR;
    return AV_PIX_FMT_NONE;
}

AVPixelFormat GetHwFormat(AVCodecContext* ctx, const AVPixelFormat* formats) {
    const bool hardwareOnly = ctx->opaque == &kHardwareOnly;
    if (hardwareOnly && FakeNvdecRefusesFrames()) return RefuseQuietly(ctx);  // DLSSVID_FAKE_NVDEC_AV1=first-frame | query-fails | start-fails (tests)
    // the height as shown, else as coded: H.264 codes it aligned (191 as 192) and crops it - its last chroma row (4:2:0) would be lost on
    // the way to the host (FFmpeg copies height / 2 chroma rows); HEVC 4:4:4 (NvdecOddHeightSafe) is not judged
    const AVPixelFormat* last = formats;  // the software format, as FFmpeg's ff_get_format reads it
    while (*last != AV_PIX_FMT_NONE && *(last + 1) != AV_PIX_FMT_NONE) ++last;
    int height = NvdecJudgedHeight(ctx->height, ctx->coded_height);
    if (height % 2 == 1 && NvdecOddHeightSafe(ctx->codec_id, *last)) height = 0;
    const AVPixelFormat f = PickNvdecFormat(formats, hardwareOnly, height);
    if (hardwareOnly && f == AV_PIX_FMT_NONE) return RefuseQuietly(ctx);
    bool cudaOffered = false;  // a codec NVDEC does not decode (ProRes) is no odd height's fault
    for (const AVPixelFormat* p = formats; *p != AV_PIX_FMT_NONE; ++p) cudaOffered = cudaOffered || *p == AV_PIX_FMT_CUDA;
    if (f != AV_PIX_FMT_CUDA && !hardwareOnly) {
        if (height % 2 == 1 && cudaOffered)
            Log()->warn("a frame of an odd height ({}): FFmpeg's NVDEC gives its chroma wrong, decoding in software", height);
        else
            Log()->warn("NVDEC pixel format not offered by decoder, falling back to software");
    }
    return f;
}

AVBufferRef* CreateCudaDevice(std::string* why) {
    // Primary CUDA context: device pointers are then usable from cudart, TensorRT and the
    // Optical Flow API in this process without context juggling.
    // TASK-0076: one thread at a time - FFmpeg's cuda_context_init reads the primary context's state, sets its flags and retains it
    // without a lock, and the flags cannot be set on a context another thread has just started (two viewport loaders of two AV1 videos)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
#ifdef DLSSVID_WITH_CUDA
    // FFmpeg retains the primary context and insists on CU_CTX_SCHED_BLOCKING_SYNC; a stage that used the CUDA
    // runtime before this decoder (TensorRT depth, OFA) has already created it with the default scheduling, and
    // FFmpeg then refuses NVDEC ("Primary context already active with incompatible flags") and the stage decodes
    // in software. cudart applies the flags to the active primary context too: align them first - only then. An inactive one FFmpeg
    // sets itself, and cudaSetDeviceFlags would start it for the runtime and keep it to the process's end (TASK-0076: the GUI's viewport
    // decodes AV1 on NVDEC, and a run must get that memory back when the decoder closes).
    unsigned flags = 0;
    int active = 0;
    CUdevice dev = 0;
    // the same test as FFmpeg's cuda_context_init (libavutil/hwcontext_cuda.c): it refuses an active context whose flags are not exactly these
    if (cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&dev, 0) == CUDA_SUCCESS && cuDevicePrimaryCtxGetState(dev, &flags, &active) == CUDA_SUCCESS && active &&
        flags != CU_CTX_SCHED_BLOCKING_SYNC)
        if (const cudaError_t e = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); e != cudaSuccess)
            Log()->debug("cudaSetDeviceFlags(BlockingSync) before NVDEC: {}", cudaGetErrorString(e));
#endif
    AVBufferRef* device = nullptr;
    if (const int err = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT); err < 0) {
        if (why) *why = AvErrorToString(err);
        return nullptr;
    }
    return device;
}

// How a decoded AVFrame layout maps onto a CpuFrame without swscale: planar YUV at 8 / 10 / 12 / 16 bits and
// 4:2:0 / 4:2:2 / 4:4:4 (the alpha plane of yuva formats is dropped) and the 16-bit semi-planar layouts (P010, P016,
// P210, P216, P410, P416 — what NVDEC and some decoders hand out). `streamDepth` is the bit depth of the stream's own
// pixel format: a decoder that returns 16-bit words for a 10-bit stream (NVDEC's yuv444p16 for yuv444p10) keeps the
// samples MSB-aligned, and the layout shifts them down to the stream's depth so NVDEC and software decoding agree.
struct NativeLayout {
    FrameDesc desc;      // width / height left at 0
    int shiftRight = 0;  // MSB-aligned 16-bit words -> LSB-aligned `desc.bitDepth` bits
    bool semiPlanar = false;
};

std::optional<NativeLayout> NativeLayoutFor(AVPixelFormat fmt, int streamDepth) {
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(fmt);
    if (!d || d->nb_components < 3) return std::nullopt;
    if (d->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BE |
                    AV_PIX_FMT_FLAG_BAYER))
        return std::nullopt;
    if (!(d->flags & AV_PIX_FMT_FLAG_PLANAR)) return std::nullopt;
    if (d->log2_chroma_w > 1 || d->log2_chroma_h > 1 || (d->log2_chroma_h == 1 && d->log2_chroma_w == 0)) return std::nullopt;  // 4:4:0, 4:1:1 ...
    const int depth = d->comp[0].depth, shift = d->comp[0].shift, step = d->comp[0].step;
    for (int c = 1; c < 3; ++c)
        if (d->comp[c].depth != depth || d->comp[c].shift != shift) return std::nullopt;
    const bool semi = d->comp[1].plane == 1 && d->comp[2].plane == 1;
    const bool planar = d->comp[0].plane == 0 && d->comp[1].plane == 1 && d->comp[2].plane == 2;
    if (!semi && !planar) return std::nullopt;
    const std::string chroma = d->log2_chroma_w == 0 ? "444" : (d->log2_chroma_h == 0 ? "422" : "420");
    if (depth == 8) {
        if (semi || shift != 0 || step != 1) return std::nullopt;  // NV12 has its own bit-exact path, NV16 / NV24 go through swscale
        return NativeLayout{FrameDesc{0, 0, PixelFormatFor(chroma, 8), 8}, 0, false};
    }
    if (step != 2 || depth < 9 || depth > 16) return std::nullopt;
    int bitDepth = depth, shiftRight = shift;  // P010: depth 10, shift 6 (the descriptor says how far above the LSB the value sits)
    if (streamDepth > 8 && streamDepth < depth) {  // 16-bit words for a 10 / 12-bit stream: MSB-aligned
        shiftRight += depth - streamDepth;
        bitDepth = streamDepth;
    }
    if (bitDepth != 10 && bitDepth != 12 && bitDepth != 16) return std::nullopt;  // 9 / 14-bit: swscale
    return NativeLayout{FrameDesc{0, 0, PixelFormatFor(chroma, bitDepth), static_cast<uint8_t>(bitDepth)}, shiftRight, semi};
}

}  // namespace

std::string_view Av1SequenceHeader(const uint8_t* data, size_t size) {
    if (!data) return {};
    if (size >= 4 && data[0] == 0x81) {  // av1C: marker 1, version 1 (an OBU header's top bit is 0)
        data += 4;
        size -= 4;
    }
    std::string_view found;
    size_t pos = 0;
    while (pos < size) {
        const uint8_t h = data[pos];
        if (h & 0x80) return {};  // obu_forbidden_bit
        const int type = (h >> 3) & 0x0F;
        const bool extension = (h >> 2) & 1, hasSize = (h >> 1) & 1;
        size_t p = pos + 1 + (extension ? 1 : 0);
        if (p > size) return {};
        uint64_t length = 0;
        if (hasSize) {  // leb128
            bool ended = false;
            for (int i = 0; i < 8 && p < size; ++i) {
                const uint8_t b = data[p++];
                length |= static_cast<uint64_t>(b & 0x7F) << (7 * i);
                if (!(b & 0x80)) {
                    ended = true;
                    break;
                }
            }
            if (!ended) return {};
        } else {
            length = size - p;  // the last OBU
        }
        if (length > size - p) return {};
        if (type == 1) found = std::string_view(reinterpret_cast<const char*>(data + p), static_cast<size_t>(length));  // OBU_SEQUENCE_HEADER
        pos = p + static_cast<size_t>(length);
    }
    return found;
}

std::optional<Av1SequenceKey> Av1SequenceKeyOf(std::string_view header) {
    size_t bit = 0;
    bool ok = true;
    const auto f = [&](int n) -> uint32_t {  // f(n) of the AV1 specification, most significant bit first
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) {
            if (bit >= header.size() * 8) {
                ok = false;
                return 0;
            }
            v = (v << 1) | ((static_cast<uint8_t>(header[bit / 8]) >> (7 - bit % 8)) & 1u);
            ++bit;
        }
        return v;
    };
    const auto uvlc = [&] {
        int zeros = 0;
        while (ok && !f(1))
            if (++zeros >= 32) ok = false;
        if (ok && zeros) f(zeros);
    };
    Av1SequenceKey k;
    k.profile = f(3);
    f(1);  // still_picture
    const bool reduced = f(1);
    bool decoderModel = false, initialDisplay = false;
    int bufferDelayLength = 0;
    if (reduced) {
        f(5);  // seq_level_idx[0]
    } else {
        if (f(1)) {  // timing_info_present_flag
            f(32);
            f(32);
            if (f(1)) uvlc();  // equal_picture_interval: num_ticks_per_picture_minus_1
            decoderModel = f(1);
            if (decoderModel) {
                bufferDelayLength = static_cast<int>(f(5)) + 1;
                f(32);
                f(5);
                f(5);
            }
        }
        initialDisplay = f(1);
        const uint32_t points = f(5) + 1;
        for (uint32_t i = 0; ok && i < points; ++i) {
            f(12);                 // operating_point_idc
            if (f(5) > 7) f(1);    // seq_level_idx, seq_tier
            if (decoderModel && f(1)) {  // operating_parameters_info - the part a repeat may change
                f(bufferDelayLength);
                f(bufferDelayLength);
                f(1);
            }
            if (initialDisplay && f(1)) f(4);
        }
    }
    const int widthBits = static_cast<int>(f(4)) + 1, heightBits = static_cast<int>(f(4)) + 1;
    k.maxWidth = f(widthBits) + 1;
    k.maxHeight = f(heightBits) + 1;
    if (!reduced && f(1)) {  // frame_id_numbers_present_flag
        f(4);
        f(3);
    }
    f(1);  // use_128x128_superblock
    f(1);  // enable_filter_intra
    f(1);  // enable_intra_edge_filter
    if (!reduced) {
        f(4);  // enable_interintra_compound, enable_masked_compound, enable_warped_motion, enable_dual_filter
        const bool orderHint = f(1);
        if (orderHint) f(2);  // enable_jnt_comp, enable_ref_frame_mvs
        const uint32_t screenContent = f(1) ? 2 : f(1);  // seq_choose_screen_content_tools, else seq_force_screen_content_tools
        if (screenContent > 0 && !f(1)) f(1);            // seq_choose_integer_mv, else seq_force_integer_mv
        if (orderHint) f(3);                              // order_hint_bits_minus_1
    }
    f(3);  // enable_superres, enable_cdef, enable_restoration
    // color_config
    const bool high = f(1);
    k.bitDepth = k.profile == 2 && high ? (f(1) ? 12 : 10) : high ? 10 : 8;
    k.mono = k.profile == 1 ? false : f(1);
    uint32_t primaries = 2, transfer = 2, matrix = 2;
    if (f(1)) {  // color_description_present_flag
        primaries = f(8);
        transfer = f(8);
        matrix = f(8);
    }
    if (k.mono) {
        k.subsamplingX = k.subsamplingY = 1;
    } else if (primaries == 1 && transfer == 13 && matrix == 0) {  // CP_BT_709, TC_SRGB, MC_IDENTITY
        k.rgb = true;
        k.subsamplingX = k.subsamplingY = 0;
    } else {
        f(1);  // color_range
        if (k.profile == 0) {
            k.subsamplingX = k.subsamplingY = 1;
        } else if (k.profile == 1) {
            k.subsamplingX = k.subsamplingY = 0;
        } else if (k.bitDepth == 12) {
            k.subsamplingX = f(1);
            k.subsamplingY = k.subsamplingX ? f(1) : 0;
        } else {
            k.subsamplingX = 1;
            k.subsamplingY = 0;
        }
    }
    if (!ok) return std::nullopt;
    return k;
}

std::string LayoutKindOf(const std::string& pixelFormat) {
    const AVPixFmtDescriptor* d = pixelFormat.empty() ? nullptr : av_pix_fmt_desc_get(av_get_pix_fmt(pixelFormat.c_str()));
    if (!d) return "other";
    if (d->flags & AV_PIX_FMT_FLAG_PAL) return "palette";  // pal8: one component, a colour palette
    if (d->flags & AV_PIX_FMT_FLAG_RGB) return "rgb";
    if (d->nb_components < 3) return "gray";
    // a YUV: its J:a:b ("420"; for a chroma "other" - 4:1:1 DV NTSC, 4:4:0 some MJPEG, 4:1:0)
    const int a = 4 >> d->log2_chroma_w;
    const int b = d->log2_chroma_h == 0 ? a : 0;
    if (a >= 1 && d->log2_chroma_h <= 2) return "4" + std::to_string(a) + std::to_string(b);
    return "other";
}

bool IsProbeVerdict(const DecodeSupport& s) {
    if (s.part || s.pixelFormat == "unknown") return false;
    if (const auto met = StreamPartOf(s.why); met && met->learned) return false;
    switch (s.block) {
        case DecodeBlock::Layout:
        case DecodeBlock::NoCuda:
        case DecodeBlock::NoNvdec:
        case DecodeBlock::NoAv1:
        case DecodeBlock::Size:
        case DecodeBlock::OddHeight: return true;
        default: return false;
    }
}

DecodeSupport VerdictFromWhy(const VideoStreamInfo& probe, const std::string& why, const std::string& where) {
    if (const auto part = StreamPartOf(why)) {
        DecodeSupport judged = CheckDecodeSupport(StreamPartInfo(probe, part->pixelFormat, part->codedWidth, part->codedHeight));
        if (!judged.Ok() && !judged.Transient()) {
            judged.why = where + ": " + why;
            judged.part = !part->learned;
            return judged;
        }
    }
    // the words RefusalText writes for a block (and a size): `why` is them, or them with « (<detail>)»
    const auto detailOf = [&](const std::string& words) -> std::optional<std::string> {
        if (why == words) return std::string();
        if (why.size() > words.size() + 3 && why.rfind(words + " (", 0) == 0 && why.back() == ')')
            return why.substr(words.size() + 2, why.size() - words.size() - 3);
        return std::nullopt;
    };
    const auto withWhere = [&](const std::string& detail) { return detail.empty() ? where : where + ": " + detail; };
    for (size_t at = why.find('x'); at != std::string::npos; at = why.find('x', at + 1)) {  // the first «<W>x<H>» names the size
        size_t from = at;
        while (from > 0 && std::isdigit(static_cast<unsigned char>(why[from - 1]))) --from;
        size_t to = at + 1;
        while (to < why.size() && std::isdigit(static_cast<unsigned char>(why[to]))) ++to;
        uint32_t w = 0, h = 0;
        if (from == at || to == at + 1 || std::from_chars(why.data() + from, why.data() + at, w).ec != std::errc() ||
            std::from_chars(why.data() + at + 1, why.data() + to, h).ec != std::errc())
            continue;
        const VideoStreamInfo sized = StreamPartInfo(probe, "", w, h);
        for (const DecodeBlock block : {DecodeBlock::Resized, DecodeBlock::OddHeight})
            if (const auto detail = detailOf(RefusedDecode(sized, block, "").why)) {
                DecodeSupport s = RefusedDecode(sized, block, withWhere(*detail));
                // an odd height where the probe's is even: a part (as VideoDecoder::ReceiveFrame marks it)
                s.part = block == DecodeBlock::OddHeight && probe.pixelFormat != "unknown" && probe.codedHeight % 2 == 0 && probe.codedHeight > 0;
                return s;
            }
        break;
    }
    if (const auto detail = detailOf(RefusedDecode(probe, DecodeBlock::NvdecFailed, "").why))
        return RefusedDecode(probe, DecodeBlock::NvdecFailed, withWhere(*detail));
    return RefusedDecode(probe, DecodeBlock::NvdecFailed, where + ": " + why);
}

bool Av1FrameBelowSequence(int width, int height, std::string_view sequenceHeader) {
    const auto key = Av1SequenceKeyOf(sequenceHeader);
    return key && width > 0 && height > 0 && (static_cast<uint32_t>(width) < key->maxWidth || static_cast<uint32_t>(height) < key->maxHeight);
}

namespace {

// "420" | "422" | "444" | "other" (RGB, grey) - the VideoStreamInfo::chroma of a pixel format
std::string ChromaOf(const AVPixFmtDescriptor* d) {
    return (d->flags & AV_PIX_FMT_FLAG_RGB) || d->nb_components < 3 ? "other"
           : d->log2_chroma_w == 0 && d->log2_chroma_h == 0        ? "444"
           : d->log2_chroma_w == 1 && d->log2_chroma_h == 0        ? "422"
           : d->log2_chroma_w == 1 && d->log2_chroma_h == 1        ? "420"
                                                                   : "other";
}

}  // namespace

VideoStreamInfo StreamPartInfo(const VideoStreamInfo& info, const std::string& pixelFormat, uint32_t codedWidth, uint32_t codedHeight) {
    VideoStreamInfo part = info;
    const AVPixelFormat fmt = pixelFormat.empty() ? AV_PIX_FMT_NONE : av_get_pix_fmt(pixelFormat.c_str());
    if (const AVPixFmtDescriptor* d = fmt != AV_PIX_FMT_NONE ? av_pix_fmt_desc_get(fmt) : nullptr) {
        part.pixelFormat = pixelFormat;
        part.bitDepth = d->comp[0].depth;
        part.chroma = ChromaOf(d);
    }
    if (codedWidth > 0) part.codedWidth = codedWidth;
    if (codedHeight > 0) part.codedHeight = codedHeight;
    return part;
}

std::string StreamPartWords(const StreamPart& part) {
    if (part.learned)
        return ", in " + part.pixelFormat +
               (part.codedWidth > 0 && part.codedHeight > 0 ? " " + std::to_string(part.codedWidth) + "x" + std::to_string(part.codedHeight) : std::string()) +
               " - a layout the probe did not learn";
    return ", on a part of the stream in other parameters (" + part.pixelFormat + " " + std::to_string(part.codedWidth) + "x" + std::to_string(part.codedHeight) +
           ")";
}

std::optional<StreamPart> StreamPartOf(const std::string& why) {
    static const std::string partHead = ", on a part of the stream in other parameters (";
    static const std::string learnedTail = " - a layout the probe did not learn";
    const size_t partAt = why.rfind(partHead);
    const size_t learnedAt = why.rfind(learnedTail);
    if (partAt != std::string::npos && (learnedAt == std::string::npos || partAt > learnedAt)) {
        const size_t from = partAt + partHead.size();
        const size_t space = why.find(' ', from), close = why.find(')', from);
        if (space == std::string::npos || close == std::string::npos || space > close) return std::nullopt;
        const std::string size = why.substr(space + 1, close - space - 1);
        const size_t x = size.find('x');
        if (x == std::string::npos) return std::nullopt;
        StreamPart part;
        part.pixelFormat = why.substr(from, space - from);
        const auto parse = [](const std::string& s, uint32_t& out) {
            return !s.empty() && std::from_chars(s.data(), s.data() + s.size(), out).ptr == s.data() + s.size();
        };
        if (!parse(size.substr(0, x), part.codedWidth) || !parse(size.substr(x + 1), part.codedHeight)) return std::nullopt;
        return part;
    }
    if (learnedAt != std::string::npos) {
        const size_t in = why.rfind(", in ", learnedAt);
        if (in == std::string::npos || learnedAt < in + 5) return std::nullopt;
        StreamPart part;
        part.learned = true;
        const std::string words = why.substr(in + 5, learnedAt - in - 5);  // «<fmt>» or «<fmt> <w>x<h>»
        const size_t space = words.find(' ');
        part.pixelFormat = words.substr(0, space);
        if (part.pixelFormat.empty()) return std::nullopt;
        if (space != std::string::npos) {
            const std::string size = words.substr(space + 1);
            const size_t x = size.find('x');
            const auto parse = [](const std::string& s, uint32_t& out) {
                return !s.empty() && std::from_chars(s.data(), s.data() + s.size(), out).ptr == s.data() + s.size();
            };
            if (x == std::string::npos || !parse(size.substr(0, x), part.codedWidth) || !parse(size.substr(x + 1), part.codedHeight)) return std::nullopt;
        }
        return part;
    }
    return std::nullopt;
}

FrameDesc DecodedFrameDescFor(const std::string& pixelFormat) {
    const AVPixelFormat fmt = av_get_pix_fmt(pixelFormat.c_str());
    if (fmt != AV_PIX_FMT_NONE) {
        if (fmt == AV_PIX_FMT_NV12) return FrameDesc{0, 0, PixelFormat::Yuv420p, 8};
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(fmt);
        if (const auto lay = NativeLayoutFor(fmt, d ? d->comp[0].depth : 8)) return lay->desc;
    }
    return FrameDesc{0, 0, PixelFormat::Yuv420p, 8};
}

FrameDesc SourceLayout(const VideoStreamInfo& info) {
    FrameDesc d = DecodedFrameDescFor(info.pixelFormat);
    d.width = info.width;
    d.height = info.height;
    return d;
}

DisplayOrientation OrientationFromDisplayMatrix(const int32_t m[9], std::string* why) {
    // fftools/cmdutils.c get_rotation and ffmpeg_filter.c (7.1): the angle the frame is turned clockwise for display, then by its sign
    // bits the transpose / flip filters the command line inserts
    const double got = av_display_rotation_get(m);
    if (std::isnan(got)) {
        if (why) *why = "a degenerate display matrix";
        return {};
    }
    double theta = -std::round(got);
    theta -= 360.0 * std::floor(theta / 360.0 + 0.9 / 360.0);
    if (std::fabs(theta - 90.0) < 1.0) return m[3] > 0 ? DisplayOrientation{90, true} : DisplayOrientation{90, false};  // cclock_flip | clock
    if (std::fabs(theta - 180.0) < 1.0) {
        const bool h = m[0] < 0, v = m[4] < 0;  // hflip, vflip
        if (h && v) return {180, false};
        if (h) return {0, true};
        if (v) return {180, true};
        return {};
    }
    if (std::fabs(theta - 270.0) < 1.0) return m[3] < 0 ? DisplayOrientation{270, true} : DisplayOrientation{270, false};  // clock_flip | cclock
    if (std::fabs(theta) < 1.0) return m[4] < 0 ? DisplayOrientation{180, true} : DisplayOrientation{};  // vflip
    if (why) *why = "a rotation by " + std::to_string(static_cast<int>(theta)) + " deg (not a multiple of 90) is not applied";
    return {};
}

DisplayOrientation OrientationFromRotateTag(std::string_view tag, std::string* why) {
    long long deg = 0;
    const auto [end, ec] = std::from_chars(tag.data(), tag.data() + tag.size(), deg);
    if (ec != std::errc() || end != tag.data() + tag.size()) {
        if (why) *why = "the rotate tag '" + std::string(tag) + "' is not a whole number of degrees: not applied";
        return {};
    }
    if (deg % 90 != 0) {
        if (why) *why = "the rotate tag '" + std::string(tag) + "' is not a multiple of 90 deg: not applied";
        return {};
    }
    return {static_cast<uint16_t>(((deg % 360) + 360) % 360), false};
}

std::map<std::string, std::string> ReadContainerTags(const std::filesystem::path& path) {
    AVFormatContext* fmt = nullptr;
    const std::string url = path.string();
    CheckAv(avformat_open_input(&fmt, url.c_str(), nullptr, nullptr), "avformat_open_input");
    std::map<std::string, std::string> tags;
    const AVDictionaryEntry* e = nullptr;
    while ((e = av_dict_iterate(fmt->metadata, e))) {
        std::string key = e->key;  // Matroska stores COMMENT, mp4 / mov comment: one spelling for the callers
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        tags[key] = e->value;
    }
    avformat_close_input(&fmt);
    return tags;
}

KeyframeScan ScanKeyframes(const std::filesystem::path& path) {
    KeyframeScan scan;
    AVFormatContext* fmt = nullptr;
    const std::string url = path.string();
    CheckAv(avformat_open_input(&fmt, url.c_str(), nullptr, nullptr), "avformat_open_input");
    AVPacket* pkt = av_packet_alloc();
    try {
        CheckAv(avformat_find_stream_info(fmt, nullptr), "avformat_find_stream_info");
        const int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vs < 0) Throw("no video stream in " + url);
        for (unsigned i = 0; i < fmt->nb_streams; ++i)
            if (static_cast<int>(i) != vs) fmt->streams[i]->discard = AVDISCARD_ALL;  // video packets only
        int64_t sinceKey = 0;
        bool anyKey = false;
        while (av_read_frame(fmt, pkt) >= 0) {
            if (pkt->stream_index == vs) {
                ++scan.frames;
                if (pkt->flags & AV_PKT_FLAG_KEY) {
                    if (anyKey) scan.maxInterval = std::max(scan.maxInterval, sinceKey);
                    ++scan.keyframes;
                    anyKey = true;
                    sinceKey = 0;
                }
                ++sinceKey;
            }
            av_packet_unref(pkt);
        }
        scan.maxInterval = std::max(scan.maxInterval, anyKey ? sinceKey : scan.frames);  // the tail after the last keyframe
    } catch (...) {
        av_packet_free(&pkt);
        avformat_close_input(&fmt);
        throw;
    }
    av_packet_free(&pkt);
    avformat_close_input(&fmt);
    return scan;
}

int64_t PtsToIndex(int64_t pts, int64_t startTime, Rational timeBase, Rational rate) {
    const double seconds = static_cast<double>(pts - startTime) * timeBase.num / timeBase.den;
    return static_cast<int64_t>(std::llround(seconds * rate.num / rate.den));
}

std::string FrameIndexScan::Why() const {
    if (noRate) return "the stream has no frame rate";
    if (firstBad < 0) return "every frame at its time";
    if (badIndex < 0) return "frame " + std::to_string(firstBad) + " has no timestamp";
    return "frame " + std::to_string(firstBad) + " carries the time of frame " + std::to_string(badIndex);
}

FrameIndexScan ScanFrameIndex(const std::filesystem::path& path, int64_t upTo, Rational indexRate) {
    FrameIndexScan scan;
    AVFormatContext* fmt = nullptr;
    const std::string url = path.string();
    CheckAv(avformat_open_input(&fmt, url.c_str(), nullptr, nullptr), "avformat_open_input");
    AVPacket* pkt = av_packet_alloc();
    try {
        CheckAv(avformat_find_stream_info(fmt, nullptr), "avformat_find_stream_info");
        const int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (vs < 0) Throw("no video stream in " + url);
        for (unsigned i = 0; i < fmt->nb_streams; ++i)
            if (static_cast<int>(i) != vs) fmt->streams[i]->discard = AVDISCARD_ALL;  // video packets only
        AVStream* st = fmt->streams[vs];
        const AVRational guessed = av_guess_frame_rate(fmt, st, nullptr);
        const Rational rate = indexRate.num > 0 && indexRate.den > 0 ? indexRate : Rational{guessed.num, guessed.den};
        const Rational tb{st->time_base.num, st->time_base.den};
        const int64_t start = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
        // the reordering of B-frames moves a frame by a few packets: 64 more than needed put every frame up to upTo in its place
        const int64_t wanted = upTo >= std::numeric_limits<int64_t>::max() - 65 ? std::numeric_limits<int64_t>::max() : std::max<int64_t>(upTo, 0) + 1 + 64;
        std::vector<int64_t> times;
        int64_t firstMissing = -1;  // the first packet with neither pts nor dts
        while (scan.frames < wanted && av_read_frame(fmt, pkt) >= 0) {  // every video packet counts: one without a time adds none
            if (pkt->stream_index == vs && !(pkt->flags & AV_PKT_FLAG_DISCARD)) {
                const int64_t t = pkt->pts != AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
                if (t == AV_NOPTS_VALUE) {
                    firstMissing = scan.frames++;  // firstBad from here on whatever follows: a raw stream is not read to its end
                    av_packet_unref(pkt);
                    break;
                }
                times.push_back(t);
                ++scan.frames;
            }
            av_packet_unref(pkt);
        }
        std::sort(times.begin(), times.end());
        if (rate.num <= 0 || tb.num <= 0) {
            scan.firstBad = 0;  // no rate: no index can be taken from pts
            scan.noRate = true;
        } else if (firstMissing >= 0) {
            scan.firstBad = firstMissing;  // a frame without a time: the ones after it cannot be found by theirs
        } else {
            const int64_t last = std::min<int64_t>(upTo, static_cast<int64_t>(times.size()) - 1);
            for (int64_t k = 0; k <= last; ++k) {
                const int64_t named = PtsToIndex(times[static_cast<size_t>(k)], start, tb, rate);
                if (named != k) {
                    scan.firstBad = k;
                    scan.badIndex = named;
                    break;
                }
            }
        }
    } catch (...) {
        av_packet_free(&pkt);
        avformat_close_input(&fmt);
        throw;
    }
    av_packet_free(&pkt);
    avformat_close_input(&fmt);
    return scan;
}

VideoDecoder::VideoDecoder(const std::filesystem::path& path, const Options& options) {
    url_ = path.string();
    const std::string& url = url_;
    CheckAv(avformat_open_input(&fmt_, url.c_str(), nullptr, nullptr), "avformat_open_input");
    CheckAv(avformat_find_stream_info(fmt_, nullptr), "avformat_find_stream_info");

    const AVCodec* decoder = nullptr;
    videoStream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (videoStream_ < 0) Throw("no video stream in " + url);
    audioStream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStream_ < 0) audioStream_ = -1;

    AVStream* vs = fmt_->streams[videoStream_];
    decoder_ = decoder;
    hwOnly_ = HardwareOnlyDecoder(decoder);
    probe_ = options.probe;
    sliceThreadsOnly_ = options.sliceThreadsOnly;
    if (options.hwaccel == HwAccel::Cuda && !hwOnly_ && vs->codecpar->height % 2 == 1 && HasNvdecHwaccel(decoder) &&
        !NvdecOddHeightSafe(decoder->id, static_cast<AVPixelFormat>(vs->codecpar->format))) {
        // TASK-0076: an odd height (VP9 1920x803, H.264 cropped to 191 rows): FFmpeg's NVDEC gives its chroma wrong - software from the
        // open (with its threads; no CUDA device held), not from get_format (one thread). GetHwFormat still guards a size met later
        // (HEVC 4:4:4 of an odd height stays on NVDEC: NvdecOddHeightSafe)
        Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info,
                   "{}: an odd height ({}): FFmpeg's NVDEC gives its chroma wrong, decoding in software", url, vs->codecpar->height);
    } else if (options.hwaccel == HwAccel::Cuda) {
        std::string why;
        hwDevice_ = hwOnly_ && FakeNvdecDeviceFails() ? nullptr : CreateCudaDevice(&why);  // (device-fails: as StartNvdec would meet it)
        // a hardware-only decoder has no software to fall back to: StartNvdec tries again and refuses with the reason
        if (!hwDevice_ && !hwOnly_) Log()->warn("CUDA hwaccel unavailable ({}), decoding in software", why);
    }
    gpuFrames_ = hwDevice_ != nullptr;
    indexRate_ = options.indexRate;
    codec_ = OpenCodec(hwDevice_);
    if (hwOnly_) nvdecSequence_.assign(Av1SequenceHeader(vs->codecpar->extradata, static_cast<size_t>(std::max(vs->codecpar->extradata_size, 0))));

    frame_ = av_frame_alloc();
    swFrame_ = av_frame_alloc();
    hwKeep_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !swFrame_ || !hwKeep_ || !packet_) Throw("av_frame_alloc/av_packet_alloc failed");

    info_.codedWidth = static_cast<uint32_t>(codec_->width);
    info_.codedHeight = static_cast<uint32_t>(codec_->height);
    // TASK-0072: how players show the stream - the container's display matrix (MP4 / MOV tkhd, Matroska's projection), else the
    // 'rotate' tag of older muxers; the frames are turned so (OrientFrame), and every size below is the shown one
    std::string orientationWhy;
    if (const AVPacketSideData* sd = av_packet_side_data_get(vs->codecpar->coded_side_data, vs->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
        sd && sd->size >= 9 * sizeof(int32_t)) {
        info_.orientation = OrientationFromDisplayMatrix(reinterpret_cast<const int32_t*>(sd->data), &orientationWhy);
        if (!info_.orientation.Identity()) info_.orientationFrom = "display matrix";
    } else if (const AVDictionaryEntry* tag = av_dict_get(vs->metadata, "rotate", nullptr, 0)) {
        info_.orientation = OrientationFromRotateTag(tag->value, &orientationWhy);
        if (!info_.orientation.Identity()) info_.orientationFrom = "rotate tag";
    }
    if (!orientationWhy.empty())  // once per real open: a probe (the plan, the viewport, the result check) says it at debug level
        Log()->log(options.probe ? spdlog::level::debug : spdlog::level::warn, "{}: {}", url, orientationWhy);
    info_.width = info_.orientation.SwapsAxes() ? info_.codedHeight : info_.codedWidth;
    info_.height = info_.orientation.SwapsAxes() ? info_.codedWidth : info_.codedHeight;
    // TASK-0073: the pixels' proportions - the container's (MP4 pasp, MKV DisplayWidth, AVI vprp, DV), else the bitstream's (VUI, the
    // sequence header); in the stored grid's axes, inverted for the turned frame by 90 / 270
    {
        AVRational guess = av_guess_sample_aspect_ratio(fmt_, vs, nullptr);
        if (guess.num > 0 && guess.den > 0) av_reduce(&guess.num, &guess.den, guess.num, guess.den, 65535);
        std::string sarWhy;
        info_.codedSar = PixelAspectFrom(guess.num, guess.den, &sarWhy);
        if (!sarWhy.empty()) Log()->log(options.probe ? spdlog::level::debug : spdlog::level::warn, "{}: {}", url, sarWhy);
        if (!info_.codedSar.Square() && !info_.codedSar.MovesAPixel(info_.codedWidth, info_.codedHeight)) {  // the same picture as square
            Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info, "{}: a pixel aspect of {} moves no pixel of {}x{}: the pixels are taken as square",
                       url, info_.codedSar.Token(), info_.codedWidth, info_.codedHeight);
            info_.codedSar = {};
        }
        // which one decided: the codec's when it says the same (a raw stream: no container field, the guess falls back to the codec
        // parameters; MP4's pasp and the VUI agreeing), else the container's (MKV DisplayWidth over FFV1, MOV pasp over ProRes)
        if (!info_.codedSar.Square())
            info_.sarFrom = PixelAspectFrom(vs->codecpar->sample_aspect_ratio.num, vs->codecpar->sample_aspect_ratio.den) == info_.codedSar ? "bitstream" : "container";
        info_.sar = info_.orientation.SwapsAxes() ? info_.codedSar.Inverse() : info_.codedSar;
    }
    const AVRational fr = av_guess_frame_rate(fmt_, vs, nullptr);
    info_.frameRate = Rational{fr.num, fr.den};
    info_.timeBase = Rational{vs->time_base.num, vs->time_base.den};
    info_.frameCount = vs->nb_frames;
    info_.durationUs = fmt_->duration > 0 ? fmt_->duration : 0;
    info_.codecName = decoder->name;
    const auto streamFmt = static_cast<AVPixelFormat>(vs->codecpar->format);
    const char* pixName = av_get_pix_fmt_name(streamFmt);
    info_.pixelFormat = pixName ? pixName : "unknown";
    if (const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(streamFmt)) {
        info_.bitDepth = d->comp[0].depth;
        info_.chroma = ChromaOf(d);
    }
    info_.decoded = DecodedFrameDescFor(info_.pixelFormat);
    info_.decoded.width = info_.codedWidth;
    info_.decoded.height = info_.codedHeight;
    info_.decoded = OrientedDesc(info_.decoded, info_.orientation);
    info_.colorSpace = static_cast<int>(vs->codecpar->color_space);
    info_.colorRange = static_cast<int>(vs->codecpar->color_range);
    info_.colorPrimaries = static_cast<int>(vs->codecpar->color_primaries);
    info_.colorTrc = static_cast<int>(vs->codecpar->color_trc);
    info_.hasAudio = audioStream_ >= 0;
    if (info_.hasAudio) {
        const AVCodecDescriptor* d = avcodec_descriptor_get(fmt_->streams[audioStream_]->codecpar->codec_id);
        info_.audioCodecName = d ? d->name : "unknown";
    }
    info_.hardwareOnly = hwOnly_;
    info_.nvdecCodec = HasNvdecHwaccel(decoder);

    Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info, "open {}: {}x{} {} {} ({}) {}/{} fps, {} frames{}{}", url, info_.width, info_.height,
               info_.codecName, info_.pixelFormat, info_.decoded.Describe(), fr.num, fr.den, info_.frameCount,
               info_.hasAudio ? ", audio " + info_.audioCodecName : "",
               hwDevice_ ? ", NVDEC" : hwOnly_ ? ", NVDEC only (no software decoder)" : "");
    if (!info_.orientation.Identity())
        Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info, "open {}: stored {}x{}, shown {} ({}): the frames are turned so", url,
                   info_.codedWidth, info_.codedHeight, info_.orientation.Describe(), info_.orientationFrom);
    if (!info_.sar.Square()) {
        const auto [dw, dh] = info_.sar.DisplaySize(info_.width, info_.height);
        Log()->log(options.probe ? spdlog::level::debug : spdlog::level::info, "open {}: pixels {} ({}), shown {}x{}", url, info_.codedSar.Token(),
                   info_.sarFrom, dw, dh);
    }
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

AVCodecContext* VideoDecoder::OpenCodec(AVBufferRef* device) const {
    AVStream* vs = fmt_->streams[videoStream_];
    AVCodecContext* c = avcodec_alloc_context3(decoder_);
    if (!c) Throw("avcodec_alloc_context3 failed");
    try {
        CheckAv(avcodec_parameters_to_context(c, vs->codecpar), "avcodec_parameters_to_context");
        c->pkt_timebase = vs->time_base;
        c->opaque = hwOnly_ ? &kHardwareOnly : nullptr;
        if (device) {
            c->hw_device_ctx = av_buffer_ref(device);
            c->get_format = GetHwFormat;
        } else {
            // software decoding: frame + slice threads, count chosen by libavcodec (viewport scrubbing at 1080p); an intra
            // proxy: slice threads only (a seek then costs one decode)
            c->thread_count = 0;
            c->thread_type = sliceThreadsOnly_ ? FF_THREAD_SLICE : FF_THREAD_FRAME | FF_THREAD_SLICE;
        }
        CheckAv(avcodec_open2(c, decoder_, nullptr), "avcodec_open2");
    } catch (...) {
        avcodec_free_context(&c);
        throw;
    }
    return c;
}

void VideoDecoder::RequireDecodable() const {
    if (!hwOnly_) return;
    if (refused_) throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
    if (const DecodeSupport s = CheckDecodeSupport(info_); !s.Ok()) throw DecodeUnavailable(s, url_ + ": " + s.why);
}

void VideoDecoder::StartNvdec() {
    // TASK-0076: before the first packet only (NextFrame is the one place packets are sent; a seek before it only flushed codec_ and
    // moved the demuxer, which stays where it is). Nothing changes until it all worked: a refusal leaves the decoder as it was
    const DecodeSupport support = CheckDecodeSupport(info_);
    if (!support.Ok()) throw DecodeUnavailable(support, url_ + ": " + support.why);
    if (!hwDevice_) {
        std::string why = "DLSSVID_FAKE_NVDEC_AV1=device-fails";
        AVBufferRef* device = FakeNvdecDeviceFails() ? nullptr : CreateCudaDevice(&why);
        if (!device) {  // CUDA was found (CheckDecodeSupport): a device that is not created now is no verdict (memory, the driver)
            const DecodeSupport s = RefusedDecode(info_, DecodeBlock::NvdecStartFailed, "the CUDA device could not be created: " + why);
            throw DecodeUnavailable(s, url_ + ": " + s.why);
        }
        AVCodecContext* reopened = nullptr;
        try {
            reopened = OpenCodec(device);
        } catch (...) {
            av_buffer_unref(&device);
            throw;
        }
        avcodec_free_context(&codec_);
        codec_ = reopened;
        hwDevice_ = device;
        instanceFrames_ = 0;
    }
    nvdecStarted_ = true;
    nvdecStartTransient_ = support.verified || (!support.caps.checked && support.caps.cudaError);
    Log()->log(probe_ ? spdlog::level::debug : spdlog::level::info, "{}: {} decodes on NVDEC only (no software decoder in this build), {}", url_, info_.codecName,
               support.why);
}

void VideoDecoder::FollowSequenceHeader() {
    const std::string_view sequence = Av1SequenceHeader(packet_->data, static_cast<size_t>(std::max(packet_->size, 0)));
    if (sequence.empty()) return;
    // the same key (size, depth, chroma): the decoder NVDEC made goes on - av1dec gives it the new header's fields frame by frame, as it
    // does without this (a repeat that differs in operating_parameters_info only, within one coded video sequence: a new decoder there
    // would lose the reference frames). A header that does not parse: by its bytes
    const auto now = Av1SequenceKeyOf(sequence), before = Av1SequenceKeyOf(nvdecSequence_);
    const bool other = now && before ? !(*now == *before) : sequence != nvdecSequence_;
    if (!nvdecSequence_.empty() && other && hwDevice_) {
        // a new coded video sequence (a recording whose size or depth changes, two videos joined): av1dec would decode it on the NVDEC
        // decoder made for the last one (av1dec.c get_pixel_format: the CUDA format is still offered - no new decoder; nvdec.c makes its
        // decoder and surfaces for the size it starts with) - a decoder of its own, on the same device, as one that started there. The
        // last one has given every frame of the packets it got: av1dec returns a temporal unit's frame as it decodes it
        // (said when this decoder has decoded under the last one: an av1C that differs from the first packet's changes nothing seen)
        const bool news = !probe_ && instanceFrames_ > 0;
        if (instanceFrames_ > 0) decodedOtherSequence_ = true;
        const std::string place = PlaceWords(packet_->pts != AV_NOPTS_VALUE ? packet_->pts : packet_->dts);
        AVCodecContext* reopened = OpenCodec(hwDevice_);
        avcodec_free_context(&codec_);
        codec_ = reopened;
        instanceFrames_ = 0;
        Log()->log(news ? spdlog::level::info : spdlog::level::debug, "{}: another AV1 sequence header ({}) - its own NVDEC decoder", url_, place);
    }
    nvdecSequence_.assign(sequence);
}

std::string VideoDecoder::PlaceWords(std::optional<int64_t> pts) const {
    int64_t index = -1;
    const Rational rate = IndexRate();
    if (seeked_ && pts && *pts != AV_NOPTS_VALUE && rate.num > 0 && info_.timeBase.num > 0) index = IndexFromPts(*pts);  // as ReceiveFrame counts
    else if (PositionKnown()) index = nextIndex_;
    if (index == 0) return "on the first frame";
    if (index > 0) return "at frame " + std::to_string(index);
    return "on the first frame after a seek";
}

void VideoDecoder::CheckDecode(int err, const char* what, std::optional<int64_t> pts) {
    if (err >= 0) return;
    if (hwOnly_ && err == AVERROR(ENOSYS)) {  // av1dec without a working hwaccel: every frame header would fail the same way - kept (this
        // decoder must not be fed again). The driver having said it takes the stream, NVDEC did not start (memory, the driver): a new
        // decoder may succeed - the refusal is transient
        // (only on this decoder's first frame: frames decoded by it, NVDEC did start - a later refusal is the stream's; a new sequence
        // header in other parameters has a decoder of its own, FollowSequenceHeader)
        // (the words name the frame's place in the stream, PlaceWords - not this decoder's count: one that started with a seek may meet a
        // later part of the stream at its first frame. av1dec has parsed its sequence header by now: other parameters than the probe's -
        // a part NVDEC does not take, the stream's verdict)
        const char* swFmt = codec_->sw_pix_fmt != AV_PIX_FMT_NONE ? av_get_pix_fmt_name(codec_->sw_pix_fmt) : nullptr;
        // the layout the probe did not learn is known now (av1dec parsed the sequence header before it asked for a format): judged as a
        // probe's would be - its own reason, or (the driver taking it) NVDEC not starting
        const bool learnedNow = swFmt && info_.pixelFormat == "unknown";
        // (the probe's layout and size compared only when it learned the layout - its size is the container's otherwise, no evidence of
        // a part; a layout learned now after frames under another sequence header is that part's, not the stream's - decodedOtherSequence_)
        const bool probeKnown = info_.pixelFormat != "unknown";
        const bool otherPart = (probeKnown && ((swFmt && info_.pixelFormat != swFmt) ||
                                               (codec_->width > 0 && static_cast<uint32_t>(codec_->width) != info_.codedWidth) ||
                                               (codec_->height > 0 && static_cast<uint32_t>(codec_->height) != info_.codedHeight))) ||
                               (learnedNow && decodedOtherSequence_);
        std::string when = PlaceWords(pts);
        bool transient = nvdecStartTransient_;
        if (otherPart || learnedNow) {  // the part judged against the GPU as the probe was: refused - that verdict with its reason; taken - as for the probe
            StreamPart part;
            part.pixelFormat = swFmt ? swFmt : "?";
            part.codedWidth = static_cast<uint32_t>(std::max(codec_->width, 0));
            part.codedHeight = static_cast<uint32_t>(std::max(codec_->height, 0));
            part.learned = !otherPart;
            when += StreamPartWords(part);
            DecodeSupport judged = CheckDecodeSupport(StreamPartInfo(info_, swFmt ? swFmt : "", codec_->width > 0 ? static_cast<uint32_t>(codec_->width) : 0,
                                                                     codec_->height > 0 ? static_cast<uint32_t>(codec_->height) : 0));
            if (!judged.Ok()) {
                judged.why += " (" + when + ")";
                judged.part = otherPart;
                refused_ = std::make_shared<const DecodeSupport>(std::move(judged));
                throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
            }
            transient = judged.verified || (!judged.caps.checked && judged.caps.cudaError);
        }
        const bool startFailed = transient && instanceFrames_ == 0;  // this decoder's first frame (one opened for a new sequence too)
        refused_ = std::make_shared<const DecodeSupport>(RefusedDecode(info_, startFailed ? DecodeBlock::NvdecStartFailed : DecodeBlock::NvdecFailed, when));
        throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
    }
    CheckAv(err, what);
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
    if (gpu && !gpuFrames_) {  // GpuFrames only when HwAccel::Cuda asked for them (a hardware-only decoder on NVDEC unasked: host frames)
        gpu = nullptr;
        cpu = true;
    }
    if (refused_) throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
    if (hwOnly_ && !nvdecStarted_) StartNvdec();  // TASK-0076: first, before a packet reaches codec_
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
            CheckDecode(avcodec_send_packet(codec_, nullptr), "avcodec_send_packet(flush)");
            eofSent_ = true;
            continue;
        }
        CheckAv(err, "av_read_frame");
        if (packet_->stream_index == videoStream_ && hwOnly_ && framesDecoded_ >= 3 && FakeNvdecRefusesLater()) {  // DLSSVID_FAKE_NVDEC_AV1=later-frame
            const int64_t pts = packet_->pts != AV_NOPTS_VALUE ? packet_->pts : packet_->dts;
            av_packet_unref(packet_);
            CheckDecode(AVERROR(ENOSYS), "avcodec_send_packet", pts);
        }
        if (packet_->stream_index == videoStream_) {
            if (hwOnly_) {
                try {
                    FollowSequenceHeader();
                } catch (...) {
                    av_packet_unref(packet_);
                    throw;
                }
            }
            const int64_t sentPts = packet_->pts != AV_NOPTS_VALUE ? packet_->pts : packet_->dts;
            const int sent = avcodec_send_packet(codec_, packet_);
            if (sent < 0) av_packet_unref(packet_);
            CheckDecode(sent, "avcodec_send_packet", sentPts);
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
    CheckDecode(err, "avcodec_receive_frame");
    // TASK-0076: the size of the current geometry is codec_'s (av1dec sets it from each frame header); the CUDA frame NVDEC hands out of
    // a codec without exported cropping has the size of its surface (even, the sequence's): a 191-row AV1 frame comes as 192 rows. A
    // delayed frame of a codec with output delay (H.264 / HEVC after a resolution change) may be of the former geometry
    const int frameW = codec_->width, frameH = codec_->height;
    const int64_t pts = frame_->best_effort_timestamp != AV_NOPTS_VALUE ? frame_->best_effort_timestamp : frame_->pts;
    if (hwOnly_ && Av1FrameBelowSequence(frameW, frameH, nvdecSequence_)) {  // NVDEC gives it wrong (Resized)
        const auto key = Av1SequenceKeyOf(nvdecSequence_);
        const std::string frame = "a frame of " + std::to_string(frameW) + "x" + std::to_string(frameH) + ", " + PlaceWords(pts);
        av_frame_unref(frame_);
        refused_ = std::make_shared<const DecodeSupport>(RefusedDecode(StreamPartInfo(info_, "", key->maxWidth, key->maxHeight), DecodeBlock::Resized, frame));
        throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
    }
    if (hwOnly_ && frameH % 2 == 1) {  // TASK-0076: a part, or a size the probe did not know (OddHeight: the chroma a row off)
        const std::string frame = PlaceWords(pts);
        const uint32_t w = static_cast<uint32_t>(frameW), h = static_cast<uint32_t>(frameH);
        av_frame_unref(frame_);
        DecodeSupport s = RefusedDecode(StreamPartInfo(info_, "", w, h), DecodeBlock::OddHeight, frame);
        // the probe's height (of the stream, its layout learned) was even, or this decoder decoded frames under another sequence header: a
        // part in other parameters
        s.part = (info_.pixelFormat != "unknown" && info_.codedHeight % 2 == 0 && info_.codedHeight > 0) || decodedOtherSequence_;
        refused_ = std::make_shared<const DecodeSupport>(std::move(s));
        throw DecodeUnavailable(*refused_, url_ + ": " + refused_->why);
    }

    int64_t index = nextIndex_;
    if (seeked_ && pts != AV_NOPTS_VALUE) index = IndexFromPts(pts);
    nextIndex_ = index + 1;
    decodedSinceSeek_ = true;
    ++framesDecoded_;
    ++instanceFrames_;
    if (!checkedFrameMatrix_) {  // an orientation in the bitstream (SEI) other than the container's: the container's is applied
        checkedFrameMatrix_ = true;
        if (const AVFrameSideData* sd = av_frame_get_side_data(frame_, AV_FRAME_DATA_DISPLAYMATRIX); sd && sd->size >= 9 * sizeof(int32_t)) {
            const DisplayOrientation own = OrientationFromDisplayMatrix(reinterpret_cast<const int32_t*>(sd->data));
            if (!(own == info_.orientation))
                Log()->warn("the frames carry a display orientation '{}' other than the stream's '{}': the stream's is applied", own.Token(),
                            info_.orientation.Token());
        }
    }
    AVFrame* src = frame_;
    if (frame_->format == AV_PIX_FMT_CUDA) {
        nvdecFrames_ = true;
        // TASK-0076: NVDEC's frame comes at its surface's size (even sides: a 853-wide AV1 as 854) - the surface's rounding cut off, as
        // FFmpeg's apply_cropping does for a hardware frame of a codec that exports cropping (the width and height; the pointers and the
        // pitch stay - the chroma of 853 and 854 is as wide). Only that one row or column: a frame larger by more is of another geometry
        // (H.264 / HEVC: a delayed frame from before a resolution change, cut by FFmpeg already) and comes as before, at its own size;
        // nor a frame FFmpeg left cropped on the left or the top (H.264 / HEVC: the cut keeps the top-left corner, it would shift the
        // picture - it comes as before, at its surface's width)
        if (frameW > 0 && frameH > 0 && frame_->crop_left == 0 && frame_->crop_top == 0 && (frameW < frame_->width || frameH < frame_->height) &&
            frame_->width - frameW <= 1 && frame_->height - frameH <= 1) {
            frame_->width = std::min(frame_->width, frameW);
            frame_->height = std::min(frame_->height, frameH);
        }
        if (gpu && frame_->width % 2 == 1) {
            // an odd width: the NV12 consumers on the device (OFA, the pass cache) take even rows - the host path, as for a turned stream
            *gpu = GpuFrame{};
            cpu = true;
            if (!warnedOddWidthGpu_) {
                Log()->info("the stream is {} wide (odd): frames go through the host, the on-device fast path is off", frame_->width);
                warnedOddWidthGpu_ = true;
            }
        } else if (gpu && !info_.orientation.Identity()) {
            // a turned stream: the NV12 frame on the device is stored, not shown - no GpuFrame (OFA re-initialises for ABGR uploads of
            // the turned CPU frame, as for 10-bit streams)
            *gpu = GpuFrame{};
            cpu = true;
            if (!warnedTurnedGpu_) {
                Log()->info("the stream is shown {}: frames are turned on the host, the on-device fast path is off", info_.orientation.Describe());
                warnedTurnedGpu_ = true;
            }
        } else if (gpu) {
            // Keep a reference so the device memory stays valid until the next frame.
            av_frame_unref(hwKeep_);
            CheckAv(av_frame_ref(hwKeep_, frame_), "av_frame_ref");
            const auto* hwFrames = reinterpret_cast<const AVHWFramesContext*>(hwKeep_->hw_frames_ctx->data);
            if (hwFrames->sw_format == AV_PIX_FMT_NV12) {
                gpu->width = static_cast<uint32_t>(hwKeep_->width);
                gpu->height = static_cast<uint32_t>(hwKeep_->height);
                gpu->y = reinterpret_cast<uintptr_t>(hwKeep_->data[0]);
                gpu->uv = reinterpret_cast<uintptr_t>(hwKeep_->data[1]);
                gpu->pitch = static_cast<size_t>(hwKeep_->linesize[0]);
                gpu->index = index;
            } else {
                // NVDEC decodes 10/12-bit and 4:4:4 streams (p010, p016, yuv444p16 ...) into other layouts. The
                // on-device NV12 fast path (OFA fed straight from NVDEC) is not available for them: hand the caller
                // a CPU frame instead — the consumers fall back to their host paths (OFA re-initialises for ABGR
                // uploads, the pass cache uploads from the CPU frame). The CPU frame keeps the source's bit depth
                // and chroma layout (ConvertFrame), so nothing is lost on the way.
                *gpu = GpuFrame{};
                cpu = true;
                if (!warnedNonNv12_) {
                    Log()->info("NVDEC decodes this stream as {} (not NV12): the on-device fast path is off, frames go through the host as {}",
                                av_get_pix_fmt_name(hwFrames->sw_format) ? av_get_pix_fmt_name(hwFrames->sw_format) : "?", info_.decoded.Describe());
                    warnedNonNv12_ = true;
                }
            }
        }
        if (!cpu) {
            out.desc = OrientedDesc(FrameDesc{static_cast<uint32_t>(frame_->width), static_cast<uint32_t>(frame_->height), info_.decoded.format,
                                              info_.decoded.bitDepth},
                                    info_.orientation);
            out.desc.format = info_.decoded.format;
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
    if (!cpu && !gpu) {  // software frame skipped by the caller (DiscardFrames: seeking): no conversion; with gpu asked it comes on the host
        out.desc = OrientedDesc(FrameDesc{static_cast<uint32_t>(src->width), static_cast<uint32_t>(src->height), info_.decoded.format, info_.decoded.bitDepth},
                                info_.orientation);
        out.desc.format = info_.decoded.format;
        out.data.clear();
        out.index = index;
        out.pts = pts;
        av_frame_unref(frame_);
        return true;
    }
    if (info_.orientation.Identity()) {
        ConvertFrame(src, out);
    } else {
        ConvertFrame(src, unturned_);
        OrientFrame(unturned_, info_.orientation, out);  // TASK-0072: as players show it
    }
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
    // Planar YUV at 10 / 12 / 16 bits or 4:2:2 / 4:4:4 (and NVDEC's 16-bit layouts): copied as it is, bit-exact
    if (const auto lay = NativeLayoutFor(fmt, info_.bitDepth)) {
        FrameDesc desc = lay->desc;
        desc.width = w;
        desc.height = h;
        if (lay->semiPlanar) {
            SemiPlanar16ToPlanar(f->data[0], f->linesize[0], f->data[1], f->linesize[1], desc, out, lay->shiftRight);
        } else {
            const uint8_t* const planes[3] = {f->data[0], f->data[1], f->data[2]};
            const int pitch[3] = {f->linesize[0], f->linesize[1], f->linesize[2]};
            CopyPlanar(planes, pitch, desc, out, lay->shiftRight);
        }
        return;
    }

    // Anything else (RGB, 4:1:1, 9 / 14-bit ...) is converted with swscale to 8-bit 4:2:0. Not bit-exact.
    if (!warnedConversion_) {
        Log()->warn("source pixel format {} converted to yuv420p with swscale (lossy)", av_get_pix_fmt_name(fmt) ? av_get_pix_fmt_name(fmt) : "?");
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
    const Rational rate = IndexRate();
    if (rate.num <= 0 || info_.timeBase.num <= 0) return nextIndex_;
    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    return PtsToIndex(pts, start, info_.timeBase, rate);
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
    const Rational rate = IndexRate();
    if (rate.num <= 0 || info_.timeBase.num <= 0) return false;
    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    const AVRational fps{rate.num, rate.den};
    const AVRational tb{info_.timeBase.num, info_.timeBase.den};
    const int64_t target = start + av_rescale_q(index, av_inv_q(fps), tb);
    if (av_seek_frame(fmt_, videoStream_, target, AVSEEK_FLAG_BACKWARD) < 0) return false;
    avcodec_flush_buffers(codec_);
    eofSent_ = false;
    drained_ = false;
    seeked_ = true;
    decodedSinceSeek_ = false;
    ++seeks_;
    nextIndex_ = 0;  // replaced by the pts-derived index of the first decoded frame
    return true;
}

bool VideoDecoder::SeekToFrame(int64_t index) {
    if (index < 0) return false;
    if (info_.frameCount > 0 && index >= info_.frameCount) return false;
    if (!drained_ && index == nextIndex_) return true;
    if (!drained_ && index > nextIndex_ && index - nextIndex_ <= kSequentialSeekFrames) return DiscardFrames(index);

    const int64_t start = fmt_->streams[videoStream_]->start_time != AV_NOPTS_VALUE ? fmt_->streams[videoStream_]->start_time : 0;
    const Rational rate = IndexRate();
    const AVRational fps{rate.num, rate.den};
    const AVRational tb{info_.timeBase.num, info_.timeBase.den};
    auto seekTo = [&](int64_t frameIndex) {
        const int64_t target = start + av_rescale_q(std::max<int64_t>(frameIndex, 0), av_inv_q(fps), tb);
        if (av_seek_frame(fmt_, videoStream_, target, AVSEEK_FLAG_BACKWARD) < 0) return false;
        avcodec_flush_buffers(codec_);
        eofSent_ = false;
        drained_ = false;
        seeked_ = true;
        decodedSinceSeek_ = false;
        ++seeks_;
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
