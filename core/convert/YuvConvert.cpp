#include "convert/YuvConvert.h"

extern "C" {
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "util/Error.h"

namespace dlssvid {

namespace {

AVPixelFormat AvFormatFor(const FrameDesc& d) {
    const std::string_view c = ChromaName(d.format);
    const int depth = d.HighBitDepth() ? d.bitDepth : 8;
    if (c == "444") return depth == 8 ? AV_PIX_FMT_YUV444P : depth == 10 ? AV_PIX_FMT_YUV444P10LE : depth == 12 ? AV_PIX_FMT_YUV444P12LE : AV_PIX_FMT_YUV444P16LE;
    if (c == "422") return depth == 8 ? AV_PIX_FMT_YUV422P : depth == 10 ? AV_PIX_FMT_YUV422P10LE : depth == 12 ? AV_PIX_FMT_YUV422P12LE : AV_PIX_FMT_YUV422P16LE;
    return depth == 8 ? AV_PIX_FMT_YUV420P : depth == 10 ? AV_PIX_FMT_YUV420P10LE : depth == 12 ? AV_PIX_FMT_YUV420P12LE : AV_PIX_FMT_YUV420P16LE;
}

}  // namespace

YuvConverter::~YuvConverter() {
    if (sws_) sws_freeContext(sws_);
}

void YuvConverter::Convert(const CpuFrame& in, FrameDesc target, CpuFrame& out) {
    target.width = in.desc.width;
    target.height = in.desc.height;
    if (!target.HighBitDepth()) target.bitDepth = 8;
    if (in.data.size() < in.desc.ByteSize()) Throw("YuvConverter: frame has no data");
    if (target == in.desc) {
        out = in;
        return;
    }
    const int w = static_cast<int>(in.desc.width), h = static_cast<int>(in.desc.height);
    sws_ = sws_getCachedContext(sws_, w, h, AvFormatFor(in.desc), w, h, AvFormatFor(target), SWS_BILINEAR | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT, nullptr,
                                nullptr, nullptr);
    if (!sws_) Throw("YuvConverter: sws_getCachedContext failed for " + in.desc.Name() + " -> " + target.Name());
    out.Allocate(target);
    out.index = in.index;
    out.pts = in.pts;
    const uint8_t* src[3] = {in.Plane(0), in.Plane(1), in.Plane(2)};
    const int srcPitch[3] = {static_cast<int>(in.desc.PlaneWidth(0)), static_cast<int>(in.desc.PlaneWidth(1)), static_cast<int>(in.desc.PlaneWidth(2))};
    uint8_t* dst[3] = {out.Plane(0), out.Plane(1), out.Plane(2)};
    const int dstPitch[3] = {static_cast<int>(out.desc.PlaneWidth(0)), static_cast<int>(out.desc.PlaneWidth(1)), static_cast<int>(out.desc.PlaneWidth(2))};
    sws_scale(sws_, src, srcPitch, 0, h, dst, dstPitch);
}

YuvResizer::~YuvResizer() {
    if (sws_) sws_freeContext(sws_);
}

void YuvResizer::Resize(const CpuFrame& in, uint32_t width, uint32_t height, CpuFrame& out) {
    if (in.data.size() < in.desc.ByteSize()) Throw("YuvResizer: frame has no data");
    if (!width || !height) Throw("YuvResizer: empty target size");
    FrameDesc target = in.desc;
    target.width = width;
    target.height = height;
    const AVPixelFormat f = AvFormatFor(in.desc);
    const int w = static_cast<int>(in.desc.width), h = static_cast<int>(in.desc.height);
    sws_ = sws_getCachedContext(sws_, w, h, f, static_cast<int>(width), static_cast<int>(height), f, SWS_LANCZOS | SWS_ACCURATE_RND | SWS_BITEXACT, nullptr,
                                nullptr, nullptr);
    if (!sws_) Throw("YuvResizer: sws_getCachedContext failed for " + in.desc.Name() + " -> " + target.Name());
    out.Allocate(target);
    out.index = in.index;
    out.pts = in.pts;
    const uint8_t* src[3] = {in.Plane(0), in.Plane(1), in.Plane(2)};
    const int srcPitch[3] = {static_cast<int>(in.desc.PlaneWidth(0)), static_cast<int>(in.desc.PlaneWidth(1)), static_cast<int>(in.desc.PlaneWidth(2))};
    uint8_t* dst[3] = {out.Plane(0), out.Plane(1), out.Plane(2)};
    const int dstPitch[3] = {static_cast<int>(out.desc.PlaneWidth(0)), static_cast<int>(out.desc.PlaneWidth(1)), static_cast<int>(out.desc.PlaneWidth(2))};
    sws_scale(sws_, src, srcPitch, 0, h, dst, dstPitch);
}

}  // namespace dlssvid
