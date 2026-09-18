#include "convert/ColorConvert.h"

#include <algorithm>
#include <cmath>

#include "util/Error.h"

namespace dlssvid {

ColorInfo ColorInfoFromStream(const VideoStreamInfo& info) {
    ColorInfo c;
    // AVCOL_SPC_*: 1 = BT.709, 5/6 = BT.601 (625/525); AVCOL_RANGE_JPEG = 2.
    if (info.colorSpace == 1) c.matrix = ColorMatrix::Bt709;
    else if (info.colorSpace == 5 || info.colorSpace == 6) c.matrix = ColorMatrix::Bt601;
    else c.matrix = info.height >= 720 ? ColorMatrix::Bt709 : ColorMatrix::Bt601;
    c.range = info.colorRange == 2 ? ColorRange::Full : ColorRange::Limited;
    return c;
}

std::string_view ToString(ColorMatrix m) { return m == ColorMatrix::Bt709 ? "bt709" : "bt601"; }
std::string_view ToString(ColorRange r) { return r == ColorRange::Full ? "full" : "limited"; }

PassImage Yuv420pToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType) {
    if (frame.desc.format != PixelFormat::Yuv420p) Throw("Yuv420pToRgb: frame is not yuv420p");
    const uint32_t w = frame.desc.width, h = frame.desc.height;
    PassImage out = MakePassImage(PassKind::ColorSource, w, h, outType);

    const bool full = info.range == ColorRange::Full;
    const float yScale = full ? 1.f / 255.f : 1.f / 219.f;
    const float yOff = full ? 0.f : 16.f;
    const float cScale = full ? 1.f / 255.f : 1.f / 224.f;
    // Y'CbCr -> R'G'B' coefficients
    const float kr = info.matrix == ColorMatrix::Bt709 ? 0.2126f : 0.299f;
    const float kb = info.matrix == ColorMatrix::Bt709 ? 0.0722f : 0.114f;
    const float kg = 1.f - kr - kb;
    const float crR = 2.f * (1.f - kr);
    const float cbB = 2.f * (1.f - kb);
    const float cbG = -2.f * kb * (1.f - kb) / kg;
    const float crG = -2.f * kr * (1.f - kr) / kg;

    const uint8_t* Y = frame.Plane(0);
    const uint8_t* U = frame.Plane(1);
    const uint8_t* V = frame.Plane(2);
    const size_t cw = frame.desc.PlaneWidth(1);
    const float scaleOut = outType == PixelType::U16 ? 65535.f : (outType == PixelType::U8 ? 255.f : 1.f);

    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const float yy = (static_cast<float>(Y[static_cast<size_t>(y) * w + x]) - yOff) * yScale;
            const size_t ci = static_cast<size_t>(y / 2) * cw + x / 2;
            const float cb = (static_cast<float>(U[ci]) - 128.f) * cScale;
            const float cr = (static_cast<float>(V[ci]) - 128.f) * cScale;
            const float r = std::clamp(yy + crR * cr, 0.f, 1.f);
            const float g = std::clamp(yy + cbG * cb + crG * cr, 0.f, 1.f);
            const float b = std::clamp(yy + cbB * cb, 0.f, 1.f);
            out.Set(x, y, 0, r * scaleOut);
            out.Set(x, y, 1, g * scaleOut);
            out.Set(x, y, 2, b * scaleOut);
        }
    return out;
}

}  // namespace dlssvid
