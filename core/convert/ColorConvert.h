#pragma once

#include "io/VideoDecoder.h"
#include "passes/PassImage.h"
#include "pipeline/Frame.h"

namespace dlssvid {

enum class ColorMatrix { Bt601, Bt709 };
enum class ColorRange { Limited, Full };

struct ColorInfo {
    ColorMatrix matrix = ColorMatrix::Bt709;
    ColorRange range = ColorRange::Limited;
};

// Stream metadata when present, otherwise BT.709 for >= 720 lines and BT.601 below (limited range).
ColorInfo ColorInfoFromStream(const VideoStreamInfo& info);
std::string_view ToString(ColorMatrix m);
std::string_view ToString(ColorRange r);

// YUV 4:2:0 8-bit -> RGB (display-referred, sRGB/BT.709 transfer, [0,1]); chroma is
// replicated (nearest) so the result is deterministic and reversible per 2x2 block.
// U16 output = round(v * 65535).
PassImage Yuv420pToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType = PixelType::F16);

}  // namespace dlssvid
