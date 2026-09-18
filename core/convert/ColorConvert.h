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

// Same conversion straight into the viewport's RGBA16F texture layout (alpha = 1). Bit-identical to
// Yuv420pToRgb(frame, info, PixelType::F16) plus alpha; uses F16C when the CPU has it.
PassImage Yuv420pToRgba16f(const CpuFrame& frame, const ColorInfo& info);

// Any colour image (u8/u16/f16/f32, 1..4 channels) -> RGBA16F with integer types scaled to [0,1].
PassImage ToRgba16f(const PassImage& img);

// Inverse of Yuv420pToRgb: display-referred RGB (float [0,1], integers scaled) -> tightly packed
// 8-bit YUV420P (BT.601/709, limited/full range), chroma = mean of each 2x2 block, rounded to nearest.
void RgbToYuv420p(const PassImage& rgb, const ColorInfo& info, CpuFrame& out);
// True when the CPU supports the F16C float <-> half instructions (used by the converters above).
bool HasF16C();

}  // namespace dlssvid
