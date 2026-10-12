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

// YUV (any CpuFrame format: 8-bit or 10/12/16-bit, 4:2:0 / 4:2:2 / 4:4:4) -> RGB (display-referred, sRGB/BT.709
// transfer, [0,1]); chroma is replicated (nearest) so the result is deterministic and reversible per block. Limited
// range scales with the bit depth (16..235 at 8 bits is 64..940 at 10). U16 output = round(v * 65535).
PassImage YuvToRgb(const CpuFrame& frame, const ColorInfo& info, PixelType outType = PixelType::F16);

// Same conversion straight into the viewport's RGBA16F texture layout (alpha = 1). Bit-identical to
// YuvToRgb(frame, info, PixelType::F16) plus alpha; uses F16C when the CPU has it.
// `threads`: bands of rows converted in parallel (0 = auto: half the cores, at most 8, from 256 rows); the result does
// not depend on it.
PassImage YuvToRgba16f(const CpuFrame& frame, const ColorInfo& info, int threads = 0);

// Any colour image (u8/u16/f16/f32, 1..4 channels) -> RGBA16F with integer types scaled to [0,1].
PassImage ToRgba16f(const PassImage& img);

// Inverse of YuvToRgb: display-referred RGB (float [0,1], integers scaled) -> tightly packed YUV of `target`'s
// format and bit depth (BT.601/709, limited/full range), chroma = mean of each block (2x2 for 4:2:0, 2x1 for
// 4:2:2, the pixel itself for 4:4:4), rounded to nearest. `target.width/height` are taken from the image.
void RgbToYuv(const PassImage& rgb, const ColorInfo& info, FrameDesc target, CpuFrame& out);
// The 8-bit 4:2:0 case (previews, tests).
void RgbToYuv420p(const PassImage& rgb, const ColorInfo& info, CpuFrame& out);
// True when the CPU supports the F16C float <-> half instructions (used by the converters above).
bool HasF16C();

}  // namespace dlssvid
