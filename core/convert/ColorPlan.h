#pragma once

#include "convert/ColorConvert.h"
#include "io/VideoEncoder.h"

namespace dlssvid {

// How one source is read, shown and written (TASK-0075, HDR sources): every consumer takes
// its colour from here, so all of them see the same thing. Each role gets its own ColorInfo because an HDR source will
// be read differently by each - the colour stages keep its codes, the guides (depth, flow, scenes) get an SDR proxy, the
// viewport decodes the matrix only and maps the curve on the GPU, and the result is encoded in the domain of its last
// pass. Until that work lands every source is read as before: all roles are ColorInfoFromStream.
struct ColorPlan {
    ColorInfo colourDecode;  // colour stages and their previews (upscale, nr, fg), export, compare, info
    ColorInfo guideDecode;   // guide stages (depth, flow, scenes)
    ColorInfo viewDecode;    // viewport and thumbnails
    ColorInfo resultEncode;  // RGB -> YUV of the result
    ColorTags resultTags;    // the colour tags of the result file
};

ColorPlan ResolveColor(const VideoStreamInfo& info);

// The tags of a stream encoded with `encode` from a source with `info`: the matrix and range used, the source's
// primaries and transfer passed through (BT.601 is written as the source said, BT470BG, else SMPTE170M).
ColorTags ResultColorTags(const ColorInfo& encode, const VideoStreamInfo& info);

}  // namespace dlssvid
