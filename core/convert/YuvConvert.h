#pragma once

#include "pipeline/Frame.h"

struct SwsContext;

namespace dlssvid {

// YUV -> YUV between CpuFrame layouts (bit depth and chroma subsampling) through swscale: the passthrough path uses it
// when the source's layout is not the one the result is written in (a 10-bit 4:4:4 source into an 8-bit 4:2:0 H.264).
// Chroma is resampled bilinearly, bit depth reduced with rounding; the identity (same layout) is a plain copy.
class YuvConverter {
public:
    YuvConverter() = default;
    ~YuvConverter();
    YuvConverter(const YuvConverter&) = delete;
    YuvConverter& operator=(const YuvConverter&) = delete;

    void Convert(const CpuFrame& in, FrameDesc target, CpuFrame& out);

private:
    SwsContext* sws_ = nullptr;
};

}  // namespace dlssvid
