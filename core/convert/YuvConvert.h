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

// The same layout at another size (TASK-0073: a pass of an anamorphic source stretched to square pixels when `process` encodes it):
// Lanczos, the same on every CPU (SWS_BITEXACT); at the same size the samples come out as they are.
class YuvResizer {
public:
    YuvResizer() = default;
    ~YuvResizer();
    YuvResizer(const YuvResizer&) = delete;
    YuvResizer& operator=(const YuvResizer&) = delete;

    void Resize(const CpuFrame& in, uint32_t width, uint32_t height, CpuFrame& out);

private:
    SwsContext* sws_ = nullptr;
};

}  // namespace dlssvid
