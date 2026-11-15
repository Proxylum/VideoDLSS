#pragma once

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace dlssvid {

// TASK-0076: VideoDecoder's get_format under NVDEC - AV_PIX_FMT_CUDA when the decoder offers it; otherwise the first format (a decoder
// with a software path then goes through FFmpeg's cascade down to software, as before), or AV_PIX_FMT_NONE for a hardware-only decoder
// (HardwareOnlyDecoder): it has no software path, choosing nothing fails at once and the failure is reported with its reason.
AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly);

}  // namespace dlssvid
