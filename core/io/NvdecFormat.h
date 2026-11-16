#pragma once

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace dlssvid {

// TASK-0076: VideoDecoder's get_format under NVDEC - AV_PIX_FMT_CUDA when the decoder offers it; otherwise the first format (a decoder
// with a software path then goes through FFmpeg's cascade down to software, as before), or AV_PIX_FMT_NONE for a hardware-only decoder
// (HardwareOnlyDecoder): it has no software path, choosing nothing fails at once and the failure is reported with its reason.
// A hardware-only decoder's software format (the list's last, as FFmpeg's ff_get_format reads it) outside Av1NvdecLayout: AV_PIX_FMT_NONE
// as well - FFmpeg offers CUDA for gray AV1, which this build leaves out; the refusal then names that layout (VideoDecoder::CheckDecode).
AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly);

// TASK-0076: the sample layouts this build decodes AV1 in (on NVDEC): 8 and 10-bit 4:2:0 - JudgeDecode's Layout and get_format both
inline bool Av1NvdecLayout(AVPixelFormat format) { return format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUV420P10LE; }

}  // namespace dlssvid
