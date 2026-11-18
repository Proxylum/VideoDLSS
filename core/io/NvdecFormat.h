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
// A codec with a software path whose frame is of an odd height (VP9 1920x803 coded so; H.264 / HEVC 320x191 coded aligned and cropped):
// its software format - FFmpeg's NVDEC gives the chroma of such a frame wrong (a row off, or its last row lost on the way to the host);
// a hardware-only one is refused for it instead.
AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly, int codedHeight = 0);

// TASK-0076: the height a frame is judged by for NVDEC: the shown one when it is odd (H.264 / HEVC code it aligned and crop it: 191 as
// 192), else the coded one (VP9 codes it as it is)
inline int NvdecJudgedHeight(int height, int codedHeight) { return height % 2 == 1 ? height : codedHeight; }

// TASK-0076: the sample layouts this build decodes AV1 in (on NVDEC): 8 and 10-bit 4:2:0 - JudgeDecode's Layout and get_format both
inline bool Av1NvdecLayout(AVPixelFormat format) { return format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUV420P10LE; }

}  // namespace dlssvid
