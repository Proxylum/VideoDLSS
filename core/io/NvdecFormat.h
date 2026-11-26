#pragma once

extern "C" {
#include <libavcodec/codec_id.h>
#include <libavutil/pixfmt.h>
}

namespace dlssvid {

// TASK-0076: VideoDecoder's get_format under NVDEC - AV_PIX_FMT_CUDA when the decoder offers it; otherwise the first format (a decoder
// with a software path then goes through FFmpeg's cascade down to software, as before), or AV_PIX_FMT_NONE for a hardware-only decoder
// (HardwareOnlyDecoder): it has no software path, choosing nothing fails at once and the failure is reported with its reason.
// A hardware-only decoder's software format (the list's last, as FFmpeg's ff_get_format reads it) outside Av1NvdecLayout: AV_PIX_FMT_NONE
// as well - FFmpeg offers CUDA for gray AV1, which this build leaves out; the refusal then names that layout (VideoDecoder::CheckDecode).
// A codec with a software path whose frame is of an odd height (VP9 1920x803 coded so; H.264 320x191 coded aligned and cropped):
// its software format - FFmpeg's NVDEC gives the chroma of such a frame wrong (a row off, or its last row lost on the way to the host);
// A hardware-only decoder has no software path: it gets AV_PIX_FMT_CUDA whatever the height (judgedHeight is not read for it); its odd
// height is refused elsewhere - by JudgeDecode before the GPU, by VideoDecoder::ReceiveFrame on the frame (DecodeBlock::OddHeight).
// (A layout NvdecOddHeightSafe takes is not judged: GetHwFormat passes 0.)
// judgedHeight: what NvdecJudgedHeight(ctx->height, ctx->coded_height) gives (the shown height when odd, else the coded one); 0 - not judged
AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly, int judgedHeight = 0);

// TASK-0076: the height a frame is judged by for NVDEC: the shown one when it is odd (VP9 codes 803 as it is; H.264 / HEVC code 191
// aligned as 192 and crop it), else the coded one - even then as well (aligned, or equal to the shown): only an odd shown height decides
inline int NvdecJudgedHeight(int height, int codedHeight) { return height % 2 == 1 ? height : codedHeight; }

// TASK-0076: an odd shown height FFmpeg 7.1.1's NVDEC gives right - HEVC 4:4:4 only (checked on a GPU, bit for bit with the software
// decode): HEVC exports its cropping (the frame mapped at its even coded height, cropped after) and nvdec_hevc keeps 4:4:4 a 4:4:4
// surface (supports_444; the copy to the host takes every chroma row). Not H.264 (nvdec_h264: no 4:4:4 surface - 4:2:2 / 4:4:4 come as
// NV12), nor HEVC 4:2:2 (hevcdec offers no CUDA for it), nor VP9 / MPEG-4 (no cropping exported: the frame mapped at its odd height,
// the next plane a row off), nor 4:2:0 (its last chroma row lost on the way to the host), nor RGB, gray or a format not known
bool NvdecOddHeightSafe(AVCodecID codec, AVPixelFormat softwareFormat);

// TASK-0076: the sample layouts this build decodes AV1 in (on NVDEC): 8 and 10-bit 4:2:0 - JudgeDecode's Layout and get_format both
inline bool Av1NvdecLayout(AVPixelFormat format) { return format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUV420P10LE; }

}  // namespace dlssvid
