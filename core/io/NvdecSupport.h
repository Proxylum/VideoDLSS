#pragma once

#include <string>

#include "io/VideoDecoder.h"
#include "util/Error.h"

struct AVCodec;

namespace dlssvid {

// TASK-0076: decoders that have no software path. This build of FFmpeg has no software AV1 decoder (no libdav1d, no libaom):
// av_find_best_stream picks the built-in `av1`, which decodes only through a hardware accelerator - without one every packet fails
// with ENOSYS («Function not implemented»). Such a stream is decoded on NVDEC whatever VideoDecoder::Options::hwaccel says, and
// one NVDEC cannot decode is refused with a reason that names it, before a stage starts.

// Whether `codec` decodes only with a hardware accelerator: the built-in av1 decoder (with a CUDA hw config). A build with libdav1d
// picks that one first and the rule switches itself off. No public FFmpeg flag says it: AV_CODEC_CAP_HARDWARE marks the *_cuvid
// wrappers, not the hwaccel-only decoders.
bool HardwareOnlyDecoder(const AVCodec* codec);
// Whether NVDEC can decode for `codec` at all: a CUDA hw config with a device context (h264, hevc, vp9, av1 ...; not prores, ffv1, dnxhd).
bool HasNvdecHwaccel(const AVCodec* codec);

enum class DecodeBlock {
    None,         // decodes: in software, or on NVDEC
    Layout,       // a sample layout this build does not decode AV1 in: 8 / 10-bit 4:2:0 only (NVDEC has no AV1 4:2:2 / 4:4:4 / 12-bit;
                  // 4:0:0 gray / gray10, which FFmpeg's NVDEC path takes, is left out by decision - TASK-0076)
    NoCuda,       // no CUDA device / driver to run NVDEC on
    NoNvdec,      // CUDA is there, the driver's video decoder (nvcuvid.dll) is not
    NoAv1,        // the GPU's NVDEC has no AV1 decoder at the stream's depth
    Size,         // the frame is outside the sizes the GPU's NVDEC decodes
    // an odd frame height: NVDEC's surfaces are even and FFmpeg 7.1.1's nvdec.c maps the chroma by the frame's own height - a chroma
    // row off (the Video Codec SDK aligns that offset by 2)
    OddHeight,
    NvdecFailed,  // NVDEC was asked and refused a frame - the first one, or a later one (why: on the first frame / after N frames)
    // a frame smaller than its sequence's maximum (an AV1 frame size override, a resize within a sequence): FFmpeg 7.1.1's NVDEC makes its
    // surfaces for the maximum and hands the frame out at that size, a lower one also with the chroma mapped by its own height (nvdec.c) -
    // width / height are the sequence's maximum, caps.why names the frame
    Resized,
    NvdecStartFailed,  // NVDEC did not start on the first frame although the driver took the stream or could not be asked for a CUDA
                       // error (NvdecAv1Caps::cudaError), or the CUDA device could not be created (out of GPU memory, a driver error):
                       // not the stream's verdict - another decoder may succeed later
};

// What the driver says of AV1 decoding on CUDA device 0 (FFmpeg's NVDEC takes that one), at one bit depth, 4:2:0.
struct NvdecAv1Caps {
    bool checked = false;    // a definite answer: no CUDA, no nvcuvid.dll (nvdecMissing), or the decoder caps were read. false: the
                             // driver could not be asked (a CUDA error: cudaError; a driver without cuvidGetDecoderCaps, which FFmpeg's
                             // loader does not reach) - not cached
    bool cuda = false;       // a CUDA device exists
    bool nvdecMissing = false;  // ... but the driver's video decoder library (nvcuvid.dll) could not be loaded
    bool cudaError = false;     // not checked because a CUDA call failed (a context not created now: memory) - not an old driver
    bool supported = false;  // cuvidGetDecoderCaps: bIsSupported
    std::string gpu;         // its name
    std::string why;         // what failed, when something did
    unsigned minWidth = 0, minHeight = 0, maxWidth = 0, maxHeight = 0, maxMacroblocks = 0;
};

struct DecodeSupport {
    bool hardwareOnly = false;  // the stream's decoder needs NVDEC (VideoStreamInfo::hardwareOnly)
    DecodeBlock block = DecodeBlock::None;
    bool verified = false;  // hardwareOnly: the GPU's NVDEC was asked and takes the stream; false: decided on the first frame
    std::string why;        // English, for the CLI and the log: the refusal, else how the stream decodes
    std::string codec;      // the decoder's name
    bool nvdecCodec = false;  // NVDEC decodes this codec at all (VideoStreamInfo::nvdecCodec)
    std::string pixelFormat;
    int bitDepth = 8;
    std::string chroma;  // VideoStreamInfo::chroma: "420" | "422" | "444" | "other" (gray among them)
    unsigned width = 0, height = 0;  // the stored frame
    // the verdict is on a part of the stream in other parameters than the probe's (a new sequence header): VideoDecoder::CheckDecode -
    // pixelFormat ... width / height are that part's -, ReceiveFrame and VerdictFromWhy for OddHeight - width / height the part's, the
    // layout the probe's -, the GUI reading the proxy build's
    bool part = false;
    NvdecAv1Caps caps;
    bool Ok() const { return block == DecodeBlock::None; }
    // A refusal that says nothing of the stream on this machine (NvdecStartFailed): a viewport tries again, the plan is not stopped
    bool Transient() const { return block == DecodeBlock::NvdecStartFailed; }
};

// The verdict from the stream and what NVDEC said (`caps` null: not asked). Order: a decoder with a software path is fine; a layout this
// build does not decode AV1 in (8 / 10-bit 4:2:0 only) is refused before the GPU is asked (CheckDecodeSupport asks nothing for it); a
// driver that could not be asked (a CUDA error on the temporary context) - an odd frame height (OddHeight) refused, else on blind (FFmpeg
// asks the driver itself on the first frame); then no CUDA, no nvcuvid, no AV1; an unknown pixel format (the probe did not learn it) is
// left to the first frame from there on (its size, the container's, too); then the size; then an odd frame height.
DecodeSupport JudgeDecode(const VideoStreamInfo& info, const NvdecAv1Caps* caps);

// The driver's answer for AV1 4:2:0 at `bitDepth` (8 / 10): cuInit, device 0, a temporary context of its own (the primary context the
// decoders share is left alone), cuvidGetDecoderCaps. Loaded at run time (nvcuda.dll, nvcuvid.dll); a definite answer is cached per depth
// for the process, no answer (checked=false: a context that could not be created ...) is asked again next time.
NvdecAv1Caps QueryNvdecAv1Caps(int bitDepth);

// JudgeDecode with the driver asked when the stream needs NVDEC. DLSSVID_FAKE_NVDEC_AV1 stands in for the driver's answer (tests, a
// support case), read on every call before the cache: no-cuda | no-nvdec | no-av1 - that refusal; first-frame - no answer (the stream
// goes on unchecked) and NVDEC then refuses its first frame; start-fails - the driver takes the stream and NVDEC still does not start
// (NvdecStartFailed, FakeNvdecRefusesFrames); device-fails - the driver takes the stream and the CUDA device is not created
// (NvdecStartFailed, FakeNvdecDeviceFails); query-fails - the driver could not be asked for a CUDA error and NVDEC then refuses the
// first frame (NvdecStartFailed: memory, not the stream); later-frame - NVDEC refuses after a few frames (NvdecFailed: the stream's).
DecodeSupport CheckDecodeSupport(const VideoStreamInfo& info);
// DLSSVID_FAKE_NVDEC_AV1=first-frame | query-fails | start-fails: VideoDecoder's get_format declines CUDA for a hardware-only decoder (the
// refusal FFmpeg reports as ENOSYS on the first frame - a GPU whose limits the driver did not tell in advance, or NVDEC failing to start).
bool FakeNvdecRefusesFrames();
// DLSSVID_FAKE_NVDEC_AV1=device-fails: VideoDecoder::StartNvdec does not get its CUDA device (out of GPU memory, as it were).
bool FakeNvdecDeviceFails();
// DLSSVID_FAKE_NVDEC_AV1=later-frame: VideoDecoder takes NVDEC refusing a frame after a few decoded by the same decoder - NvdecFailed,
// the stream's. A guard: FFmpeg 7.1.1 refuses with ENOSYS when it asks for a format after a sequence header, and a new one in other
// parameters opens a decoder of its own (VideoDecoder::FollowSequenceHeader), whose refusal is judged as a part of the stream or as a
// start failure; a header Av1SequenceHeader does not find would come here.
bool FakeNvdecRefusesLater();

// The refusal `block` of the stream, `detail` saying what failed (VideoDecoder: no CUDA device when it is created, NVDEC refusing the
// first frame).
DecodeSupport RefusedDecode(const VideoStreamInfo& info, DecodeBlock block, const std::string& detail);

// One line for `dlssvid info`: «software (h264); NVDEC with --hwaccel cuda when the GPU decodes this layout», «software (prores); NVDEC
// does not decode this codec», «software (vp9); not NVDEC even with --hwaccel cuda: an odd frame height (1920x803, the frame as stored) - …», «NVDEC only (av1: no software AV1 decoder in this build), on '<gpu>', AV1 8-bit 4:2:0 up to WxH» (or
// «..., not checked: <why>»), «unavailable - <why>».
std::string DescribeDecode(const DecodeSupport& support);

// The refusal of a stream that cannot be decoded here: CheckDecodeSupport (RequireDecodable, StartNvdec, process), the CUDA device not
// created, or NVDEC refusing a frame (the first or a later one; why says which, and a part of the stream in other parameters).
class DecodeUnavailable : public Error {
public:
    DecodeUnavailable(DecodeSupport s, const std::string& message) : Error(message), support(std::move(s)) {}
    DecodeSupport support;
};

}  // namespace dlssvid
