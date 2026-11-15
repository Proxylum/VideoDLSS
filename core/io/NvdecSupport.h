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
    Layout,       // the stream's sample layout is one NVDEC does not decode for this codec (AV1: 8 / 10-bit 4:2:0 only)
    NoCuda,       // no CUDA device / driver to run NVDEC on
    NoNvdec,      // CUDA is there, the driver's video decoder (nvcuvid.dll) is not
    NoAv1,        // the GPU's NVDEC has no AV1 decoder at the stream's depth
    Size,         // the frame is outside the sizes the GPU's NVDEC decodes
    NvdecFailed,  // NVDEC was asked and refused on the first frame (FFmpeg's reason is in its log)
};

// What the driver says of AV1 decoding on CUDA device 0 (FFmpeg's NVDEC takes that one), at one bit depth, 4:2:0.
struct NvdecAv1Caps {
    bool checked = false;    // a definite answer: no CUDA, or the decoder caps were read. false: the driver could not be asked
    bool cuda = false;       // a CUDA device exists
    bool nvdecMissing = false;  // ... but the driver's video decoder library (nvcuvid.dll) could not be loaded
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
    NvdecAv1Caps caps;
    bool Ok() const { return block == DecodeBlock::None; }
};

// The verdict from the stream and what NVDEC said (`caps` null: not asked). Order: a decoder with a software path is fine; an unknown
// pixel format is left to NVDEC (decided on the first frame); a layout this build does not decode AV1 in (8 / 10-bit 4:2:0 only) is
// refused before the GPU is asked; no answer from the driver goes on blind (as FFmpeg does); then no CUDA, no nvcuvid, no AV1, the size.
DecodeSupport JudgeDecode(const VideoStreamInfo& info, const NvdecAv1Caps* caps);

// The driver's answer for AV1 4:2:0 at `bitDepth` (8 / 10): cuInit, device 0, a temporary context of its own (the primary context the
// decoders share is left alone), cuvidGetDecoderCaps. Loaded at run time (nvcuda.dll, nvcuvid.dll); cached per depth for the process.
NvdecAv1Caps QueryNvdecAv1Caps(int bitDepth);

// JudgeDecode with the driver asked when the stream needs NVDEC. DLSSVID_FAKE_NVDEC_AV1 stands in for the driver's answer (tests, a
// support case), read on every call before the cache: no-cuda | no-nvdec | no-av1 - that refusal; first-frame - no answer (the stream
// goes on unchecked) and NVDEC then refuses its first frame (FakeNvdecRefusesFrames).
DecodeSupport CheckDecodeSupport(const VideoStreamInfo& info);
// DLSSVID_FAKE_NVDEC_AV1=first-frame: VideoDecoder's get_format declines CUDA for a hardware-only decoder (the refusal FFmpeg reports
// as ENOSYS on the first frame - a GPU whose limits the driver did not tell in advance).
bool FakeNvdecRefusesFrames();

// The refusal `block` of the stream, `detail` saying what failed (VideoDecoder: no CUDA device when it is created, NVDEC refusing the
// first frame).
DecodeSupport RefusedDecode(const VideoStreamInfo& info, DecodeBlock block, const std::string& detail);

// One line for `dlssvid info`: «software (h264); NVDEC with --hwaccel cuda when the GPU decodes this layout», «software (prores); NVDEC
// does not decode this codec», «NVDEC only (av1: ...): NVDEC on '<gpu>', ...», «unavailable - ...».
std::string DescribeDecode(const DecodeSupport& support);

// The refusal of a stream that cannot be decoded here (CheckDecodeSupport, or NVDEC refusing the first frame).
class DecodeUnavailable : public Error {
public:
    DecodeUnavailable(DecodeSupport s, const std::string& message) : Error(message), support(std::move(s)) {}
    DecodeSupport support;
};

}  // namespace dlssvid
