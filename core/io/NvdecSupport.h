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

enum class DecodeBlock {
    None,         // decodes: in software, or on NVDEC
    Layout,       // the stream's sample layout is one NVDEC does not decode for this codec (AV1: 8 / 10-bit 4:2:0 only)
    NoCuda,       // no CUDA device / driver to run NVDEC on
    NoAv1,        // the GPU's NVDEC has no AV1 decoder at the stream's depth
    Size,         // the frame is outside the sizes the GPU's NVDEC decodes
    NvdecFailed,  // NVDEC was asked and refused on the first frame (FFmpeg's reason is in its log)
};

// What the driver says of AV1 decoding on CUDA device 0 (FFmpeg's NVDEC takes that one), at one bit depth, 4:2:0.
struct NvdecAv1Caps {
    bool checked = false;    // a definite answer: no CUDA, or the decoder caps were read. false: the driver could not be asked
    bool cuda = false;       // a CUDA device exists
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
    std::string pixelFormat;
    int bitDepth = 8;
    unsigned width = 0, height = 0;  // the stored frame
    NvdecAv1Caps caps;
    bool Ok() const { return block == DecodeBlock::None; }
};

// The verdict from the stream and what NVDEC said (`caps` null: not asked). Order: a decoder with a software path is fine; an unknown
// pixel format is left to NVDEC (decided on the first frame); a layout NVDEC does not decode for AV1 is refused before the GPU is asked;
// no answer from the driver goes on blind (as FFmpeg does); then no CUDA, no AV1, the size limits.
DecodeSupport JudgeDecode(const VideoStreamInfo& info, const NvdecAv1Caps* caps);

// The driver's answer for AV1 4:2:0 at `bitDepth` (8 / 10): cuInit, device 0, a temporary context of its own (the primary context the
// decoders share is left alone), cuvidGetDecoderCaps. Loaded at run time (nvcuda.dll, nvcuvid.dll); cached per depth for the process.
NvdecAv1Caps QueryNvdecAv1Caps(int bitDepth);

// JudgeDecode with the driver asked when the stream needs NVDEC. DLSSVID_FAKE_NVDEC_AV1 = no-cuda | no-av1 stands in for the driver's
// answer (tests, a support case); read on every call, before the cache.
DecodeSupport CheckDecodeSupport(const VideoStreamInfo& info);

// The refusal `block` of the stream, `detail` saying what failed (VideoDecoder: no CUDA device when it is created, NVDEC refusing the
// first frame).
DecodeSupport RefusedDecode(const VideoStreamInfo& info, DecodeBlock block, const std::string& detail);

// One line for `dlssvid info`: «software (h264), NVDEC with --hwaccel cuda», «NVDEC only (av1: ...) on <gpu>, ...», «unavailable - ...».
std::string DescribeDecode(const DecodeSupport& support);

// The refusal of a stream that cannot be decoded here (CheckDecodeSupport, or NVDEC refusing the first frame).
class DecodeUnavailable : public Error {
public:
    DecodeUnavailable(DecodeSupport s, const std::string& message) : Error(message), support(std::move(s)) {}
    DecodeSupport support;
};

}  // namespace dlssvid
