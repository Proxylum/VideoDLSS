// TASK-0076: which streams decode only on NVDEC, and whether NVDEC takes them. The driver is asked through ffnvcodec's run-time loader
// (dynlink_loader.h: nvcuda.dll and nvcuvid.dll loaded on demand) with FFmpeg's load flags, never through <cuda.h> or the CUDA runtime -
// this unit must not be mixed with them, and the loader's types stay inside it.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <cstdarg>
#include <cstdio>

// nvcuda.dll / nvcuvid.dll as FFmpeg's NVDEC loads them (its compat/cuda/dynlink_loader.h over compat/w32dlfcn.h: the program's folder and
// System32, not the current folder or PATH): the probe judges the library the decoder will use, and runs no other one
#define FFNV_LOAD_FUNC(path) LoadLibraryExA((path), nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)
#define FFNV_SYM_FUNC(lib, sym) GetProcAddress((lib), (sym))
#define FFNV_FREE_FUNC(lib) FreeLibrary(lib)

namespace dlssvid::nvdec_detail {
void LoaderLog(const char* format, ...);  // the loader's messages at debug level (by default it prints to stderr)
}  // namespace dlssvid::nvdec_detail
#define FFNV_LOG_FUNC(logctx, msg, ...) ::dlssvid::nvdec_detail::LoaderLog((msg), __VA_ARGS__)
#define FFNV_DEBUG_LOG_FUNC(logctx, msg, ...) ::dlssvid::nvdec_detail::LoaderLog((msg), __VA_ARGS__)
#include <ffnvcodec/dynlink_loader.h>

#include "io/NvdecFormat.h"
#include "io/NvdecSupport.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
}

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <set>

#include "util/Log.h"

namespace dlssvid {

namespace nvdec_detail {
void LoaderLog(const char* format, ...) {
    char buf[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof buf, format, args);
    va_end(args);
    std::string s = buf;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    Log()->debug("nvdec loader: {}", s);
}
}  // namespace nvdec_detail

bool HasNvdecHwaccel(const AVCodec* codec) {
    if (!codec) return false;
    for (int i = 0;; ++i) {
        const AVCodecHWConfig* c = avcodec_get_hw_config(codec, i);
        if (!c) return false;
        if (c->device_type == AV_HWDEVICE_TYPE_CUDA && (c->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) return true;
    }
}

bool HardwareOnlyDecoder(const AVCodec* codec) {
    return codec && codec->id == AV_CODEC_ID_AV1 && std::string(codec->name) == "av1" && HasNvdecHwaccel(codec);
}

AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly, int codedHeight) {
    const AVPixelFormat* last = formats;  // the software format, as FFmpeg's ff_get_format reads it
    while (*last != AV_PIX_FMT_NONE && *(last + 1) != AV_PIX_FMT_NONE) ++last;
    if (hardwareOnly && !Av1NvdecLayout(*last)) return AV_PIX_FMT_NONE;  // gray, 4:4:4, 12-bit: refused with that layout's reason
    if (!hardwareOnly && codedHeight % 2 == 1 && *last != AV_PIX_FMT_NONE) return *last;  // the chroma a row off on NVDEC: software
    for (const AVPixelFormat* p = formats; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    // no CUDA on offer: a decoder with a software path goes on with FFmpeg's cascade from the first format (as before); a hardware-only
    // one has nothing to fall back to - choosing nothing ends it at once (ENOSYS, reported with the reason)
    return hardwareOnly ? AV_PIX_FMT_NONE : formats[0];
}

namespace {

std::string Depth(int bitDepth) { return std::to_string(bitDepth) + "-bit"; }

std::string Layout(const DecodeSupport& s) {  // «10-bit 4:4:4 (yuv444p10le)», «8-bit monochrome / other (gray)»
    const std::string chroma = s.chroma == "420" ? "4:2:0" : s.chroma == "422" ? "4:2:2" : s.chroma == "444" ? "4:4:4" : "monochrome / other";
    return Depth(s.bitDepth) + " " + chroma + " (" + s.pixelFormat + ")";
}

std::string RefusalText(const DecodeSupport& s) {
    const std::string codec = "AV1";
    switch (s.block) {
        case DecodeBlock::Layout:
            return codec + " " + Layout(s) + " cannot be decoded here - this build has no software AV1 decoder and decodes AV1 on NVDEC only as 8 or 10-bit 4:2:0";
        case DecodeBlock::NoCuda:
            return codec + " decodes only on an NVIDIA GPU (NVDEC) in this build - there is no software AV1 decoder - and CUDA is unavailable: " + s.caps.why;
        case DecodeBlock::NoNvdec:
            return codec + " decodes only on NVDEC in this build - there is no software AV1 decoder - and the NVIDIA driver's video decoder is unavailable: " +
                   s.caps.why;
        case DecodeBlock::NoAv1:
            if (s.pixelFormat == "unknown")  // asked for 8 bits: the stream's own depth is not known before the first frame
                return "the GPU '" + s.caps.gpu + "' has no NVDEC AV1 decoder (asked for 8-bit 4:2:0 - the stream's layout is not known before the "
                       "first frame; for example, GeForce RTX 30 series and newer have one), and this build has no software AV1 decoder" +
                       (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
            return "the GPU '" + s.caps.gpu + "' has no NVDEC AV1 decoder for " + Depth(s.bitDepth) +
                   " 4:2:0 (for example, GeForce RTX 30 series and newer have one), and this build has no software AV1 decoder" +
                   (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::Size:
            return codec + " " + std::to_string(s.width) + "x" + std::to_string(s.height) + " (the frame as stored) is outside what NVDEC of '" + s.caps.gpu +
                   "' decodes (" + std::to_string(s.caps.minWidth) + "x" + std::to_string(s.caps.minHeight) + " to " + std::to_string(s.caps.maxWidth) + "x" +
                   std::to_string(s.caps.maxHeight) + ", at most " + std::to_string(s.caps.maxMacroblocks) +
                   " macroblocks), and this build has no software AV1 decoder";
        case DecodeBlock::OddHeight:
            return codec + " " + std::to_string(s.width) + "x" + std::to_string(s.height) +
                   " (the frame as stored) has an odd frame height - FFmpeg's NVDEC reads the chroma of such a frame a row off (its surfaces are "
                   "even), and this build has no software AV1 decoder" +
                   (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::NvdecFailed:
            return "NVDEC refused this AV1 stream and this build has no software AV1 decoder" + (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::Resized:
            return codec + " frames smaller than the sequence's " + std::to_string(s.width) + "x" + std::to_string(s.height) +
                   " (a frame size override or a resize within it) cannot be decoded here - FFmpeg's NVDEC makes its surfaces for the "
                   "sequence's maximum and hands such a frame out at that size, not its own (a lower one also with the chroma mapped by its "
                   "own height), and this build has no software AV1 decoder" +
                   (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::NvdecStartFailed:
            return "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 decoder" +
                   (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::None: break;
    }
    return {};
}

DecodeSupport Refuse(DecodeSupport s, DecodeBlock block) {
    s.block = block;
    s.verified = false;
    s.why = RefusalText(s);
    return s;
}

std::string FakeValue() {
    const char* v = std::getenv("DLSSVID_FAKE_NVDEC_AV1");
    return v ? std::string(v) : std::string();
}

// Each value of DLSSVID_FAKE_NVDEC_AV1 is said once (a test process sets several in turn)
bool FirstTimeFor(const std::string& value) {
    static std::mutex mutex;
    static std::set<std::string> said;
    std::lock_guard<std::mutex> lock(mutex);
    return said.insert(value).second;
}

// DLSSVID_FAKE_NVDEC_AV1: the driver's answer stood in for (no-cuda | no-nvdec | no-av1 | first-frame | query-fails | start-fails |
// device-fails | later-frame); nullopt: not set (or not understood - said once per value)
std::optional<NvdecAv1Caps> FakeCaps() {
    const std::string value = FakeValue();
    if (value.empty()) return std::nullopt;
    NvdecAv1Caps c;
    c.checked = true;
    c.why = "DLSSVID_FAKE_NVDEC_AV1=" + value;
    if (value == "no-cuda") {
    } else if (value == "no-nvdec") {
        c.cuda = true;
        c.nvdecMissing = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
    } else if (value == "no-av1") {
        c.cuda = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
    } else if (value == "first-frame") {  // no answer: the stream goes on unchecked, and NVDEC refuses it on its first frame
        c.checked = false;
        c.cuda = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
    } else if (value == "query-fails") {  // no answer for a CUDA error (memory), and NVDEC refuses the first frame
        c.checked = false;
        c.cuda = true;
        c.cudaError = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
    } else if (value == "start-fails" || value == "device-fails" || value == "later-frame") {  // the driver says yes, and NVDEC / the CUDA device still does not start
        c.cuda = true;
        c.supported = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
        c.minWidth = c.minHeight = 1;
        c.maxWidth = c.maxHeight = 65535;
        c.maxMacroblocks = 0xffffffffu;
    } else {
        if (FirstTimeFor(value))
            Log()->warn("DLSSVID_FAKE_NVDEC_AV1='{}' is not no-cuda | no-nvdec | no-av1 | first-frame | query-fails | start-fails | device-fails | "
                        "later-frame: ignored",
                        value);
        return std::nullopt;
    }
    if (FirstTimeFor(value)) Log()->warn("DLSSVID_FAKE_NVDEC_AV1={}: NVDEC's AV1 answer is faked", value);
    return c;
}

std::string CuError(const CudaFunctions* cu, CUresult r) {
    const char* name = nullptr;
    if (cu && cu->cuGetErrorName && cu->cuGetErrorName(r, &name) == CUDA_SUCCESS && name) return name;
    return "CUresult " + std::to_string(static_cast<int>(r));
}

NvdecAv1Caps QueryLive(int bitDepth) {
    NvdecAv1Caps c;
    CudaFunctions* cu = nullptr;
    CuvidFunctions* cv = nullptr;
    struct Free {
        CudaFunctions*& cu;
        CuvidFunctions*& cv;
        ~Free() {
            cuvid_free_functions(&cv);
            cuda_free_functions(&cu);
        }
    } freeLater{cu, cv};
    if (cuda_load_functions(&cu, nullptr) < 0) {
        c.checked = true;
        c.why = "nvcuda.dll could not be loaded (no NVIDIA driver)";
        return c;
    }
    if (const CUresult r = cu->cuInit(0); r != CUDA_SUCCESS) {
        c.checked = true;
        c.why = "cuInit failed: " + CuError(cu, r);
        return c;
    }
    int count = 0;
    if (const CUresult r = cu->cuDeviceGetCount(&count); r != CUDA_SUCCESS || count <= 0) {
        c.checked = true;
        c.why = r != CUDA_SUCCESS ? "cuDeviceGetCount failed: " + CuError(cu, r) : std::string("no CUDA device");
        return c;
    }
    CUdevice dev = 0;
    if (const CUresult r = cu->cuDeviceGet(&dev, 0); r != CUDA_SUCCESS) {
        c.checked = true;
        c.why = "cuDeviceGet(0) failed: " + CuError(cu, r);
        return c;
    }
    c.cuda = true;
    char name[256] = {};
    if (cu->cuDeviceGetName(name, static_cast<int>(sizeof name) - 1, dev) == CUDA_SUCCESS) c.gpu = name;
    if (cuvid_load_functions(&cv, nullptr) < 0) {  // the driver has no video decoder library: NVDEC is not there
        c.checked = true;
        c.nvdecMissing = true;
        c.why = "nvcuvid.dll could not be loaded (missing, or a driver too old for FFmpeg's NVDEC)";
        return c;
    }
    if (!cv->cuvidGetDecoderCaps) {  // a guard: not reached with ffnvcodec's loader (it requires cuvidGetDecodeStatus, newer than this)
        c.why = "the driver does not report NVDEC capabilities";
        return c;
    }
    CUcontext ctx = nullptr;
    if (const CUresult r = cu->cuCtxCreate(&ctx, 0, dev); r != CUDA_SUCCESS) {
        c.why = "cuCtxCreate failed: " + CuError(cu, r);
        c.cudaError = true;
        return c;
    }
    CUVIDDECODECAPS caps{};
    caps.eCodecType = cudaVideoCodec_AV1;
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = static_cast<unsigned>(bitDepth > 8 ? bitDepth - 8 : 0);
    const CUresult r = cv->cuvidGetDecoderCaps(&caps);
    CUcontext popped = nullptr;
    cu->cuCtxPopCurrent(&popped);
    cu->cuCtxDestroy(ctx);
    if (r != CUDA_SUCCESS) {
        c.why = "cuvidGetDecoderCaps failed: " + CuError(cu, r);
        c.cudaError = true;
        return c;
    }
    c.checked = true;
    c.supported = caps.bIsSupported != 0;
    c.minWidth = caps.nMinWidth;
    c.minHeight = caps.nMinHeight;
    c.maxWidth = caps.nMaxWidth;
    c.maxHeight = caps.nMaxHeight;
    c.maxMacroblocks = caps.nMaxMBCount;
    return c;
}

}  // namespace

DecodeSupport JudgeDecode(const VideoStreamInfo& info, const NvdecAv1Caps* caps) {
    DecodeSupport s;
    s.hardwareOnly = info.hardwareOnly;
    s.codec = info.codecName;
    s.nvdecCodec = info.nvdecCodec;
    s.pixelFormat = info.pixelFormat;
    s.bitDepth = info.bitDepth;
    s.chroma = info.chroma;
    s.width = info.codedWidth;
    s.height = info.codedHeight;
    if (caps) s.caps = *caps;
    if (!info.hardwareOnly) {
        s.why = "software (" + info.codecName + ")";
        return s;
    }
    const bool unknown = info.pixelFormat == "unknown";  // the probe did not learn it: NVDEC decides on the first frame (CUDA is asked)
    if (!unknown && !Av1NvdecLayout(av_get_pix_fmt(info.pixelFormat.c_str()))) return Refuse(s, DecodeBlock::Layout);
    // an odd height (OddHeight) after the GPU's answer: a machine whose GPU has no AV1 says that first - the even height would not help
    // there; without an answer (VideoDecoder: a part, NVDEC refusing a frame) it is refused before going on blind
    // (a layout the probe did not learn: its size is the container's, the sequence header unread - the first frame judges the height, as
    // it does the size; ReceiveFrame has the frame's own)
    const bool oddHeight = !unknown && info.codedHeight % 2 == 1;
    const auto refuseOdd = [&] {  // the driver's own words (a CUDA error, a stand-in) are not this refusal's reason
        s.caps.why.clear();
        return Refuse(s, DecodeBlock::OddHeight);
    };
    if (!caps || !caps->checked) {  // the driver could not be asked (a CUDA error on the temporary context): FFmpeg asks it on the first frame
        if (oddHeight) return refuseOdd();
        s.why = "not checked" + (caps && !caps->why.empty() ? ": " + caps->why : std::string());
        return s;
    }
    if (!caps->cuda) return Refuse(s, DecodeBlock::NoCuda);
    if (caps->nvdecMissing) return Refuse(s, DecodeBlock::NoNvdec);
    if (!caps->supported) return Refuse(s, DecodeBlock::NoAv1);
    if (unknown) {  // (no odd height either: the probe's size is the container's)
        s.why = "on '" + caps->gpu + "', the sample layout is not known before the first frame";
        return s;
    }
    // as FFmpeg 7.1.1's nvdec_test_capabilities (libavcodec/nvdec.c): the coded area / 256, rounded down - neither looser (a promise
    // FFmpeg then breaks on the first frame) nor stricter (a refusal FFmpeg would not make)
    const uint64_t macroblocks = static_cast<uint64_t>(info.codedWidth) * info.codedHeight / 256;
    if (info.codedWidth < caps->minWidth || info.codedHeight < caps->minHeight || info.codedWidth > caps->maxWidth || info.codedHeight > caps->maxHeight ||
        macroblocks > caps->maxMacroblocks)
        return Refuse(s, DecodeBlock::Size);  // before an odd height: an even one would not bring the frame within the GPU's limits
    if (oddHeight) return refuseOdd();
    s.verified = true;
    s.why = "on '" + caps->gpu + "', AV1 " + Depth(info.bitDepth) + " 4:2:0 up to " + std::to_string(caps->maxWidth) + "x" + std::to_string(caps->maxHeight);
    return s;
}

NvdecAv1Caps QueryNvdecAv1Caps(int bitDepth) {
    static std::mutex mutex;
    static std::map<int, NvdecAv1Caps> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(bitDepth); it != cache.end()) return it->second;
    NvdecAv1Caps c = QueryLive(bitDepth);
    if (!c.checked) {  // no answer (a context that could not be created now ...): asked again next time, not remembered for the process
        Log()->debug("NVDEC AV1 {}: not checked: {}", Depth(bitDepth), c.why);
        return c;
    }
    Log()->debug("NVDEC AV1 {}: checked={} cuda={} supported={} gpu='{}' {}x{}..{}x{} mb<={} {}", Depth(bitDepth), c.checked, c.cuda, c.supported, c.gpu,
                 c.minWidth, c.minHeight, c.maxWidth, c.maxHeight, c.maxMacroblocks, c.why);
    cache.emplace(bitDepth, c);
    return c;
}

DecodeSupport CheckDecodeSupport(const VideoStreamInfo& info) {
    if (!info.hardwareOnly) return JudgeDecode(info, nullptr);
    // the layout first: no GPU asked (an odd height waits for the GPU's answer - JudgeDecode's order)
    if (const DecodeSupport pre = JudgeDecode(info, nullptr); pre.block == DecodeBlock::Layout) return pre;
    if (const auto fake = FakeCaps()) return JudgeDecode(info, &*fake);
    const NvdecAv1Caps caps = QueryNvdecAv1Caps(info.pixelFormat == "unknown" ? 8 : info.bitDepth);  // unknown: what 8 bits says
    return JudgeDecode(info, &caps);
}

DecodeSupport RefusedDecode(const VideoStreamInfo& info, DecodeBlock block, const std::string& detail) {
    DecodeSupport s = JudgeDecode(info, nullptr);
    s.caps.why = detail;
    return Refuse(s, block);
}

bool FakeNvdecRefusesFrames() {
    const std::string v = FakeValue();
    return v == "first-frame" || v == "start-fails" || v == "query-fails";
}

bool FakeNvdecDeviceFails() { return FakeValue() == "device-fails"; }

bool FakeNvdecRefusesLater() { return FakeValue() == "later-frame"; }

std::string DescribeDecode(const DecodeSupport& s) {
    if (!s.hardwareOnly) {
        if (!s.nvdecCodec) return s.why + "; NVDEC does not decode this codec";
        if (s.height % 2 == 1)  // as VideoDecoder decides it (an odd height: software also with --hwaccel cuda)
            return s.why + "; not NVDEC even with --hwaccel cuda: an odd frame height (" + std::to_string(s.height) +
                   ") - FFmpeg's NVDEC gives its chroma wrong";
        return s.why + "; NVDEC with --hwaccel cuda when the GPU decodes this layout";
    }
    if (!s.Ok()) return "unavailable - " + s.why;
    return "NVDEC only (" + s.codec + ": no software AV1 decoder in this build), " + s.why;
}

}  // namespace dlssvid
