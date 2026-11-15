// TASK-0076: which streams decode only on NVDEC, and whether NVDEC takes them. The driver is asked through FFmpeg's own run-time loader
// (ffnvcodec's dynlink_loader.h: nvcuda.dll and nvcuvid.dll loaded on demand), never through <cuda.h> or the CUDA runtime - this
// unit must not be mixed with them, and the loader's types stay inside it.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <cstdarg>
#include <cstdio>

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
}

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>

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

bool HardwareOnlyDecoder(const AVCodec* codec) {
    if (!codec || codec->id != AV_CODEC_ID_AV1 || std::string(codec->name) != "av1") return false;
    for (int i = 0;; ++i) {
        const AVCodecHWConfig* c = avcodec_get_hw_config(codec, i);
        if (!c) return false;
        if (c->device_type == AV_HWDEVICE_TYPE_CUDA && (c->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) return true;
    }
}

AVPixelFormat PickNvdecFormat(const AVPixelFormat* formats, bool hardwareOnly) {
    for (const AVPixelFormat* p = formats; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    // no CUDA on offer: a decoder with a software path goes on with FFmpeg's cascade from the first format (as before); a hardware-only
    // one has nothing to fall back to - choosing nothing ends it at once (ENOSYS, reported with the reason)
    return hardwareOnly ? AV_PIX_FMT_NONE : formats[0];
}

namespace {

std::string Depth(int bitDepth) { return std::to_string(bitDepth) + "-bit"; }

std::string RefusalText(const DecodeSupport& s) {
    const std::string codec = "AV1";
    switch (s.block) {
        case DecodeBlock::Layout:
            return codec + " " + s.pixelFormat + " cannot be decoded here - this build has no software AV1 decoder and NVDEC decodes AV1 only as 8 or 10-bit 4:2:0";
        case DecodeBlock::NoCuda:
            return codec + " decodes only on an NVIDIA GPU (NVDEC) in this build - there is no software AV1 decoder - and CUDA is unavailable: " + s.caps.why;
        case DecodeBlock::NoAv1:
            return "the GPU '" + s.caps.gpu + "' has no NVDEC AV1 decoder for " + Depth(s.bitDepth) +
                   " 4:2:0 (for example, GeForce RTX 30 series and newer have one), and this build has no software AV1 decoder" +
                   (s.caps.why.empty() ? std::string() : " (" + s.caps.why + ")");
        case DecodeBlock::Size:
            return codec + " " + std::to_string(s.width) + "x" + std::to_string(s.height) + " is outside what NVDEC of '" + s.caps.gpu + "' decodes (" +
                   std::to_string(s.caps.minWidth) + "x" + std::to_string(s.caps.minHeight) + " to " + std::to_string(s.caps.maxWidth) + "x" +
                   std::to_string(s.caps.maxHeight) + ", at most " + std::to_string(s.caps.maxMacroblocks) + " macroblocks)";
        case DecodeBlock::NvdecFailed:
            return "NVDEC did not start decoding this AV1 stream and this build has no software AV1 decoder (FFmpeg's reason is in its log)" +
                   (s.caps.why.empty() ? std::string() : ": " + s.caps.why);
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

// DLSSVID_FAKE_NVDEC_AV1: the driver's answer stood in for (no-cuda | no-av1); nullopt: not set (or not understood - said once)
std::optional<NvdecAv1Caps> FakeCaps() {
    const char* v = std::getenv("DLSSVID_FAKE_NVDEC_AV1");
    if (!v || !*v) return std::nullopt;
    static std::once_flag said;
    const std::string value = v;
    NvdecAv1Caps c;
    c.checked = true;
    if (value == "no-cuda") {
        c.why = "DLSSVID_FAKE_NVDEC_AV1=no-cuda";
    } else if (value == "no-av1") {
        c.cuda = true;
        c.gpu = "DLSSVID_FAKE_NVDEC_AV1";
        c.why = "DLSSVID_FAKE_NVDEC_AV1=no-av1";
    } else {
        std::call_once(said, [&] { Log()->warn("DLSSVID_FAKE_NVDEC_AV1='{}' is not no-cuda | no-av1: ignored", value); });
        return std::nullopt;
    }
    std::call_once(said, [&] { Log()->warn("DLSSVID_FAKE_NVDEC_AV1={}: NVDEC's AV1 answer is faked", value); });
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
        c.why = "nvcuvid.dll could not be loaded";
        return c;
    }
    if (!cv->cuvidGetDecoderCaps) {  // an old driver: decide on the first frame
        c.why = "the driver does not report NVDEC capabilities";
        return c;
    }
    CUcontext ctx = nullptr;
    if (const CUresult r = cu->cuCtxCreate(&ctx, 0, dev); r != CUDA_SUCCESS) {
        c.why = "cuCtxCreate failed: " + CuError(cu, r);
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
    s.pixelFormat = info.pixelFormat;
    s.bitDepth = info.bitDepth;
    s.width = info.codedWidth;
    s.height = info.codedHeight;
    if (caps) s.caps = *caps;
    if (!info.hardwareOnly) {
        s.why = "software (" + info.codecName + ")";
        return s;
    }
    if (info.pixelFormat == "unknown") {  // the probe did not learn it: NVDEC decides on the first frame
        s.why = "NVDEC (" + info.codecName + "), the sample layout is not known before the first frame";
        return s;
    }
    if (info.pixelFormat != "yuv420p" && info.pixelFormat != "yuv420p10le") return Refuse(s, DecodeBlock::Layout);
    if (!caps || !caps->checked) {  // the driver could not be asked: go on blind, as FFmpeg does
        s.why = "NVDEC (" + info.codecName + "), not checked" + (caps && !caps->why.empty() ? ": " + caps->why : std::string());
        return s;
    }
    if (!caps->cuda) return Refuse(s, DecodeBlock::NoCuda);
    if (!caps->supported) return Refuse(s, DecodeBlock::NoAv1);
    const uint64_t macroblocks = static_cast<uint64_t>((info.codedWidth + 15) / 16) * ((info.codedHeight + 15) / 16);
    if (info.codedWidth < caps->minWidth || info.codedHeight < caps->minHeight || info.codedWidth > caps->maxWidth || info.codedHeight > caps->maxHeight ||
        macroblocks > caps->maxMacroblocks)
        return Refuse(s, DecodeBlock::Size);
    s.verified = true;
    s.why = "NVDEC on '" + caps->gpu + "', AV1 " + Depth(info.bitDepth) + " 4:2:0 up to " + std::to_string(caps->maxWidth) + "x" + std::to_string(caps->maxHeight);
    return s;
}

NvdecAv1Caps QueryNvdecAv1Caps(int bitDepth) {
    static std::mutex mutex;
    static std::map<int, NvdecAv1Caps> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(bitDepth); it != cache.end()) return it->second;
    NvdecAv1Caps c = QueryLive(bitDepth);
    Log()->debug("NVDEC AV1 {}: checked={} cuda={} supported={} gpu='{}' {}x{}..{}x{} mb<={} {}", Depth(bitDepth), c.checked, c.cuda, c.supported, c.gpu,
                 c.minWidth, c.minHeight, c.maxWidth, c.maxHeight, c.maxMacroblocks, c.why);
    cache.emplace(bitDepth, c);
    return c;
}

DecodeSupport CheckDecodeSupport(const VideoStreamInfo& info) {
    if (!info.hardwareOnly) return JudgeDecode(info, nullptr);
    if (const DecodeSupport pre = JudgeDecode(info, nullptr); !pre.Ok() || info.pixelFormat == "unknown") return pre;  // the layout first: no GPU asked
    if (const auto fake = FakeCaps()) return JudgeDecode(info, &*fake);
    const NvdecAv1Caps caps = QueryNvdecAv1Caps(info.bitDepth);
    return JudgeDecode(info, &caps);
}

DecodeSupport RefusedDecode(const VideoStreamInfo& info, DecodeBlock block, const std::string& detail) {
    DecodeSupport s = JudgeDecode(info, nullptr);
    s.caps.why = detail;
    return Refuse(s, block);
}

std::string DescribeDecode(const DecodeSupport& s) {
    if (!s.hardwareOnly) return s.why + ", NVDEC with --hwaccel cuda";
    if (!s.Ok()) return "unavailable - " + s.why;
    return "NVDEC only (" + s.codec + ": no software AV1 decoder in this build): " + s.why;
}

}  // namespace dlssvid
