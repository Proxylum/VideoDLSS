// TASK-0076: AV1 has no software decoder in this build - which decoders need NVDEC, whether NVDEC takes a stream (JudgeDecode), the pixel
// format VideoDecoder asks FFmpeg for (PickNvdecFormat), the DLSSVID_FAKE_NVDEC_AV1 stand-in for the driver and the `info` line. No GPU
// work: CheckDecodeSupport asks the driver for its AV1 caps once (a temporary context, no decoding), whatever it answers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <cstdlib>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "io/NvdecFormat.h"
#include "io/NvdecSupport.h"
#include "io/VideoDecoder.h"
#include "util/Log.h"

#include <spdlog/sinks/ostream_sink.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

using namespace dlssvid;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

namespace {

// DLSSVID_FAKE_NVDEC_AV1 for one scope (nullptr: unset); the value before it comes back afterwards.
struct FakeNvdecEnv {
    std::optional<std::string> before;
    explicit FakeNvdecEnv(const char* value) {
        if (const char* e = std::getenv("DLSSVID_FAKE_NVDEC_AV1")) before = e;
        _putenv_s("DLSSVID_FAKE_NVDEC_AV1", value ? value : "");
    }
    ~FakeNvdecEnv() { _putenv_s("DLSSVID_FAKE_NVDEC_AV1", before ? before->c_str() : ""); }
};

// VideoStreamInfo::chroma as VideoDecoder names it for a pixel format
std::string ChromaOf(const std::string& pixelFormat) {
    if (pixelFormat.rfind("yuv420p", 0) == 0) return "420";
    if (pixelFormat.rfind("yuv422p", 0) == 0) return "422";
    if (pixelFormat.rfind("yuv444p", 0) == 0) return "444";
    return "other";
}

// What VideoDecoder reports for an AV1 stream of this build (VideoStreamInfo::hardwareOnly), 320x192 8-bit 4:2:0 unless told otherwise.
VideoStreamInfo Av1Info(const std::string& pixelFormat = "yuv420p", int bitDepth = 8, uint32_t w = 320, uint32_t h = 192) {
    VideoStreamInfo i;
    i.codecName = "av1";
    i.hardwareOnly = true;
    i.nvdecCodec = true;
    i.pixelFormat = pixelFormat;
    i.bitDepth = bitDepth;
    i.chroma = ChromaOf(pixelFormat);
    i.width = i.codedWidth = w;
    i.height = i.codedHeight = h;
    return i;
}

// A codec with a software path that NVDEC also decodes (h264), and one NVDEC does not decode at all (ffv1)
VideoStreamInfo SoftwareInfo(const std::string& codec, bool nvdecCodec) {
    VideoStreamInfo i;
    i.codecName = codec;
    i.hardwareOnly = false;
    i.nvdecCodec = nvdecCodec;
    i.pixelFormat = "yuv420p";
    i.chroma = "420";
    i.width = i.codedWidth = 320;
    i.height = i.codedHeight = 192;
    return i;
}
VideoStreamInfo H264Info() { return SoftwareInfo("h264", true); }
VideoStreamInfo Ffv1Info() { return SoftwareInfo("ffv1", false); }

// A GPU whose NVDEC decodes AV1 from 64x64 up to 8192x8192
NvdecAv1Caps GoodCaps() {
    NvdecAv1Caps c;
    c.checked = true;
    c.cuda = true;
    c.supported = true;
    c.gpu = "Test GPU";
    c.minWidth = 64;
    c.minHeight = 64;
    c.maxWidth = 8192;
    c.maxHeight = 8192;
    c.maxMacroblocks = 262144;
    return c;
}

const char* const kAllBlocks[] = {"None", "Layout", "NoCuda", "NoNvdec", "NoAv1", "Size", "NvdecFailed", "NvdecStartFailed"};

std::string BlockName(DecodeBlock b) {
    const int i = static_cast<int>(b);
    return i >= 0 && i < static_cast<int>(std::size(kAllBlocks)) ? kAllBlocks[i] : "?";
}

// Every refusal names AV1 and NVDEC, and never FFmpeg's bare «Function not implemented» (the message before TASK-0076).
void CheckRefusal(const DecodeSupport& s, DecodeBlock block) {
    INFO("block " << BlockName(s.block) << ", why: " << s.why);
    CHECK(s.block == block);
    CHECK_FALSE(s.Ok());
    CHECK_FALSE(s.verified);
    CHECK(s.hardwareOnly);
    CHECK_THAT(s.why, ContainsSubstring("AV1"));
    CHECK_THAT(s.why, ContainsSubstring("NVDEC"));
    CHECK_THAT(s.why, !ContainsSubstring("Function not implemented"));
}

}  // namespace

TEST_CASE("HardwareOnlyDecoder: only the built-in av1 decoder of this build has no software path", "[unit][av1]") {
    const AVCodec* av1 = avcodec_find_decoder_by_name("av1");
    REQUIRE(av1);
    CHECK(HardwareOnlyDecoder(av1));
    // what av_find_best_stream hands VideoDecoder for an AV1 stream is that one
    CHECK(HardwareOnlyDecoder(avcodec_find_decoder(AV_CODEC_ID_AV1)));
    for (const char* name : {"h264", "hevc", "ffv1", "prores", "av1_cuvid", "h264_cuvid", "hevc_cuvid"}) {
        INFO(name);
        CHECK_FALSE(HardwareOnlyDecoder(avcodec_find_decoder_by_name(name)));  // a decoder this build lacks: nullptr, false as well
    }
    CHECK_FALSE(HardwareOnlyDecoder(nullptr));
}

TEST_CASE("HasNvdecHwaccel: the codecs NVDEC decodes at all, through a CUDA device context", "[unit][av1]") {
    for (const char* name : {"h264", "hevc", "av1"}) {
        INFO(name);
        const AVCodec* c = avcodec_find_decoder_by_name(name);
        REQUIRE(c);
        CHECK(HasNvdecHwaccel(c));
    }
    for (const char* name : {"ffv1", "prores", "dnxhd"}) {
        INFO(name);
        const AVCodec* c = avcodec_find_decoder_by_name(name);
        REQUIRE(c);
        CHECK_FALSE(HasNvdecHwaccel(c));
    }
    CHECK_FALSE(HasNvdecHwaccel(nullptr));
}

TEST_CASE("Build premise: FFmpeg has no software AV1 decoder, its av1 decodes through a CUDA hwaccel", "[unit][av1]") {
    // If this fails, the build of FFmpeg changed (libdav1d / libaom added): HardwareOnlyDecoder switches itself off for the decoder FFmpeg
    // then picks, and TASK-0076's NVDEC-only path, its texts («no software AV1 decoder in this build») and these tests need a look.
    const AVCodec* best = avcodec_find_decoder(AV_CODEC_ID_AV1);
    REQUIRE(best);
    CHECK(std::string(best->name) == "av1");
    CHECK(avcodec_find_decoder_by_name("libdav1d") == nullptr);
    CHECK(avcodec_find_decoder_by_name("libaom-av1") == nullptr);
    bool cuda = false;
    for (int i = 0;; ++i) {
        const AVCodecHWConfig* c = avcodec_get_hw_config(best, i);
        if (!c) break;
        cuda = cuda || (c->device_type == AV_HWDEVICE_TYPE_CUDA && c->pix_fmt == AV_PIX_FMT_CUDA && (c->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX));
    }
    CHECK(cuda);
}

TEST_CASE("JudgeDecode: a decoder with a software path is fine whatever NVDEC says", "[unit][av1]") {
    NvdecAv1Caps noCuda;
    noCuda.checked = true;
    noCuda.why = "no CUDA device";
    VideoStreamInfo odd = H264Info();
    odd.pixelFormat = "yuv444p12le";  // a layout NVDEC would not take for AV1: not looked at without hardwareOnly
    odd.chroma = "444";
    for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), static_cast<const NvdecAv1Caps*>(&noCuda)}) {
        const DecodeSupport s = JudgeDecode(H264Info(), caps);
        CHECK(s.Ok());
        CHECK_FALSE(s.hardwareOnly);
        CHECK_FALSE(s.verified);
        CHECK(s.nvdecCodec);
        CHECK(s.why == "software (h264)");
        CHECK(JudgeDecode(odd, caps).Ok());
        const DecodeSupport f = JudgeDecode(Ffv1Info(), caps);
        CHECK(f.Ok());
        CHECK_FALSE(f.nvdecCodec);
        CHECK(f.why == "software (ffv1)");
    }
}

TEST_CASE("JudgeDecode: the layout is judged before NVDEC is asked; an unknown layout and an unanswered driver go on unverified", "[unit][av1]") {
    const NvdecAv1Caps good = GoodCaps();
    NvdecAv1Caps noCuda;
    noCuda.checked = true;
    noCuda.why = "no CUDA device";

    // the layouts NVDEC decodes AV1 in: 8 and 10-bit 4:2:0
    CHECK(JudgeDecode(Av1Info("yuv420p", 8), &good).Ok());
    CHECK(JudgeDecode(Av1Info("yuv420p10le", 10), &good).Ok());
    // any other is refused as a layout - whatever the driver said, and before it (a GPU without CUDA too); the text names it in words
    struct LayoutCase {
        const char* pix;
        int depth;
        const char* words;
    };
    const LayoutCase layouts[] = {
        {"yuv444p", 8, "8-bit 4:4:4 (yuv444p)"},
        {"yuv422p10le", 10, "10-bit 4:2:2 (yuv422p10le)"},
        {"yuv420p12le", 12, "12-bit 4:2:0 (yuv420p12le)"},
        {"gray", 8, "8-bit monochrome / other (gray)"},
        {"yuv444p10le", 10, "10-bit 4:4:4 (yuv444p10le)"},
    };
    for (const LayoutCase& c : layouts) {
        INFO(c.pix);
        for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), &good, static_cast<const NvdecAv1Caps*>(&noCuda)}) {
            const DecodeSupport s = JudgeDecode(Av1Info(c.pix, c.depth), caps);
            CheckRefusal(s, DecodeBlock::Layout);
            CHECK(s.why == std::string("AV1 ") + c.words +
                               " cannot be decoded here - this build has no software AV1 decoder and decodes AV1 on NVDEC only as 8 or 10-bit 4:2:0");
            CHECK(s.pixelFormat == c.pix);
            CHECK(s.chroma == ChromaOf(c.pix));
        }
    }

    // an unknown layout (the probe did not learn it): what the driver says of CUDA, NVDEC and AV1 still refuses it; past that the first
    // frame decides (the layout, the size) - Ok, not verified
    {
        NvdecAv1Caps noNvdec = GoodCaps();
        noNvdec.nvdecMissing = true;
        noNvdec.supported = false;
        noNvdec.why = "nvcuvid.dll could not be loaded (missing, or a driver too old for FFmpeg's NVDEC)";
        NvdecAv1Caps noAv1 = GoodCaps();
        noAv1.supported = false;
        CheckRefusal(JudgeDecode(Av1Info("unknown"), &noCuda), DecodeBlock::NoCuda);
        CheckRefusal(JudgeDecode(Av1Info("unknown"), &noNvdec), DecodeBlock::NoNvdec);
        CheckRefusal(JudgeDecode(Av1Info("unknown"), &noAv1), DecodeBlock::NoAv1);
        const DecodeSupport s = JudgeDecode(Av1Info("unknown"), &good);
        CHECK(s.Ok());
        CHECK_FALSE(s.verified);
        CHECK(s.hardwareOnly);
        CHECK(s.why == "on 'Test GPU', the sample layout is not known before the first frame");
        CHECK(DescribeDecode(s) == "NVDEC only (av1: no software AV1 decoder in this build), on 'Test GPU', the sample layout is not known "
                                   "before the first frame");
        // the size is the first frame's to judge too (the probe's size of a stream it could not read is no basis for a refusal)
        NvdecAv1Caps narrow = GoodCaps();
        narrow.minWidth = 400;
        CHECK(JudgeDecode(Av1Info("unknown"), &narrow).Ok());
        // the driver not asked, or no answer from it: unverified as any stream
        const DecodeSupport blind = JudgeDecode(Av1Info("unknown"), nullptr);
        CHECK(blind.Ok());
        CHECK_FALSE(blind.verified);
        CHECK(blind.why == "not checked");
    }

    // the driver not asked, or no definite answer from it: on blind, as FFmpeg does - not verified, the reason kept
    const DecodeSupport notAsked = JudgeDecode(Av1Info(), nullptr);
    CHECK(notAsked.Ok());
    CHECK_FALSE(notAsked.verified);
    CHECK(notAsked.why == "not checked");
    NvdecAv1Caps unanswered = GoodCaps();
    unanswered.checked = false;
    unanswered.supported = false;  // not read: must not count
    unanswered.cuda = false;
    unanswered.nvdecMissing = true;
    unanswered.why = "the driver does not report NVDEC capabilities";
    const DecodeSupport blind = JudgeDecode(Av1Info(), &unanswered);
    CHECK(blind.Ok());
    CHECK_FALSE(blind.verified);
    CHECK(blind.why == "not checked: the driver does not report NVDEC capabilities");
    CHECK_FALSE(blind.caps.cudaError);  // an old driver: NVDEC refusing the first frame is the stream's verdict then
    NvdecAv1Caps cudaFailed = unanswered;
    cudaFailed.cudaError = true;
    cudaFailed.why = "cuCtxCreate failed: CUDA_ERROR_OUT_OF_MEMORY";
    const DecodeSupport memory = JudgeDecode(Av1Info(), &cudaFailed);
    CHECK(memory.Ok());
    CHECK_FALSE(memory.verified);
    CHECK(memory.caps.cudaError);  // VideoDecoder takes NVDEC refusing the first frame then for a start failure
    CHECK(memory.why == "not checked: cuCtxCreate failed: CUDA_ERROR_OUT_OF_MEMORY");
}

TEST_CASE("JudgeDecode: no CUDA, no NVDEC, no AV1 decoder, the size limits each refuse with the reason, in that order", "[unit][av1]") {
    // no CUDA
    NvdecAv1Caps noCuda = GoodCaps();
    noCuda.cuda = false;
    noCuda.supported = false;
    noCuda.gpu.clear();
    noCuda.why = "nvcuda.dll could not be loaded (no NVIDIA driver)";
    const DecodeSupport nc = JudgeDecode(Av1Info(), &noCuda);
    CheckRefusal(nc, DecodeBlock::NoCuda);
    CHECK(nc.why == "AV1 decodes only on an NVIDIA GPU (NVDEC) in this build - there is no software AV1 decoder - and CUDA is unavailable: "
                    "nvcuda.dll could not be loaded (no NVIDIA driver)");
    // CUDA, but the driver's video decoder library is missing
    NvdecAv1Caps noNvdec = GoodCaps();
    noNvdec.supported = false;
    noNvdec.nvdecMissing = true;
    noNvdec.gpu = "NVIDIA GeForce RTX 3060";
    noNvdec.why = "nvcuvid.dll could not be loaded (missing, or a driver too old for FFmpeg's NVDEC)";
    const DecodeSupport nn = JudgeDecode(Av1Info(), &noNvdec);
    CheckRefusal(nn, DecodeBlock::NoNvdec);
    CHECK(nn.why == "AV1 decodes only on NVDEC in this build - there is no software AV1 decoder - and the NVIDIA driver's video decoder is unavailable: "
                    "nvcuvid.dll could not be loaded (missing, or a driver too old for FFmpeg's NVDEC)");
    // CUDA and NVDEC but no AV1 in it: the GPU and the depth named
    NvdecAv1Caps noAv1 = GoodCaps();
    noAv1.supported = false;
    noAv1.gpu = "NVIDIA GeForce GTX 1080";
    const DecodeSupport na8 = JudgeDecode(Av1Info(), &noAv1);
    CheckRefusal(na8, DecodeBlock::NoAv1);
    CHECK_THAT(na8.why, ContainsSubstring("'NVIDIA GeForce GTX 1080'"));
    CHECK_THAT(na8.why, ContainsSubstring("8-bit 4:2:0"));
    CHECK_THAT(na8.why, ContainsSubstring("no software AV1 decoder"));
    const DecodeSupport na10 = JudgeDecode(Av1Info("yuv420p10le", 10), &noAv1);
    CheckRefusal(na10, DecodeBlock::NoAv1);
    CHECK_THAT(na10.why, ContainsSubstring("10-bit 4:2:0"));
    // the order: no CUDA before no NVDEC before no AV1 (a machine without the first has none of the others either)
    NvdecAv1Caps neither = noNvdec;
    neither.cuda = false;
    CHECK(JudgeDecode(Av1Info(), &neither).block == DecodeBlock::NoCuda);
    NvdecAv1Caps nvdecButSupported = noNvdec;
    nvdecButSupported.supported = true;  // no library, whatever else was read
    CHECK(JudgeDecode(Av1Info(), &nvdecButSupported).block == DecodeBlock::NoNvdec);

    // the size limits, each one alone (320x192 = 20 x 12 = 240 macroblocks)
    struct SizeCase {
        const char* what;
        unsigned minW, minH, maxW, maxH, maxMb;
        bool ok;
    };
    const SizeCase cases[] = {
        {"at every limit exactly", 320, 192, 320, 192, 240, true},
        {"well inside", 64, 64, 8192, 8192, 262144, true},
        {"narrower than the minimum", 321, 64, 8192, 8192, 262144, false},
        {"lower than the minimum", 64, 193, 8192, 8192, 262144, false},
        {"wider than the maximum", 64, 64, 319, 8192, 262144, false},
        {"taller than the maximum", 64, 64, 8192, 191, 262144, false},
        {"more macroblocks than the maximum", 64, 64, 8192, 8192, 239, false},
        // a limit swapped onto the other side would refuse these
        {"the minimum below, the maximum above", 16, 16, 4096, 2304, 36864, true},
    };
    for (const SizeCase& c : cases) {
        INFO(c.what);
        NvdecAv1Caps caps = GoodCaps();
        caps.gpu = "NVIDIA GeForce RTX 4090";
        caps.minWidth = c.minW;
        caps.minHeight = c.minH;
        caps.maxWidth = c.maxW;
        caps.maxHeight = c.maxH;
        caps.maxMacroblocks = c.maxMb;
        const DecodeSupport s = JudgeDecode(Av1Info(), &caps);
        if (c.ok) {
            CHECK(s.Ok());
            CHECK(s.verified);
        } else {
            CheckRefusal(s, DecodeBlock::Size);
            CHECK_THAT(s.why, ContainsSubstring("320x192"));
            CHECK_THAT(s.why, ContainsSubstring("'NVIDIA GeForce RTX 4090'"));
            CHECK_THAT(s.why, ContainsSubstring(std::to_string(c.minW) + "x" + std::to_string(c.minH) + " to " + std::to_string(c.maxW) + "x" +
                                                std::to_string(c.maxH) + ", at most " + std::to_string(c.maxMb) + " macroblocks"));
        }
    }
    // macroblocks as FFmpeg 7.1.1's nvdec_test_capabilities counts them (libavcodec/nvdec.c): the coded area / 256, rounded down.
    // 330x200 is 66000 / 256 = 257 - a count by whole blocks (20 x 12 = 240) would let through a frame FFmpeg refuses on the first
    // frame, one rounded up per side (21 x 13 = 273) would refuse one FFmpeg decodes
    NvdecAv1Caps mb = GoodCaps();
    mb.maxMacroblocks = 257;
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 330, 200), &mb).Ok());
    mb.maxMacroblocks = 256;
    CheckRefusal(JudgeDecode(Av1Info("yuv420p", 8, 330, 200), &mb), DecodeBlock::Size);
    // 336x200 is 67200 / 256 = 262.5: rounded down
    mb.maxMacroblocks = 262;
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 336, 200), &mb).Ok());
    mb.maxMacroblocks = 261;
    CheckRefusal(JudgeDecode(Av1Info("yuv420p", 8, 336, 200), &mb), DecodeBlock::Size);
    // the stored frame is what is judged, not the frame as shown (a turned phone video)
    VideoStreamInfo turned = Av1Info();
    turned.width = 192;
    turned.height = 320;
    NvdecAv1Caps wide = GoodCaps();
    wide.maxHeight = 200;  // the stored 192 rows fit, the shown 320 would not
    CHECK(JudgeDecode(turned, &wide).Ok());
}

TEST_CASE("JudgeDecode: a stream NVDEC takes is verified and described; RefusedDecode names the failure", "[unit][av1]") {
    const NvdecAv1Caps good = GoodCaps();
    const DecodeSupport s = JudgeDecode(Av1Info("yuv420p10le", 10, 1920, 1080), &good);
    CHECK(s.Ok());
    CHECK(s.block == DecodeBlock::None);
    CHECK(s.verified);
    CHECK(s.hardwareOnly);
    CHECK(s.nvdecCodec);
    CHECK(s.codec == "av1");
    CHECK(s.pixelFormat == "yuv420p10le");
    CHECK(s.bitDepth == 10);
    CHECK(s.chroma == "420");
    CHECK(s.width == 1920);
    CHECK(s.height == 1080);
    CHECK(s.caps.gpu == "Test GPU");
    CHECK(s.why == "on 'Test GPU', AV1 10-bit 4:2:0 up to 8192x8192");

    // VideoDecoder's refusals: NVDEC refusing the stream (ENOSYS, the stream's verdict) or not starting (NvdecStartFailed: ENOSYS on the
    // first frame of a stream the driver took, or the CUDA device not created - memory, the driver; no verdict)
    const DecodeSupport failed = RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "");
    CheckRefusal(failed, DecodeBlock::NvdecFailed);
    CHECK(failed.why == "NVDEC refused this AV1 stream and this build has no software AV1 decoder");
    CHECK(RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "on the first frame").why ==
          "NVDEC refused this AV1 stream and this build has no software AV1 decoder (on the first frame)");
    CHECK(RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "after 12 frames").why ==
          "NVDEC refused this AV1 stream and this build has no software AV1 decoder (after 12 frames)");
    CHECK_FALSE(failed.Transient());
    // the driver having said yes, NVDEC did not start (memory, the driver): no verdict on the stream - transient
    const DecodeSupport start = RefusedDecode(Av1Info(), DecodeBlock::NvdecStartFailed, "on the first frame");
    CheckRefusal(start, DecodeBlock::NvdecStartFailed);
    CHECK(start.Transient());
    CHECK(start.why == "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 "
                       "decoder (on the first frame)");
    CHECK(RefusedDecode(Av1Info(), DecodeBlock::NvdecStartFailed, "").why ==
          "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 decoder");
    CHECK(RefusedDecode(Av1Info(), DecodeBlock::NvdecStartFailed, "the CUDA device could not be created: x").why ==
          "NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software AV1 decoder (the "
          "CUDA device could not be created: x)");
    for (const DecodeBlock b : {DecodeBlock::None, DecodeBlock::Layout, DecodeBlock::NoCuda, DecodeBlock::NoNvdec, DecodeBlock::NoAv1, DecodeBlock::Size,
                                DecodeBlock::NvdecFailed}) {
        DecodeSupport d;
        d.block = b;
        INFO(BlockName(b));
        CHECK_FALSE(d.Transient());
    }
    const DecodeSupport device =
        RefusedDecode(Av1Info(), DecodeBlock::NvdecStartFailed, "the CUDA device could not be created: Generic error in an external library");
    CheckRefusal(device, DecodeBlock::NvdecStartFailed);
    CHECK(device.Transient());
    CHECK_THAT(device.why, ContainsSubstring("the CUDA device could not be created: Generic error in an external library"));
    CHECK(device.codec == "av1");
    CHECK(device.width == 320);
}

TEST_CASE("PickNvdecFormat: CUDA when offered; else a hardware-only decoder chooses nothing, the others go on in software", "[unit][av1]") {
    // av1dec's list with this build's hwaccels: the D3D ones first, CUDA among them, the software format last
    const AVPixelFormat av1List[] = {AV_PIX_FMT_DXVA2_VLD, AV_PIX_FMT_D3D11VA_VLD, AV_PIX_FMT_D3D11, AV_PIX_FMT_CUDA, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(av1List, true) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(av1List, false) == AV_PIX_FMT_CUDA);
    // CUDA struck from the list (its hwaccel failed to start): the software format would decode nothing for AV1 (ENOSYS)
    const AVPixelFormat noCuda[] = {AV_PIX_FMT_D3D11, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(noCuda, true) == AV_PIX_FMT_NONE);
    CHECK(PickNvdecFormat(noCuda, false) == AV_PIX_FMT_D3D11);  // as before TASK-0076: the first, FFmpeg's cascade down to software
    const AVPixelFormat software[] = {AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(software, true) == AV_PIX_FMT_NONE);
    CHECK(PickNvdecFormat(software, false) == AV_PIX_FMT_YUV420P10LE);
}

TEST_CASE("CheckDecodeSupport: DLSSVID_FAKE_NVDEC_AV1 stands in for the driver on every call, also after the driver was asked", "[unit][av1][gpu]") {
    FakeNvdecEnv unset(nullptr);
    const VideoStreamInfo av1 = Av1Info();
    // the driver's own answer first (it fills the per-process cache), whatever this machine says
    const DecodeSupport real = CheckDecodeSupport(av1);
    INFO("this machine: " << BlockName(real.block) << " - " << real.why);
    CHECK(real.hardwareOnly);
    CHECK((real.block == DecodeBlock::None || real.block == DecodeBlock::NoCuda || real.block == DecodeBlock::NoNvdec || real.block == DecodeBlock::NoAv1 ||
           real.block == DecodeBlock::Size));
    if (real.Ok()) CHECK(real.verified == real.caps.checked);
    CHECK(CheckDecodeSupport(av1).block == real.block);  // the cache answers the same
    CHECK_FALSE(FakeNvdecRefusesFrames());
    // an unknown layout (at any depth the probe guessed): the driver's 8-bit answer refuses it as it would refuse 8-bit AV1; the size is
    // left to the first frame
    const DecodeSupport realUnknown = CheckDecodeSupport(Av1Info("unknown", 10));
    if (real.block == DecodeBlock::Size) CHECK(realUnknown.Ok());
    else CHECK(realUnknown.block == real.block);
    if (realUnknown.Ok() && real.caps.checked) CHECK_THAT(realUnknown.why, ContainsSubstring("the sample layout is not known before the first frame"));

    {
        FakeNvdecEnv fake("no-av1");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoAv1);
        CHECK(s.caps.gpu == "DLSSVID_FAKE_NVDEC_AV1");
        CHECK_THAT(s.why, ContainsSubstring("DLSSVID_FAKE_NVDEC_AV1=no-av1"));
        CheckRefusal(CheckDecodeSupport(Av1Info("yuv420p10le", 10)), DecodeBlock::NoAv1);
        // not consulted where the driver would not be: software decoders, layouts refused first; an unknown layout is refused by it
        CHECK(CheckDecodeSupport(H264Info()).Ok());
        CHECK(CheckDecodeSupport(H264Info()).why == "software (h264)");
        CHECK(CheckDecodeSupport(Av1Info("yuv444p")).block == DecodeBlock::Layout);
        CheckRefusal(CheckDecodeSupport(Av1Info("unknown")), DecodeBlock::NoAv1);
        CHECK_FALSE(FakeNvdecRefusesFrames());
    }
    {
        FakeNvdecEnv fake("no-cuda");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoCuda);
        CHECK_THAT(s.why, ContainsSubstring("DLSSVID_FAKE_NVDEC_AV1=no-cuda"));
        CHECK(CheckDecodeSupport(H264Info()).Ok());
        CheckRefusal(CheckDecodeSupport(Av1Info("unknown", 10)), DecodeBlock::NoCuda);
        CHECK_FALSE(FakeNvdecRefusesFrames());
    }
    {
        FakeNvdecEnv fake("no-nvdec");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoNvdec);
        CHECK(s.caps.gpu == "DLSSVID_FAKE_NVDEC_AV1");
        CHECK(s.why == "AV1 decodes only on NVDEC in this build - there is no software AV1 decoder - and the NVIDIA driver's video decoder is unavailable: "
                       "DLSSVID_FAKE_NVDEC_AV1=no-nvdec");
        CheckRefusal(CheckDecodeSupport(Av1Info("unknown")), DecodeBlock::NoNvdec);
        CHECK_FALSE(FakeNvdecRefusesFrames());
    }
    {
        // no answer from the driver: the stream goes on unchecked - and VideoDecoder's NVDEC then refuses its first frame
        FakeNvdecEnv fake("first-frame");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CHECK(s.Ok());
        CHECK_FALSE(s.verified);
        CHECK_FALSE(s.caps.checked);
        CHECK(s.why == "not checked: DLSSVID_FAKE_NVDEC_AV1=first-frame");
        CHECK(FakeNvdecRefusesFrames());
        CHECK(CheckDecodeSupport(Av1Info("yuv444p")).block == DecodeBlock::Layout);  // the layout is still refused first
        const DecodeSupport unknown = CheckDecodeSupport(Av1Info("unknown"));
        CHECK(unknown.Ok());
        CHECK(unknown.why == "not checked: DLSSVID_FAKE_NVDEC_AV1=first-frame");
    }
    {
        // the driver says yes (verified, limits wide open) - and NVDEC then does not start: VideoDecoder's get_format declines CUDA
        FakeNvdecEnv fake("start-fails");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CHECK(s.Ok());
        CHECK(s.verified);
        CHECK(s.caps.checked);
        CHECK(s.why == "on 'DLSSVID_FAKE_NVDEC_AV1', AV1 8-bit 4:2:0 up to 65535x65535");
        CHECK(CheckDecodeSupport(Av1Info("yuv420p", 8, 8192, 4320)).verified);
        CHECK(FakeNvdecRefusesFrames());
        CHECK_FALSE(FakeNvdecDeviceFails());
        CHECK(CheckDecodeSupport(Av1Info("yuv444p")).block == DecodeBlock::Layout);
    }
    {
        // the driver says yes - and the CUDA device is not created (VideoDecoder::StartNvdec): no frame is ever asked of NVDEC
        FakeNvdecEnv fake("device-fails");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CHECK(s.Ok());
        CHECK(s.verified);
        CHECK(s.why == "on 'DLSSVID_FAKE_NVDEC_AV1', AV1 8-bit 4:2:0 up to 65535x65535");
        CHECK(FakeNvdecDeviceFails());
        CHECK_FALSE(FakeNvdecRefusesFrames());
    }
    {
        // the driver could not be asked for a CUDA error (a context not created now: memory) - unchecked, and NVDEC then refuses the first
        // frame: VideoDecoder takes that for a start failure, not a verdict (cudaError)
        FakeNvdecEnv fake("query-fails");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CHECK(s.Ok());
        CHECK_FALSE(s.verified);
        CHECK_FALSE(s.caps.checked);
        CHECK(s.caps.cudaError);
        CHECK(s.why == "not checked: DLSSVID_FAKE_NVDEC_AV1=query-fails");
        CHECK(FakeNvdecRefusesFrames());
        CHECK_FALSE(FakeNvdecDeviceFails());
        CHECK_FALSE(FakeNvdecRefusesLater());
    }
    {
        // the driver takes the stream; NVDEC refuses a frame after a few (VideoDecoder): the stream's verdict
        FakeNvdecEnv fake("later-frame");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CHECK(s.Ok());
        CHECK(s.verified);
        CHECK(FakeNvdecRefusesLater());
        CHECK_FALSE(FakeNvdecRefusesFrames());
        CHECK_FALSE(FakeNvdecDeviceFails());
    }
    for (const char* other : {"no-cuda", "no-nvdec", "no-av1", "first-frame", "start-fails", "query-fails", "device-fails"}) {
        INFO(other);
        FakeNvdecEnv fake(other);
        CHECK_FALSE(FakeNvdecRefusesLater());
    }
    for (const char* other : {"no-cuda", "no-nvdec", "no-av1", "first-frame", "start-fails", "query-fails", "later-frame"}) {
        INFO(other);
        FakeNvdecEnv fake(other);
        CHECK_FALSE(FakeNvdecDeviceFails());
    }
    {
        FakeNvdecEnv fake("bogus");  // not understood: ignored, the driver answers
        CHECK(CheckDecodeSupport(av1).block == real.block);
        CHECK_FALSE(FakeNvdecRefusesFrames());
        CHECK_FALSE(FakeNvdecDeviceFails());
    }
    CHECK(CheckDecodeSupport(av1).block == real.block);  // gone with its scope
    CHECK_FALSE(FakeNvdecRefusesFrames());
    CHECK_FALSE(FakeNvdecDeviceFails());
}

TEST_CASE("DescribeDecode: the decode line of dlssvid info in its forms", "[unit][av1]") {
    CHECK(DescribeDecode(JudgeDecode(H264Info(), nullptr)) == "software (h264); NVDEC with --hwaccel cuda when the GPU decodes this layout");
    CHECK(DescribeDecode(JudgeDecode(Ffv1Info(), nullptr)) == "software (ffv1); NVDEC does not decode this codec");
    const NvdecAv1Caps good = GoodCaps();
    CHECK(DescribeDecode(JudgeDecode(Av1Info(), &good)) == "NVDEC only (av1: no software AV1 decoder in this build), on 'Test GPU', AV1 8-bit 4:2:0 up to 8192x8192");
    CHECK(DescribeDecode(JudgeDecode(Av1Info(), nullptr)) == "NVDEC only (av1: no software AV1 decoder in this build), not checked");
    NvdecAv1Caps noAv1 = GoodCaps();
    noAv1.supported = false;
    const DecodeSupport refused = JudgeDecode(Av1Info(), &noAv1);
    CHECK(DescribeDecode(refused) == "unavailable - " + refused.why);
    CHECK_THAT(DescribeDecode(refused), StartsWith("unavailable - the GPU 'Test GPU' has no NVDEC AV1 decoder for 8-bit 4:2:0"));
    CHECK_THAT(DescribeDecode(JudgeDecode(Av1Info("yuv444p"), nullptr)), StartsWith("unavailable - AV1 8-bit 4:4:4 (yuv444p) cannot be decoded here"));
}

namespace {

// The process's log into `out` for one scope ("<level> <message>")
struct LogCapture {
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;
    explicit LogCapture(std::ostringstream& out) : sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(out)) {
        sink->set_pattern("%l %v");
        Log()->sinks().push_back(sink);
    }
    ~LogCapture() {
        auto& sinks = Log()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    }
};

size_t Count(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
}

constexpr const char* kFakeWarningsChild = "DLSSVID_FAKE_NVDEC_AV1 warnings in a fresh process - the child of the case that starts it";

}  // namespace

// Hidden ([.]): run by the case below in a process of its own - the warnings are said once per process and value, so a process that
// already set a value (any case before this one) would say nothing of it
TEST_CASE(kFakeWarningsChild, "[.][av1-child]") {
    std::ostringstream text;
    {
        LogCapture capture(text);
        for (const char* value : {"no-av1", "no-cuda", "no-av1", "start-fails", "no-cuda", "start-fails", "bogus-one", "bogus-two", "bogus-one"}) {
            FakeNvdecEnv fake(value);
            (void)CheckDecodeSupport(Av1Info());
        }
    }
    const std::string log = text.str();
    INFO(log);
    for (const char* value : {"no-av1", "no-cuda", "start-fails"}) {
        INFO(value);
        CHECK(Count(log, std::string("warning DLSSVID_FAKE_NVDEC_AV1=") + value + ": NVDEC's AV1 answer is faked") == 1);  // each value, once
    }
    CHECK(Count(log, "NVDEC's AV1 answer is faked") == 3);
    for (const char* value : {"bogus-one", "bogus-two"}) {
        INFO(value);
        CHECK(Count(log, std::string("warning DLSSVID_FAKE_NVDEC_AV1='") + value + "' is not no-cuda | ") == 1);
    }
    CHECK(Count(log, "later-frame: ignored") == 2);
}

// [gpu]: the child asks the driver for the values it does not understand (a CUDA context)
TEST_CASE("DLSSVID_FAKE_NVDEC_AV1 is said once per value, not once per process: three values faked, two not understood - five warnings", "[unit][av1][gpu]") {
    char* self = nullptr;
    REQUIRE(_get_pgmptr(&self) == 0);
    REQUIRE(self);
    const std::filesystem::path log = std::filesystem::path(DLSSVID_TEST_TMP) / "fake_warnings_child.log";
    std::filesystem::create_directories(log.parent_path());
    std::filesystem::remove(log);
    FakeNvdecEnv none(nullptr);  // the child starts without a stand-in
    const std::string cmd = std::string("\"\"") + self + "\" \"" + kFakeWarningsChild + "\" -r compact > \"" + log.string() + "\" 2>&1\"";
    const int code = std::system(cmd.c_str());
    std::ifstream in(log);
    const std::string out((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    INFO(out);
    CHECK(code == 0);
    CHECK_THAT(out, ContainsSubstring("All tests passed"));
    CHECK_THAT(out, ContainsSubstring("1 test case"));
}
