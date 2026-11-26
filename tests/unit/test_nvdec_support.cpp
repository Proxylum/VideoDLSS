// TASK-0076: AV1 has no software decoder in this build - which decoders need NVDEC, whether NVDEC takes a stream (JudgeDecode), the pixel
// format VideoDecoder asks FFmpeg for (PickNvdecFormat), the DLSSVID_FAKE_NVDEC_AV1 stand-in for the driver and the `info` line. No GPU
// work: CheckDecodeSupport asks the driver for its AV1 caps once (a temporary context, no decoding), whatever it answers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
}

#include <cstdlib>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
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

const char* const kAllBlocks[] = {"None", "Layout", "NoCuda", "NoNvdec", "NoAv1", "Size", "OddHeight", "NvdecFailed", "Resized", "NvdecStartFailed"};

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
        {"gray", 8, "8-bit monochrome (gray)"},
        {"gbrp", 8, "8-bit RGB (gbrp)"},
        {"gbrp10le", 10, "10-bit RGB (gbrp10le)"},
        {"pal8", 8, "8-bit palette (pal8)"},
        {"yuv411p", 8, "8-bit 4:1:1 (yuv411p)"},
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
        // the depth asked about is the probe's default, not the stream's: said so
        CHECK(JudgeDecode(Av1Info("unknown"), &noAv1).why ==
              "the GPU 'Test GPU' has no NVDEC AV1 decoder (asked for 8-bit 4:2:0 - the probe did not learn the stream's layout; for example, "
              "GeForce RTX 30 series and newer have one), and this build has no software AV1 decoder");
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
            CHECK_THAT(s.why, ContainsSubstring("320x192 (the frame as stored)"));  // a turned video is shown the other way round
            CHECK_THAT(s.why, ContainsSubstring("no software AV1 decoder"));
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
    CHECK(RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "at frame 12").why ==
          "NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame 12)");
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
    // TASK-0076: CUDA offered for a layout this build leaves out (FFmpeg offers it for gray AV1; the list's last is the software format, as
    // ff_get_format reads it): a hardware-only decoder chooses nothing - the refusal names the layout; a decoder with a software path takes
    // CUDA as before
    for (const AVPixelFormat sw : {AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY10LE, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV420P12LE}) {
        INFO(av_get_pix_fmt_name(sw));
        const AVPixelFormat list[] = {AV_PIX_FMT_D3D11, AV_PIX_FMT_CUDA, sw, AV_PIX_FMT_NONE};
        CHECK(PickNvdecFormat(list, true) == AV_PIX_FMT_NONE);
        CHECK(PickNvdecFormat(list, false) == AV_PIX_FMT_CUDA);
        CHECK_FALSE(Av1NvdecLayout(sw));
    }
    for (const AVPixelFormat sw : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE}) {
        INFO(av_get_pix_fmt_name(sw));
        const AVPixelFormat list[] = {AV_PIX_FMT_D3D11, AV_PIX_FMT_CUDA, sw, AV_PIX_FMT_NONE};
        CHECK(PickNvdecFormat(list, true) == AV_PIX_FMT_CUDA);
        CHECK(PickNvdecFormat(list, false) == AV_PIX_FMT_CUDA);
        CHECK(Av1NvdecLayout(sw));
    }
    // the software format is the last one, whatever comes first: a 4:2:0 hardware format in front does not make gray decodable
    const AVPixelFormat grayAfter420[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_CUDA, AV_PIX_FMT_GRAY8, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(grayAfter420, true) == AV_PIX_FMT_NONE);
    const AVPixelFormat onlyCuda[] = {AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE};  // no software format: nothing to judge the layout by
    CHECK(PickNvdecFormat(onlyCuda, true) == AV_PIX_FMT_NONE);
    const AVPixelFormat empty[] = {AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(empty, true) == AV_PIX_FMT_NONE);
    // TASK-0076: a codec with a software path whose frame is of an odd height (VP9 1920x803, MPEG-2 ...): FFmpeg's NVDEC maps its chroma a
    // row off - the software format (the list's last) though CUDA is offered; an even height, or none known: CUDA as before
    const AVPixelFormat vp9List[] = {AV_PIX_FMT_D3D11, AV_PIX_FMT_CUDA, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    // the height judged (NvdecJudgedHeight): the shown one when odd - H.264 / HEVC code 191 rows as 192 and crop them -, else the coded one
    CHECK(NvdecJudgedHeight(191, 192) == 191);
    CHECK(NvdecJudgedHeight(803, 803) == 803);
    CHECK(NvdecJudgedHeight(192, 192) == 192);
    CHECK(NvdecJudgedHeight(1080, 1088) == 1088);
    const AVPixelFormat h264List[] = {AV_PIX_FMT_D3D11VA_VLD, AV_PIX_FMT_D3D11, AV_PIX_FMT_CUDA, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(h264List, false, NvdecJudgedHeight(191, 192)) == AV_PIX_FMT_YUV420P);  // cropped to an odd height: software
    CHECK(PickNvdecFormat(h264List, false, NvdecJudgedHeight(803, 803)) == AV_PIX_FMT_YUV420P);
    CHECK(PickNvdecFormat(h264List, false, NvdecJudgedHeight(192, 192)) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(h264List, false, NvdecJudgedHeight(1080, 1088)) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(vp9List, false, 803) == AV_PIX_FMT_YUV420P);
    CHECK(PickNvdecFormat(vp9List, false, 191) == AV_PIX_FMT_YUV420P);
    CHECK(PickNvdecFormat(vp9List, false, 804) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(vp9List, false, 0) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(vp9List, false) == AV_PIX_FMT_CUDA);
    const AVPixelFormat tenBitList[] = {AV_PIX_FMT_CUDA, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_NONE};
    CHECK(PickNvdecFormat(tenBitList, false, 803) == AV_PIX_FMT_YUV420P10LE);
    // a hardware-only decoder (AV1): no software path to take - CUDA; its odd height is refused elsewhere (JudgeDecode, ReceiveFrame)
    CHECK(PickNvdecFormat(vp9List, true, 191) == AV_PIX_FMT_CUDA);
    CHECK(PickNvdecFormat(tenBitList, true, 803) == AV_PIX_FMT_CUDA);
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
        // the layout is judged before the driver (here: its stand-in) is asked - no caps read, none said
        for (const VideoStreamInfo& early : {Av1Info("yuv444p"), Av1Info("yuv420p12le", 12)}) {
            INFO(early.pixelFormat);
            const DecodeSupport e = CheckDecodeSupport(early);
            CHECK(e.block == DecodeBlock::Layout);
            CHECK_FALSE(e.caps.checked);
            CHECK(e.caps.why.empty());
            CHECK(e.caps.gpu.empty());
        }
        // an odd height waits for the driver's answer: a GPU without AV1 says that first (an even height would not help there)
        const DecodeSupport oddNoAv1 = CheckDecodeSupport(Av1Info("yuv420p", 8, 320, 191));
        CheckRefusal(oddNoAv1, DecodeBlock::NoAv1);
        CHECK(oddNoAv1.caps.checked);
        CHECK(oddNoAv1.caps.gpu == "DLSSVID_FAKE_NVDEC_AV1");
        CHECK(oddNoAv1.height == 191);
        // a layout refused and an odd height: the layout, before the driver
        const DecodeSupport oddLayout = CheckDecodeSupport(Av1Info("yuv444p", 8, 320, 191));
        CHECK(oddLayout.block == DecodeBlock::Layout);
        CHECK_FALSE(oddLayout.caps.checked);
        CheckRefusal(CheckDecodeSupport(Av1Info("unknown")), DecodeBlock::NoAv1);
        CHECK_FALSE(FakeNvdecRefusesFrames());
    }
    {
        FakeNvdecEnv fake("no-cuda");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoCuda);
        CHECK_THAT(s.why, ContainsSubstring("DLSSVID_FAKE_NVDEC_AV1=no-cuda"));
        CheckRefusal(CheckDecodeSupport(Av1Info("yuv420p", 8, 320, 191)), DecodeBlock::NoCuda);  // no CUDA first, also for an odd height
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
        // no answer from the driver: an odd height is refused rather than gone on blind - the stand-in's words are not its reason
        const DecodeSupport odd = CheckDecodeSupport(Av1Info("yuv420p", 8, 320, 191));
        CheckRefusal(odd, DecodeBlock::OddHeight);
        CHECK_FALSE(odd.caps.checked);
        CHECK_THAT(odd.why, Catch::Matchers::EndsWith("has no software AV1 decoder"));
        CHECK_THAT(odd.why, !ContainsSubstring(" (DLSSVID_FAKE"));
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
        // the driver takes AV1: an odd height is refused after its answer (the caps read) - with its own reason only
        const DecodeSupport odd = CheckDecodeSupport(Av1Info("yuv420p", 8, 320, 191));
        CheckRefusal(odd, DecodeBlock::OddHeight);
        CHECK(odd.caps.checked);
        CHECK_THAT(odd.why, Catch::Matchers::EndsWith("has no software AV1 decoder"));
        CHECK_THAT(odd.why, !ContainsSubstring(" (DLSSVID_FAKE"));
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

TEST_CASE("StreamPartInfo: a part of the stream as it decodes - the pixel format's depth and chroma from FFmpeg, the size; an unknown format or size keeps the probe's",
          "[unit][av1]") {
    const VideoStreamInfo probe = Av1Info();  // 320x192 yuv420p, 8-bit 4:2:0
    {
        const VideoStreamInfo part = StreamPartInfo(probe, "yuv420p12le", 384, 224);
        CHECK(part.pixelFormat == "yuv420p12le");
        CHECK(part.bitDepth == 12);
        CHECK(part.chroma == "420");
        CHECK(part.codedWidth == 384);
        CHECK(part.codedHeight == 224);
        CHECK(part.width == probe.width);  // the shown size is the probe's
        CHECK(part.codecName == "av1");
        CHECK(part.hardwareOnly);
    }
    CHECK(StreamPartInfo(probe, "yuv444p10le", 0, 0).chroma == "444");
    CHECK(StreamPartInfo(probe, "yuv444p10le", 0, 0).bitDepth == 10);
    CHECK(StreamPartInfo(probe, "yuv422p", 0, 0).chroma == "422");
    CHECK(StreamPartInfo(probe, "gray", 0, 0).chroma == "other");
    CHECK(StreamPartInfo(probe, "gbrp", 0, 0).chroma == "other");  // RGB
    for (const char* unknown : {"", "?", "not-a-format"}) {
        INFO("'" << unknown << "'");
        const VideoStreamInfo part = StreamPartInfo(probe, unknown, 0, 0);
        CHECK(part.pixelFormat == "yuv420p");
        CHECK(part.bitDepth == 8);
        CHECK(part.chroma == "420");
        CHECK(part.codedWidth == 320);
        CHECK(part.codedHeight == 192);
    }
    // one side known: that one taken
    CHECK(StreamPartInfo(probe, "", 640, 0).codedWidth == 640);
    CHECK(StreamPartInfo(probe, "", 640, 0).codedHeight == 192);
    CHECK(StreamPartInfo(probe, "", 0, 360).codedWidth == 320);
    CHECK(StreamPartInfo(probe, "", 0, 360).codedHeight == 360);
    // judged as the probe is: the part's layout refused, the probe's taken
    NvdecAv1Caps good = GoodCaps();
    CHECK(JudgeDecode(StreamPartInfo(probe, "yuv420p12le", 0, 0), &good).block == DecodeBlock::Layout);
    CHECK(JudgeDecode(StreamPartInfo(probe, "yuv420p10le", 0, 0), &good).Ok());
}

TEST_CASE("StreamPartWords / StreamPartOf: the part a refusal names, written and read back - a part in other parameters, a layout the probe did not learn",
          "[unit][av1]") {
    auto same = [](const StreamPart& a, const StreamPart& b) {
        return a.pixelFormat == b.pixelFormat && a.codedWidth == b.codedWidth && a.codedHeight == b.codedHeight && a.learned == b.learned;
    };
    // written into a why as VideoDecoder::CheckDecode does, read back
    StreamPart part;
    part.pixelFormat = "yuv420p12le";
    part.codedWidth = 384;
    part.codedHeight = 224;
    CHECK(StreamPartWords(part) == ", on a part of the stream in other parameters (yuv420p12le 384x224)");
    const auto partBack = StreamPartOf("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (at frame 16" + StreamPartWords(part) + ")");
    REQUIRE(partBack);
    CHECK(same(*partBack, part));
    StreamPart learned;
    learned.pixelFormat = "yuv420p12le";
    learned.learned = true;
    CHECK(StreamPartWords(learned) == ", in yuv420p12le - a layout the probe did not learn");  // no size: the old words
    const auto learnedBack = StreamPartOf("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (on the first frame" + StreamPartWords(learned) + ")");
    REQUIRE(learnedBack);
    CHECK(same(*learnedBack, learned));  // learned, the size 0 (not said)
    // with the decoder's size (VideoDecoder::CheckDecode: the probe's is the container's), read back with it
    StreamPart learnedSized = learned;
    learnedSized.codedWidth = 320;
    learnedSized.codedHeight = 191;
    CHECK(StreamPartWords(learnedSized) == ", in yuv420p12le 320x191 - a layout the probe did not learn");
    const auto sizedBack = StreamPartOf("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (on the first frame" + StreamPartWords(learnedSized) + ")");
    REQUIRE(sizedBack);
    CHECK(same(*sizedBack, learnedSized));
    // no part
    CHECK_FALSE(StreamPartOf(""));
    CHECK_FALSE(StreamPartOf("NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame 3)"));
    // the decoder not knowing the format
    const auto unknown = StreamPartOf("NVDEC refused this AV1 stream and this build has no software AV1 decoder (at frame 3, on a part of the stream in other parameters (? 320x192))");
    REQUIRE(unknown);
    CHECK(unknown->pixelFormat == "?");
    CHECK(unknown->codedWidth == 320);
    CHECK(unknown->codedHeight == 192);
    CHECK_FALSE(unknown->learned);
    // as the proxy build's line says it in the GUI (the process's whole reason after «the proxy build: »)
    const auto proxied = StreamPartOf("the proxy build: AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (at frame 32, on a part of the stream in other "
                                      "parameters (yuv420p12le 320x192))");
    REQUIRE(proxied);
    CHECK(proxied->pixelFormat == "yuv420p12le");
    CHECK(proxied->codedWidth == 320);
    CHECK(proxied->codedHeight == 192);
    // malformed sizes: nothing read
    for (const char* size : {"320x", "x192", "320y192", "abcx192", "320x19z", "", "-320x192"}) {
        INFO(size);
        CHECK_FALSE(StreamPartOf(std::string("x (at frame 1, on a part of the stream in other parameters (yuv420p ") + size + "))"));
    }
    CHECK_FALSE(StreamPartOf("x (at frame 1, on a part of the stream in other parameters (yuv420p320x192))"));  // no space
    CHECK_FALSE(StreamPartOf("x (on the first frame, in - a layout the probe did not learn)"));  // the head and the tail overlapping
    // the place of a frame after a seek that knows no index (PlaceWords): read as any other
    const auto afterSeek = StreamPartOf("NVDEC could not start decoding this AV1 stream now - out of GPU memory or a driver error; this build has no software "
                                        "AV1 decoder (on the first frame after a seek, on a part of the stream in other parameters (yuv420p 384x224))");
    REQUIRE(afterSeek);
    CHECK(afterSeek->pixelFormat == "yuv420p");
    CHECK(afterSeek->codedWidth == 384);
    CHECK(afterSeek->codedHeight == 224);
    CHECK_FALSE(afterSeek->learned);
    StreamPart empty;  // the words of an empty format, as StreamPartWords writes them
    empty.learned = true;
    CHECK_FALSE(StreamPartOf("x (on the first frame" + StreamPartWords(empty) + ")"));
    CHECK_FALSE(StreamPartOf("x (on the first frame, in yuv 420 - a layout the probe did not learn)"));            // not one word
    for (const char* size : {"320x", "x191", "320", "320y191", "320x19z", "-320x191"}) {  // a size not read whole: nothing
        INFO(size);
        CHECK_FALSE(StreamPartOf(std::string("x (on the first frame, in yuv420p ") + size + " - a layout the probe did not learn)"));
    }
}

TEST_CASE("A layout the probe did not learn, read back from a refusal: judged before the GPU is asked - 12-bit is the layout's refusal", "[unit][av1]") {
    VideoStreamInfo info;  // as VideoDecoder probes an AV1 stream whose sequence header it did not meet
    info.codecName = "av1";
    info.hardwareOnly = true;
    info.nvdecCodec = true;
    info.pixelFormat = "unknown";
    info.width = info.codedWidth = 320;
    info.height = info.codedHeight = 192;
    const auto part = StreamPartOf("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (on the first frame, in yuv420p12le 384x224 - a layout the probe "
                                   "did not learn)");
    REQUIRE(part);
    REQUIRE(part->learned);
    const VideoStreamInfo judged = StreamPartInfo(info, part->pixelFormat, part->codedWidth, part->codedHeight);
    CHECK(judged.codedWidth == 384);  // the decoder's size - the probe's is the container's
    CHECK(judged.codedHeight == 224);
    CHECK(judged.bitDepth == 12);
    // the old words, without a size: the probe's
    const auto unsized = StreamPartOf("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here (on the first frame, in yuv420p12le - a layout the probe did not "
                                      "learn)");
    REQUIRE(unsized);
    CHECK(StreamPartInfo(info, unsized->pixelFormat, unsized->codedWidth, unsized->codedHeight).codedWidth == 320);
    const DecodeSupport verdict = CheckDecodeSupport(judged);
    CHECK(verdict.block == DecodeBlock::Layout);
    CHECK_THAT(verdict.why, StartsWith("AV1 12-bit 4:2:0 (yuv420p12le) cannot be decoded here"));
    // the driver not asked at all: under a stand-in that would refuse AV1 the layout is still the verdict, with no caps read or said
    FakeNvdecEnv fake("no-av1");
    for (const char* pix : {"yuv420p12le", "yuv444p"}) {
        INFO(pix);
        const DecodeSupport layout = CheckDecodeSupport(StreamPartInfo(info, pix, 0, 0));
        CHECK(layout.block == DecodeBlock::Layout);
        CHECK(layout.pixelFormat == pix);
        CHECK_FALSE(layout.caps.checked);
        CHECK(layout.caps.why.empty());
        CHECK(layout.caps.gpu.empty());
    }
}

TEST_CASE("JudgeDecode: an odd frame height is refused after the layout and the GPU's answer - no CUDA, NVDEC or AV1 says that first",
          "[unit][av1]") {
    const NvdecAv1Caps good = GoodCaps();
    NvdecAv1Caps noCuda = GoodCaps();
    noCuda.cuda = false;
    noCuda.supported = false;
    noCuda.why = "nvcuda.dll could not be loaded (no NVIDIA driver)";
    NvdecAv1Caps noNvdec = GoodCaps();
    noNvdec.supported = false;
    noNvdec.nvdecMissing = true;
    noNvdec.why = "nvcuvid.dll could not be loaded (missing, or a driver too old for FFmpeg's NVDEC)";
    NvdecAv1Caps noAv1 = GoodCaps();
    noAv1.supported = false;
    NvdecAv1Caps unchecked;  // the driver could not be asked: no answer to wait for
    unchecked.checked = false;
    const std::string odd = "AV1 320x191 (the frame as stored) has an odd frame height - FFmpeg's NVDEC reads the chroma of such a frame a row off "
                            "(its surfaces are even), and this build has no software AV1 decoder";
    // a machine that could not decode it anyway says why first: no CUDA, no NVDEC, no AV1 (an even height would not help there)
    CheckRefusal(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &noCuda), DecodeBlock::NoCuda);
    CheckRefusal(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &noNvdec), DecodeBlock::NoNvdec);
    // their reasons say the odd height too: the GPU or the driver mended alone would not open it (the size the probe learned)
    for (const NvdecAv1Caps* caps : {&noCuda, &noNvdec}) {
        INFO((caps == &noCuda ? "no CUDA" : "no NVDEC"));
        const std::string even = JudgeDecode(Av1Info("yuv420p", 8, 320, 192), caps).why;
        CHECK_THAT(even, !ContainsSubstring("odd"));  // an even one: as before
        CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), caps).why ==
              even + "; its frame height (320x191, the frame as stored) is odd too, which this build refuses on any GPU");
        // a layout the probe did not learn: the size is the container's - the height not said
        CHECK(JudgeDecode(Av1Info("unknown", 8, 320, 191), caps).why == JudgeDecode(Av1Info("unknown"), caps).why);
        CHECK_THAT(JudgeDecode(Av1Info("unknown", 8, 320, 191), caps).why, !ContainsSubstring("odd"));
    }
    const DecodeSupport noAv1Odd = JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &noAv1);
    CheckRefusal(noAv1Odd, DecodeBlock::NoAv1);
    CHECK(noAv1Odd.height == 191);
    // the odd height said too: no GPU takes it in this build - another computer alone would not help
    CHECK_THAT(noAv1Odd.why, ContainsSubstring("; its frame height (320x191, the frame as stored) is odd too, which this build refuses on any GPU"));
    CHECK_THAT(JudgeDecode(Av1Info("yuv420p", 8, 320, 192), &noAv1).why, !ContainsSubstring("odd"));  // an even one: as before
    // a layout the probe did not learn: the size is the container's - the height not said
    CHECK(JudgeDecode(Av1Info("unknown", 8, 320, 191), &noAv1).why == JudgeDecode(Av1Info("unknown"), &noAv1).why);
    // a size outside the GPU's limits is said before the odd height (an even one would not bring it within them) - the odd height too
    NvdecAv1Caps below = GoodCaps();
    below.maxHeight = 190;
    const DecodeSupport tall = JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &below);
    CheckRefusal(tall, DecodeBlock::Size);
    CHECK_THAT(tall.why, ContainsSubstring("; its frame height (320x191, the frame as stored) is odd too, which this build refuses on any GPU"));
    const DecodeSupport tallEven = JudgeDecode(Av1Info("yuv420p", 8, 320, 192), &below);
    CheckRefusal(tallEven, DecodeBlock::Size);
    CHECK_THAT(tallEven.why, !ContainsSubstring("odd"));  // an even one: as before
    CHECK_THAT(tallEven.why, Catch::Matchers::EndsWith("and this build has no software AV1 decoder"));
    // a layout the probe did not learn (a verdict of its size from elsewhere): the container's height - not said
    CHECK_THAT(RefusedDecode(Av1Info("unknown", 8, 320, 191), DecodeBlock::Size, "").why, !ContainsSubstring("odd"));
    // no answer (not asked, or none), or a GPU that decodes AV1: the odd height
    for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), static_cast<const NvdecAv1Caps*>(&unchecked), &good}) {
        INFO((caps ? (caps->checked ? "good caps" : "unchecked caps") : "no caps"));
        const DecodeSupport s = JudgeDecode(Av1Info("yuv420p", 8, 320, 191), caps);
        CheckRefusal(s, DecodeBlock::OddHeight);
        CHECK(s.why == odd);  // the height is this refusal's reason: no words of the driver's in it, whatever the caps carry
        CHECK(s.width == 320);
        CHECK(s.height == 191);
        CHECK_FALSE(s.part);
        CHECK_FALSE(s.Transient());
        CHECK(s.caps.checked == (caps && caps->checked));  // the caps as given, if any
    }
    CHECK_FALSE(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), nullptr).caps.checked);
    // the size before the odd height: an even height would not bring the frame within the GPU's limits - the size is said; within them,
    // the odd height
    NvdecAv1Caps low = GoodCaps();
    low.maxHeight = 128;
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &low).block == DecodeBlock::Size);
    NvdecAv1Caps narrow = GoodCaps();
    narrow.maxWidth = 256;
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &narrow).block == DecodeBlock::Size);
    NvdecAv1Caps wide = GoodCaps();
    wide.maxHeight = 192;  // 191 within the limits
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &wide).block == DecodeBlock::OddHeight);
    // caps with words of their own - the driver not asked for a CUDA error, or a GPU that answered with a note: the height is this refusal's
    // reason, the driver's words are not said in it (they are for «not checked»)
    NvdecAv1Caps cudaError;
    cudaError.checked = false;
    cudaError.cudaError = true;
    cudaError.why = "cuCtxCreate failed: x";
    NvdecAv1Caps noted = GoodCaps();
    noted.why = "a note of the driver";
    for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(&cudaError), static_cast<const NvdecAv1Caps*>(&noted)}) {
        INFO(caps->why);
        const DecodeSupport s = JudgeDecode(Av1Info("yuv420p", 8, 320, 191), caps);
        CheckRefusal(s, DecodeBlock::OddHeight);
        CHECK(s.why == odd);
        CHECK_THAT(s.why, Catch::Matchers::EndsWith("has no software AV1 decoder"));
        CHECK_THAT(s.why, !ContainsSubstring(" (cuCtxCreate"));
        CHECK_THAT(s.why, !ContainsSubstring("a note of the driver"));
        CHECK(s.caps.checked == caps->checked);  // the rest of the answer kept
    }
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &cudaError).caps.cudaError);
    CHECK(JudgeDecode(Av1Info(), &cudaError).why == "not checked: cuCtxCreate failed: x");  // an even height: the words where they belong
    // 10-bit, and a layout the probe did not learn: the same
    CHECK(JudgeDecode(Av1Info("yuv420p10le", 10, 1920, 1081), &good).block == DecodeBlock::OddHeight);
    // a layout the probe did not learn: its size is the container's (the sequence header unread) - the first frame judges the height
    // (VideoDecoder::ReceiveFrame), as it does the layout and the size
    const DecodeSupport unknownOdd = JudgeDecode(Av1Info("unknown", 8, 320, 191), &good);
    CHECK(unknownOdd.Ok());
    CHECK_FALSE(unknownOdd.verified);
    CHECK(unknownOdd.why == "on 'Test GPU', the sample layout is not known before the first frame");
    CHECK(JudgeDecode(Av1Info("unknown", 8, 320, 191), nullptr).Ok());
    CHECK(JudgeDecode(Av1Info("unknown", 8, 320, 191), nullptr).why == "not checked");
    // the layout first: a 4:4:4 stream of an odd height is refused for its layout
    CHECK(JudgeDecode(Av1Info("yuv444p", 8, 320, 191), &good).block == DecodeBlock::Layout);
    // an even height decodes; so does an odd width - NVDEC hands the frame out even-sided (322 wide) and VideoDecoder cuts it to the video's
    // own width (integration: «An AV1 stream of an odd frame width ...»)
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 190), &good).Ok());
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 321, 192), &good).Ok());
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 320, 0), nullptr).Ok());  // a height not known: nothing to judge
    // a decoder with a software path decodes any height
    VideoStreamInfo h264 = H264Info();
    h264.height = h264.codedHeight = 191;
    CHECK(JudgeDecode(h264, &good).Ok());
    // VideoDecoder's run-time refusal names when; RefusalText says it
    const DecodeSupport later = RefusedDecode(Av1Info("yuv420p", 8, 320, 191), DecodeBlock::OddHeight, "at frame 16");
    CheckRefusal(later, DecodeBlock::OddHeight);
    CHECK(later.why == odd + " (at frame 16)");
    CHECK_FALSE(later.Transient());
    CHECK(RefusedDecode(Av1Info("yuv420p", 8, 320, 191), DecodeBlock::OddHeight, "").why == odd);
    CHECK(BlockName(DecodeBlock::OddHeight) == "OddHeight");
}

TEST_CASE("VerdictFromWhy reads another process's refusal back with its own block: a part, a learned layout, smaller frames, an odd height; else NVDEC's refusal",
          "[unit][av1]") {
    const VideoStreamInfo probe = Av1Info();  // 320x192 8-bit 4:2:0
    // a part of the stream in other parameters (12-bit): the part's verdict - its layout, judged without the GPU, said where it was met
    StreamPart twelve;
    twelve.pixelFormat = "yuv420p12le";
    twelve.codedWidth = 320;
    twelve.codedHeight = 192;
    const std::string layoutWords = JudgeDecode(Av1Info("yuv420p12le", 12), nullptr).why;
    const std::string partWhy = layoutWords + " (at frame 16" + StreamPartWords(twelve) + ")";
    const DecodeSupport part = VerdictFromWhy(probe, partWhy, "the proxy build");
    CheckRefusal(part, DecodeBlock::Layout);
    CHECK(part.part);
    CHECK(part.bitDepth == 12);
    CHECK(part.pixelFormat == "yuv420p12le");
    CHECK_FALSE(part.caps.checked);
    CHECK(part.why == "the proxy build: " + partWhy);
    // the layout the probe did not learn, met at the first frame: the stream's own layout - no part
    VideoStreamInfo unknown = probe;
    unknown.pixelFormat = "unknown";
    StreamPart learned;
    learned.pixelFormat = "yuv420p12le";
    learned.learned = true;
    const std::string learnedWhy = layoutWords + " (on the first frame" + StreamPartWords(learned) + ")";
    const DecodeSupport layout = VerdictFromWhy(unknown, learnedWhy, "the proxy build");
    CheckRefusal(layout, DecodeBlock::Layout);
    CHECK_FALSE(layout.part);
    CHECK(layout.bitDepth == 12);
    CHECK(layout.why == "the proxy build: " + learnedWhy);
    // frames smaller than the sequence's (Resized), with and without VideoDecoder's detail: the sequence's size named, the detail kept
    const VideoStreamInfo big = StreamPartInfo(probe, "", 640, 480);
    const std::string resizedWords = RefusedDecode(big, DecodeBlock::Resized, "").why;
    for (const std::string& detail : {std::string("a frame of 320x240, at frame 5"), std::string()}) {
        INFO("detail: " << detail);
        const DecodeSupport resized = VerdictFromWhy(probe, RefusedDecode(big, DecodeBlock::Resized, detail).why, "the proxy build");
        CheckRefusal(resized, DecodeBlock::Resized);
        CHECK(resized.width == 640);
        CHECK(resized.height == 480);
        CHECK_FALSE(resized.part);  // a size of the sequence, not a part judged apart
        CHECK_FALSE(resized.Transient());
        CHECK(resized.why == resizedWords + (detail.empty() ? " (the proxy build)" : " (the proxy build: " + detail + ")"));
    }
    // an odd frame height (VideoDecoder's run-time refusal): its size
    const VideoStreamInfo oddSize = StreamPartInfo(probe, "", 320, 191);
    const std::string oddWords = RefusedDecode(oddSize, DecodeBlock::OddHeight, "").why;
    REQUIRE(oddWords == "AV1 320x191 (the frame as stored) has an odd frame height - FFmpeg's NVDEC reads the chroma of such a frame a row off (its "
                        "surfaces are even), and this build has no software AV1 decoder");
    REQUIRE_THAT(resizedWords, StartsWith("AV1 frames smaller than the sequence's 640x480 (a frame size override or a resize within it) cannot be "
                                          "decoded here - FFmpeg's NVDEC makes its surfaces for the sequence's maximum"));
    for (const std::string& detail : {std::string("at frame 16"), std::string("on the first frame"), std::string()}) {
        INFO("detail: " << detail);
        const DecodeSupport odd = VerdictFromWhy(probe, RefusedDecode(oddSize, DecodeBlock::OddHeight, detail).why, "the proxy build");
        CheckRefusal(odd, DecodeBlock::OddHeight);
        CHECK(odd.width == 320);
        CHECK(odd.height == 191);
        CHECK(odd.part);  // the probe's height is even: a part of the stream (as VideoDecoder::ReceiveFrame marks it)
        CHECK(odd.why == oddWords + (detail.empty() ? " (the proxy build)" : " (the proxy build: " + detail + ")"));
    }
    // the probe did not learn the layout (its size the container's): the stream's own height met at its first frame - no part
    {
        VideoStreamInfo noLayout = probe;
        noLayout.pixelFormat = "unknown";
        const DecodeSupport whole = VerdictFromWhy(noLayout, RefusedDecode(oddSize, DecodeBlock::OddHeight, "on the first frame").why, "the proxy build");
        CHECK(whole.block == DecodeBlock::OddHeight);
        CHECK(whole.height == 191);
        CHECK_FALSE(whole.part);
    }
    // the probe's own height odd (the stream as a whole) or not known: the stream's verdict, no part
    for (const uint32_t probeHeight : {191u, 0u}) {
        INFO("probe height " << probeHeight);
        VideoStreamInfo other = probe;
        other.height = other.codedHeight = probeHeight;
        const DecodeSupport whole = VerdictFromWhy(other, RefusedDecode(oddSize, DecodeBlock::OddHeight, "at frame 16").why, "the proxy build");
        CHECK(whole.block == DecodeBlock::OddHeight);
        CHECK(whole.height == 191);
        CHECK_FALSE(whole.part);
    }
    // the words name a part (VideoDecoder::ReceiveFrame's, after frames under another sequence header): a part also where this probe
    // learned no layout - judged by the part's own words with its format known, by the odd height's words with «?»
    {
        FakeNvdecEnv fake("start-fails");  // the driver takes AV1 (no word of this machine's GPU in the verdict)
        VideoStreamInfo noLayout = probe;
        noLayout.pixelFormat = "unknown";
        for (const char* format : {"yuv420p", "?"}) {
            INFO(format);
            StreamPart named;
            named.pixelFormat = format;
            named.codedWidth = 320;
            named.codedHeight = 191;
            const std::string why = RefusedDecode(oddSize, DecodeBlock::OddHeight, "at frame 24" + StreamPartWords(named)).why;
            REQUIRE_THAT(why, Catch::Matchers::EndsWith(std::string("(at frame 24, on a part of the stream in other parameters (") + format + " 320x191))"));
            const DecodeSupport v = VerdictFromWhy(noLayout, why, "the proxy build");
            CheckRefusal(v, DecodeBlock::OddHeight);
            CHECK(v.part);
            CHECK(v.width == 320);
            CHECK(v.height == 191);
            CHECK_FALSE(v.Transient());
        }
    }
    // NVDEC refusing the stream: its detail said as the place's
    const std::string failedWords = RefusedDecode(probe, DecodeBlock::NvdecFailed, "").why;
    const DecodeSupport failed = VerdictFromWhy(probe, RefusedDecode(probe, DecodeBlock::NvdecFailed, "at frame 7").why, "the proxy build");
    CheckRefusal(failed, DecodeBlock::NvdecFailed);
    CHECK(failed.width == 320);
    CHECK(failed.height == 192);
    CHECK(failed.why == failedWords + " (the proxy build: at frame 7)");
    CHECK(VerdictFromWhy(probe, failedWords, "the proxy build").why == failedWords + " (the proxy build)");
    CHECK(VerdictFromWhy(probe, failedWords, "elsewhere").why == failedWords + " (elsewhere)");
    // any other text: NVDEC's refusal with the text whole - also one naming a size that is no refusal's
    for (const std::string text : {std::string("something else entirely"), std::string("the proxy failed at 320x191 for its own reasons"),
                                   std::string("AV1 640x480 (the frame as stored) is outside what NVDEC decodes")}) {
        INFO(text);
        const DecodeSupport other = VerdictFromWhy(probe, text, "the proxy build");
        CheckRefusal(other, DecodeBlock::NvdecFailed);
        CHECK(other.why == failedWords + " (the proxy build: " + text + ")");
        CHECK(other.height == 192);
    }
}

TEST_CASE("VerdictFromWhy under DLSSVID_FAKE_NVDEC_AV1: a part the GPU takes is NVDEC refusing the stream; a part it does not take is that verdict",
          "[unit][av1]") {
    const VideoStreamInfo probe = Av1Info();  // 320x192 8-bit 4:2:0
    StreamPart bigger;  // an 8-bit part of another size: its layout decodes here
    bigger.pixelFormat = "yuv420p";
    bigger.codedWidth = 384;
    bigger.codedHeight = 224;
    const std::string failedWords = RefusedDecode(probe, DecodeBlock::NvdecFailed, "").why;
    const std::string when = "at frame 16" + StreamPartWords(bigger);
    const std::string why = failedWords + " (" + when + ")";  // VideoDecoder::CheckDecode's, the proxy build's line
    // the driver takes the part (start-fails: verified) or does not answer (first-frame): not the part's verdict - NVDEC refused the
    // stream, the whole detail kept, the probe's size
    for (const char* value : {"start-fails", "first-frame"}) {
        INFO(value);
        FakeNvdecEnv fake(value);
        const DecodeSupport v = VerdictFromWhy(probe, why, "the proxy build");
        CheckRefusal(v, DecodeBlock::NvdecFailed);
        CHECK_FALSE(v.part);
        CHECK(v.why == failedWords + " (the proxy build: " + when + ")");
        CHECK(v.why == failedWords + " (the proxy build: at frame 16, on a part of the stream in other parameters (yuv420p 384x224))");
        CHECK(v.width == 320);
        CHECK(v.height == 192);
        CHECK_FALSE(v.Transient());
    }
    // the place after a seek that knows no index: the same, its words kept
    {
        FakeNvdecEnv fake("start-fails");
        const std::string seekWhen = "on the first frame after a seek" + StreamPartWords(bigger);
        const DecodeSupport v = VerdictFromWhy(probe, failedWords + " (" + seekWhen + ")", "the proxy build");
        CheckRefusal(v, DecodeBlock::NvdecFailed);
        CHECK_FALSE(v.part);
        CHECK(v.why == failedWords + " (the proxy build: on the first frame after a seek, on a part of the stream in other parameters (yuv420p 384x224))");
    }
    // a layout learned with the decoder's size (VideoDecoder::CheckDecode): judged with that size - a frame outside the GPU's limits is
    // that verdict, not NVDEC refusing the stream (the probe, which learned no layout, has the container's size)
    {
        FakeNvdecEnv fake("start-fails");  // the driver takes AV1 up to 65535x65535
        VideoStreamInfo unknown = probe;
        unknown.pixelFormat = "unknown";
        const DecodeSupport huge = CheckDecodeSupport(Av1Info("yuv420p", 8, 70000, 5000));
        REQUIRE(huge.block == DecodeBlock::Size);
        StreamPart big;
        big.pixelFormat = "yuv420p";
        big.codedWidth = 70000;
        big.codedHeight = 5000;
        big.learned = true;
        const std::string sizeWhy = huge.why + " (on the first frame" + StreamPartWords(big) + ")";
        REQUIRE_THAT(sizeWhy, Catch::Matchers::EndsWith("(on the first frame, in yuv420p 70000x5000 - a layout the probe did not learn)"));
        const DecodeSupport v = VerdictFromWhy(unknown, sizeWhy, "the proxy build");
        CheckRefusal(v, DecodeBlock::Size);
        CHECK(v.width == 70000);
        CHECK(v.height == 5000);
        CHECK_FALSE(v.part);
        CHECK(v.why == "the proxy build: " + sizeWhy);
    }
    // a GPU without AV1 (no-av1): the part's verdict - the driver's answer, a part
    FakeNvdecEnv fake("no-av1");
    const DecodeSupport v = VerdictFromWhy(probe, why, "the proxy build");
    CheckRefusal(v, DecodeBlock::NoAv1);
    CHECK(v.part);
    CHECK(v.caps.checked);
    CHECK(v.width == 384);
    CHECK(v.height == 224);
    CHECK(v.why == "the proxy build: " + why);
}

namespace {
std::string Bytes(const char* hex) {
    std::string out;
    for (const char* c = hex; c[0] && c[1]; c += 2) out.push_back(static_cast<char>(std::stoi(std::string(c, 2), nullptr, 16)));
    return out;
}

// Bits of a sequence header, most significant first, ended with the trailing bits (AV1 5.3.4)
struct HeaderBits {
    std::string out;
    size_t pos = 0;
    HeaderBits& put(uint32_t v, int n) {
        for (int i = n - 1; i >= 0; --i, ++pos) {
            if (pos % 8 == 0) out.push_back(0);
            if ((v >> i) & 1u) out.back() = static_cast<char>(static_cast<uint8_t>(out.back()) | (0x80u >> (pos % 8)));
        }
        return *this;
    }
    std::string done() {
        put(1, 1);
        while (pos % 8) put(0, 1);
        return out;
    }
};

// A sequence header with timing_info, decoder_model_info and operating_parameters_info (AV1 5.5): operating point 0's buffer delays as
// given - what a repeat of the header within a coded video sequence may change (AV1 7.5); 1920x1080 4:2:0, 10-bit or 8-bit
std::string ModelHeader(uint32_t decoderDelay, uint32_t encoderDelay, bool tenBit = true) {
    HeaderBits b;
    b.put(0, 3).put(0, 1).put(0, 1);  // seq_profile, still_picture, reduced_still_picture_header
    b.put(1, 1);                      // timing_info_present_flag
    b.put(1001, 32).put(60000, 32);   // num_units_in_display_tick, time_scale
    b.put(1, 1).put(1, 1);            // equal_picture_interval, num_ticks_per_picture_minus_1 = 0 (uvlc: 1)
    b.put(1, 1);                      // decoder_model_info_present_flag
    b.put(9, 5);                      // buffer_delay_length_minus_1: 10-bit delays
    b.put(1001, 32);                  // num_units_in_decoding_tick
    b.put(31, 5).put(31, 5);          // buffer_removal_time_length_minus_1, frame_presentation_time_length_minus_1
    b.put(1, 1);                      // initial_display_delay_present_flag
    b.put(0, 5);                      // operating_points_cnt_minus_1
    b.put(0, 12).put(8, 5).put(0, 1);  // operating_point_idc, seq_level_idx (> 7: seq_tier)
    b.put(1, 1).put(decoderDelay, 10).put(encoderDelay, 10).put(0, 1);  // operating_parameters_info: the delays, low_delay_mode_flag
    b.put(1, 1).put(9, 4);            // initial_display_delay_present_for_this_op, initial_display_delay_minus_1
    b.put(10, 4).put(10, 4);          // frame_width_bits_minus_1, frame_height_bits_minus_1
    b.put(1919, 11).put(1079, 11);    // max_frame_width_minus_1, max_frame_height_minus_1
    b.put(0, 1);                      // frame_id_numbers_present_flag
    b.put(0, 1).put(1, 1).put(1, 1);  // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    b.put(0, 4);                      // interintra, masked, warped motion, dual filter
    b.put(1, 1).put(1, 1).put(1, 1);  // enable_order_hint, enable_jnt_comp, enable_ref_frame_mvs
    b.put(1, 1).put(1, 1);            // seq_choose_screen_content_tools (SELECT), seq_choose_integer_mv
    b.put(6, 3);                      // order_hint_bits_minus_1
    b.put(0, 1).put(1, 1).put(1, 1);  // enable_superres, enable_cdef, enable_restoration
    b.put(tenBit ? 1 : 0, 1);         // high_bitdepth
    b.put(0, 1);                      // mono_chrome
    b.put(1, 1).put(1, 8).put(1, 8).put(1, 8);  // color_description_present_flag: BT.709 primaries, transfer, matrix (not the sRGB triple)
    b.put(0, 1).put(0, 2).put(0, 1);  // color_range, chroma_sample_position, separate_uv_delta_q
    b.put(0, 1);                      // film_grain_params_present
    return b.done();
}

// A reduced still picture header (AV1 5.5): 64x48 8-bit 4:2:0
std::string StillHeader() {
    HeaderBits b;
    b.put(0, 3).put(1, 1).put(1, 1);  // seq_profile, still_picture, reduced_still_picture_header
    b.put(0, 5);                      // seq_level_idx[0]
    b.put(6, 4).put(5, 4);            // frame_width_bits_minus_1, frame_height_bits_minus_1
    b.put(63, 7).put(47, 6);          // max_frame_width_minus_1, max_frame_height_minus_1
    b.put(0, 1).put(0, 1).put(0, 1);  // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
    b.put(0, 1).put(0, 1).put(0, 1);  // enable_superres, enable_cdef, enable_restoration
    b.put(0, 1).put(0, 1).put(0, 1);  // high_bitdepth, mono_chrome, color_description_present_flag
    b.put(0, 1).put(0, 2).put(0, 1);  // color_range, chroma_sample_position, separate_uv_delta_q
    b.put(0, 1);                      // film_grain_params_present
    return b.done();
}

std::string SequenceHeaderIn(const std::string& data) {
    return std::string(Av1SequenceHeader(reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}
}  // namespace

TEST_CASE("Av1SequenceKeyOf: what NVDEC makes its decoder for - av1_nvenc's headers, a repeat with other buffer delays, a reduced still picture; a cut one is none",
          "[unit][av1]") {
    // av1_nvenc's own headers (from its IVF output, an RTX 40 card): 320x192 8-bit, the same in Main 10, 384x224 8-bit
    const auto k8 = Av1SequenceKeyOf(Bytes("000000fa1e7f7e010cc020"));
    const auto k10 = Av1SequenceKeyOf(Bytes("000000fa1e7f7e010ce020"));
    const auto kBig = Av1SequenceKeyOf(Bytes("0000000c3dff7c02198040"));
    REQUIRE(k8);
    REQUIRE(k10);
    REQUIRE(kBig);
    CHECK(k8->profile == 0);
    CHECK(k8->bitDepth == 8);
    CHECK(k8->subsamplingX == 1);
    CHECK(k8->subsamplingY == 1);
    CHECK(k8->maxWidth == 320);
    CHECK(k8->maxHeight == 192);
    CHECK_FALSE(k8->mono);
    CHECK_FALSE(k8->rgb);
    CHECK(k10->bitDepth == 10);
    CHECK(k10->maxWidth == 320);
    CHECK(kBig->maxWidth == 384);
    CHECK(kBig->maxHeight == 224);
    CHECK_FALSE(*k8 == *k10);
    CHECK_FALSE(*k8 == *kBig);
    // a decoder model: two repeats differing in operating_parameters_info only - one key
    const std::string a = ModelHeader(100, 200), b = ModelHeader(300, 400);
    REQUIRE(a != b);
    const auto ka = Av1SequenceKeyOf(a), kb = Av1SequenceKeyOf(b);
    REQUIRE(ka);
    REQUIRE(kb);
    CHECK(*ka == *kb);
    CHECK(ka->bitDepth == 10);
    CHECK(ka->maxWidth == 1920);
    CHECK(ka->maxHeight == 1080);
    CHECK(ka->subsamplingX == 1);
    CHECK(ka->subsamplingY == 1);
    CHECK_FALSE(ka->rgb);
    const auto k8bit = Av1SequenceKeyOf(ModelHeader(100, 200, false));
    REQUIRE(k8bit);
    CHECK_FALSE(*k8bit == *ka);  // another depth: another decoder
    // a reduced still picture header
    const auto still = Av1SequenceKeyOf(StillHeader());
    REQUIRE(still);
    CHECK(still->maxWidth == 64);
    CHECK(still->maxHeight == 48);
    CHECK(still->bitDepth == 8);
    CHECK(still->subsamplingX == 1);
    // cut short: none
    CHECK_FALSE(Av1SequenceKeyOf(a.substr(0, 6)));
    CHECK_FALSE(Av1SequenceKeyOf(Bytes("000000fa1e7f")));
    CHECK_FALSE(Av1SequenceKeyOf(""));
}

TEST_CASE("Av1SequenceHeader finds the sequence header OBU in low-overhead OBUs and in an av1C record; garbage has none", "[unit][av1]") {
    const std::string payload = Bytes("000000fa1e7f7e010cc020");
    const std::string obu = std::string(1, '\x0A') + std::string(1, static_cast<char>(payload.size())) + payload;  // OBU_SEQUENCE_HEADER, with its size
    const std::string delimiter("\x12\x00", 2);                                                                   // OBU_TEMPORAL_DELIMITER
    CHECK(SequenceHeaderIn(delimiter + obu) == payload);
    CHECK(SequenceHeaderIn(std::string("\x81\x00\x0C\x00", 4) + obu) == payload);  // av1C: marker and version, then its configOBUs
    CHECK(SequenceHeaderIn(delimiter).empty());                                       // no sequence header
    CHECK(SequenceHeaderIn(std::string("\xFF\xFF\xFF", 3)).empty());                 // obu_forbidden_bit
    CHECK(SequenceHeaderIn(obu.substr(0, obu.size() - 3)).empty());                    // its size beyond the data
    CHECK(SequenceHeaderIn("").empty());
    CHECK(Av1SequenceHeader(nullptr, 0).empty());
}

TEST_CASE("Av1FrameBelowSequence: a frame smaller than its sequence header's maximum (a frame size override, a resize within it); the Resized refusal",
          "[unit][av1]") {
    const std::string header = Bytes("000000fa1e7f7e010cc020");  // av1_nvenc's: at most 320x192
    REQUIRE(Av1SequenceKeyOf(header));
    CHECK(Av1FrameBelowSequence(319, 192, header));  // narrower
    CHECK(Av1FrameBelowSequence(320, 191, header));  // lower
    CHECK(Av1FrameBelowSequence(160, 96, header));
    CHECK_FALSE(Av1FrameBelowSequence(320, 192, header));  // the sequence's own size
    CHECK_FALSE(Av1FrameBelowSequence(640, 480, header));  // not below (another sequence's frame: FollowSequenceHeader's)
    CHECK_FALSE(Av1FrameBelowSequence(0, 0, header));      // no size
    CHECK_FALSE(Av1FrameBelowSequence(0, 100, header));
    CHECK_FALSE(Av1FrameBelowSequence(100, 0, header));
    CHECK_FALSE(Av1FrameBelowSequence(160, 96, ""));                     // no header
    CHECK_FALSE(Av1FrameBelowSequence(160, 96, Bytes("000000fa1e7f")));  // one that does not parse
    // the refusal VideoDecoder makes of it: the sequence's size, the frame named
    const DecodeSupport s = RefusedDecode(StreamPartInfo(Av1Info(), "", 640, 480), DecodeBlock::Resized, "a frame of 320x240, at frame 5");
    CheckRefusal(s, DecodeBlock::Resized);
    CHECK_FALSE(s.Transient());
    CHECK(s.width == 640);
    CHECK(s.height == 480);
    CHECK(s.why == "AV1 frames smaller than the sequence's 640x480 (a frame size override or a resize within it) cannot be decoded here - FFmpeg's "
                   "NVDEC makes its surfaces for the sequence's maximum and hands such a frame out at that size, not its own (a lower one also with the "
                   "chroma mapped by its own height), and this build has no software AV1 decoder (a frame of 320x240, at frame 5)");
}

TEST_CASE("DescribeDecode: the decode line of dlssvid info in its forms", "[unit][av1]") {
    CHECK(DescribeDecode(JudgeDecode(H264Info(), nullptr)) == "software (h264); NVDEC with --hwaccel cuda where FFmpeg offers it for this layout and the GPU decodes it");
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
    // TASK-0076: a codec NVDEC decodes, of an odd height - software also with --hwaccel cuda (as VideoDecoder decides it)
    VideoStreamInfo h264Odd = H264Info();
    h264Odd.height = h264Odd.codedHeight = 191;
    CHECK(DescribeDecode(JudgeDecode(h264Odd, nullptr)) ==
          "software (h264); not NVDEC even with --hwaccel cuda: an odd frame height (320x191, the frame as stored) - FFmpeg's NVDEC gives its "
          "chroma wrong");
    VideoStreamInfo vp9 = SoftwareInfo("vp9", true);
    vp9.width = vp9.codedWidth = 1920;
    vp9.height = vp9.codedHeight = 803;
    CHECK(DescribeDecode(JudgeDecode(vp9, nullptr)) ==
          "software (vp9); not NVDEC even with --hwaccel cuda: an odd frame height (1920x803, the frame as stored) - FFmpeg's NVDEC gives its "
          "chroma wrong");
    vp9.height = vp9.codedHeight = 804;
    CHECK(DescribeDecode(JudgeDecode(vp9, nullptr)) == "software (vp9); NVDEC with --hwaccel cuda where FFmpeg offers it for this layout and the GPU decodes it");
    VideoStreamInfo ffv1Odd = Ffv1Info();  // NVDEC does not decode it at all: that, not the height
    ffv1Odd.height = ffv1Odd.codedHeight = 191;
    CHECK(DescribeDecode(JudgeDecode(ffv1Odd, nullptr)) == "software (ffv1); NVDEC does not decode this codec");
}

TEST_CASE("IsProbeVerdict: the verdicts a probe gives - the layout, no CUDA / NVDEC / AV1, the size, an odd height; not a decoder's", "[unit][av1]") {
    const NvdecAv1Caps good = GoodCaps();
    NvdecAv1Caps noCuda = GoodCaps();
    noCuda.cuda = false;
    NvdecAv1Caps noNvdec = GoodCaps();
    noNvdec.nvdecMissing = true;
    NvdecAv1Caps noAv1 = GoodCaps();
    noAv1.supported = false;
    NvdecAv1Caps low = GoodCaps();
    low.maxHeight = 128;
    struct Case {
        const char* what;
        DecodeSupport s;
        DecodeBlock block;
    };
    const Case probes[] = {
        {"layout", JudgeDecode(Av1Info("yuv444p"), nullptr), DecodeBlock::Layout},
        {"no CUDA", JudgeDecode(Av1Info(), &noCuda), DecodeBlock::NoCuda},
        {"no NVDEC", JudgeDecode(Av1Info(), &noNvdec), DecodeBlock::NoNvdec},
        {"no AV1", JudgeDecode(Av1Info(), &noAv1), DecodeBlock::NoAv1},
        {"size", JudgeDecode(Av1Info(), &low), DecodeBlock::Size},
        {"odd height", JudgeDecode(Av1Info("yuv420p", 8, 320, 191), &good), DecodeBlock::OddHeight},
    };
    for (const Case& c : probes) {
        INFO(c.what);
        REQUIRE(c.s.block == c.block);
        CHECK(IsProbeVerdict(c.s));
        // the same block on a part of the stream: a decoder met it
        DecodeSupport part = c.s;
        part.part = true;
        CHECK_FALSE(IsProbeVerdict(part));
        // of a stream whose layout the probe did not learn: no probe gave it
        DecodeSupport unknown = c.s;
        unknown.pixelFormat = "unknown";
        CHECK_FALSE(IsProbeVerdict(unknown));
    }
    // a layout learned at the first frame (VideoDecoder::CheckDecode's words): the decoder's
    StreamPart learned;
    learned.pixelFormat = "yuv420p12le";
    learned.learned = true;
    DecodeSupport late = JudgeDecode(Av1Info("yuv420p12le", 12), nullptr);
    REQUIRE(late.block == DecodeBlock::Layout);
    late.why += " (on the first frame" + StreamPartWords(learned) + ")";
    CHECK_FALSE(IsProbeVerdict(late));
    // NVDEC refusing frames, frames of another size, not starting, nothing at all: no probe's
    for (const DecodeBlock b : {DecodeBlock::NvdecFailed, DecodeBlock::Resized, DecodeBlock::NvdecStartFailed}) {
        INFO(BlockName(b));
        CHECK_FALSE(IsProbeVerdict(RefusedDecode(Av1Info(), b, "on the first frame")));
    }
    CHECK_FALSE(IsProbeVerdict(JudgeDecode(Av1Info(), &good)));
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

TEST_CASE("LayoutKindOf: the kind a chroma other is told by - a palette, RGB, one component (grey), a YUV as its J:a:b digits; an unknown format is other",
          "[unit][av1]") {
    for (const char* rgb : {"gbrp", "bgr0", "rgb24", "gbrp10le"}) {
        INFO(rgb);
        CHECK(LayoutKindOf(rgb) == "rgb");
    }
    for (const char* gray : {"gray", "gray10le", "gray12le"}) {
        INFO(gray);
        CHECK(LayoutKindOf(gray) == "gray");
    }
    CHECK(LayoutKindOf("pal8") == "palette");  // one component, a colour palette: not grey
    // a YUV: its J:a:b digits - the common ones and the rarer subsamplings
    struct Yuv {
        const char* format;
        const char* kind;
    };
    for (const Yuv& y : {Yuv{"yuv420p", "420"}, Yuv{"yuv422p10le", "422"}, Yuv{"yuv444p12le", "444"}, Yuv{"yuv411p", "411"}, Yuv{"yuv440p", "440"},
                         Yuv{"yuv410p", "410"}}) {
        INFO(y.format);
        CHECK(LayoutKindOf(y.format) == y.kind);
    }
    // a format not known: other
    for (const char* other : {"", "unknown", "nonsense"}) {
        INFO(other);
        CHECK(LayoutKindOf(other) == "other");
    }
}

TEST_CASE("NvdecOddHeightSafe: the odd heights FFmpeg's NVDEC gives right - HEVC 4:4:4 only; DescribeDecode says it", "[unit][av1]") {
    CHECK(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_YUV444P));
    CHECK(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_YUV444P10LE));
    // FFmpeg 7.1.1: hevcdec offers no CUDA for 4:2:2; nvdec_h264 has no 4:4:4 surface (4:2:2 / 4:4:4 come as NV12)
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_YUV422P10LE));
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_H264, AV_PIX_FMT_YUV444P));
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_H264, AV_PIX_FMT_YUV422P));
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_GBRP));  // RGB
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_YUV420P));  // its last chroma row lost on the way to the host
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_VP9, AV_PIX_FMT_YUV444P));   // no cropping exported: mapped at its odd height
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_GRAY8));
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_NONE));     // a format not known
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_AV1, AV_PIX_FMT_YUV444P));
    CHECK_FALSE(NvdecOddHeightSafe(AV_CODEC_ID_HEVC, AV_PIX_FMT_CUDA));     // a hardware format is no layout
    // dlssvid info: an HEVC 4:4:4 of an odd height may decode on NVDEC; an H.264 4:2:0 of one does not
    VideoStreamInfo hevc = SoftwareInfo("hevc", true);
    hevc.pixelFormat = "yuv444p";
    hevc.chroma = "444";
    hevc.height = hevc.codedHeight = 191;
    CHECK(DescribeDecode(JudgeDecode(hevc, nullptr)) == "software (hevc); NVDEC with --hwaccel cuda where FFmpeg offers it for this layout and the GPU decodes it");
    VideoStreamInfo h264 = H264Info();
    h264.height = h264.codedHeight = 191;
    CHECK_THAT(DescribeDecode(JudgeDecode(h264, nullptr)), ContainsSubstring("not NVDEC even with --hwaccel cuda: an odd frame height (320x191"));
    h264.pixelFormat = "yuv444p";  // an H.264 4:4:4 of one neither: NVDEC gives it as NV12
    h264.chroma = "444";
    CHECK_THAT(DescribeDecode(JudgeDecode(h264, nullptr)), ContainsSubstring("not NVDEC even with --hwaccel cuda: an odd frame height (320x191"));
    hevc.pixelFormat = "yuv420p";
    hevc.chroma = "420";
    CHECK_THAT(DescribeDecode(JudgeDecode(hevc, nullptr)), ContainsSubstring("not NVDEC even with --hwaccel cuda"));
}
