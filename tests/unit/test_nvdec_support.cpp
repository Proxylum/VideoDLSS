// TASK-0076: AV1 has no software decoder in this build - which decoders need NVDEC, whether NVDEC takes a stream (JudgeDecode), the pixel
// format VideoDecoder asks FFmpeg for (PickNvdecFormat), the DLSSVID_FAKE_NVDEC_AV1 stand-in for the driver and the `info` line. No GPU
// work: CheckDecodeSupport asks the driver for its AV1 caps once (a temporary context, no decoding), whatever it answers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "io/NvdecFormat.h"
#include "io/NvdecSupport.h"
#include "io/VideoDecoder.h"

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

// What VideoDecoder reports for an AV1 stream of this build (VideoStreamInfo::hardwareOnly), 320x192 8-bit 4:2:0 unless told otherwise.
VideoStreamInfo Av1Info(const std::string& pixelFormat = "yuv420p", int bitDepth = 8, uint32_t w = 320, uint32_t h = 192) {
    VideoStreamInfo i;
    i.codecName = "av1";
    i.hardwareOnly = true;
    i.pixelFormat = pixelFormat;
    i.bitDepth = bitDepth;
    i.width = i.codedWidth = w;
    i.height = i.codedHeight = h;
    return i;
}

VideoStreamInfo H264Info() {
    VideoStreamInfo i;
    i.codecName = "h264";
    i.hardwareOnly = false;
    i.pixelFormat = "yuv420p";
    i.width = i.codedWidth = 320;
    i.height = i.codedHeight = 192;
    return i;
}

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

const std::string kAllBlocks[] = {"None", "Layout", "NoCuda", "NoAv1", "Size", "NvdecFailed"};

std::string BlockName(DecodeBlock b) { return kAllBlocks[static_cast<int>(b)]; }

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
    for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), static_cast<const NvdecAv1Caps*>(&noCuda)}) {
        const DecodeSupport s = JudgeDecode(H264Info(), caps);
        CHECK(s.Ok());
        CHECK_FALSE(s.hardwareOnly);
        CHECK_FALSE(s.verified);
        CHECK(s.why == "software (h264)");
        CHECK(JudgeDecode(odd, caps).Ok());
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
    // any other is refused as a layout - whatever the driver said, and before it (a GPU without CUDA too)
    for (const char* pix : {"yuv444p", "yuv422p10le", "yuv420p12le", "gray", "yuv444p10le"}) {
        INFO(pix);
        for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), &good, static_cast<const NvdecAv1Caps*>(&noCuda)}) {
            const DecodeSupport s = JudgeDecode(Av1Info(pix), caps);
            CheckRefusal(s, DecodeBlock::Layout);
            CHECK_THAT(s.why, ContainsSubstring(pix));
            CHECK(s.pixelFormat == pix);
        }
    }

    // an unknown layout (the probe did not learn it): NVDEC decides on the first frame - not refused, even with no CUDA said
    for (const NvdecAv1Caps* caps : {static_cast<const NvdecAv1Caps*>(nullptr), &good, static_cast<const NvdecAv1Caps*>(&noCuda)}) {
        const DecodeSupport s = JudgeDecode(Av1Info("unknown"), caps);
        CHECK(s.Ok());
        CHECK_FALSE(s.verified);
        CHECK(s.hardwareOnly);
        CHECK_THAT(s.why, ContainsSubstring("not known before the first frame"));
    }

    // the driver not asked, or no definite answer from it: on blind, as FFmpeg does - not verified, the reason kept
    const DecodeSupport notAsked = JudgeDecode(Av1Info(), nullptr);
    CHECK(notAsked.Ok());
    CHECK_FALSE(notAsked.verified);
    CHECK(notAsked.why == "NVDEC (av1), not checked");
    NvdecAv1Caps unanswered = GoodCaps();
    unanswered.checked = false;
    unanswered.supported = false;  // not read: must not count
    unanswered.cuda = false;
    unanswered.why = "the driver does not report NVDEC capabilities";
    const DecodeSupport blind = JudgeDecode(Av1Info(), &unanswered);
    CHECK(blind.Ok());
    CHECK_FALSE(blind.verified);
    CHECK(blind.why == "NVDEC (av1), not checked: the driver does not report NVDEC capabilities");
}

TEST_CASE("JudgeDecode: no CUDA, no AV1 decoder, the size limits each refuse with the reason", "[unit][av1]") {
    // no CUDA
    NvdecAv1Caps noCuda = GoodCaps();
    noCuda.cuda = false;
    noCuda.supported = false;
    noCuda.gpu.clear();
    noCuda.why = "nvcuda.dll could not be loaded (no NVIDIA driver)";
    const DecodeSupport nc = JudgeDecode(Av1Info(), &noCuda);
    CheckRefusal(nc, DecodeBlock::NoCuda);
    CHECK_THAT(nc.why, ContainsSubstring("nvcuda.dll could not be loaded (no NVIDIA driver)"));
    CHECK_THAT(nc.why, ContainsSubstring("no software AV1 decoder"));
    // CUDA but no AV1 in its NVDEC: the GPU and the depth named
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
    // no CUDA is said before no AV1 (a GPU-less machine has no AV1 caps either)
    NvdecAv1Caps neither = noAv1;
    neither.cuda = false;
    CHECK(JudgeDecode(Av1Info(), &neither).block == DecodeBlock::NoCuda);

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
    // a partial macroblock counts as a whole one: 330x200 is 21 x 13 = 273
    NvdecAv1Caps mb = GoodCaps();
    mb.maxMacroblocks = 273;
    CHECK(JudgeDecode(Av1Info("yuv420p", 8, 330, 200), &mb).Ok());
    mb.maxMacroblocks = 272;
    CheckRefusal(JudgeDecode(Av1Info("yuv420p", 8, 330, 200), &mb), DecodeBlock::Size);
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
    CHECK(s.codec == "av1");
    CHECK(s.pixelFormat == "yuv420p10le");
    CHECK(s.bitDepth == 10);
    CHECK(s.width == 1920);
    CHECK(s.height == 1080);
    CHECK(s.caps.gpu == "Test GPU");
    CHECK(s.why == "NVDEC on 'Test GPU', AV1 10-bit 4:2:0 up to 8192x8192");

    // VideoDecoder's refusals on the first frame: NVDEC refused (ENOSYS), or no CUDA device to create
    const DecodeSupport failed = RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "");
    CheckRefusal(failed, DecodeBlock::NvdecFailed);
    CHECK(failed.why == "NVDEC did not start decoding this AV1 stream and this build has no software AV1 decoder (FFmpeg's reason is in its log)");
    const DecodeSupport failedWhy = RefusedDecode(Av1Info(), DecodeBlock::NvdecFailed, "boom");
    CHECK(failedWhy.why == failed.why + ": boom");
    const DecodeSupport device = RefusedDecode(Av1Info(), DecodeBlock::NoCuda, "the CUDA device could not be created: Generic error in an external library");
    CheckRefusal(device, DecodeBlock::NoCuda);
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

TEST_CASE("CheckDecodeSupport: DLSSVID_FAKE_NVDEC_AV1 stands in for the driver on every call, also after the driver was asked", "[unit][av1]") {
    FakeNvdecEnv unset(nullptr);
    const VideoStreamInfo av1 = Av1Info();
    // the driver's own answer first (it fills the per-process cache), whatever this machine says
    const DecodeSupport real = CheckDecodeSupport(av1);
    INFO("this machine: " << BlockName(real.block) << " - " << real.why);
    CHECK(real.hardwareOnly);
    CHECK((real.block == DecodeBlock::None || real.block == DecodeBlock::NoCuda || real.block == DecodeBlock::NoAv1 || real.block == DecodeBlock::Size));
    if (real.Ok()) CHECK(real.verified == real.caps.checked);
    CHECK(CheckDecodeSupport(av1).block == real.block);  // the cache answers the same

    {
        FakeNvdecEnv fake("no-av1");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoAv1);
        CHECK(s.caps.gpu == "DLSSVID_FAKE_NVDEC_AV1");
        CHECK_THAT(s.why, ContainsSubstring("DLSSVID_FAKE_NVDEC_AV1=no-av1"));
        CheckRefusal(CheckDecodeSupport(Av1Info("yuv420p10le", 10)), DecodeBlock::NoAv1);
        // not consulted where the driver would not be: software decoders, layouts refused first, an unknown layout
        CHECK(CheckDecodeSupport(H264Info()).Ok());
        CHECK(CheckDecodeSupport(H264Info()).why == "software (h264)");
        CHECK(CheckDecodeSupport(Av1Info("yuv444p")).block == DecodeBlock::Layout);
        CHECK(CheckDecodeSupport(Av1Info("unknown")).Ok());
    }
    {
        FakeNvdecEnv fake("no-cuda");
        const DecodeSupport s = CheckDecodeSupport(av1);
        CheckRefusal(s, DecodeBlock::NoCuda);
        CHECK_THAT(s.why, ContainsSubstring("DLSSVID_FAKE_NVDEC_AV1=no-cuda"));
        CHECK(CheckDecodeSupport(H264Info()).Ok());
    }
    {
        FakeNvdecEnv fake("bogus");  // not understood: ignored, the driver answers
        CHECK(CheckDecodeSupport(av1).block == real.block);
    }
    CHECK(CheckDecodeSupport(av1).block == real.block);  // gone with its scope
}

TEST_CASE("DescribeDecode: the decode line of dlssvid info in its three forms", "[unit][av1]") {
    CHECK(DescribeDecode(JudgeDecode(H264Info(), nullptr)) == "software (h264), NVDEC with --hwaccel cuda");
    const NvdecAv1Caps good = GoodCaps();
    CHECK(DescribeDecode(JudgeDecode(Av1Info(), &good)) == "NVDEC only (av1: no software AV1 decoder in this build): NVDEC on 'Test GPU', AV1 8-bit 4:2:0 up to 8192x8192");
    CHECK(DescribeDecode(JudgeDecode(Av1Info(), nullptr)) == "NVDEC only (av1: no software AV1 decoder in this build): NVDEC (av1), not checked");
    NvdecAv1Caps noAv1 = GoodCaps();
    noAv1.supported = false;
    const DecodeSupport refused = JudgeDecode(Av1Info(), &noAv1);
    CHECK(DescribeDecode(refused) == "unavailable - " + refused.why);
    CHECK_THAT(DescribeDecode(refused), StartsWith("unavailable - the GPU 'Test GPU' has no NVDEC AV1 decoder for 8-bit 4:2:0"));
    CHECK_THAT(DescribeDecode(JudgeDecode(Av1Info("yuv444p"), nullptr)), StartsWith("unavailable - AV1 yuv444p cannot be decoded here"));
}
