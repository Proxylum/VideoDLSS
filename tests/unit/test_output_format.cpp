// The result's bit depth and chroma (TASK-0034): the automatic choice from the codec and the source's decoded layout,
// explicit requests checked against what the codec encodes, the containers and the bitrate rule for ProRes.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "io/EncodeDefaults.h"
#include "io/OutputFormat.h"

using namespace dlssvid;
using Catch::Matchers::ContainsSubstring;

namespace {
FrameDesc Src(PixelFormat f, int depth) { return FrameDesc{1920, 1080, f, static_cast<uint8_t>(depth)}; }
}  // namespace

TEST_CASE("Automatic output format follows the source within the codec", "[unit][encode][format]") {
    const FrameDesc s8 = Src(PixelFormat::Yuv420p, 8), s10 = Src(PixelFormat::Yuv444p16, 10), s12 = Src(PixelFormat::Yuv422p16, 12);
    const OutputFormatRequest autoReq;
    OutputFormat r = ResolveOutputFormat("hevc_nvenc", autoReq, s10);
    CHECK(r.desc.format == PixelFormat::Yuv420p16);  // a 10-bit 4:4:4 source -> 10-bit, 4:2:0 for NVENC (the user's default)
    CHECK(r.desc.bitDepth == 10);
    CHECK(r.autoBitDepth);
    CHECK(r.autoChroma);
    CHECK(r.desc.Describe() == "10-bit 4:2:0");
    CHECK(r.desc.Name() == "yuv420p10");
    CHECK_FALSE(r.reason.empty());
    r = ResolveOutputFormat("hevc_nvenc", autoReq, s8);
    CHECK(r.desc.format == PixelFormat::Yuv420p);
    CHECK(r.desc.bitDepth == 8);
    r = ResolveOutputFormat("h264_nvenc", autoReq, s10);  // H.264 NVENC: 8-bit only, the source is reduced
    CHECK(r.desc.format == PixelFormat::Yuv420p);
    CHECK(r.desc.bitDepth == 8);
    r = ResolveOutputFormat("av1_nvenc", autoReq, s12);  // a 12-bit source: the codec's 10
    CHECK(r.desc.bitDepth == 10);
    CHECK(ChromaName(r.desc.format) == "420");
    r = ResolveOutputFormat("prores_ks", autoReq, s8);  // ProRes is 10-bit: an 8-bit source goes up; 4:2:2 (422 HQ)
    CHECK(r.desc.format == PixelFormat::Yuv422p16);
    CHECK(r.desc.bitDepth == 10);
    r = ResolveOutputFormat("prores_ks", autoReq, s10);  // a 4:4:4 source -> 4444
    CHECK(r.desc.format == PixelFormat::Yuv444p16);
    r = ResolveOutputFormat("ffv1", autoReq, s12);  // FFV1 keeps everything
    CHECK(r.desc.format == PixelFormat::Yuv422p16);
    CHECK(r.desc.bitDepth == 12);
    r = ResolveOutputFormat("ffv1", autoReq, s8);
    CHECK(r.desc == FrameDesc{0, 0, PixelFormat::Yuv420p, 8});
    r = ResolveOutputFormat("libx264", autoReq, s10);  // an encoder outside the table: 8-bit 4:2:0 unless told otherwise
    CHECK(r.desc.format == PixelFormat::Yuv420p);
    CHECK(r.desc.bitDepth == 8);
}

TEST_CASE("Explicit output formats are checked against the codec", "[unit][encode][format]") {
    const FrameDesc s10 = Src(PixelFormat::Yuv420p16, 10);
    OutputFormatRequest req;
    req.bitDepth = 10;
    req.chroma = "444";
    OutputFormat r = ResolveOutputFormat("hevc_nvenc", req, s10);
    CHECK(r.desc.format == PixelFormat::Yuv444p16);
    CHECK_FALSE(r.autoChroma);
    CHECK_FALSE(r.autoBitDepth);
    req.chroma = "422";
    CHECK_THROWS_WITH(ResolveOutputFormat("hevc_nvenc", req, s10), ContainsSubstring("4:2:2"));
    req.chroma = "444";
    CHECK_THROWS_WITH(ResolveOutputFormat("av1_nvenc", req, s10), ContainsSubstring("4:4:4"));
    req.chroma = "auto";
    CHECK_THROWS_WITH(ResolveOutputFormat("h264_nvenc", req, s10), ContainsSubstring("8-bit only"));
    req.bitDepth = 8;
    CHECK_THROWS_WITH(ResolveOutputFormat("prores_ks", req, s10), ContainsSubstring("10-bit only"));
    req.bitDepth = 12;
    CHECK_THROWS_WITH(ResolveOutputFormat("hevc_nvenc", req, s10), ContainsSubstring("12-bit"));
    r = ResolveOutputFormat("ffv1", req, s10);
    CHECK(r.desc.bitDepth == 12);
    req.bitDepth = 9;
    CHECK_THROWS(ResolveOutputFormat("ffv1", req, s10));
    req.bitDepth = 0;
    req.chroma = "4:2:0";
    CHECK_THROWS(ResolveOutputFormat("ffv1", req, s10));
    req.chroma = "420";
    req.bitDepth = 10;
    r = ResolveOutputFormat("libx265", req, s10);  // unknown encoder: the request is passed on as it is
    CHECK(r.desc == FrameDesc{0, 0, PixelFormat::Yuv420p16, 10});

    CHECK(ParseBitDepthChoice("auto") == 0);
    CHECK(ParseBitDepthChoice("") == 0);
    CHECK(ParseBitDepthChoice("10") == 10);
    CHECK_FALSE(ParseBitDepthChoice("11"));
    CHECK(BitDepthChoiceName(0) == "auto");
    CHECK(BitDepthChoiceName(10) == "10");
    CHECK(ValidChromaChoice("422"));
    CHECK_FALSE(ValidChromaChoice("411"));
}

TEST_CASE("Codec table: containers, bitrate control and result paths", "[unit][encode][format]") {
    CHECK(ResultExtensionFor("hevc_nvenc") == ".mp4");
    CHECK(ResultExtensionFor("prores_ks") == ".mov");
    CHECK(ResultExtensionFor("ffv1") == ".mkv");
    CHECK(ResultExtensionFor("whatever") == ".mkv");
    CHECK(ResultPathFor("C:/x/clip_result.mp4", "prores_ks") == std::filesystem::path("C:/x/clip_result.mov"));
    CHECK(ResultPathFor("C:/x/clip_result.MP4", "hevc_nvenc") == std::filesystem::path("C:/x/clip_result.MP4"));  // the same container: untouched
    CHECK(ResultPathFor("C:/x/clip_result.mkv", "h264_nvenc") == std::filesystem::path("C:/x/clip_result.mp4"));
    CHECK(ResultPathFor("C:/x/clip.avi", "prores_ks") == std::filesystem::path("C:/x/clip.avi"));  // not a result container: untouched
    CHECK(ResultPathFor("", "ffv1").empty());
    CHECK(HasBitrateControl("hevc_nvenc"));
    CHECK_FALSE(HasBitrateControl("prores_ks"));
    CHECK_FALSE(HasBitrateControl("ffv1"));
    CHECK(HasBitrateControl("libx264"));
    CHECK(LossyCodec("prores_ks"));                                     // lossy ...
    CHECK(RecommendedBitrateMbps("prores_ks", 1920, 1080, 24.0) == 0.0);  // ... but the profile sets the rate: no `b`
    CHECK(EffectiveCodecOptions("prores_ks", {}, 1920, 1080, 24.0).empty());
    REQUIRE(FindCodecCaps("prores_ks"));
    CHECK(FindCodecCaps("prores_ks")->title == "ProRes");
    CHECK(FindCodecCaps("prores_ks")->bitDepths == std::vector<int>{10});
    CHECK(FindCodecCaps("h264_nvenc")->bitDepths == std::vector<int>{8});
    CHECK_FALSE(FindCodecCaps("x"));
    CHECK(KnownCodecs().size() == 5);
}
