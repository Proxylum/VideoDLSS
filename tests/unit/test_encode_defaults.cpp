// Bitrate defaults of the encode step (TASK-0016): without `b` a 3840×1600 result came out at ~1.4 Mbit/s from NVENC's
// own default; a run now picks the bitrate from the output size, rate and codec, and the GUI / CLI values round-trip.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "io/EncodeDefaults.h"

using namespace dlssvid;
using Catch::Approx;

TEST_CASE("Recommended bitrate follows the output size, rate and codec", "[unit][encode]") {
    CHECK(RecommendedBitrateMbps("h264_nvenc", 1280, 720, 30.0) == Approx(10.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 1920, 1080, 30.0) == Approx(16.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 1920, 1080, 60.0) == Approx(24.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 3840, 2160, 30.0) == Approx(45.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 3840, 2160, 60.0) == Approx(70.0));   // 67.5 rounded to fives
    CHECK(RecommendedBitrateMbps("h264_nvenc", 3840, 1600, 48.0) == Approx(45.0));   // the face / pan results
    CHECK(RecommendedBitrateMbps("hevc_nvenc", 3840, 2160, 30.0) == Approx(29.0));
    CHECK(RecommendedBitrateMbps("av1_nvenc", 3840, 2160, 30.0) == Approx(25.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 7680, 4320, 30.0) == Approx(160.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 640, 360, 30.0) == Approx(2.5));      // proportional below 720p
    CHECK(RecommendedBitrateMbps("h264_nvenc", 1920, 1080, 120.0) > RecommendedBitrateMbps("h264_nvenc", 1920, 1080, 60.0));
    CHECK(RecommendedBitrateMbps("h264_nvenc", 1920, 1080, 0.0) == Approx(16.0));    // unknown rate: as 30 fps
    CHECK(RecommendedBitrateMbps("ffv1", 1920, 1080, 30.0) == 0.0);                   // lossless: no bitrate
    CHECK(RecommendedBitrateMbps("h264_nvenc", 0, 0, 30.0) == 0.0);
    CHECK(LossyCodec("h264_nvenc"));
    CHECK(!LossyCodec("ffv1"));
}

TEST_CASE("Bitrate options parse and format both ways", "[unit][encode]") {
    CHECK(ParseBitrateMbps("50M") == Approx(50.0));
    CHECK(ParseBitrateMbps("50000k") == Approx(50.0));
    CHECK(ParseBitrateMbps("50000000") == Approx(50.0));
    CHECK(ParseBitrateMbps("1.5M") == Approx(1.5));
    CHECK(ParseBitrateMbps(" 12m ") == Approx(12.0));
    CHECK(ParseBitrateMbps("0.5G") == Approx(500.0));
    CHECK(!ParseBitrateMbps("abc"));
    CHECK(!ParseBitrateMbps(""));
    CHECK(!ParseBitrateMbps("50 Mbit"));
    CHECK(!ParseBitrateMbps("-5M"));
    CHECK(BitrateOption(50.0) == "50M");
    CHECK(BitrateOption(45.0) == "45M");
    CHECK(BitrateOption(1.5) == "1500k");
    CHECK(BitrateOption(0.0).empty());
    CHECK(ParseBitrateMbps(BitrateOption(2.5)) == Approx(2.5));
}

TEST_CASE("Effective encoder options add the bitrate only when a lossy codec has none", "[unit][encode]") {
    const std::map<std::string, std::string> none;
    auto auto264 = EffectiveCodecOptions("h264_nvenc", none, 3840, 1600, 48.0);
    REQUIRE(auto264.count("b") == 1);
    CHECK(auto264["b"] == "45M");
    const std::map<std::string, std::string> given{{"b", "20M"}, {"preset", "p7"}};
    auto kept = EffectiveCodecOptions("h264_nvenc", given, 3840, 1600, 48.0);
    CHECK(kept["b"] == "20M");
    CHECK(kept["preset"] == "p7");
    CHECK(kept.size() == 2);
    auto lossless = EffectiveCodecOptions("ffv1", none, 3840, 1600, 48.0);
    CHECK(lossless.empty());
    auto unknownSize = EffectiveCodecOptions("h264_nvenc", none, 0, 0, 48.0);
    CHECK(unknownSize.empty());
}
