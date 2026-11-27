#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "passes/Manifest.h"

using namespace dlssvid;

namespace {
std::filesystem::path Tmp(const char* name) {
    const std::filesystem::path d = std::filesystem::path(DLSSVID_TEST_TMP) / "manifest" / name;
    std::filesystem::create_directories(d);
    return d;
}
}  // namespace

TEST_CASE("Manifest::ForPass fills canonical defaults", "[passes][manifest]") {
    const Manifest m = Manifest::ForPass(PassKind::DepthRaw, 1920, 1080, FileFormat::Exr);
    CHECK(m.pass == "depth_raw");
    CHECK(m.pixelType == PixelType::F32);
    CHECK(m.channels == std::vector<std::string>{"Z"});
    CHECK(m.filePattern == "depth_raw_%06d.exr");
    CHECK(m.FrameFileName(42) == "depth_raw_000042.exr");
    CHECK(m.convention == Convention::Raw);

    const Manifest c = Manifest::ForPass(PassKind::ColorSource, 100, 50, FileFormat::Png);
    CHECK(c.pixelType == PixelType::U16);
    CHECK(c.colorspace == "srgb");
    CHECK(c.filePattern == "color_source_%06d.png");

    const Manifest mv = Manifest::ForPass(PassKind::MvDlss, 100, 50, FileFormat::Raw);
    CHECK(mv.pixelType == PixelType::F16);
    CHECK(mv.filePattern == "mv_dlss_%06d.rg16f");
    CHECK(mv.mv.direction == "backward");

    CHECK(Manifest::ForPass(PassKind::DepthDlss, 1, 1, FileFormat::Raw).filePattern == "depth_dlss_%06d.r32");
    CHECK(ExtensionFor(FileFormat::Raw, PixelType::U8, 1) == ".r8");
    CHECK(ExtensionFor(FileFormat::Raw, PixelType::F32, 2) == ".rg32f");
}

TEST_CASE("Manifest JSON round trip", "[passes][manifest]") {
    Manifest m = Manifest::ForPass(PassKind::MvRaw, 640, 360, FileFormat::Npz);
    m.fps = Rational{30000, 1001};
    m.frameCount = 12;
    m.firstFrame = 3;
    m.lastFrame = 14;
    m.model = "sea-raft";
    m.modelVersion = "1.0";
    m.sourceHash = "sha256:abc";
    m.sourceFile = "in.mp4";
    m.stageParams = {{"iters", 12}, {"mode", "quality"}};
    m.depth.relative = true;
    m.depth.minValue = 0.5f;
    m.depth.maxValue = 9.f;
    m.mv.yUp = true;
    m.mv.refWidth = 1280;

    const nlohmann::json j = m.ToJson();
    CHECK(j["schema_version"] == 1);
    CHECK(j["fps"]["num"] == 30000);
    const Manifest r = Manifest::FromJson(j);
    CHECK(r.pass == "mv_raw");
    CHECK(r.width == 640);
    CHECK(r.fps.den == 1001);
    CHECK(r.frameCount == 12);
    CHECK(r.firstFrame == 3);
    CHECK(r.lastFrame == 14);
    CHECK(r.format == FileFormat::Npz);
    CHECK(r.pixelType == PixelType::F32);
    CHECK(r.channels == m.channels);
    CHECK(r.model == "sea-raft");
    CHECK(r.stageParams["iters"] == 12);
    CHECK(r.depth.relative);
    CHECK(r.depth.maxValue == 9.f);
    CHECK(r.mv.yUp);
    CHECK(r.mv.refWidth == 1280);
    CHECK(r.ToJson() == j);
}

TEST_CASE("Manifest save/load and error cases", "[passes][manifest]") {
    const auto dir = Tmp("save");
    Manifest m = Manifest::ForPass(PassKind::DepthDlss, 32, 16, FileFormat::Raw);
    m.frameCount = 2;
    m.lastFrame = 1;
    m.Save(dir);
    REQUIRE(Manifest::Exists(dir));
    const Manifest l = Manifest::Load(dir);
    CHECK(l.pass == "depth_dlss");
    CHECK(l.FrameFileName(1) == "depth_dlss_000001.r32");

    const auto bad = Tmp("bad");
    std::ofstream(bad / Manifest::kFileName) << "{\"schema_version\": 99, \"width\": 1, \"height\": 1}";
    CHECK_THROWS(Manifest::Load(bad));
    std::ofstream(bad / Manifest::kFileName) << "not json";
    CHECK_THROWS(Manifest::Load(bad));
    CHECK_THROWS(Manifest::Load(Tmp("empty")));
}

TEST_CASE("File formats parse and map to extensions", "[passes][manifest]") {
    CHECK(ParseFileFormat("png16") == FileFormat::Png);
    CHECK(ParseFileFormat("tif") == FileFormat::Tiff);
    CHECK(ParseFileFormat("xyz") == std::nullopt);
    CHECK(FormatFromExtension("a/b_000001.EXR") == FileFormat::Exr);
    CHECK(FormatFromExtension("x.rg16f") == FileFormat::Raw);
    // TASK-0044: every raw name ExtensionFor gives - a colour pass in raw (.rgb16f) could not be written
    for (PixelType t : {PixelType::U8, PixelType::U16, PixelType::F16, PixelType::F32})
        for (size_t c = 1; c <= 4; ++c) {
            INFO(ExtensionFor(FileFormat::Raw, t, c));
            CHECK(FormatFromExtension("x" + ExtensionFor(FileFormat::Raw, t, c)) == FileFormat::Raw);
        }
    CHECK(FormatFromExtension("x.rgbx16f") == std::nullopt);
    CHECK(FormatFromExtension("x.txt") == std::nullopt);
    CHECK(ToString(FileFormat::Png) == "png16");
}

// PR-8: mask_control is one frame 0 for every frame of the clip
TEST_CASE("Manifest IsStatic: static one frame at 0 only", "[unit][manifest]") {
    Manifest m;
    m.frameCount = 1;
    m.firstFrame = m.lastFrame = 0;
    CHECK(!m.IsStatic());  // no stage_params.static
    m.stageParams = {{"static", true}};
    CHECK(m.IsStatic());
    m.stageParams["static"] = false;
    CHECK(!m.IsStatic());
    m.stageParams["static"] = true;
    m.frameCount = 2;
    m.lastFrame = 1;
    CHECK(!m.IsStatic());
    m.frameCount = 1;
    m.firstFrame = m.lastFrame = 1;
    CHECK(!m.IsStatic());
    m.firstFrame = m.lastFrame = 0;
    m.stageParams = nlohmann::json::array();
    CHECK(!m.IsStatic());
    Manifest s;
    s.pass = "mask_control";
    s.width = 4;
    s.height = 2;
    s.frameCount = 1;
    s.firstFrame = s.lastFrame = 0;
    s.stageParams = {{"static", true}};
    CHECK(Manifest::FromJson(s.ToJson()).IsStatic());  // survives the manifest file
}

TEST_CASE("Manifest::IsYUpMv only for motion-vector passes with mv.y_up", "[passes][manifest][mv][y_up]") {
    for (const PassKind k : {PassKind::MvRaw, PassKind::MvDlss}) {
        Manifest m = Manifest::ForPass(k, 8, 4, FileFormat::Exr);
        CHECK_FALSE(m.IsYUpMv());
        m.mv.yUp = true;
        CHECK(m.IsYUpMv());
    }
    Manifest depth = Manifest::ForPass(PassKind::DepthRaw, 8, 4, FileFormat::Exr);
    depth.mv.yUp = true;
    CHECK_FALSE(depth.IsYUpMv());
    Manifest custom = Manifest::ForPass(PassKind::MvRaw, 8, 4, FileFormat::Exr);
    custom.pass = "layered";
    custom.mv.yUp = true;
    CHECK_FALSE(custom.IsYUpMv());
}

// TASK-0080 (TC-1, guard of the SDR path): the text of a colour pass's manifest from an SDR source is frozen - fields TC adds for non-SDR
// sources ("primaries") are written only when set, so an SDR manifest stays byte for byte what it was.
TEST_CASE("A legacy colour manifest keeps its text", "[passes][manifest][hdr][regression]") {
    Manifest m = Manifest::ForPass(PassKind::ColorSr, 192, 108, FileFormat::Exr);
    m.fps = Rational{24, 1};
    m.frameCount = 6;
    m.firstFrame = 0;
    m.lastFrame = 5;
    m.sourceFile = "clip.mkv";
    m.sourceHash = "sha256:src";
    m.sourceFormat = "yuv420p10";
    m.stageParams = {{"backend", "bicubic"}, {"scale", 2.0}};
    m.fingerprint = "sha256:fp";
    m.inputs = {{"color_source", "sha256:in"}};
    m.tool = {{"app", "0.1.0"}, {"backend", "bicubic"}};
    m.paramsCanonical = {{"backend", "bicubic"}, {"scale", 2}};
    m.created = "2026-11-27T00:00:00Z";
    CHECK(m.ToJson().dump(2) == R"({
  "channels": [
    "R",
    "G",
    "B"
  ],
  "colorspace": "srgb",
  "convention": "raw",
  "created": "2026-11-27T00:00:00Z",
  "depth": {
    "far": 1000.0,
    "max": 0.0,
    "min": 0.0,
    "near": 0.10000000149011612,
    "relative": false,
    "units": "meters"
  },
  "file_pattern": "color_sr_%06d.exr",
  "fingerprint": "sha256:fp",
  "format": "exr",
  "fps": {
    "den": 1,
    "num": 24
  },
  "frame_count": 6,
  "frame_range": {
    "first": 0,
    "last": 5
  },
  "height": 108,
  "inputs": {
    "color_source": "sha256:in"
  },
  "model": {
    "name": "",
    "version": ""
  },
  "mv": {
    "direction": "forward",
    "ref_height": 0,
    "ref_width": 0,
    "y_up": false
  },
  "params_canonical": {
    "backend": "bicubic",
    "scale": 2
  },
  "pass": "color_sr",
  "pixel_type": "f16",
  "schema_version": 1,
  "source": {
    "file": "clip.mkv",
    "format": "yuv420p10",
    "hash": "sha256:src"
  },
  "stage_params": {
    "backend": "bicubic",
    "scale": 2.0
  },
  "tool": {
    "app": "0.1.0",
    "backend": "bicubic"
  },
  "width": 192
})");
}

// TASK-0080 (HDR step TC-1): the primaries of a non-SDR source's pass round-trip; an SDR pass (empty) writes no key.
TEST_CASE("Manifest primaries round-trip; an SDR pass writes none", "[passes][manifest][hdr]") {
    Manifest m = Manifest::ForPass(PassKind::ColorSr, 64, 48, FileFormat::Exr);
    CHECK(m.primaries.empty());
    CHECK_FALSE(m.ToJson().contains("primaries"));
    CHECK(Manifest::FromJson(m.ToJson()).primaries.empty());
    m.colorspace = "pq";
    m.primaries = "bt2020";
    const nlohmann::json j = m.ToJson();
    CHECK(j["primaries"] == "bt2020");
    const Manifest back = Manifest::FromJson(j);
    CHECK(back.primaries == "bt2020");
    CHECK(back.colorspace == "pq");
    const auto dir = Tmp("primaries");
    std::filesystem::remove_all(dir);
    m.Save(dir);
    CHECK(Manifest::Load(dir).primaries == "bt2020");
    std::filesystem::remove_all(dir);
}

TEST_CASE("PassColourMismatch over legacy and HDR readings", "[passes][manifest][hdr]") {
    auto pass = [](const char* colorspace, const char* primaries) {
        Manifest m = Manifest::ForPass(PassKind::ColorSr, 8, 8, FileFormat::Exr);
        m.colorspace = colorspace;
        m.primaries = primaries;
        return m;
    };
    const PassColour legacy;  // an SDR source: {srgb, ""}
    CHECK(legacy.colorspace == "srgb");
    CHECK(legacy.primaries.empty());
    // the legacy reading takes everything an SDR source's stages read today
    for (const char* cs : {"", "linear", "srgb", "bt709"}) {
        INFO(cs);
        CHECK(PassColourMismatch(pass(cs, ""), legacy, "color_sr").empty());
    }
    // ... and refuses a pass said to hold another source's colour, naming it
    CHECK(PassColourMismatch(pass("pq", ""), legacy, "color_sr") == "color_sr holds pq colour of a source that is not SDR, this source reads as SDR - recompute it");
    CHECK(PassColourMismatch(pass("hlg", "bt2020"), legacy, "color_nr") ==
          "color_nr holds hlg / bt2020 colour of a source that is not SDR, this source reads as SDR - recompute it");
    CHECK_FALSE(PassColourMismatch(pass("log", ""), legacy, "color_sr").empty());
    CHECK(PassColourMismatch(pass("hlg", ""), legacy, "color_sr") == "color_sr holds hlg colour of a source that is not SDR, this source reads as SDR - recompute it");
    CHECK(PassColourMismatch(pass("pq", "bt2020"), legacy, "color_sr") ==
          "color_sr holds pq / bt2020 colour of a source that is not SDR, this source reads as SDR - recompute it");
    CHECK_FALSE(PassColourMismatch(pass("log", "bt2020"), legacy, "color_sr").empty());
    CHECK(PassColourMismatch(pass("srgb", "bt709"), legacy, "color_sr") ==
          "color_sr holds srgb / bt709 colour of a source that is not SDR, this source reads as SDR - recompute it");
    // SDR from HDR (TC-2): srgb under BT.709 primaries
    const PassColour sdr1{"srgb", "bt709"};
    CHECK(PassColourMismatch(pass("srgb", ""), sdr1, "color_sr") == "color_sr was made before HDR and wide-gamut sources were read - recompute it");
    CHECK(PassColourMismatch(pass("", ""), sdr1, "color_sr") == "color_sr was made before HDR and wide-gamut sources were read - recompute it");
    CHECK(PassColourMismatch(pass("pq", "bt2020"), sdr1, "color_sr") == "color_sr holds pq / bt2020 colour, this source reads as srgb / bt709 - recompute it");
    CHECK(PassColourMismatch(pass("srgb", "bt2020"), sdr1, "color_sr") == "color_sr holds srgb / bt2020 colour, this source reads as srgb / bt709 - recompute it");
    CHECK(PassColourMismatch(pass("srgb", "bt709"), sdr1, "color_sr").empty());
    CHECK(PassColourMismatch(pass("", "bt709"), sdr1, "color_sr").empty());  // an empty colorspace is srgb
    // codes of a wide gamut and of a log curve
    CHECK(PassColourMismatch(pass("log", "bt2020"), PassColour{"log", "bt2020"}, "color_sr").empty());
    CHECK_FALSE(PassColourMismatch(pass("srgb", "bt2020"), PassColour{"log", "bt2020"}, "color_sr").empty());
}
