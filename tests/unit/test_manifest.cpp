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
    CHECK(FormatFromExtension("x.txt") == std::nullopt);
    CHECK(ToString(FileFormat::Png) == "png16");
}
