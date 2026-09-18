// CLI contract for stage 1: export / import / convert as processes.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "TestClips.h"
#include "convert/DepthConvert.h"
#include "convert/MvConvert.h"
#include "passes/PassSequence.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}

std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "cli_passes" / name;
    std::filesystem::remove_all(d);
    return d;
}

PassImage Depth(int64_t f, uint32_t w, uint32_t h) {
    PassImage d = MakePassImage(PassKind::DepthRaw, w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) d.Set(x, y, 0, 2.f + x * 0.05f + f * 0.5f + (y % 3) * 0.01f);
    return d;
}

std::filesystem::path WriteDepthFolder(const char* name, uint32_t w, uint32_t h, int64_t frames) {
    const auto dir = Dir(name);
    Manifest m = Manifest::ForPass(PassKind::DepthRaw, w, h, FileFormat::Exr);
    ExportPass(dir, m, [&](int64_t f, PassImage& out) {
        if (f >= frames) return false;
        out = Depth(f, w, h);
        return true;
    }, FrameRange{});
    return dir;
}

}  // namespace

TEST_CASE("dlssvid export writes color_source from a clip with manifest and hash", "[integration][cli][passes]") {
    ClipSpec spec;
    spec.frames = 5;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(TempDir() / "cli_export.mkv", spec);
    const auto out = Dir("export_exr");
    REQUIRE(Run("export -i " + Q(clip) + " -o " + Q(out)) == 0);
    const PassReader r = PassReader::Open(out);
    CHECK(r.Man().pass == "color_source");
    CHECK(r.Man().frameCount == 5);
    CHECK(r.Man().width == 48);
    CHECK(r.Man().fps.num == 24);
    CHECK(r.Man().sourceHash.rfind("sha256:", 0) == 0);
    CHECK(r.Man().sourceHash.size() == 7 + 64);
    CHECK(r.Man().pixelType == PixelType::F16);
    CHECK(r.ReadFrame(4).channels == std::vector<std::string>{"R", "G", "B"});

    // range + png16 + comfyui preset
    const auto png = Dir("export_png");
    REQUIRE(Run("export -i " + Q(clip) + " -o " + Q(png) + " --preset comfyui --range 1-3") == 0);
    const PassReader rp = PassReader::Open(png);
    CHECK(rp.Man().format == FileFormat::Png);
    CHECK(rp.Man().pixelType == PixelType::U16);
    CHECK(rp.Man().firstFrame == 1);
    CHECK(rp.Man().lastFrame == 3);
    CHECK(rp.Man().frameCount == 3);
    CHECK(std::filesystem::exists(png / "color_source_000002.png"));
    CHECK(rp.ReadFrame(2).Get(3, 3, 0) <= 65535.f);

    // re-export a folder into another format
    const auto tif = Dir("export_tif");
    REQUIRE(Run("export --from-dir " + Q(out) + " -o " + Q(tif) + " --format tiff") == 0);
    const PassReader rt = PassReader::Open(tif);
    CHECK(rt.Man().format == FileFormat::Tiff);
    CHECK(rt.ReadFrame(0).data == r.ReadFrame(0).data);

    // errors: both / neither sources, bad preset
    CHECK(Run("export -o " + Q(Dir("x"))) != 0);
    CHECK(Run("export -i " + Q(clip) + " --from-dir " + Q(out) + " -o " + Q(Dir("x"))) != 0);
    CHECK(Run("export -i " + Q(clip) + " -o " + Q(Dir("x")) + " --preset bogus") != 0);
}

TEST_CASE("dlssvid import validates a pass folder", "[integration][cli][passes]") {
    const auto depth = WriteDepthFolder("import_depth", 40, 24, 6);
    CHECK(Run("import -i " + Q(depth)) == 0);
    CHECK(Run("import -i " + Q(depth) + " --expect-size 40x24 --expect-frames 6") == 0);
    CHECK(Run("import -i " + Q(depth) + " --expect-size 41x24") != 0);
    CHECK(Run("import -i " + Q(depth) + " --expect-frames 5") != 0);
    std::filesystem::remove(depth / "depth_raw_000003.exr");
    CHECK(Run("import -i " + Q(depth)) != 0);  // missing frame

    // folder without manifest: needs --pass, can write one
    const auto bare = WriteDepthFolder("import_bare", 20, 12, 3);
    std::filesystem::remove(bare / Manifest::kFileName);
    CHECK(Run("import -i " + Q(bare)) != 0);
    CHECK(Run("import -i " + Q(bare) + " --pass depth_raw --write-manifest") == 0);
    CHECK(Manifest::Exists(bare));
    CHECK(PassReader::Open(bare).Man().frameCount == 3);
    CHECK(Run("import -i " + Q(Dir("nonexistent"))) != 0);
}

TEST_CASE("dlssvid convert depth_raw -> depth_dlss -> depth_raw", "[integration][cli][passes]") {
    const auto raw = WriteDepthFolder("conv_depth", 30, 20, 4);
    const auto dlss = Dir("conv_depth_dlss");
    REQUIRE(Run("convert -i " + Q(raw) + " -o " + Q(dlss) + " --to depth_dlss --near 0.5 --far 100") == 0);
    const PassReader rd = PassReader::Open(dlss);
    CHECK(rd.Man().pass == "depth_dlss");
    CHECK(rd.Man().convention == Convention::Dlss);
    CHECK(rd.Man().depth.zNear == 0.5f);
    CHECK(rd.Man().depth.zFar == 100.f);
    CHECK(rd.Man().frameCount == 4);
    const PassImage d1 = rd.ReadFrame(1);
    const float expected = LinearToReverseZ(Depth(1, 30, 20).Get(5, 5, 0), 0.5f, 100.f);
    CHECK(std::fabs(d1.Get(5, 5, 0) - expected) < 1e-6f);

    const auto back = Dir("conv_depth_back");
    REQUIRE(Run("convert -i " + Q(dlss) + " -o " + Q(back) + " --to depth_raw") == 0);
    const PassReader rb = PassReader::Open(back);
    CHECK(rb.Man().pass == "depth_raw");
    for (uint32_t y = 0; y < 20; ++y)
        for (uint32_t x = 0; x < 30; ++x) {
            const float a = rb.ReadFrame(2).Get(x, y, 0), b = Depth(2, 30, 20).Get(x, y, 0);
            CHECK(std::fabs(a - b) <= b * 2e-5f);
        }
    // import --to-dlss shortcut and rawdlss re-export
    const auto viaImport = Dir("conv_via_import");
    REQUIRE(Run("import -i " + Q(raw) + " --to-dlss " + Q(viaImport) + " --near 0.5 --far 100") == 0);
    CHECK(PassReader::Open(viaImport).ReadFrame(1).data == d1.data);
    const auto rawdlss = Dir("conv_rawdlss");
    REQUIRE(Run("export --from-dir " + Q(dlss) + " -o " + Q(rawdlss) + " --preset rawdlss") == 0);
    CHECK(std::filesystem::exists(rawdlss / "depth_dlss_000000.r32"));
    CHECK(PassReader::Open(rawdlss).ReadFrame(1).data == d1.data);
    CHECK(Run("export --from-dir " + Q(raw) + " -o " + Q(Dir("x")) + " --preset rawdlss") != 0);  // raw pass not allowed
    CHECK(Run("convert -i " + Q(raw) + " -o " + Q(Dir("x")) + " --to color_source") != 0);
}

TEST_CASE("dlssvid convert mv_raw -> mv_dlss shifts by one frame and scales", "[integration][cli][passes]") {
    const uint32_t w = 20, h = 10;
    const auto raw = Dir("conv_mv");
    Manifest m = Manifest::ForPass(PassKind::MvRaw, w, h, FileFormat::Exr);
    ExportPass(raw, m, [&](int64_t f, PassImage& out) {
        if (f >= 3) return false;
        out = MakePassImage(PassKind::MvRaw, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                out.Set(x, y, 0, 1.f + static_cast<float>(f));
                out.Set(x, y, 1, 0.f);
            }
        return true;
    }, FrameRange{});
    const auto dlss = Dir("conv_mv_dlss");
    REQUIRE(Run("convert -i " + Q(raw) + " -o " + Q(dlss) + " --to mv_dlss --target 40x20 --dilate 0") == 0);
    const PassReader r = PassReader::Open(dlss);
    CHECK(r.Man().pass == "mv_dlss");
    CHECK(r.Man().width == 40);
    CHECK(r.Man().height == 20);
    CHECK(r.Man().mv.direction == "backward");
    CHECK(r.Man().frameCount == 3);
    CHECK(r.ReadFrame(0).Get(10, 10, 0) == 0.f);   // first frame: no previous -> zero
    CHECK(r.ReadFrame(1).Get(10, 10, 0) == -2.f);  // from mv_raw[0] = +1 px, scaled x2, negated
    CHECK(r.ReadFrame(2).Get(10, 10, 0) == -4.f);  // from mv_raw[1] = +2 px
    const auto back = Dir("conv_mv_back");
    REQUIRE(Run("convert -i " + Q(dlss) + " -o " + Q(back) + " --to mv_raw --target 20x10 --dilate 0") == 0);
    const PassReader rb = PassReader::Open(back);
    CHECK(rb.Man().pass == "mv_raw");
    CHECK(rb.ReadFrame(0).Get(5, 5, 0) == 1.f);
    CHECK(rb.ReadFrame(1).Get(5, 5, 0) == 2.f);
    CHECK(rb.ReadFrame(2).Get(5, 5, 0) == 0.f);  // last frame: no next -> zero
}
