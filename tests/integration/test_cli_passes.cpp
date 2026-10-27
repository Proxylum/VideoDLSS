// CLI contract for stage 1: export / import / convert as processes.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "TestClips.h"
#include "convert/DepthConvert.h"
#include "convert/MvConvert.h"
#include "passes/PassSequence.h"
#include "passes/formats/ExrIO.h"
#include "passes/formats/PngIO.h"
#include "pipeline/ControlMask.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}

std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }

// The exit code and what the command printed (stdout and stderr).
int RunOut(const std::string& args, const std::filesystem::path& log, std::string& out) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " > \"" + log.string() + "\" 2>&1\"";
    const int rc = std::system(cmd.c_str());
    std::ifstream in(log);
    std::stringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return rc;
}

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

// PR-8: the control mask from one image. The CLI and «Загрузить…» of the GUI call the same core function: one fingerprint for one file.
TEST_CASE("dlssvid import --pass mask_control --static gives the fingerprint the GUI path gives for the same file; import --stamp of the folder keeps it", "[integration][cli][nr][mask]") {
    const auto dir = Dir("mask_static");
    std::filesystem::create_directories(dir);
    PassImage img;
    img.Allocate(8, 6, PixelType::U8, {"R", "G", "B", "A"});
    for (uint32_t y = 0; y < 6; ++y)
        for (uint32_t x = 0; x < 8; ++x)
            for (size_t c = 0; c < 4; ++c) img.Set(x, y, c, c == 3 ? (x ? 255.f : 128.f) : static_cast<float>((x * 30 + y * 5 + c * 60) % 256));
    WritePng(dir / "mask.png", img);
    std::string out;
    REQUIRE(RunOut("import -i " + Q(dir / "mask.png") + " --pass mask_control --static -o " + Q(dir / "passes"), dir / "a.log", out) == 0);
    CHECK(out.find("(imported)") != std::string::npos);
    CHECK(out.find("alpha ignored") != std::string::npos);
    CHECK(out.find("x3 (R, G, B)") != std::string::npos);
    const Manifest m = Manifest::Load(dir / "passes" / "mask_control");
    CHECK(out.find(m.fingerprint) != std::string::npos);
    const ControlMaskImport gui = ImportControlMask(dir / "mask.png", dir / "gui_passes");  // the GUI's path
    CHECK(gui.mask.fingerprint == m.fingerprint);

    // the folder as a pass: import validates it, --stamp writes the same fingerprint (the same function)
    REQUIRE(Run("import -i " + Q(dir / "passes" / "mask_control") + " --stamp") == 0);
    CHECK(Manifest::Load(dir / "passes" / "mask_control").fingerprint == m.fingerprint);
    CHECK(ReadControlMaskInfo(dir / "passes" / "mask_control"));

    // the same image again: unchanged, nothing written
    REQUIRE(RunOut("import -i " + Q(dir / "mask.png") + " --pass mask_control --static -o " + Q(dir / "passes"), dir / "b.log", out) == 0);
    CHECK(out.find("unchanged") != std::string::npos);
    CHECK(!std::filesystem::exists(dir / "passes" / "mask_control.v"));

    // review of PR-8: the frame file edited in place - --stamp would take it under the old pixels_sha256; refused, the image is loaded again
    PassImage edited;
    edited.Allocate(8, 6, PixelType::U16, {"R", "G", "B"});
    WritePng(dir / "passes" / "mask_control" / m.FrameFileName(0), edited);
    CHECK(RunOut("import -i " + Q(dir / "passes" / "mask_control") + " --stamp", dir / "c.log", out) != 0);
    CHECK(out.find("load the image again") != std::string::npos);
    CHECK(Manifest::Load(dir / "passes" / "mask_control").fingerprint == m.fingerprint);
    REQUIRE(Run("import -i " + Q(dir / "mask.png") + " --pass mask_control --static -o " + Q(dir / "passes")) == 0);  // not «unchanged»
    CHECK(ReadControlMaskInfo(dir / "passes" / "mask_control"));
}

// Review of PR-8: design §4 - «Экспорт пассов» exports mask_control as its one frame; the CLI refused it as a custom pass, so the GUI's
// export task of the mask always failed.
TEST_CASE("dlssvid export of mask_control writes its one static frame: EXR as the codes / 65535 in f32, PNG16 as the stored codes; presets refused", "[integration][cli][mask][regression]") {
    const auto dir = Dir("mask_export");
    std::filesystem::create_directories(dir);
    PassImage img;
    img.Allocate(8, 6, PixelType::U16, {"R", "G", "B"});
    for (uint32_t y = 0; y < 6; ++y)
        for (uint32_t x = 0; x < 8; ++x)
            for (size_t c = 0; c < 3; ++c) img.Set(x, y, c, static_cast<float>((x * 7919 + y * 104729 + c * 31337) % 65536));
    WritePng(dir / "mask.png", img);
    REQUIRE(Run("import -i " + Q(dir / "mask.png") + " --pass mask_control --static -o " + Q(dir / "passes")) == 0);
    const auto cur = dir / "passes" / "mask_control";
    const Manifest src = Manifest::Load(cur);
    const PassImage stored = ReadPng(cur / src.FrameFileName(0));

    std::string out;
    REQUIRE(RunOut("export --from-dir " + Q(cur) + " -o " + Q(dir / "exr") + " --format exr", dir / "exr.log", out) == 0);  // the GUI's call
    const Manifest em = Manifest::Load(dir / "exr");
    CHECK(em.pass == "mask_control");
    CHECK(em.IsStatic());
    CHECK(em.channels == src.channels);
    CHECK(em.format == FileFormat::Exr);
    CHECK(em.pixelType == PixelType::F32);
    CHECK(em.fingerprint.empty());  // the stamp was of the PNG16 folder
    const PassImage e = ReadExr(dir / "exr" / em.FrameFileName(0));
    REQUIRE(e.width == stored.width);
    REQUIRE(e.height == stored.height);
    int off = 0;
    for (uint32_t y = 0; y < e.height; ++y)
        for (uint32_t x = 0; x < e.width; ++x)
            for (size_t c = 0; c < 3; ++c)
                if (std::lround(e.Get(x, y, c) * 65535.0) != std::lround(stored.Get(x, y, c))) ++off;
    CHECK(off == 0);

    REQUIRE(Run("export --from-dir " + Q(cur) + " -o " + Q(dir / "png") + " --format png16") == 0);
    const Manifest pm = Manifest::Load(dir / "png");
    CHECK(pm.pixelType == PixelType::U16);
    CHECK(ReadPng(dir / "png" / pm.FrameFileName(0)).data == stored.data);
    CHECK(RunOut("export --from-dir " + Q(cur) + " -o " + Q(dir / "nuke") + " --preset nuke", dir / "nuke.log", out) != 0);
    CHECK(out.find("without a preset") != std::string::npos);
    CHECK(RunOut("export --from-dir " + Q(cur) + " -o " + Q(dir / "npz") + " --format npz", dir / "npz.log", out) != 0);
}

TEST_CASE("dlssvid import --static refuses another pass, the conversion flags, an explicit --dilate, a folder as -i and a missing -o; --pass mask_control without --static explains", "[integration][cli][mask]") {
    const auto dir = Dir("mask_static_refuse");
    std::filesystem::create_directories(dir);
    PassImage img;
    img.Allocate(4, 4, PixelType::U8, {"A"});
    WritePng(dir / "mask.png", img);
    const std::string base = "import -i " + Q(dir / "mask.png") + " --static -o " + Q(dir / "passes");
    std::string out;
    CHECK(RunOut(base, dir / "1.log", out) != 0);  // no --pass
    CHECK(out.find("--pass mask_control") != std::string::npos);
    CHECK(Run(base + " --pass mask_skin") != 0);
    for (const char* flag : {"--to-dlss x", "--to-raw x", "--write-manifest", "--stamp", "--y-up", "--expect-frames 1", "--expect-size 4x4", "--convention raw",
                             "--size 4x4", "--near 0.1", "--far 10", "--relative", "--depth-dir x", "--target 4x4", "--dilate 1"}) {
        INFO(flag);
        CHECK(RunOut(base + " --pass mask_control " + flag, dir / "2.log", out) != 0);
        CHECK(out.find("does not apply to a static mask") != std::string::npos);
    }
    CHECK(RunOut("import -i " + Q(dir) + " --static --pass mask_control -o " + Q(dir / "passes"), dir / "3.log", out) != 0);
    CHECK(out.find("must be an image file") != std::string::npos);
    CHECK(RunOut("import -i " + Q(dir / "mask.png") + " --static --pass mask_control", dir / "4.log", out) != 0);
    CHECK(out.find("-o <passes root> is required") != std::string::npos);
    CHECK(RunOut("import -i " + Q(dir / "mask.png") + " --pass mask_control", dir / "5.log", out) != 0);
    CHECK(out.find("add --static") != std::string::npos);
    CHECK(!std::filesystem::exists(dir / "passes" / "mask_control"));
    // and the refusals of the core reach the exit code with their English text
    { std::ofstream(dir / "text.png") << "not an image"; }
    CHECK(RunOut("import -i " + Q(dir / "text.png") + " --static --pass mask_control -o " + Q(dir / "passes"), dir / "6.log", out) != 0);
    CHECK(out.find("is not a PNG, TIFF or EXR image") != std::string::npos);
    CHECK(Run("import -i " + Q(dir / "mask.png") + " --static --pass mask_control -o " + Q(dir / "passes")) == 0);  // the valid call works
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

// Audit pass-mv-yup (TASK-0038): a y-up motion-vector pass (another tool's convention) gives exactly what the same
// vectors stored y down give - in convert, export --from-dir and import - and nothing dlssvid writes carries the flag.
namespace {
std::filesystem::path WriteMvFolder(const char* name, float v, bool markYUp) {
    const auto dir = Dir(name);
    ExportPass(dir, Manifest::ForPass(PassKind::MvRaw, 20, 10, FileFormat::Exr), [&](int64_t f, PassImage& out) {
        if (f >= 3) return false;
        out = MakePassImage(PassKind::MvRaw, 20, 10);
        for (uint32_t y = 0; y < 10; ++y)
            for (uint32_t x = 0; x < 20; ++x) {
                out.Set(x, y, 0, 1.f + static_cast<float>(f));
                out.Set(x, y, 1, v);
            }
        return true;
    }, FrameRange{});
    if (markYUp) {
        Manifest m = Manifest::Load(dir);
        m.mv.yUp = true;
        m.Save(dir);
    }
    return dir;
}

void CheckSameFrames(const std::filesystem::path& got, const std::filesystem::path& want) {
    const PassReader g = PassReader::Open(got);
    const PassReader w = PassReader::Open(want);
    CHECK_FALSE(g.Man().mv.yUp);
    REQUIRE(g.Man().frameCount == w.Man().frameCount);
    for (int64_t f = g.Man().firstFrame; f <= g.Man().lastFrame; ++f) CHECK(g.ReadFrame(f).data == w.ReadFrame(f).data);
}
}  // namespace

TEST_CASE("dlssvid convert, export and import honour mv.y_up: a y-up pass gives the result of the same vectors y down",
          "[integration][cli][passes][mv][y_up]") {
    const auto down = WriteMvFolder("yup_down", +1.f, false);  // v = +1 px: moving down, y down
    const auto up = WriteMvFolder("yup_up", -1.f, true);       // the same motion stored y up

    SECTION("convert, at the source size and resampled") {
        for (const std::string target : {"", " --target 40x20"}) {
            INFO("target:" << target);
            const auto a = Dir("yup_conv_a");
            const auto b = Dir("yup_conv_b");
            REQUIRE(Run("convert -i " + Q(down) + " -o " + Q(a) + " --to mv_dlss --dilate 0" + target) == 0);
            REQUIRE(Run("convert -i " + Q(up) + " -o " + Q(b) + " --to mv_dlss --dilate 0" + target) == 0);
            CheckSameFrames(b, a);
            CHECK(PassReader::Open(b).ReadFrame(1).Get(5, 5, 1) == (target.empty() ? -1.f : -2.f));  // backward: moved down -> v < 0
        }
    }
    SECTION("export --from-dir writes y down, without the flag (no second flip on the next read)") {
        const auto e = Dir("yup_export");
        REQUIRE(Run("export --from-dir " + Q(up) + " --format exr -o " + Q(e)) == 0);
        CheckSameFrames(e, down);
    }
    SECTION("import --y-up of a folder without a manifest, then convert") {
        const auto bare = Dir("yup_bare");
        std::filesystem::copy(up, bare, std::filesystem::copy_options::recursive);
        std::filesystem::remove(bare / Manifest::kFileName);
        REQUIRE(Run("import -i " + Q(bare) + " --pass mv_raw --y-up --write-manifest") == 0);
        CHECK(Manifest::Load(bare).mv.yUp);  // the manifest tells the truth about the files
        const auto a = Dir("yup_bare_a");
        const auto b = Dir("yup_bare_b");
        REQUIRE(Run("convert -i " + Q(down) + " -o " + Q(a) + " --to mv_dlss --dilate 0") == 0);
        REQUIRE(Run("convert -i " + Q(bare) + " -o " + Q(b) + " --to mv_dlss --dilate 0") == 0);
        CheckSameFrames(b, a);
    }
    SECTION("export --from-dir --y-up of a folder without a manifest") {
        const auto bare = Dir("yup_bare_export");
        std::filesystem::copy(up, bare, std::filesystem::copy_options::recursive);
        std::filesystem::remove(bare / Manifest::kFileName);
        const auto e = Dir("yup_bare_export_out");
        REQUIRE(Run("export --from-dir " + Q(bare) + " --scan-pass mv_raw --y-up --format exr -o " + Q(e)) == 0);
        CheckSameFrames(e, down);
    }
    SECTION("--y-up on a folder whose manifest says y_up = false applies to the run only") {
        const auto flagless = Dir("yup_flagless");
        std::filesystem::copy(up, flagless, std::filesystem::copy_options::recursive);
        Manifest m = Manifest::Load(flagless);
        m.mv.yUp = false;
        m.Save(flagless);
        const auto a = Dir("yup_flagless_a");
        const auto b = Dir("yup_flagless_b");
        REQUIRE(Run("convert -i " + Q(down) + " -o " + Q(a) + " --to mv_dlss --dilate 0") == 0);
        REQUIRE(Run("convert -i " + Q(flagless) + " -o " + Q(b) + " --to mv_dlss --dilate 0 --y-up") == 0);
        CheckSameFrames(b, a);
        CHECK_FALSE(Manifest::Load(flagless).mv.yUp);  // not rewritten
    }
    SECTION("a folder with a manifest keeps it under --pass: no scan, no overwrite (review of TASK-0038, yup-1)") {
        const auto kept = Dir("yup_kept");
        std::filesystem::copy(up, kept, std::filesystem::copy_options::recursive);
        const std::string before = Manifest::Load(kept).ToJson().dump();
        REQUIRE(Run("import -i " + Q(kept) + " --pass mv_raw --write-manifest") == 0);
        CHECK(Manifest::Load(kept).ToJson().dump() == before);  // before: rescanned and overwritten, y_up lost
        const auto a = Dir("yup_kept_a");
        const auto b = Dir("yup_kept_b");
        REQUIRE(Run("convert -i " + Q(down) + " -o " + Q(a) + " --to mv_dlss --dilate 0") == 0);
        REQUIRE(Run("convert -i " + Q(kept) + " --pass mv_raw -o " + Q(b) + " --to mv_dlss --dilate 0") == 0);
        CheckSameFrames(b, a);  // y_up of the manifest applied, as without --pass
        CHECK(Run("import -i " + Q(kept) + " --pass depth_raw") != 0);  // the manifest names another pass
    }
    SECTION("convert refuses a pair that is not a conversion (review of TASK-0038, yup-3)") {
        CHECK(Run("convert -i " + Q(up) + " -o " + Q(Dir("yup_same")) + " --to mv_raw --y-up") != 0);  // before: shifted, inverted garbage
        CHECK(Run("convert -i " + Q(up) + " -o " + Q(Dir("yup_cross")) + " --to depth_dlss") != 0);
    }
    SECTION("--y-up is refused for passes that are not motion vectors") {
        const auto depth = WriteDepthFolder("yup_depth", 8, 4, 2);
        CHECK(Run("import -i " + Q(depth) + " --y-up") != 0);
        CHECK(Run("export --from-dir " + Q(depth) + " --y-up --format exr -o " + Q(Dir("yup_depth_out"))) != 0);
        CHECK(Run("convert -i " + Q(depth) + " -o " + Q(Dir("yup_depth_conv")) + " --to depth_dlss --y-up") != 0);
    }
}
