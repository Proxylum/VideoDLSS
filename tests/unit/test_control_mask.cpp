// The control mask of NR (PR-8, docs/conventions.md §6): import (formats, channels, normalisation, limits, the lock, the atomic
// replace), reading it back with the content check, removal and the texels NR uploads at the frame size.

#include <catch2/catch_test_macros.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>

#include "MaskImages.h"
#include "passes/ControlMaskPass.h"
#include "passes/Manifest.h"
#include "passes/formats/ExrIO.h"
#include "passes/formats/ImageProbe.h"
#include "passes/formats/PngIO.h"
#include "passes/formats/TiffIO.h"
#include "pipeline/ControlMask.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"

using namespace dlssvid;
using namespace dlssvid::test;
namespace fs = std::filesystem;

namespace {

fs::path Dir(const char* name) {
    const auto d = fs::path(DLSSVID_TEST_TMP) / "control_mask" / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

PassImage Image(uint32_t w, uint32_t h, PixelType t, std::vector<std::string> channels, const std::function<float(uint32_t, uint32_t, size_t)>& v) {
    PassImage img;
    img.Allocate(w, h, t, std::move(channels));
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < img.channels.size(); ++c) img.Set(x, y, c, v(x, y, c));
    return img;
}

PassImage Stored(const fs::path& root) {
    const Manifest m = Manifest::Load(root / kControlMaskPass);
    return ReadPng(root / kControlMaskPass / m.FrameFileName(0));
}

ControlMaskIssue IssueOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const ControlMaskError& e) {
        return e.issue;
    }
    FAIL_CHECK("no ControlMaskError");
    return {ControlMaskIssue::Kind::Write, {}};
}
ControlMaskIssue::Kind KindOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const ControlMaskError& e) {
        return e.issue.kind;
    }
    FAIL_CHECK("no ControlMaskError");
    return ControlMaskIssue::Kind::Write;
}

}  // namespace

// TASK-0042's trap (8 bit read as a fraction of 65535: an almost black mask) does not apply: codes are scaled x257 exactly.
TEST_CASE("ImportControlMask: an 8-bit RGB PNG is stored as png16 x257 without darkening; one static frame 0 with created and fingerprint", "[unit][mask][regression]") {
    const auto dir = Dir("png8");
    const auto root = dir / "passes";
    WritePng(dir / "mask.png", Image(6, 4, PixelType::U8, {"R", "G", "B"}, [](uint32_t x, uint32_t y, size_t c) { return static_cast<float>((x * 40 + y * 10 + c * 70) % 256); }));
    const ControlMaskImport r = ImportControlMask(dir / "mask.png", root);
    CHECK(r.outcome == ControlMaskImport::Outcome::Imported);
    CHECK(r.retiredVersion.empty());
    CHECK(r.sourceFormat == "png");
    CHECK(r.sourceType == "u8");
    const Manifest m = Manifest::Load(root / kControlMaskPass);
    CHECK(m.pass == "mask_control");
    CHECK(m.IsStatic());
    CHECK(m.frameCount == 1);
    CHECK(m.firstFrame == 0);
    CHECK(m.lastFrame == 0);
    CHECK(m.fps.num == 0);
    CHECK(m.format == FileFormat::Png);
    CHECK(m.pixelType == PixelType::U16);
    CHECK(m.channels == std::vector<std::string>{"R", "G", "B"});
    CHECK(!m.created.empty());
    CHECK(m.fingerprint.rfind("sha256:", 0) == 0);
    CHECK(m.stageParams["source_name"] == "mask.png");
    CHECK(m.stageParams["source_format"] == "png");
    CHECK(m.stageParams["source_type"] == "u8");
    CHECK(m.stageParams["source_size"] == nlohmann::json::array({6, 4}));
    CHECK(m.stageParams["grayscale"] == false);
    CHECK(!m.stageParams.contains("clamped"));  // integers are never clamped
    const PassImage s = Stored(root);
    REQUIRE(s.type == PixelType::U16);
    for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 6; ++x)
            for (size_t c = 0; c < 3; ++c) CHECK(s.Get(x, y, c) == static_cast<float>(((x * 40 + y * 10 + c * 70) % 256) * 257));
    CHECK(r.mask.fingerprint == m.fingerprint);
    CHECK(r.mask.width == 6);
    CHECK(r.mask.height == 4);
    CHECK(!r.mask.grayscale);
    CHECK(fs::is_directory(root / ".import"));                  // the lock's folder stays, empty
    CHECK(fs::is_empty(root / ".import"));
    CHECK(PassesUnderRoot(root) == std::vector<std::string>{"mask_control"});  // .import is no pass
}

TEST_CASE("ImportControlMask: grey (one channel, grey with alpha, R=G=B exactly) is one channel Y; almost grey RGB stays RGB", "[unit][mask]") {
    const auto dir = Dir("grey");
    auto grey = [](uint32_t x, uint32_t y, size_t) { return static_cast<float>((x * 30 + y * 7) % 256); };
    WritePng(dir / "one.png", Image(5, 3, PixelType::U8, {"A"}, grey));
    WritePng(dir / "ya.png", Image(5, 3, PixelType::U8, {"Y", "A"}, [&](uint32_t x, uint32_t y, size_t c) { return c ? 255.f : grey(x, y, 0); }));
    WritePng(dir / "rgb.png", Image(5, 3, PixelType::U16, {"R", "G", "B"}, [&](uint32_t x, uint32_t y, size_t) { return grey(x, y, 0) * 200.f; }));
    PassImage almost = Image(5, 3, PixelType::U16, {"R", "G", "B"}, [&](uint32_t x, uint32_t y, size_t) { return grey(x, y, 0) * 200.f; });
    almost.Set(4, 2, 1, almost.Get(4, 2, 1) + 1.f);  // one code off in one pixel
    WritePng(dir / "almost.png", almost);
    for (const char* name : {"one.png", "ya.png", "rgb.png"}) {
        INFO(name);
        const auto root = dir / (std::string(name) + ".passes");
        const ControlMaskImport r = ImportControlMask(dir / name, root);
        CHECK(r.mask.grayscale);
        const Manifest m = Manifest::Load(root / kControlMaskPass);
        CHECK(m.channels == std::vector<std::string>{"Y"});
        CHECK(m.stageParams["grayscale"] == true);
        CHECK(m.stageParams["bands_differ"] == false);
        const PassImage s = Stored(root);
        REQUIRE(s.channels.size() == 1);
        const float scale = std::string(name) == "rgb.png" ? 200.f : 257.f;
        CHECK(s.Get(3, 2, 0) == grey(3, 2, 0) * scale);
    }
    const auto root = dir / "almost.passes";
    const ControlMaskImport r = ImportControlMask(dir / "almost.png", root);
    CHECK(!r.mask.grayscale);  // no tolerance: almost grey is colour (v, v, v) -> v^2
    CHECK(Manifest::Load(root / kControlMaskPass).channels.size() == 3);
    CHECK(r.mask.bandsDiffer);  // G differs from B in that pixel
}

TEST_CASE("ImportControlMask: alpha is ignored and noted; a one-channel PNG read as A is the value", "[unit][mask]") {
    const auto dir = Dir("alpha");
    WritePng(dir / "rgba.png", Image(4, 4, PixelType::U8, {"R", "G", "B", "A"}, [](uint32_t x, uint32_t, size_t c) { return c == 3 ? (x < 2 ? 0.f : 255.f) : 200.f; }));
    WritePng(dir / "opaque.png", Image(4, 4, PixelType::U8, {"R", "G", "B", "A"}, [](uint32_t x, uint32_t, size_t c) { return c == 3 ? 255.f : static_cast<float>(x * 60 + c); }));
    WritePng(dir / "value.png", Image(4, 4, PixelType::U8, {"A"}, [](uint32_t x, uint32_t, size_t) { return x * 80.f; }));
    const ControlMaskImport a = ImportControlMask(dir / "rgba.png", dir / "a");
    CHECK(a.mask.alphaIgnored);
    CHECK(Manifest::Load(dir / "a" / kControlMaskPass).stageParams["alpha_ignored"] == true);
    CHECK(Stored(dir / "a").Get(0, 0, 0) == 200.f * 257.f);  // RGB as it is where the alpha is 0: the mask is white-ish there too
    const ControlMaskImport o = ImportControlMask(dir / "opaque.png", dir / "o");
    CHECK(!o.mask.alphaIgnored);
    CHECK(!Manifest::Load(dir / "o" / kControlMaskPass).stageParams.contains("alpha_ignored"));
    const ControlMaskImport v = ImportControlMask(dir / "value.png", dir / "v");
    CHECK(!v.mask.alphaIgnored);  // ReadPng names the only channel "A": it is the value, not an alpha
    CHECK(v.mask.grayscale);
    CHECK(Stored(dir / "v").Get(2, 0, 0) == 160.f * 257.f);
}

TEST_CASE("ImportControlMask: 16 bit is kept bit for bit; float EXR and TIFF clamp to 0..1, NaN and Inf counted; an EXR data window smaller than the display window is padded with zeros at its place", "[unit][mask]") {
    const auto dir = Dir("depth");
    const PassImage p16 = Image(7, 3, PixelType::U16, {"R", "G", "B"}, [](uint32_t x, uint32_t y, size_t c) { return static_cast<float>((x * 9337 + y * 1777 + c * 31) % 65536); });
    WritePng(dir / "p16.png", p16);
    ImportControlMask(dir / "p16.png", dir / "p16");
    CHECK(PassImage::MaxAbsDiff(Stored(dir / "p16"), p16) == 0.0);

    // float: -0.5, 0.25, 1.5, NaN, +Inf, -Inf, 1, 0 in one row (four clamped values per channel would be too many: one channel each)
    const std::vector<float> row{-0.5f, 0.25f, 1.5f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(), 1.f, 0.f};
    const std::vector<uint16_t> want{0, 16384, 65535, 0, 65535, 0, 65535, 0};
    WriteExr(dir / "f.exr", Image(8, 1, PixelType::F32, {"Y"}, [&](uint32_t x, uint32_t, size_t) { return row[x]; }));
    WriteTiffWith(dir / "f.tif", {8, 1, 1, 32, SAMPLEFORMAT_IEEEFP, PHOTOMETRIC_MINISBLACK}, [&](uint32_t x, uint32_t, uint32_t) { return row[x]; });
    for (const char* name : {"f.exr", "f.tif"}) {
        INFO(name);
        const auto root = dir / (std::string(name) + ".passes");
        const ControlMaskImport r = ImportControlMask(dir / name, root);
        CHECK(r.mask.clamped == 5);  // -0.5, 1.5, NaN, +Inf, -Inf
        CHECK(Manifest::Load(root / kControlMaskPass).stageParams["clamped"] == 5);
        const PassImage s = Stored(root);
        for (uint32_t x = 0; x < 8; ++x) CHECK(s.Get(x, 0, 0) == static_cast<float>(want[x]));
    }

    // review of PR-8: R = G = B by the source's codes - NaN pixels of one code keep a float mask grey (as floats NaN != NaN made it RGB)
    WriteExr(dir / "grey_nan.exr", Image(6, 2, PixelType::F32, {"R", "G", "B"}, [](uint32_t x, uint32_t, size_t) {
                 return x == 3 ? std::numeric_limits<float>::quiet_NaN() : 0.5f;
             }));
    const ControlMaskImport gn = ImportControlMask(dir / "grey_nan.exr", dir / "grey_nan");
    CHECK(gn.mask.grayscale);
    CHECK(Manifest::Load(dir / "grey_nan" / kControlMaskPass).channels == std::vector<std::string>{"Y"});
    CHECK(gn.mask.clamped == 2);  // the NaN pixels of both rows: one value each (stored grey)

    // a matte of 3x2 at (2, 1) in a 6x4 display window: zeros around it, the mask the display window's size
    WriteExrWindows(dir / "bbox.exr", 6, 4, 2, 1, 3, 2, {"R", "G", "B"}, [](int, int, size_t) { return 1.f; });
    const ControlMaskImport b = ImportControlMask(dir / "bbox.exr", dir / "bbox");
    CHECK(b.exrPadded);
    CHECK(b.mask.width == 6);
    CHECK(b.mask.height == 4);
    CHECK(Manifest::Load(dir / "bbox" / kControlMaskPass).stageParams["exr_padded"] == true);
    const PassImage s = Stored(dir / "bbox");
    for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 6; ++x) {
            const bool inside = x >= 2 && x < 5 && y >= 1 && y < 3;
            CHECK(s.Get(x, y, 0) == (inside ? 65535.f : 0.f));
        }
    // review of PR-8: an EXR from a folder with Cyrillic in its name (path.string() was the ANSI code page - «не читается»)
    const auto cyr = dir / std::filesystem::path(u8"маски");
    fs::create_directories(cyr);
    WriteExr(cyr / "m.exr", Image(4, 2, PixelType::F16, {"R", "G", "B"}, [](uint32_t x, uint32_t, size_t c) { return 0.25f * static_cast<float>(x) + 0.1f * c; }));
    CHECK(ProbeImage(cyr / "m.exr").format == ImageProbe::Format::Exr);
    const ControlMaskImport u8mask = ImportControlMask(cyr / "m.exr", cyr / "passes");
    CHECK(u8mask.mask.width == 4);
    CHECK(u8mask.sourceFormat == "exr");
    // an EXR without R, G, B, Y or a single channel
    WriteExr(dir / "uv.exr", Image(2, 2, PixelType::F16, {"u", "v"}, [](uint32_t, uint32_t, size_t) { return 0.5f; }));
    CHECK(KindOf([&] { ImportControlMask(dir / "uv.exr", dir / "uv"); }) == ControlMaskIssue::Kind::Channels);
}

// The limits come from the header: a file whose pixels would not even decode is refused as too large, not as unreadable.
TEST_CASE("ImportControlMask refuses a side above 16384 or more than 2^26 pixels from the header, before decoding", "[unit][mask][regression]") {
    const auto dir = Dir("large");
    WritePngHeaderOnly(dir / "wide.png", 16385, 1);
    WritePngHeaderOnly(dir / "huge.png", 8193, 8193);  // 67 125 249 > 2^26
    WritePngHeaderOnly(dir / "fits.png", 16384, 4096);  // exactly 2^26: passes the header, fails decoding
    for (const char* name : {"wide.png", "huge.png"}) {
        INFO(name);
        try {
            ImportControlMask(dir / name, dir / "passes");
            FAIL("imported");
        } catch (const ControlMaskError& e) {
            CHECK(e.issue.kind == ControlMaskIssue::Kind::TooLarge);
            CHECK(e.issue.width > 0);
        }
    }
    CHECK(KindOf([&] { ImportControlMask(dir / "fits.png", dir / "passes"); }) == ControlMaskIssue::Kind::Unreadable);
    CHECK(!fs::exists(dir / "passes" / kControlMaskPass));
}

TEST_CASE("ImportControlMask: palette, tiled and planar TIFF refused, min-is-white inverted; the type comes from the signature", "[unit][mask]") {
    const auto dir = Dir("tiff");
    auto code = [](uint32_t x, uint32_t, uint32_t) { return static_cast<double>(x * 50); };
    WriteTiffWith(dir / "palette.tif", {6, 2, 1, 8, SAMPLEFORMAT_UINT, PHOTOMETRIC_PALETTE}, code);
    WriteTiffWith(dir / "tiled.tif", {20, 20, 1, 8, SAMPLEFORMAT_UINT, PHOTOMETRIC_MINISBLACK, true, false}, code);
    WriteTiffWith(dir / "planar.tif", {6, 2, 3, 8, SAMPLEFORMAT_UINT, PHOTOMETRIC_RGB, false, true}, code);
    WriteTiffWith(dir / "white.tif", {6, 2, 1, 8, SAMPLEFORMAT_UINT, PHOTOMETRIC_MINISWHITE}, code);
    // review of PR-8: samples ReadTiff does not take (a bilevel mask of an editor, 32-bit integers) are «not supported», not «corrupted»
    WriteTiffWith(dir / "bilevel.tif", {6, 2, 1, 1, SAMPLEFORMAT_UINT, PHOTOMETRIC_MINISWHITE}, code);
    WriteTiffWith(dir / "int32.tif", {6, 2, 1, 32, SAMPLEFORMAT_UINT, PHOTOMETRIC_MINISBLACK}, code);
    // review of PR-8: RGBA and a saved selection (5 samples) - «not supported, save as PNG», not «corrupted»
    WriteTiffWith(dir / "rgba5.tif", {6, 2, 5, 8, SAMPLEFORMAT_UINT, PHOTOMETRIC_RGB}, code);
    for (const char* name : {"palette.tif", "tiled.tif", "planar.tif", "bilevel.tif", "int32.tif", "rgba5.tif"}) {
        INFO(name);
        CHECK(KindOf([&] { ImportControlMask(dir / name, dir / "passes"); }) == ControlMaskIssue::Kind::Tiff);
    }
    ImportControlMask(dir / "white.tif", dir / "white");
    const PassImage w = Stored(dir / "white");
    for (uint32_t x = 0; x < 6; ++x) CHECK(w.Get(x, 0, 0) == static_cast<float>((255 - x * 50) * 257));  // what any viewer shows
    // review of PR-8: a grey document with transparency and a saved selection (3 samples) is grey - sample 0 the value, inverted for
    // min-is-white; sample 1 its alpha; read by count it was R, G, B of unrelated planes
    auto planes = [](uint32_t x, uint32_t, uint32_t c) { return c == 0 ? static_cast<double>(x * 50) : c == 1 ? 100.0 : 255.0 - x * 7; };
    for (const bool white : {false, true}) {
        INFO("min-is-white " << white);
        const std::string name = white ? "white3" : "black3";
        WriteTiffWith(dir / (name + ".tif"), {6, 2, 3, 8, SAMPLEFORMAT_UINT, white ? uint16_t{PHOTOMETRIC_MINISWHITE} : uint16_t{PHOTOMETRIC_MINISBLACK}}, planes);
        const ControlMaskImport g = ImportControlMask(dir / (name + ".tif"), dir / name);
        CHECK(g.mask.grayscale);
        CHECK(g.mask.alphaIgnored);  // sample 1 is not opaque
        CHECK(Manifest::Load(dir / name / kControlMaskPass).channels == std::vector<std::string>{"Y"});
        const PassImage s = Stored(dir / name);
        for (uint32_t x = 0; x < 6; ++x) CHECK(s.Get(x, 1, 0) == static_cast<float>((white ? 255 - x * 50 : x * 50) * 257));
    }
    // a TIFF named .png is read as the TIFF it is; text is no image
    fs::copy_file(dir / "white.tif", dir / "named.png");
    const ControlMaskImport n = ImportControlMask(dir / "named.png", dir / "named");
    CHECK(n.sourceFormat == "tiff");
    { std::ofstream(dir / "text.png") << "not an image"; }
    CHECK(KindOf([&] { ImportControlMask(dir / "text.png", dir / "passes"); }) == ControlMaskIssue::Kind::Unsupported);
    CHECK(KindOf([&] { ImportControlMask(dir / "missing.png", dir / "passes"); }) == ControlMaskIssue::Kind::NotAFile);
    {  // review of PR-8: held by another program without share-read (or denied by its ACL) - «не читается», not «формат не поддерживается»
        const HANDLE held = CreateFileW((dir / "white.tif").wstring().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(held != INVALID_HANDLE_VALUE);
        const ControlMaskIssue i = IssueOf([&] { ImportControlMask(dir / "white.tif", dir / "held"); });
        CloseHandle(held);
        CHECK(i.kind == ControlMaskIssue::Kind::Unreadable);
        CHECK(i.noAccess);  // the GUI says it does not open, not that it is corrupted
    }
    {  // review of PR-8: opens, but its first bytes do not read (another program's byte-range lock; an offline cloud placeholder)
        const HANDLE locker = CreateFileW((dir / "white.tif").wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(locker != INVALID_HANDLE_VALUE);
        OVERLAPPED at{};
        REQUIRE(LockFileEx(locker, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 4, 0, &at));
        const ControlMaskIssue i = IssueOf([&] { ImportControlMask(dir / "white.tif", dir / "locked"); });
        UnlockFileEx(locker, 0, 4, 0, &at);
        CloseHandle(locker);
        CHECK(i.kind == ControlMaskIssue::Kind::Unreadable);
        CHECK(i.noAccess);
        CHECK(!fs::exists(dir / "locked" / kControlMaskPass));
    }
    { std::ofstream(dir / "short.png") << "abc"; }  // shorter than any signature: no format, as before
    CHECK(KindOf([&] { ImportControlMask(dir / "short.png", dir / "passes"); }) == ControlMaskIssue::Kind::Unsupported);
    {  // a PNG signature with nothing readable after it: corrupted (its content), not a matter of access
        std::ofstream f(dir / "cut.png", std::ios::binary);
        f.write("\x89PNG\r\n\x1a\n", 8);
    }
    const ControlMaskIssue cut = IssueOf([&] { ImportControlMask(dir / "cut.png", dir / "passes"); });
    CHECK(cut.kind == ControlMaskIssue::Kind::Unreadable);
    CHECK_FALSE(cut.noAccess);
    CHECK(KindOf([&] { ImportControlMask(dir, dir / "passes"); }) == ControlMaskIssue::Kind::NotAFile);
    CHECK(KindOf([&] { ImportControlMask(dir / "white.tif", {}); }) == ControlMaskIssue::Kind::NoRoot);
}

TEST_CASE("ImportControlMask: the fingerprint is ContentFingerprint of the folder and follows the pixels; the same pixels are Unchanged and write nothing; other pixels retire the old mask into mask_control.v", "[unit][mask][fingerprint]") {
    const auto dir = Dir("same");
    const auto root = dir / "passes";
    auto value = [](uint32_t x, uint32_t y, size_t c) { return static_cast<float>((x * 20 + y * 3 + c) % 256); };
    WritePng(dir / "a8.png", Image(5, 5, PixelType::U8, {"R", "G", "B"}, value));
    WritePng(dir / "a16.png", Image(5, 5, PixelType::U16, {"R", "G", "B"}, [&](uint32_t x, uint32_t y, size_t c) { return value(x, y, c) * 257.f; }));
    WritePng(dir / "b.png", Image(5, 5, PixelType::U8, {"R", "G", "B"}, [&](uint32_t x, uint32_t y, size_t c) { return 255.f - value(x, y, c); }));
    const ControlMaskImport a = ImportControlMask(dir / "a8.png", root);
    const Manifest m = Manifest::Load(root / kControlMaskPass);
    CHECK(a.mask.fingerprint == ContentFingerprint(root / kControlMaskPass, m));
    const auto written = fs::last_write_time(root / kControlMaskPass / Manifest::kFileName);

    // the same pixels from another file (PNG16 of the same codes x257): nothing written, the fingerprint stays, NR is not recomputed
    const ControlMaskImport again = ImportControlMask(dir / "a16.png", root);
    CHECK(again.outcome == ControlMaskImport::Outcome::Unchanged);
    CHECK(again.mask.fingerprint == a.mask.fingerprint);
    CHECK(again.mask.sourceName == "a8.png");  // the loaded mask's name: nothing changed
    CHECK(fs::last_write_time(root / kControlMaskPass / Manifest::kFileName) == written);
    CHECK(!fs::exists(PassHistoryDir(root, kControlMaskPass)));

    // other pixels: another fingerprint, the old mask in the history
    const ControlMaskImport b = ImportControlMask(dir / "b.png", root);
    CHECK(b.outcome == ControlMaskImport::Outcome::Replaced);
    CHECK(b.mask.fingerprint != a.mask.fingerprint);
    REQUIRE(!b.retiredVersion.empty());
    const auto versions = ListPassVersions(root, kControlMaskPass);
    REQUIRE(versions.size() == 2);
    CHECK(versions[0].current);
    CHECK(versions[0].fingerprint == b.mask.fingerprint);
    CHECK(versions[1].id == b.retiredVersion);
    CHECK(versions[1].fingerprint == a.mask.fingerprint);
}

TEST_CASE("ImportControlMask: a file held at the retire fails as Locked and keeps the current mask without touching the history; held at the final rename it brings the previous mask back by its id; a held lock is Busy; old leftovers of .import are cleaned, a fresh foreign one is not", "[unit][mask][regression]") {
    const auto dir = Dir("locks");
    const auto root = dir / "passes";
    WritePng(dir / "a.png", Image(4, 4, PixelType::U8, {"A"}, [](uint32_t x, uint32_t, size_t) { return x * 60.f; }));
    WritePng(dir / "b.png", Image(4, 4, PixelType::U8, {"A"}, [](uint32_t x, uint32_t, size_t) { return 255.f - x * 60.f; }));
    const ControlMaskImport a = ImportControlMask(dir / "a.png", root);
    const Manifest m = Manifest::Load(root / kControlMaskPass);
    {
        std::ifstream held(root / kControlMaskPass / m.FrameFileName(0), std::ios::binary);  // a viewer holding the frame file
        REQUIRE(held);
        CHECK(KindOf([&] { ImportControlMask(dir / "b.png", root); }) == ControlMaskIssue::Kind::Locked);
    }
    CHECK(ReadControlMaskInfo(root / kControlMaskPass)->fingerprint == a.mask.fingerprint);
    CHECK(ListPassVersions(root, kControlMaskPass).size() == 1);  // the history untouched
    {  // review of PR-8: held for a moment only (the GUI reading the manifest, an antivirus) - retried, imported
        std::atomic<bool> opened{false};
        std::thread reader([&] {
            std::ifstream f(root / kControlMaskPass / m.FrameFileName(0), std::ios::binary);
            opened = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        });
        while (!opened) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const ControlMaskImport moment = ImportControlMask(dir / "b.png", root);
        reader.join();
        CHECK(moment.outcome == ControlMaskImport::Outcome::Replaced);
        UsePassVersion(root, kControlMaskPass, moment.retiredVersion);  // a back as current for the rest of the test
        REQUIRE(ReadControlMaskInfo(root / kControlMaskPass)->fingerprint == a.mask.fingerprint);
        for (const auto& v : ListPassVersions(root, kControlMaskPass))
            if (!v.current) std::filesystem::remove_all(v.dir);
    }

    // the final rename fails (a file of the new folder held): the previous mask comes back from the history by its id
    std::unique_ptr<std::ifstream> held;
    ControlMaskHooksForTests().beforePlace = [&](const fs::path& newFolder) {
        held = std::make_unique<std::ifstream>(newFolder / m.FrameFileName(0), std::ios::binary);
    };
    const ControlMaskIssue::Kind k = KindOf([&] { ImportControlMask(dir / "b.png", root); });
    ControlMaskHooksForTests().beforePlace = nullptr;
    held.reset();
    CHECK(k == ControlMaskIssue::Kind::Locked);
    REQUIRE(ReadControlMaskInfo(root / kControlMaskPass));
    CHECK(ReadControlMaskInfo(root / kControlMaskPass)->fingerprint == a.mask.fingerprint);
    CHECK(ListPassVersions(root, kControlMaskPass).size() == 1);  // brought back, not left in the history

    // review of PR-8: the current folder broken (deleted at the replace step) and the final rename failing - there is no mask now
    { std::ofstream(root / kControlMaskPass / Manifest::kFileName) << "{ broken"; }
    ControlMaskHooksForTests().beforePlace = [&](const fs::path& newFolder) {
        held = std::make_unique<std::ifstream>(newFolder / m.FrameFileName(0), std::ios::binary);
    };
    try {
        ImportControlMask(dir / "b.png", root);
        FAIL_CHECK("imported");
    } catch (const ControlMaskError& e) {
        CHECK(e.issue.kind == ControlMaskIssue::Kind::Locked);
        CHECK_FALSE(e.issue.previousKept);
    }
    ControlMaskHooksForTests().beforePlace = nullptr;
    held.reset();
    CHECK(!fs::exists(root / kControlMaskPass));
    REQUIRE(ImportControlMask(dir / "a.png", root).outcome == ControlMaskImport::Outcome::Imported);

    // a lock of another import is Busy; one older than 10 minutes is a crashed process's and is taken over
    fs::create_directories(root / ".import" / "lock");
    CHECK(KindOf([&] { ImportControlMask(dir / "b.png", root); }) == ControlMaskIssue::Kind::Busy);
    const auto old = fs::file_time_type::clock::now() - std::chrono::minutes(20);
    fs::last_write_time(root / ".import" / "lock", old);
    fs::create_directories(root / ".import" / "1-1" / kControlMaskPass);  // a crashed import's working folder
    fs::last_write_time(root / ".import" / "1-1", old);
    fs::create_directories(root / ".import" / "2-2");  // a working folder of an import running now (fresh)
    CHECK(ImportControlMask(dir / "b.png", root).outcome == ControlMaskImport::Outcome::Replaced);
    CHECK(!fs::exists(root / ".import" / "lock"));
    CHECK(!fs::exists(root / ".import" / "1-1"));
    CHECK(fs::exists(root / ".import" / "2-2"));
}

// Review of PR-8: a frame file held by another program (an editor, a sync client) was reported as edited in place - and `process` refused
// to reuse NR over a mask whose content had not changed.
TEST_CASE("ReadControlMaskInfo: a frame file another program holds open is not readable, not changed outside dlssvid; a missing one is", "[unit][mask][regression]") {
    const auto dir = Dir("held_frame");
    const auto root = dir / "passes";
    const auto cur = root / kControlMaskPass;
    WritePng(dir / "a.png", Image(4, 4, PixelType::U8, {"R", "G", "B"}, [](uint32_t x, uint32_t, size_t c) { return x * 60.f + c; }));
    ImportControlMask(dir / "a.png", root);  // it hashed the folder it wrote under another name: no cached hash of this frame yet
    const auto frame = cur / Manifest::Load(cur).FrameFileName(0);
    std::string why;
    {
        const HANDLE held = CreateFileW(frame.wstring().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(held != INVALID_HANDLE_VALUE);
        CHECK(!ReadControlMaskInfo(cur, &why));
        CloseHandle(held);
    }
    CHECK_THAT(why, Catch::Matchers::StartsWith("frame not readable"));
    CHECK(ReadControlMaskInfo(cur));  // let go: the mask as it was
    fs::remove(frame);
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK(why == "changed outside dlssvid");
}

TEST_CASE("RemoveControlMask retires the mask; a folder without a loadable manifest is deleted, not retired as _broken or _partial; ReadControlMaskInfo names why a folder is no control mask, a frame file edited in place reads as changed outside dlssvid and is not Unchanged on re-import", "[unit][mask]") {
    const auto dir = Dir("remove");
    const auto root = dir / "passes";
    const auto cur = root / kControlMaskPass;
    WritePng(dir / "a.png", Image(4, 4, PixelType::U8, {"R", "G", "B"}, [](uint32_t x, uint32_t, size_t c) { return x * 60.f + c; }));
    std::string why;
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK(why == "absent");
    CHECK(!RemoveControlMask(root));  // nothing to remove

    const ControlMaskImport a = ImportControlMask(dir / "a.png", root);
    const auto id = RemoveControlMask(root);
    REQUIRE(id);
    CHECK(!fs::exists(cur));
    const auto history = ListPassVersions(root, kControlMaskPass);
    REQUIRE(history.size() == 1);
    CHECK(history[0].id == *id);
    CHECK(history[0].fingerprint == a.mask.fingerprint);
    UsePassVersion(root, kControlMaskPass, *id);  // «Вернуть»
    CHECK(ReadControlMaskInfo(cur)->fingerprint == a.mask.fingerprint);

    // why a folder is no control mask
    auto edit = [&](const std::function<void(Manifest&)>& f) {
        Manifest m = Manifest::Load(cur);
        f(m);
        m.Save(cur);
    };
    const Manifest good = Manifest::Load(cur);
    edit([](Manifest& m) { m.stageParams["static"] = false; });
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK(why == "not static one-frame");
    good.Save(cur);
    edit([](Manifest& m) { m.channels = {"R", "G"}; });
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK_THAT(why, Catch::Matchers::StartsWith("channels"));
    good.Save(cur);
    edit([](Manifest& m) { m.fingerprint.clear(); });
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK(why == "not stamped");
    good.Save(cur);
    REQUIRE(ReadControlMaskInfo(cur));
    // review of PR-8: values of another type (a hand-made or script-written manifest) - a reason or an absent value, not a JSON exception
    for (const nlohmann::json& v : {nlohmann::json(1), nlohmann::json("true"), nlohmann::json(nullptr)}) {
        INFO(v.dump());
        edit([&](Manifest& m) { m.stageParams["static"] = v; });
        CHECK_NOTHROW(ReadControlMaskInfo(cur, &why));
        CHECK(!ReadControlMaskInfo(cur, &why));
        CHECK(why == "not static one-frame");
        CHECK_NOTHROW(ImportControlMask(dir / "a.png", root));  // loading the image again repairs it
        CHECK(ReadControlMaskInfo(cur));
    }
    edit([](Manifest& m) {  // outside the fingerprint: the stamp holds, the values read as absent
        m.stageParams["source_name"] = nullptr;
        m.stageParams["pixels_sha256"] = 7;
        m.stageParams["bands_differ"] = "yes";
        m.stageParams["alpha_ignored"] = 1;
        m.stageParams["clamped"] = "many";
    });
    std::optional<ControlMaskInfo> odd;
    CHECK_NOTHROW(odd = ReadControlMaskInfo(cur, &why));
    REQUIRE(odd);
    CHECK(odd->sourceName.empty());
    CHECK(odd->pixelsSha256.empty());
    CHECK_FALSE(odd->bandsDiffer);
    CHECK_FALSE(odd->alphaIgnored);
    CHECK(odd->clamped == 0);
    good.Save(cur);
    REQUIRE(ReadControlMaskInfo(cur));

    // the frame file edited in place (a graphics editor in passes/mask_control): no mask until loaded again; the same source then writes
    WritePng(cur / good.FrameFileName(0), Image(4, 4, PixelType::U16, {"R", "G", "B"}, [](uint32_t, uint32_t, size_t) { return 1000.f; }));
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK(why == "changed outside dlssvid");
    const ControlMaskImport again = ImportControlMask(dir / "a.png", root);
    CHECK(again.outcome == ControlMaskImport::Outcome::Replaced);
    CHECK(again.mask.fingerprint == a.mask.fingerprint);
    CHECK(ReadControlMaskInfo(cur)->fingerprint == a.mask.fingerprint);

    // a folder without a loadable manifest: deleted (the history would hide it as _broken / _partial)
    { std::ofstream(cur / Manifest::kFileName) << "{ broken"; }
    CHECK(!ReadControlMaskInfo(cur, &why));
    CHECK_THAT(why, Catch::Matchers::StartsWith("unreadable"));
    CHECK(ControlMaskFolderBroken(root));
    const size_t before = ListPassVersions(root, kControlMaskPass).size();
    CHECK(!RemoveControlMask(root));
    CHECK(!fs::exists(cur));
    CHECK(ListPassVersions(root, kControlMaskPass).size() == before);
    for (const auto& e : fs::directory_iterator(PassHistoryDir(root, kControlMaskPass))) {
        const std::string name = e.path().filename().string();
        CHECK(name.find("_broken") == std::string::npos);
        CHECK(name.find("_partial") == std::string::npos);
    }
    // and replaced by an import the same way
    fs::create_directories(cur);
    { std::ofstream(cur / "stray.png") << "x"; }
    CHECK(ImportControlMask(dir / "a.png", root).outcome == ControlMaskImport::Outcome::Replaced);
    CHECK(!fs::exists(cur / "stray.png"));
}

TEST_CASE("LoadControlMaskForFrame: the frame size is a copy, larger is area-averaged, smaller is bilinear with half-pixel centres, each axis on its own (wider and shorter than the frame); white stays 65535; grey stays one channel; split only where G differs from B; another fingerprint than expected is refused", "[unit][mask]") {
    const auto dir = Dir("texels");
    auto load = [&](const char* name, const PassImage& img, uint32_t w, uint32_t h) {
        WritePng(dir / (std::string(name) + ".png"), img);
        const auto root = dir / name;
        const ControlMaskImport r = ImportControlMask(dir / (std::string(name) + ".png"), root);
        return LoadControlMaskForFrame(root / kControlMaskPass, w, h, r.mask.fingerprint);
    };
    // the same size: a copy; RGB with A = 65535
    const PassImage rgb = Image(3, 2, PixelType::U16, {"R", "G", "B"}, [](uint32_t x, uint32_t y, size_t c) { return static_cast<float>(1000 * (x + 3 * y) + c); });
    const ControlMaskTexels same = load("same", rgb, 3, 2);
    REQUIRE(same.rgb);
    REQUIRE(same.data.size() == 3 * 2 * 4);
    for (uint32_t i = 0; i < 6; ++i) {
        CHECK(same.data[i * 4] == 1000 * i);
        CHECK(same.data[i * 4 + 1] == 1000 * i + 1);
        CHECK(same.data[i * 4 + 3] == 65535);
    }
    CHECK(same.split);  // G = B + ... differs: 1000i + 1 vs 1000i + 2

    // a grey 4x2 -> 2x1: area means of 2x2 blocks; 2x1 -> 4x1: bilinear at (i + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, 1.25 (clamped)
    const ControlMaskTexels down = load("down", Image(4, 2, PixelType::U16, {"A"}, [](uint32_t x, uint32_t y, size_t) { return static_cast<float>(1000 * x + 400 * y); }), 2, 1);
    REQUIRE(!down.rgb);
    REQUIRE(down.data.size() == 2);
    CHECK(down.data[0] == 700);   // (0 + 1000 + 400 + 1400) / 4
    CHECK(down.data[1] == 2700);  // (2000 + 3000 + 2400 + 3400) / 4
    const ControlMaskTexels up = load("up", Image(2, 1, PixelType::U16, {"A"}, [](uint32_t x, uint32_t, size_t) { return x ? 4000.f : 0.f; }), 4, 1);
    REQUIRE(up.data.size() == 4);
    CHECK(up.data[0] == 0);
    CHECK(up.data[1] == 1000);
    CHECK(up.data[2] == 3000);
    CHECK(up.data[3] == 4000);
    // wider and shorter than the frame: x shrinks (4 -> 2, means), y grows (1 -> 2, bilinear of one row = the row)
    const ControlMaskTexels mixed = load("mixed", Image(4, 1, PixelType::U16, {"A"}, [](uint32_t x, uint32_t, size_t) { return 1000.f * x; }), 2, 2);
    REQUIRE(mixed.data.size() == 4);
    CHECK(mixed.data[0] == 500);
    CHECK(mixed.data[1] == 2500);
    CHECK(mixed.data[2] == 500);
    CHECK(mixed.data[3] == 2500);
    // white stays white at any size
    const ControlMaskTexels white = load("white", Image(7, 5, PixelType::U8, {"R", "G", "B"}, [](uint32_t, uint32_t, size_t) { return 255.f; }), 3, 11);
    CHECK(!white.rgb);  // R = G = B: stored grey
    for (uint16_t v : white.data) CHECK(v == 65535);
    // G = B everywhere: no split
    const ControlMaskTexels even = load("even", Image(4, 4, PixelType::U16, {"R", "G", "B"}, [](uint32_t x, uint32_t, size_t c) { return c ? 3000.f : 1000.f * x; }), 2, 2);
    CHECK(even.rgb);
    CHECK(!even.split);
    // the folder changed after Init read its fingerprint
    const auto root = dir / "even";
    CHECK_THROWS_WITH(LoadControlMaskForFrame(root / kControlMaskPass, 2, 2, "sha256:other"), Catch::Matchers::ContainsSubstring("changed while nr started"));
}
