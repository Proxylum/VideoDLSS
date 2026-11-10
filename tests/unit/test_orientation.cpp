// TASK-0072: a source shown turned (a phone video kept landscape with a display matrix, or the old 'rotate' tag) is decoded as players
// show it - the orientation as FFmpeg's command line reads it, the samples permuted exactly, the passes of such a source labelled.

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/display.h>
}

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "pipeline/Estimates.h"
#include "pipeline/Frame.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/SettingsNote.h"
#include "stages/upscale/IUpscaler.h"

using namespace dlssvid;

namespace {

const std::vector<DisplayOrientation> kAll = {{0, false}, {90, false}, {180, false}, {270, false}, {0, true}, {90, true}, {180, true}, {270, true}};

// The stored sample shown at (x, y) of a stored w x h plane - FFmpeg's filters, written out case by case (transpose clock / cclock /
// cclock_flip / clock_flip, hflip, vflip)
std::pair<uint32_t, uint32_t> StoredOf(const std::string& token, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    if (token == "r90") return {y, h - 1 - x};
    if (token == "r180") return {w - 1 - x, h - 1 - y};
    if (token == "r270") return {w - 1 - y, x};
    if (token == "h") return {w - 1 - x, y};
    if (token == "r90h") return {y, x};
    if (token == "r180h") return {x, h - 1 - y};
    if (token == "r270h") return {w - 1 - y, h - 1 - x};
    return {x, y};
}

uint32_t Sample(const CpuFrame& f, size_t plane, size_t col, size_t row) {
    const size_t i = row * f.desc.PlaneCols(plane) + col;
    return f.desc.HighBitDepth() ? f.Plane16(plane)[i] : f.Plane(plane)[i];
}

// A frame whose every sample is distinct (within the depth): a permutation that goes wrong shows
CpuFrame Distinct(const FrameDesc& d) {
    CpuFrame f;
    f.Allocate(d);
    const uint32_t mask = d.HighBitDepth() ? (1u << d.bitDepth) - 1u : 255u;
    uint32_t v = 1;
    for (size_t p = 0; p < 3; ++p)
        for (size_t i = 0; i < d.PlaneCols(p) * d.PlaneHeight(p); ++i, v = v * 37 + 11) {
            if (d.HighBitDepth()) f.Plane16(p)[i] = static_cast<uint16_t>(v & mask);
            else f.Plane(p)[i] = static_cast<uint8_t>(v & mask);
        }
    return f;
}

// What the shown frame must hold: each plane turned on its own grid where the chroma grid maps onto itself (an exact permutation of the
// stored chroma), else - 4:4:4 - the stored chroma under each shown luma position repeated
void CheckOriented(const CpuFrame& in, DisplayOrientation o, const CpuFrame& out) {
    const std::string t = o.Token();
    const bool swap = o.SwapsAxes();
    REQUIRE(out.desc.width == (swap ? in.desc.height : in.desc.width));
    REQUIRE(out.desc.height == (swap ? in.desc.width : in.desc.height));
    REQUIRE(out.desc.bitDepth == in.desc.bitDepth);
    const bool exact = ChromaName(out.desc.format) == ChromaName(in.desc.format);
    for (size_t p = 0; p < 3; ++p) {
        INFO("plane " << p << " " << t);
        const uint32_t sw = static_cast<uint32_t>(in.desc.PlaneCols(p)), sh = static_cast<uint32_t>(in.desc.PlaneHeight(p));
        bool same = true;
        for (size_t y = 0; y < out.desc.PlaneHeight(p) && same; ++y)
            for (size_t x = 0; x < out.desc.PlaneCols(p) && same; ++x) {
                uint32_t want;
                if (p == 0 || exact) {
                    const auto [sx, sy] = StoredOf(t, static_cast<uint32_t>(x), static_cast<uint32_t>(y), sw, sh);
                    want = Sample(in, p, sx, sy);
                } else {
                    const auto [lx, ly] = StoredOf(t, static_cast<uint32_t>(x), static_cast<uint32_t>(y), in.desc.width, in.desc.height);
                    want = Sample(in, p, lx >> ChromaShiftX(in.desc.format), ly >> ChromaShiftY(in.desc.format));
                }
                same = Sample(out, p, x, y) == want;
                if (!same) INFO("at " << x << "," << y);
            }
        CHECK(same);
    }
}

}  // namespace

TEST_CASE("DisplayOrientation: tokens, their inverse, descriptions", "[orientation]") {
    std::vector<std::string> tokens;
    for (const auto& o : kAll) tokens.push_back(o.Token());
    CHECK(tokens == std::vector<std::string>{"", "r90", "r180", "r270", "h", "r90h", "r180h", "r270h"});
    for (const auto& o : kAll) {
        INFO(o.Token());
        REQUIRE(OrientationFromToken(o.Token()).has_value());
        CHECK(*OrientationFromToken(o.Token()) == o);
        // turned and turned back: the frame as it was
        const CpuFrame f = Distinct(FrameDesc{6, 4, PixelFormat::Yuv420p});
        CpuFrame there, back;
        OrientFrame(f, o, there);
        OrientFrame(there, o.Inverse(), back);
        CHECK(back.desc == f.desc);
        CHECK(back.data == f.data);
    }
    CHECK_FALSE(OrientationFromToken("r45").has_value());
    CHECK_FALSE(OrientationFromToken("r90x").has_value());
    CHECK(DisplayOrientation{}.Describe().empty());
    CHECK(DisplayOrientation{90, false}.Describe() == "rotated 90 deg clockwise");
    CHECK(DisplayOrientation{180, true}.Describe() == "rotated 180 deg and mirrored");
    CHECK(DisplayOrientation{0, true}.Describe() == "mirrored");
}

TEST_CASE("OrientationFromDisplayMatrix reads a matrix as FFmpeg's command line applies it", "[orientation]") {
    int32_t m[9];
    for (const int deg : {0, 90, 180, 270, -90, 450})
        for (const bool h : {false, true}) {
            av_display_rotation_set(m, deg);  // clockwise (FFmpeg 7.1)
            if (h) av_display_matrix_flip(m, 1, 0);
            INFO(deg << (h ? " hflip" : ""));
            std::string why;
            CHECK(OrientationFromDisplayMatrix(m, &why) == DisplayOrientation{static_cast<uint16_t>(((deg % 360) + 360) % 360), h});
            CHECK(why.empty());
        }
    av_display_rotation_set(m, 0);
    av_display_matrix_flip(m, 0, 1);  // vflip alone: a mirror turned by 180
    CHECK(OrientationFromDisplayMatrix(m) == DisplayOrientation{180, true});
    // what a phone writes for a portrait video (ffprobe: "rotation of -90.00 degrees"): shown turned clockwise
    const int32_t phone[9] = {0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30};
    CHECK(OrientationFromDisplayMatrix(phone) == DisplayOrientation{90, false});
    // the scale and the translation of the matrix do not matter
    int32_t scaled[9];
    av_display_rotation_set(scaled, 90);
    for (const int i : {0, 1, 3, 4}) scaled[i] *= 2;
    scaled[6] = 100 << 16;
    CHECK(OrientationFromDisplayMatrix(scaled) == DisplayOrientation{90, false});
    // not a multiple of 90, degenerate: not applied, said why
    std::string why;
    av_display_rotation_set(m, 45);
    CHECK(OrientationFromDisplayMatrix(m, &why).Identity());
    CHECK(why.find("not a multiple of 90") != std::string::npos);
    const int32_t zero[9] = {};
    why.clear();
    CHECK(OrientationFromDisplayMatrix(zero, &why).Identity());
    CHECK(why == "a degenerate display matrix");
}

TEST_CASE("OrientationFromRotateTag reads the old rotate tag: degrees clockwise", "[orientation]") {
    CHECK(OrientationFromRotateTag("90") == DisplayOrientation{90, false});
    CHECK(OrientationFromRotateTag("-90") == DisplayOrientation{270, false});
    CHECK(OrientationFromRotateTag("270") == DisplayOrientation{270, false});
    CHECK(OrientationFromRotateTag("450") == DisplayOrientation{90, false});
    std::string why;
    CHECK(OrientationFromRotateTag("0", &why).Identity());
    CHECK(why.empty());
    for (const char* bad : {"45", "x", "", "90deg"}) {
        INFO(bad);
        why.clear();
        CHECK(OrientationFromRotateTag(bad, &why).Identity());
        CHECK_FALSE(why.empty());
    }
}

TEST_CASE("OrientFrame turns a 4x2 frame as players show it, by hand", "[orientation][frame]") {
    CpuFrame f;
    f.Allocate(FrameDesc{4, 2, PixelFormat::Yuv444p});
    for (uint8_t i = 0; i < 8; ++i) f.Plane(0)[i] = i;  // rows 0 1 2 3 / 4 5 6 7
    f.index = 5;
    f.pts = 77;
    auto rows = [](DisplayOrientation o, const CpuFrame& in) {
        CpuFrame out;
        OrientFrame(in, o, out);
        std::vector<std::vector<int>> r(out.desc.height);
        for (uint32_t y = 0; y < out.desc.height; ++y)
            for (uint32_t x = 0; x < out.desc.width; ++x) r[y].push_back(out.Plane(0)[y * out.desc.width + x]);
        CHECK(out.index == 5);
        CHECK(out.pts == 77);
        return r;
    };
    using R = std::vector<std::vector<int>>;
    CHECK(rows({90, false}, f) == R{{4, 0}, {5, 1}, {6, 2}, {7, 3}});    // clockwise: the left column, bottom up, is the top row
    CHECK(rows({270, false}, f) == R{{3, 7}, {2, 6}, {1, 5}, {0, 4}});   // counterclockwise
    CHECK(rows({180, false}, f) == R{{7, 6, 5, 4}, {3, 2, 1, 0}});
    CHECK(rows({0, true}, f) == R{{3, 2, 1, 0}, {7, 6, 5, 4}});          // mirrored left-right
    CHECK(rows({180, true}, f) == R{{4, 5, 6, 7}, {0, 1, 2, 3}});        // upside down
    CHECK(rows({90, true}, f) == R{{0, 4}, {1, 5}, {2, 6}, {3, 7}});     // transposed
    CHECK(rows({270, true}, f) == R{{7, 3}, {6, 2}, {5, 1}, {4, 0}});
    CHECK(rows({}, f) == R{{0, 1, 2, 3}, {4, 5, 6, 7}});
}

TEST_CASE("OrientFrame permutes the samples exactly in every layout; 4:4:4 where the chroma grid does not map onto itself", "[orientation][frame]") {
    struct Case {
        FrameDesc in;
        std::vector<std::pair<std::string, PixelFormat>> formats;  // token -> the shown layout
    };
    const std::vector<Case> cases = {
        {FrameDesc{6, 4, PixelFormat::Yuv420p, 8}, {}},  // even sides: 4:2:0 stays for every orientation
        {FrameDesc{6, 4, PixelFormat::Yuv420p16, 10}, {}},
        {FrameDesc{5, 3, PixelFormat::Yuv444p, 8}, {}},
        {FrameDesc{6, 4, PixelFormat::Yuv422p16, 10}, {{"r90", PixelFormat::Yuv444p16}, {"r270", PixelFormat::Yuv444p16}, {"r90h", PixelFormat::Yuv444p16},
                                                      {"r270h", PixelFormat::Yuv444p16}}},  // 4:4:0 is not a format: turned 4:2:2 is 4:4:4
        // an odd width: reversed by 180, h, 270 (4:4:4); 90 and the transpose keep the stored x order (4:2:0 stays)
        {FrameDesc{5, 4, PixelFormat::Yuv420p, 8}, {{"r180", PixelFormat::Yuv444p}, {"h", PixelFormat::Yuv444p}, {"r270", PixelFormat::Yuv444p},
                                                    {"r270h", PixelFormat::Yuv444p}}},
        // an odd height: reversed by 90, 180, 180h, 270h
        {FrameDesc{4, 5, PixelFormat::Yuv420p, 8}, {{"r90", PixelFormat::Yuv444p}, {"r180", PixelFormat::Yuv444p}, {"r180h", PixelFormat::Yuv444p},
                                                    {"r270h", PixelFormat::Yuv444p}}},
    };
    for (const Case& c : cases)
        for (const auto& o : kAll) {
            INFO(c.in.Name() << " " << c.in.width << "x" << c.in.height << " " << o.Token());
            PixelFormat want = c.in.format;
            for (const auto& [t, f] : c.formats)
                if (t == o.Token()) want = f;
            const FrameDesc shown = OrientedDesc(c.in, o);
            CHECK(shown.format == want);
            const CpuFrame f = Distinct(c.in);
            CpuFrame out;
            OrientFrame(f, o, out);
            CHECK(out.desc == shown);
            CheckOriented(f, o, out);
        }
}

TEST_CASE("SourceFormatLabel: a turned source's passes carry the turn, an unturned one keeps its label and fingerprints", "[orientation][fingerprint]") {
    const FrameDesc d420{1920, 1080, PixelFormat::Yuv420p, 8}, d10{1920, 1080, PixelFormat::Yuv420p16, 10};
    CHECK(SourceFormatLabel(d420, {}) == "yuv420p");
    CHECK(SourceFormatLabel(d10, {}) == "yuv420p10");
    CHECK(SourceFormatLabel(d420, {90, false}) == "yuv420p+r90");
    CHECK(SourceFormatLabel(d10, {270, true}) == "yuv420p10+r270h");
    CHECK(OrientationOfSourceFormat("yuv420p10+r270h") == DisplayOrientation{270, true});
    CHECK(OrientationOfSourceFormat("yuv420p+r180") == DisplayOrientation{180, false});
    CHECK(OrientationOfSourceFormat("yuv420p").Identity());
    CHECK(OrientationOfSourceFormat("").Identity());
    CHECK(OrientationOfSourceFormat("yuv420p+r45").Identity());
    const nlohmann::json p = {{"scale", 2.0}};
    const std::string plain = ComputeFingerprint("upscale", "sha256:00", p, {}, {}, SourceFormatLabel(d420, {})).value;
    CHECK(plain == ComputeFingerprint("upscale", "sha256:00", p, {}, {}).value);  // as before: no source format in the hash for 8-bit 4:2:0
    for (const auto& o : kAll) {
        if (o.Identity()) continue;
        INFO(o.Token());
        CHECK(ComputeFingerprint("upscale", "sha256:00", p, {}, {}, SourceFormatLabel(d420, o)).value != plain);
    }
}

TEST_CASE("ColorInfoFromStream guesses an untagged matrix by the stored height: a turned 960x540 stays BT.601", "[orientation][color]") {
    VideoStreamInfo info;
    info.width = 540;
    info.height = 960;
    info.codedWidth = 960;
    info.codedHeight = 540;
    CHECK(ColorInfoFromStream(info).matrix == ColorMatrix::Bt601);
    info.codedWidth = info.codedHeight = 0;  // a VideoStreamInfo filled by hand: by the height as before
    CHECK(ColorInfoFromStream(info).matrix == ColorMatrix::Bt709);
    info.colorSpace = 6;
    info.codedHeight = 1080;
    CHECK(ColorInfoFromStream(info).matrix == ColorMatrix::Bt601);  // a tag wins
}

TEST_CASE("ResolveUpscaleTarget caps a portrait output by its long side: 1080x1920 x2 is 2160x3840", "[orientation][upscale]") {
    UpscaleTarget t = ResolveUpscaleTarget(1080, 1920, 2.0);
    CHECK(t.width == 2160);
    CHECK(t.height == 3840);
    CHECK_FALSE(t.capped);
    t = ResolveUpscaleTarget(1080, 1920, 2.0, 0, 0, 3840, 2160, false);  // by the axes, as before
    CHECK(t.width == 1216);
    CHECK(t.height == 2160);
    CHECK(t.capped);
    t = ResolveUpscaleTarget(1080, 1920, 3.0);
    CHECK(t.width == 2160);
    CHECK(t.height == 3840);
    CHECK(t.capped);
    t = ResolveUpscaleTarget(1080, 1920, 0.0, 0, 3840);  // an explicit portrait target
    CHECK(t.width == 2160);
    CHECK(t.height == 3840);
    CHECK_FALSE(t.capped);
    for (const bool longSide : {true, false}) {  // landscape: as it was
        INFO(longSide);
        t = ResolveUpscaleTarget(1920, 1080, 3.0, 0, 0, 3840, 2160, longSide);
        CHECK(t.width == 3840);
        CHECK(t.height == 2160);
        CHECK(t.capped);
        t = ResolveUpscaleTarget(1920, 800, 3.0, 0, 0, 3840, 2160, longSide);
        CHECK(t.width == 3840);
        CHECK(t.height == 1600);
    }
    const nlohmann::json p = {{"scale", 2.0}};
    CHECK(UpscaleOutputFor(1080, 1920, p).height == 3840);
    CHECK(UpscaleOutputFor(1080, 1920, p, false).height == 2160);
}

TEST_CASE("Settings note: a turned source says so first - source: rotate=90; an unturned one as before", "[orientation][encode]") {
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "settings_note" / "turned";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    auto save = [&](const std::string& format) {
        Manifest m = Manifest::ForPass(PassKind::ColorSr, 2160, 3840, FileFormat::Exr);
        m.stageParams = {{"backend", "nis"}, {"scale", 2.0}};
        m.sourceFormat = format;
        m.Save(root / "color_sr");
    };
    save("yuv420p+r90");
    CHECK(BuildSettingsNote(root, "color_sr") == std::string("VideoDLSS ") + DLSSVID_VERSION + " | source: rotate=90 | upscale: nis x2");
    CHECK(BuildSettingsJson(root, "color_sr")["source"] == nlohmann::json{{"rotate", 90}});
    save("yuv420p10+r270h");
    CHECK(BuildSettingsNote(root, "color_sr").find("| source: rotate=270 hflip |") != std::string::npos);
    save("yuv420p+h");
    CHECK(BuildSettingsNote(root, "color_sr").find("| source: hflip |") != std::string::npos);
    save("yuv420p");
    CHECK(BuildSettingsNote(root, "color_sr") == std::string("VideoDLSS ") + DLSSVID_VERSION + " | upscale: nis x2");
    CHECK_FALSE(BuildSettingsJson(root, "color_sr").contains("source"));
}
