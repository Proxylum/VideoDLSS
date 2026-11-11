// TASK-0073: the proportions of a pixel (PixelAspect), the upscale target of the frame as shown, the estimate, DLSS-G's camera, the resize
// `process` stretches an anamorphic source's result with, the settings note.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "convert/YuvConvert.h"
#include "passes/Manifest.h"
#include "pipeline/Estimates.h"
#include "pipeline/Frame.h"
#include "pipeline/PassVersions.h"
#include "pipeline/SettingsNote.h"
#include "stages/ParamSchema.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/upscale/IUpscaler.h"

using namespace dlssvid;
using Catch::Approx;

TEST_CASE("PixelAspect: reduced, plausible or square; the frame as shown is never narrowed", "[unit][aspect]") {
    CHECK(PixelAspect{}.Square());
    CHECK(PixelAspect{7, 7}.Square());
    CHECK(PixelAspect{} == PixelAspect{5, 5});
    CHECK(!(PixelAspect{32, 27} == PixelAspect{}));
    CHECK(PixelAspect{32, 27}.Token() == "32:27");
    CHECK(PixelAspect{}.Token().empty());
    CHECK(PixelAspect{32, 27}.Inverse() == PixelAspect{27, 32});
    CHECK(PixelAspect{}.Inverse().Square());
    CHECK(PixelAspect{32, 27}.Value() == Approx(32.0 / 27.0));
    CHECK(PixelAspect{}.Value() == 1.0);

    CHECK(PixelAspectFrom(64, 54) == PixelAspect{32, 27});  // reduced
    CHECK(PixelAspectFrom(0, 1).Square());
    CHECK(PixelAspectFrom(-4, 3).Square());
    CHECK(PixelAspectFrom(4, 0).Square());
    CHECK(PixelAspectFrom(40, 40).Square());
    std::string why;
    CHECK(PixelAspectFrom(4, 1, &why) == PixelAspect{4, 1});  // the edge of plausible
    CHECK(why.empty());
    CHECK(PixelAspectFrom(1, 4, &why) == PixelAspect{1, 4});
    CHECK(PixelAspectFrom(1000, 1, &why).Square());  // a broken file
    CHECK(why == "a pixel aspect of 1000:1 is not plausible: the pixels are taken as square");
    why.clear();
    CHECK(PixelAspectFrom(1, 5, &why).Square());
    CHECK(!why.empty());
    CHECK(PixelAspectFrom(65537, 65536).Square());  // sides above 16 bits

    // wider pixels widen the frame, narrower ones make it higher; even sides
    CHECK(PixelAspect{32, 27}.DisplaySize(720, 480) == std::pair<uint32_t, uint32_t>{854, 480});
    CHECK(PixelAspect{8, 9}.DisplaySize(720, 480) == std::pair<uint32_t, uint32_t>{720, 540});
    CHECK(PixelAspect{10, 11}.DisplaySize(720, 480) == std::pair<uint32_t, uint32_t>{720, 528});
    CHECK(PixelAspect{4, 3}.DisplaySize(1440, 1080) == std::pair<uint32_t, uint32_t>{1920, 1080});
    CHECK(PixelAspect{64, 45}.DisplaySize(720, 576) == std::pair<uint32_t, uint32_t>{1024, 576});
    CHECK(PixelAspect{32, 27}.DisplaySize(54, 36) == std::pair<uint32_t, uint32_t>{64, 36});
    CHECK(PixelAspect{}.DisplaySize(721, 481) == std::pair<uint32_t, uint32_t>{721, 481});  // square: as is, odd too
    // a pixel aspect that moves no pixel of the frame (review of TASK-0073): half a pixel or more moves one
    CHECK(PixelAspect{32, 27}.MovesAPixel(720, 480));
    CHECK(PixelAspect{8, 9}.MovesAPixel(720, 480));
    CHECK(!PixelAspect{14400, 14399}.MovesAPixel(1694, 720));  // 0.12 px
    CHECK(PixelAspect{14400, 14399}.MovesAPixel(7200, 720));   // 0.5 px
    CHECK(!PixelAspect{10000, 10001}.MovesAPixel(720, 720));   // narrower: 0.07 px of the height
    CHECK(PixelAspect{9000, 9001}.MovesAPixel(720, 4510));     // 0.501 px of the height
    CHECK(!PixelAspect{}.MovesAPixel(720, 480));
    CHECK(PixelAspect{32, 27}.DisplaySize(0, 0) == std::pair<uint32_t, uint32_t>{0, 0});  // a size not known yet
}

TEST_CASE("PixelAspect stamps of colour passes and the pixels of a pass read back", "[unit][aspect]") {
    CHECK(PixelAspect{}.Stamp().empty());
    CHECK(PixelAspect{1, 1}.Stamp() == "1:1");
    CHECK(PixelAspect{32, 27}.Stamp() == "32:27");
    CHECK(PixelAspectFromStamp("32:27") == PixelAspect{32, 27});
    CHECK(PixelAspectFromStamp("64:54") == PixelAspect{32, 27});
    REQUIRE(PixelAspectFromStamp("1:1"));
    CHECK(PixelAspectFromStamp("1:1")->num == 1);  // known square, not unknown
    for (const char* bad : {"", "32", "32:", ":27", "a:b", "32:27x", "-4:3", "0:1", "1000:1", " 32:27"}) {
        INFO(bad);
        CHECK(!PixelAspectFromStamp(bad));
    }
    const PixelAspect dvd{32, 27};
    Manifest m = Manifest::ForPass(PassKind::ColorSr, 108, 72, FileFormat::Exr);
    CHECK(PassPixelAspect(m, {}, 54, 36).num == 0);   // a square source: nothing
    CHECK(PassPixelAspect(m, dvd, 54, 36) == dvd);    // the grid
    m.width = 128;                                     // the frame as shown, no record: an explicit target of the build before
    CHECK(PassPixelAspect(m, dvd, 54, 36).num == 1);
    CHECK(PassPixelAspect(m, dvd, 54, 36).Square());
    m.width = 144;                                     // 2:1, another target of both sides then: square too (review of TASK-0073)
    CHECK(PassPixelAspect(m, dvd, 54, 36).num == 1);
    CHECK(PassPixelAspect(m, PixelAspect{8, 9}, 72, 48).num == 1);  // NTSC 4:3 with a 16:9 target: neither the grid's nor as shown
    m.width = 128;
    m.stageParams["pixel_aspect"] = "32:27";          // the record wins over the guess
    CHECK(PassPixelAspect(m, dvd, 54, 36) == dvd);
    m.width = 108;
    m.stageParams["pixel_aspect"] = "1:1";
    CHECK(PassPixelAspect(m, dvd, 54, 36).num == 1);
    m.stageParams["pixel_aspect"] = 5;                // not a text: the guess
    CHECK(PassPixelAspect(m, dvd, 54, 36) == dvd);
    CHECK(StageParamText(m, "pixel_aspect").empty());
    CHECK(ColourPixelAspect({}, dvd, 54, 36) == dvd);  // the source's frames
    CHECK(ColourPixelAspect(std::filesystem::path(DLSSVID_TEST_TMP) / "aspect_nowhere", dvd, 54, 36) == dvd);
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "aspect_colour" / "color_sr";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    m.stageParams["pixel_aspect"] = "1:1";
    m.Save(dir);
    CHECK(ColourPixelAspect(dir, dvd, 54, 36).num == 1);
    CHECK(ColourPixelAspect(dir, {}, 54, 36).num == 0);
}

TEST_CASE("PassResultSize: the frame as shown a pass records, else the pass stretched", "[unit][aspect]") {
    using Size = std::pair<uint32_t, uint32_t>;
    const PixelAspect dvd{32, 27};
    Manifest m = Manifest::ForPass(PassKind::ColorSr, 82, 54, FileFormat::Exr);
    m.stageParams = {{"pixel_aspect", "32:27"}};
    CHECK(!RecordedShownSize(m));
    CHECK(PassResultSize(m, dvd, 54, 36) == Size{98, 54});  // 82 x 32/27 = 97.2
    m.stageParams["shown_size"] = {96, 54};
    REQUIRE(RecordedShownSize(m));
    CHECK(*RecordedShownSize(m) == Size{96, 54});
    CHECK(PassResultSize(m, dvd, 54, 36) == Size{96, 54});  // the target
    m.stageParams["shown_size"] = {200, 54};                // of another frame: not taken
    CHECK(PassResultSize(m, dvd, 54, 36) == Size{98, 54});
    for (const nlohmann::json bad : {nlohmann::json{96}, nlohmann::json{96, -54}, nlohmann::json{"96", 54}, nlohmann::json{0, 54}, nlohmann::json{96, 54, 1},
                                     nlohmann::json("96x54")}) {
        INFO(bad.dump());
        m.stageParams["shown_size"] = bad;
        CHECK(!RecordedShownSize(m));
    }
    m.stageParams["shown_size"] = {96, 54};
    CHECK(PassResultSize(m, {}, 54, 36) == Size{82, 54});  // a square source: the pass as it is
    m.stageParams["pixel_aspect"] = "1:1";
    CHECK(PassResultSize(m, dvd, 54, 36) == Size{82, 54});  // square pixels: as it is
    m.stageParams["shown_size"] = {84, 54};                 // ... whatever a record near it says
    CHECK(PassResultSize(m, dvd, 54, 36) == Size{82, 54});
    CHECK(PassResultSize(m, {}, 54, 36) == Size{82, 54});
    m.stageParams["shown_size"] = {96, 54};
    // passed on by NR and FG: the record of their colour of their size
    const auto dir = std::filesystem::path(DLSSVID_TEST_TMP) / "aspect_shown" / "color_sr";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    m.stageParams["pixel_aspect"] = "32:27";
    m.Save(dir);
    CHECK(ColourShownSize(dir, 82, 54) == std::optional<Size>(Size{96, 54}));
    CHECK(!ColourShownSize(dir, 164, 108));  // another frame
    CHECK(!ColourShownSize({}, 82, 54));
    CHECK(!ColourShownSize(dir.parent_path() / "nowhere", 82, 54));
    // the frame as shown for the viewport: the record, else the pass stretched (no cap: the viewport shows it as it is)
    m.stageParams["shown_size"] = {96, 54};
    CHECK(PassShownSize(m, dvd) == Size{96, 54});
    m.stageParams.erase("shown_size");
    CHECK(PassShownSize(m, dvd) == Size{98, 54});
    CHECK(PassShownSize(m, {}) == Size{82, 54});
    CHECK(ShownSizeMatches({3840, 2160}, {3836, 2160}));
    CHECK(ShownSizeMatches({3840, 2160}, {3832, 2160}));   // 8 px: within 1 % (a wide SAR's capped grid rounded down)
    CHECK(!ShownSizeMatches({3840, 2160}, {3790, 2160}));
    CHECK(!ShownSizeMatches({200, 54}, {98, 54}));
}

TEST_CASE("StretchedResultSize: the result of an anamorphic pass in square pixels, within 3840x2160 by the long side", "[unit][aspect]") {
    using Size = std::pair<uint32_t, uint32_t>;
    CHECK(StretchedResultSize({}, 4096, 2160) == Size{4096, 2160});  // square: as is, beyond the cap too
    CHECK(StretchedResultSize(PixelAspect{32, 27}, 1440, 960) == Size{1706, 960});
    CHECK(StretchedResultSize(PixelAspect{4, 3}, 2880, 2160) == Size{3840, 2160});  // at the cap: as is
    CHECK(StretchedResultSize(PixelAspect{4, 3}, 3840, 2160) == Size{3840, 1620});  // 5120x2160 fit
    CHECK(StretchedResultSize(PixelAspect{4, 1}, 1000, 540) == Size{3840, 518});
    CHECK(StretchedResultSize(PixelAspect{4, 1}, 1000, 552) == Size{3840, 530});  // the free side 529.92 rounds to even (not down to 528)
    CHECK(StretchedResultSize(PixelAspect{27, 32}, 2160, 3240) == Size{2160, 3840});  // portrait at its cap
    CHECK(StretchedResultSize(PixelAspect{1, 2}, 2160, 3840) == Size{1080, 3840});   // portrait 2160x7680 fit
    CHECK(StretchedResultSize(PixelAspect{32, 27}, 0, 0) == Size{0, 0});
}

TEST_CASE("ResolveUpscaleTarget of an anamorphic input: the scale, the target and the cap are of the frame as shown", "[unit][aspect][upscale]") {
    const PixelAspect dvd{32, 27};
    UpscaleTarget t = ResolveUpscaleTarget(720, 480, 2.0, 0, 0, 3840, 2160, true, dvd);
    CHECK(t.width == 1440);  // the grid: x2 of the stored pixels
    CHECK(t.height == 960);
    CHECK(t.displayWidth == 1706);
    CHECK(t.displayHeight == 960);
    CHECK(t.scale == Approx(2.0));
    CHECK(!t.capped);
    t = ResolveUpscaleTarget(720, 480, 0.0, 1920, 1080, 3840, 2160, true, dvd);  // 16:9 as shown: x2.25 of the grid on both axes
    CHECK(t.width == 1620);
    CHECK(t.height == 1080);
    CHECK(t.displayWidth == 1920);
    CHECK(t.displayHeight == 1080);
    t = ResolveUpscaleTarget(720, 480, 0.0, 1920, 0, 3840, 2160, true, dvd);  // the width alone: the shown proportions
    CHECK(t.width == 1620);
    CHECK(t.height == 1080);
    t = ResolveUpscaleTarget(720, 480, 0.0, 0, 1080, 3840, 2160, true, dvd);
    CHECK(t.width == 1620);
    CHECK(t.displayWidth == 1920);
    // the target itself as shown, though the grid stretched back misses it (review of TASK-0073): 10:11, 1920x1440 - the grid 1920x1310,
    // stretched 1441; 32:27, 1440 wide - the grid 1216, stretched 1441.2
    t = ResolveUpscaleTarget(720, 480, 0.0, 1920, 1440, 3840, 2160, true, PixelAspect{10, 11});
    CHECK(t.width == 1920);
    CHECK(t.height == 1310);
    CHECK(t.displayWidth == 1920);
    CHECK(t.displayHeight == 1440);
    t = ResolveUpscaleTarget(720, 480, 0.0, 1440, 0, 3840, 2160, true, dvd);
    CHECK(t.width == 1216);
    CHECK(t.displayWidth == 1440);
    CHECK(t.displayHeight == 810);
    // near the cap, not capped: the frame as shown at the target, not 2 past the cap (97:80, 3840x2160 - the grid 3168 stretched is 3841.2)
    t = ResolveUpscaleTarget(720, 480, 0.0, 3840, 2160, 3840, 2160, true, PixelAspect{97, 80});
    CHECK(!t.capped);
    CHECK(t.width == 3168);
    CHECK(t.displayWidth == 3840);
    CHECK(t.displayHeight == 2160);
    // HDV x3: 1920x1080 as shown is capped at 3840x2160 - the grid 2880x2160
    t = ResolveUpscaleTarget(1440, 1080, 3.0, 0, 0, 3840, 2160, true, PixelAspect{4, 3});
    CHECK(t.capped);
    CHECK(t.width == 2880);
    CHECK(t.height == 2160);
    CHECK(t.displayWidth == 3840);
    CHECK(t.displayHeight == 2160);
    // PAL 16:9 x4: capped, the frame as shown within the cap
    t = ResolveUpscaleTarget(720, 576, 4.0, 0, 0, 3840, 2160, true, PixelAspect{64, 45});
    CHECK(t.capped);
    CHECK(t.displayWidth <= 3840);
    CHECK(t.displayHeight == 2160);
    CHECK(t.width == 2700);
    // narrower pixels: the frame as shown is higher
    t = ResolveUpscaleTarget(720, 480, 2.0, 0, 0, 3840, 2160, true, PixelAspect{8, 9});
    CHECK(t.width == 1440);
    CHECK(t.height == 960);
    CHECK(t.displayWidth == 1440);
    CHECK(t.displayHeight == 1080);
    // a portrait anamorphic frame (a turned one): capped by its long side as shown
    t = ResolveUpscaleTarget(480, 720, 5.0, 0, 0, 3840, 2160, true, PixelAspect{27, 32});
    CHECK(t.capped);
    CHECK(t.width == 2160);
    CHECK(t.height == 3240);
    CHECK(t.displayWidth == 2160);
    CHECK(t.displayHeight == 3840);

    // square pixels: exactly as before, sar or not
    const std::vector<std::vector<double>> cases = {{1920, 1080, 2}, {1920, 1080, 3}, {1280, 720, 3}, {1001, 501, 2}, {1080, 1920, 3}, {1920, 800, 3}, {640, 360, 1.5}};
    for (const auto& c : cases) {
        const auto a = ResolveUpscaleTarget(static_cast<uint32_t>(c[0]), static_cast<uint32_t>(c[1]), c[2]);
        const auto b = ResolveUpscaleTarget(static_cast<uint32_t>(c[0]), static_cast<uint32_t>(c[1]), c[2], 0, 0, 3840, 2160, true, PixelAspect{9, 9});
        CHECK(a.width == b.width);
        CHECK(a.height == b.height);
        CHECK(a.displayWidth == a.width);
        CHECK(a.displayHeight == a.height);
    }
    // capped: the frame as shown never above the cap, whatever the source and its pixels (the grid rounded down along the stretched axis)
    int capped = 0;
    for (uint32_t w = 700; w <= 740; ++w)
        for (uint32_t h = 470; h <= 490; ++h)
            for (const PixelAspect sar : {PixelAspect{32, 27}, PixelAspect{8, 9}, PixelAspect{10, 11}, PixelAspect{64, 45}, PixelAspect{16, 15},
                                          PixelAspect{4, 3}, PixelAspect{40, 33}, PixelAspect{12, 11}}) {
                const UpscaleTarget c = ResolveUpscaleTarget(w, h, 6.0, 0, 0, 3840, 2160, true, sar);
                capped += c.capped ? 1 : 0;
                CHECK(c.displayWidth <= 3840);
                CHECK(c.displayHeight <= 2160);
                const UpscaleTarget p = ResolveUpscaleTarget(h, w, 6.0, 0, 0, 3840, 2160, true, sar);  // portrait: by the long side
                CHECK(p.displayWidth <= 2160);
                CHECK(p.displayHeight <= 3840);
            }
    CHECK(capped == 41 * 21 * 8);
    // a grid that does not come out whole: rounded down along the stretched axis (701x470 32:27 x6 - 3818.2 as shown, 3221.6 of the grid)
    const UpscaleTarget n = ResolveUpscaleTarget(701, 470, 6.0, 0, 0, 3840, 2160, true, PixelAspect{32, 27});
    CHECK(n.capped);
    CHECK(n.width == 3220);
    CHECK(n.height == 2160);
    CHECK(n.displayWidth == 3818);
    // by the scale alone, not capped: the grid is the stored side times the scale exactly as for square pixels - no +sr.sar for it (review
    // of TASK-0073: 706 x1.5 at 32:27 came out 1058, the square rule's 1060)
    CHECK(ResolveUpscaleTarget(706, 480, 1.5, 0, 0, 3840, 2160, true, PixelAspect{32, 27}).width == 1060);
    int uncapped = 0;
    for (uint32_t w = 690; w <= 740; ++w)
        for (uint32_t h = 470; h <= 490; ++h)
            for (const double scale : {1.5, 2.0, 2.5, 3.0})
                for (const PixelAspect sar : {PixelAspect{32, 27}, PixelAspect{8, 9}, PixelAspect{10, 11}, PixelAspect{64, 45}, PixelAspect{16, 15},
                                              PixelAspect{40, 33}, PixelAspect{12, 11}, PixelAspect{16, 11}}) {
                    const UpscaleTarget a = ResolveUpscaleTarget(w, h, scale, 0, 0, 3840, 2160, true, sar);
                    if (a.capped) continue;
                    ++uncapped;
                    const UpscaleTarget b = ResolveUpscaleTarget(w, h, scale);
                    if (a.width != b.width || a.height != b.height) FAIL(w << "x" << h << " x" << scale << " at " << sar.Token());
                }
    CHECK(uncapped > 20000);
    // a target of one side on the axis the pixels do not stretch: the other side as for square pixels (review of TASK-0073: 720x576 at
    // 64:45 with the height 1060 gave 1324, the square rule's 1326)
    CHECK(ResolveUpscaleTarget(720, 576, 0.0, 0, 1060, 3840, 2160, true, PixelAspect{64, 45}).width == 1326);
    int oneSide = 0;
    for (uint32_t w = 690; w <= 740; ++w)
        for (uint32_t h = 470; h <= 590; h += 3)
            for (uint32_t side = 600; side <= 1600; side += 17)
                for (const PixelAspect sar : {PixelAspect{32, 27}, PixelAspect{64, 45}, PixelAspect{16, 15}, PixelAspect{12, 11}, PixelAspect{8, 9},
                                              PixelAspect{10, 11}}) {
                    const bool wider = sar.Value() > 1.0;
                    const UpscaleTarget a = ResolveUpscaleTarget(w, h, 0.0, wider ? 0 : side, wider ? side : 0, 3840, 2160, true, sar);
                    if (a.capped) continue;
                    ++oneSide;
                    const UpscaleTarget b = ResolveUpscaleTarget(w, h, 0.0, wider ? 0 : side, wider ? side : 0);
                    if (a.width != b.width || a.height != b.height) FAIL(w << "x" << h << " side " << side << " at " << sar.Token());
                }
    CHECK(oneSide > 10000);
    // ... capped: the grid from the frame as shown within the cap, as for any capped target (review of TASK-0073)
    const UpscaleTarget ch = ResolveUpscaleTarget(720, 480, 0.0, 0, 2400, 3840, 2160, true, PixelAspect{32, 27});
    CHECK(ch.capped);
    CHECK(ch.width == 3240);
    CHECK(ch.height == 2160);
    CHECK(ch.displayWidth == 3840);
    // a source already past the cap as shown: the grid stays the input's (no upscaler goes down), shown fit into the cap (review)
    const UpscaleTarget past = ResolveUpscaleTarget(3840, 2160, 2.0, 0, 0, 3840, 2160, true, PixelAspect{4, 3});
    CHECK(past.capped);
    CHECK(past.width == 3840);
    CHECK(past.height == 2160);
    CHECK(past.displayWidth == 3840);
    CHECK(past.displayHeight == 1620);
    CHECK(ResolveUpscaleTarget(3840, 2160, 2.0).width == 3840);  // the square rule: the same grid - no +sr.sar
    // ... whatever the target: the frame as shown is what the encode writes of that grid (review: 7680x4320 said 3840x2160)
    const UpscaleTarget pastT = ResolveUpscaleTarget(3840, 2160, 0.0, 7680, 4320, 3840, 2160, true, PixelAspect{4, 3});
    CHECK(pastT.width == 3840);
    CHECK(pastT.height == 2160);
    CHECK(pastT.displayWidth == 3840);
    CHECK(pastT.displayHeight == 1620);
    CHECK(ResolveUpscaleTarget(3840, 1600, 0.0, 7680, 4320, 3840, 2160, true, PixelAspect{4, 3}).displayHeight == 1200);
    const UpscaleTarget past2 = ResolveUpscaleTarget(2880, 2160, 2.0, 0, 0, 3840, 2160, true, PixelAspect{2, 1});
    CHECK(past2.width == 2880);
    CHECK(past2.height == 2160);
    CHECK(past2.displayWidth == 3840);
    CHECK(past2.displayHeight == 1440);
    const UpscaleTarget cw = ResolveUpscaleTarget(720, 480, 0.0, 4000, 0, 3840, 2160, true, PixelAspect{8, 9});
    CHECK(cw.capped);
    CHECK(cw.width == 2880);
    CHECK(cw.height == 1920);
    CHECK(cw.displayHeight == 2160);
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}}, true, dvd).displayWidth == 1706);
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"target_width", 1920}}, true, dvd).width == 1620);
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"target_width", 1920}}).width == 1920);  // the square rule (the build before)
    const UpscaleTarget x1 = UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"artifact_reduction_only", true}}, true, dvd);
    CHECK(x1.width == 720);
    CHECK(x1.displayWidth == 854);
    // x1 of an anamorphic 4K: as `process` writes it - fit into 3840x2160 (5120x2160 stretched)
    const UpscaleTarget x1k = UpscaleOutputFor(3840, 2160, {{"scale", 2.0}, {"artifact_reduction_only", true}}, true, PixelAspect{4, 3});
    CHECK(x1k.displayWidth == 3840);
    CHECK(x1k.displayHeight == 1620);
}

TEST_CASE("EstimateRun of an anamorphic source: the stages make the grid, the result is it stretched", "[unit][aspect][estimates]") {
    ProcessPlan plan;
    plan.passesRoot = "Z:/nowhere";
    for (const char* name : {"upscale", "nr"}) {
        StageDecision d;
        d.name = name;
        d.action = "run";
        // NR's model at half the frame: its time then depends on the frame it runs at (at model_scale 1 one pass cancels out, review)
        d.params = EffectiveStageParams(name, std::string(name) == "nr" ? nlohmann::json{{"model_scale", 0.5}} : nlohmann::json::object());
        plan.stages.push_back(d);
    }
    std::vector<ProcessStage> stages = DefaultProcessStages();
    for (auto& s : stages) s.enabled = s.name == "upscale" || s.name == "nr";
    SourceInfo src;
    src.width = 54;
    src.height = 36;
    src.fps = 24.0;
    src.frames = 10;
    src.sar = PixelAspect{32, 27};
    const RunOutcome o = EstimateRun(plan, stages, src);
    CHECK(o.width == 128);  // before TASK-0073: 108
    CHECK(o.height == 72);
    CHECK(o.gridWidth == 108);
    CHECK(o.gridHeight == 72);
    REQUIRE(o.stages.size() == 3);
    CHECK(o.stages[1].bytes == BytesPerFrame("nr", 108, 72, 1.0, 1) * 10);  // NR writes the grid
    // ... and its time is of the grid: as for the square twin of that grid
    SourceInfo twin = src;
    twin.sar = {};
    const RunOutcome t = EstimateRun(plan, stages, twin);
    CHECK(t.width == 108);
    CHECK(o.stages[1].msPerFrame == Approx(t.stages[1].msPerFrame));
    // a target: of the frame as shown
    for (auto& s : stages)
        if (s.name == "upscale") s.params["target_width"] = 128;
    const RunOutcome targeted = EstimateRun(plan, stages, src);
    CHECK(targeted.width == 128);  // the square rule: 128 of the grid, 152 as shown
    CHECK(targeted.gridWidth == 108);
    for (auto& s : stages)
        if (s.name == "upscale") s.params["target_width"] = 96;
    CHECK(EstimateRun(plan, stages, src).width == 96);  // the target itself, not the grid 82 stretched (98)
    for (auto& s : stages) s.params.erase("target_width");
    // no colour stage: the source passes through with its proportions - the grid
    for (auto& s : stages) s.enabled = false;
    plan.stages.clear();
    const RunOutcome pass = EstimateRun(plan, stages, src);
    CHECK(pass.width == 54);
    CHECK(pass.gridWidth == 54);
    // square: as before
    src.sar = {};
    for (auto& s : stages) s.enabled = s.name == "upscale";
    const RunOutcome square = EstimateRun(plan, stages, src);
    CHECK(square.width == 108);
    CHECK(square.gridWidth == 108);
    // NR alone on a grid the stretch would take beyond 4K: fit into 3840x2160 as process writes it
    SourceInfo wide;
    wide.width = 1000;
    wide.height = 540;
    wide.fps = 24.0;
    wide.frames = 2;
    wide.sar = PixelAspect{4, 1};
    for (auto& s : stages) s.enabled = s.name == "nr";
    const RunOutcome fit = EstimateRun(plan, stages, wide);
    CHECK(fit.width == 3840);
    CHECK(fit.height == 518);
    CHECK(fit.gridWidth == 1000);
    src.sar = PixelAspect{32, 27};
    // an upscale pass reused (of the build before: no shown_size record) - the result is its grid stretched, as the encode writes it,
    // not the target as shown (review of TASK-0073: 702x480 at 32:27 x1.5 - 1250 written, 1248 said)
    const auto reusedRoot = std::filesystem::path(DLSSVID_TEST_TMP) / "aspect_reused_pass";
    std::filesystem::remove_all(reusedRoot);
    std::filesystem::create_directories(reusedRoot / "color_sr");
    Manifest old = Manifest::ForPass(PassKind::ColorSr, 1054, 720, FileFormat::Exr);
    old.Save(reusedRoot / "color_sr");
    SourceInfo dvd = src;
    dvd.width = 702;
    dvd.height = 480;
    dvd.sar = PixelAspect{32, 27};
    ProcessPlan reused;
    reused.passesRoot = reusedRoot;
    StageDecision up;
    up.name = "upscale";
    up.action = "reuse";
    up.params = EffectiveStageParams("upscale", {{"scale", 1.5}});
    reused.stages.push_back(up);
    for (auto& s : stages) {
        s.enabled = s.name == "upscale";
        if (s.name == "upscale") s.params["scale"] = 1.5;
    }
    CHECK(EstimateRun(reused, stages, dvd).width == 1250);
    reused.stages[0].action = "run";  // made now: it records the target as shown
    CHECK(EstimateRun(reused, stages, dvd).width == 1248);
    reused.stages[0].action = "reuse";
    old.stageParams = {{"pixel_aspect", "32:27"}, {"shown_size", {1248, 720}}};  // a pass of this build
    old.Save(reusedRoot / "color_sr");
    CHECK(EstimateRun(reused, stages, dvd).width == 1248);
    // restored from the history: that version (review of TASK-0073)
    const auto version = PassHistoryDir(reusedRoot, "color_sr") / "v1";
    std::filesystem::create_directories(version);
    Manifest restored = Manifest::ForPass(PassKind::ColorSr, 1054, 720, FileFormat::Exr);
    restored.Save(version);
    reused.stages[0].action = "restore";
    reused.stages[0].version = "v1";
    CHECK(EstimateRun(reused, stages, dvd).width == 1250);
}

TEST_CASE("DLSS-G's camera takes the frame as shown: the grid of 32:27 pixels is 16:9", "[unit][aspect][fg]") {
    CHECK(BuildFgCamera(54, 36, PixelAspect{32, 27}).aspect == Approx(64.0 / 36.0));
    CHECK(BuildFgCamera(1440, 960, PixelAspect{32, 27}).aspect == Approx(16.0 / 9.0).epsilon(1e-3));
    CHECK(BuildFgCamera(1920, 1080).aspect == Approx(1920.0 / 1080.0));
    const FgCamera c = BuildFgCamera(720, 480, PixelAspect{8, 9});
    CHECK(c.aspect == Approx(4.0 / 3.0));
    CHECK(c.viewToClip[0] == Approx(c.viewToClip[5] / c.aspect));
}

TEST_CASE("YuvResizer: the same layout at another size, Lanczos, repeatable; the same size a copy", "[unit][aspect][convert]") {
    for (const FrameDesc desc : {FrameDesc{54, 36, PixelFormat::Yuv420p, 8}, FrameDesc{54, 36, PixelFormat::Yuv420p16, 10}, FrameDesc{54, 36, PixelFormat::Yuv444p, 8}}) {
        INFO(desc.Name());
        CpuFrame in;
        in.Allocate(desc);
        in.index = 7;
        in.pts = 9;
        // a flat grey stays flat; a vertical edge stays where it is as shown
        for (size_t p = 0; p < 3; ++p)
            for (size_t y = 0; y < desc.PlaneHeight(p); ++y)
                for (size_t x = 0; x < desc.PlaneCols(p); ++x) {
                    const uint32_t v8 = p == 0 ? (x < desc.PlaneCols(p) / 2 ? 40u : 200u) : 128u;
                    if (desc.HighBitDepth()) in.Plane16(p)[y * desc.PlaneCols(p) + x] = static_cast<uint16_t>(v8 << (desc.bitDepth - 8));
                    else in.Plane(p)[y * desc.PlaneCols(p) + x] = static_cast<uint8_t>(v8);
                }
        YuvResizer r;
        CpuFrame same;
        r.Resize(in, 54, 36, same);
        CHECK(same.data == in.data);
        CpuFrame out, again;
        r.Resize(in, 64, 36, out);
        r.Resize(in, 64, 36, again);
        CHECK(out.desc.width == 64);
        CHECK(out.desc.height == 36);
        CHECK(out.desc.format == desc.format);
        CHECK(out.desc.bitDepth == desc.bitDepth);
        CHECK(out.index == 7);
        CHECK(out.pts == 9);
        CHECK(out.data == again.data);
        auto at = [&](size_t p, size_t x, size_t y) -> uint32_t {
            const uint32_t v = desc.HighBitDepth() ? out.Plane16(p)[y * out.desc.PlaneCols(p) + x] : out.Plane(p)[y * out.desc.PlaneCols(p) + x];
            return desc.HighBitDepth() ? v >> (desc.bitDepth - 8) : v;
        };
        CHECK(at(1, 3, 3) == 128);  // flat chroma stays exact
        uint32_t lowest = 255;  // Lanczos rings at the edge: below the dark level just before it (bilinear would not)
        for (uint32_t x = 0; x < 64; ++x) lowest = std::min(lowest, at(0, x, 10));
        CHECK(lowest <= 33);  // Lanczos-3: about 31 at x 30; swscale's bicubic about 36 (review of TASK-0073)
        CHECK(at(0, 4, 10) == 40);  // away from the edge: the level kept
        CHECK(at(0, 60, 10) == 200);
        uint32_t edge = 0;  // the edge moved from 27 of 54 to 32 of 64: the half-way crossing between 31 and 33
        for (uint32_t x = 0; x + 1 < 64; ++x)
            if (at(0, x, 10) < 120 && at(0, x + 1, 10) >= 120) edge = x + 1;
        CHECK(edge >= 31);
        CHECK(edge <= 33);
    }
    CpuFrame empty, out;
    YuvResizer r;
    CHECK_THROWS(r.Resize(empty, 64, 36, out));
    CpuFrame described;  // a layout without its samples: refused before swscale reads past them
    described.desc = FrameDesc{54, 36, PixelFormat::Yuv420p, 8};
    CHECK_THROWS_WITH(r.Resize(described, 64, 36, out), Catch::Matchers::ContainsSubstring("has no data"));
}

TEST_CASE("Settings note: an anamorphic source says its pixels - source: sar=32:27; a square one as before", "[unit][aspect][encode]") {
    const auto root = std::filesystem::path(DLSSVID_TEST_TMP) / "settings_note" / "aspect";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    auto save = [&](const std::string& format) {
        Manifest m = Manifest::ForPass(PassKind::ColorSr, 1440, 960, FileFormat::Exr);
        m.stageParams = {{"backend", "nis"}, {"scale", 2.0}};
        m.sourceFormat = format;
        m.Save(root / "color_sr");
    };
    save("yuv420p");
    CHECK(BuildSettingsNote(root, "color_sr", {}, PixelAspect{32, 27}) == std::string("VideoDLSS ") + DLSSVID_VERSION + " | source: sar=32:27 | upscale: nis x2");
    CHECK(BuildSettingsJson(root, "color_sr", {}, PixelAspect{32, 27})["source"] == nlohmann::json{{"sar", "32:27"}});
    CHECK(BuildSettingsNote(root, "color_sr") == std::string("VideoDLSS ") + DLSSVID_VERSION + " | upscale: nis x2");
    save("yuv420p+r90");
    CHECK(BuildSettingsNote(root, "color_sr", {}, PixelAspect{32, 27}).find("| source: rotate=90 sar=32:27 |") != std::string::npos);
    CHECK_FALSE(BuildSettingsJson(root, "missing", {}, PixelAspect{32, 27}).contains("source"));  // no encoded pass: nothing said
    // a note read back: only a text is taken
    CHECK(SettingsNoteText({{"source", {{"sar", 5}}}}).find("sar") == std::string::npos);
}
