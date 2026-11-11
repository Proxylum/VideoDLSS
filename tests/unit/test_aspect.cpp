// TASK-0073: the proportions of a pixel (PixelAspect), the upscale target of the frame as shown, the estimate, DLSS-G's camera, the resize
// `process` stretches an anamorphic source's result with, the settings note.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "convert/YuvConvert.h"
#include "passes/Manifest.h"
#include "pipeline/Estimates.h"
#include "pipeline/Frame.h"
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
    CHECK(PixelAspect{32, 27}.DisplaySize(0, 0) == std::pair<uint32_t, uint32_t>{0, 0});  // a size not known yet
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
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}}, true, dvd).displayWidth == 1706);
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"target_width", 1920}}, true, dvd).width == 1620);
    CHECK(UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"target_width", 1920}}).width == 1920);  // the square rule (the build before)
    const UpscaleTarget x1 = UpscaleOutputFor(720, 480, {{"scale", 2.0}, {"artifact_reduction_only", true}}, true, dvd);
    CHECK(x1.width == 720);
    CHECK(x1.displayWidth == 854);
}

TEST_CASE("EstimateRun of an anamorphic source: the stages make the grid, the result is it stretched", "[unit][aspect][estimates]") {
    ProcessPlan plan;
    plan.passesRoot = "Z:/nowhere";
    for (const char* name : {"upscale", "nr"}) {
        StageDecision d;
        d.name = name;
        d.action = "run";
        d.params = EffectiveStageParams(name, nlohmann::json::object());
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
}

TEST_CASE("DLSS-G's camera takes the frame as shown: the grid of 32:27 pixels is 16:9", "[unit][aspect][fg]") {
    CHECK(BuildFgCamera(54, 36, PixelAspect{32, 27}).aspect == Approx(64.0 / 36.0));
    CHECK(BuildFgCamera(1440, 960, PixelAspect{32, 27}).aspect == Approx(16.0 / 9.0).epsilon(1e-3));
    CHECK(BuildFgCamera(1920, 1080).aspect == Approx(1920.0 / 1080.0));
    const FgCamera c = BuildFgCamera(720, 480, PixelAspect{8, 9});
    CHECK(c.aspect == Approx(4.0 / 3.0));
    CHECK(c.viewToClip[0] == Approx(c.viewToClip[5] / c.aspect));
}

TEST_CASE("YuvResizer: the same layout at another size, Lanczos, deterministic; the same size a copy", "[unit][aspect][convert]") {
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
        CHECK(lowest < 40);
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
