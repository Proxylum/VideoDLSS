// TASK-0072: a source shown turned - a phone video kept landscape with a display matrix, or the old 'rotate' tag - is processed as players
// show it: the decoder turns the frames, the passes carry the turn, the result is written upright without a matrix, the proxies follow.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "io/ResultFile.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "passes/PassImage.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "pipeline/ControlMask.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "util/Sha256.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/nr/INrBackend.h"
#include "stages/upscale/IUpscaler.h"
#include "viewport/ViewportProxy.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const std::string& name) {
    const auto d = TempDir() / "orientation" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

const std::vector<DisplayOrientation> kAll = {{0, false}, {90, false}, {180, false}, {270, false}, {0, true}, {90, true}, {180, true}, {270, true}};

std::string Name(DisplayOrientation o) { return o.Identity() ? "id" : o.Token(); }

// The shown frame i of a clip written from SyntheticFrame(i, stored) with orientation `o` (OrientFrame: the unit tests hold it to
// FFmpeg's filters)
CpuFrame Shown(int64_t i, const FrameDesc& stored, DisplayOrientation o) {
    CpuFrame t;
    OrientFrame(SyntheticFrame(i, stored), o, t);
    return t;
}

bool Same(const CpuFrame& a, const CpuFrame& b) { return a.desc == b.desc && a.data == b.data; }

std::vector<ProcessStage> StubStages() {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend",
                             "fg.multiplier=2"})
        ApplyParamSpec(s, spec);
    return s;
}

const ProcessStageReport& Report(const ProcessResult& r, const std::string& name) {
    for (const auto& s : r.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no report for " + name);
}

const StageDecision& Decision(const ProcessPlan& p, const std::string& name) {
    for (const auto& s : p.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no decision for " + name);
}

ProcessOptions Options(const std::filesystem::path& clip, const std::filesystem::path& dir, const std::string& result = "result.mkv") {
    ProcessOptions o;
    o.input = clip;
    o.output = dir / result;
    o.passesRoot = dir / "passes";
    o.stages = StubStages();
    o.codec = "ffv1";
    o.warp = true;
    return o;
}

void WritePass(const std::filesystem::path& dir, PassKind kind, uint32_t w, uint32_t h, int64_t frames, FileFormat format = FileFormat::Exr) {
    PassWriter wr(dir, Manifest::ForPass(kind, w, h, format));
    for (int64_t f = 0; f < frames; ++f) wr.WriteFrame(f, MakePassImage(kind, w, h, kind == PassKind::Mask ? std::optional<PixelType>(PixelType::U8) : std::nullopt));
    wr.Finish();
}

}  // namespace

TEST_CASE("VideoDecoder shows a turned clip as players do: every orientation of the display matrix, after a seek too", "[integration][orientation]") {
    const auto d = Dir("decode");
    const FrameDesc stored{64, 36, PixelFormat::Yuv420p, 8};
    for (const auto& o : kAll) {
        INFO(o.Token());
        ClipSpec spec;
        spec.frames = 10;
        spec.width = stored.width;
        spec.height = stored.height;
        spec.orientation = o;
        const auto clip = WriteClip(d / ("clip_" + Name(o) + ".mkv"), spec);
        VideoDecoder probe(clip, VideoDecoder::Probe());
        const VideoStreamInfo& info = probe.Info();
        CHECK(info.orientation == o);
        CHECK(info.orientationFrom == (o.Identity() ? "" : "display matrix"));
        CHECK(info.codedWidth == 64);
        CHECK(info.codedHeight == 36);
        CHECK(info.width == (o.SwapsAxes() ? 36u : 64u));
        CHECK(info.height == (o.SwapsAxes() ? 64u : 36u));
        CHECK(info.decoded == OrientedDesc(stored, o));
        const auto frames = DecodeAll(clip);
        REQUIRE(frames.size() == 10);
        bool all = true;
        for (int64_t i = 0; i < 10; ++i) all = all && Same(frames[static_cast<size_t>(i)], Shown(i, stored, o));
        CHECK(all);
        VideoDecoder seek(clip);
        REQUIRE(seek.SeekToFrame(7));
        CpuFrame f;
        REQUIRE(seek.NextFrame(f));
        CHECK(f.index == 7);
        CHECK(Same(f, Shown(7, stored, o)));
        VideoDecoder skip(clip);  // a frame skipped while seeking carries the shown size too
        CpuFrame s;
        REQUIRE(skip.NextFrame(s, nullptr, false));
        CHECK(s.desc.width == info.width);
        CHECK(s.desc.height == info.height);
    }
}

TEST_CASE("VideoDecoder turns a clip by the rotate tag of older muxers; a display matrix wins over the tag", "[integration][orientation]") {
    const auto d = Dir("tag");
    const FrameDesc stored{64, 36, PixelFormat::Yuv420p, 8};
    ClipSpec spec;
    spec.frames = 3;
    spec.width = stored.width;
    spec.height = stored.height;
    spec.rotateTag = "90";
    const auto tagged = WriteClip(d / "tagged.mkv", spec);
    {
        VideoDecoder probe(tagged, VideoDecoder::Probe());
        CHECK(probe.Info().orientation == DisplayOrientation{90, false});
        CHECK(probe.Info().orientationFrom == "rotate tag");
    }
    CHECK(Same(DecodeAll(tagged)[1], Shown(1, stored, {90, false})));
    spec.orientation = {180, false};
    const auto both = WriteClip(d / "both.mkv", spec);
    {
        VideoDecoder probe(both, VideoDecoder::Probe());
        CHECK(probe.Info().orientation == DisplayOrientation{180, false});
        CHECK(probe.Info().orientationFrom == "display matrix");
    }
    spec.orientation = {};
    spec.rotateTag = "45";  // not a multiple of 90: not applied
    const auto odd = WriteClip(d / "odd.mkv", spec);
    VideoDecoder probe(odd, VideoDecoder::Probe());
    CHECK(probe.Info().orientation.Identity());
    CHECK(probe.Info().orientationFrom.empty());
    CHECK(probe.Info().width == 64);
}

TEST_CASE("A turned 4:2:2 clip decodes as 4:4:4 by 90 deg and stays 4:2:2 by 180 - exact; its own layout stays 4:2:2", "[integration][orientation][formats]") {
    const auto d = Dir("chroma");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.pixFmt = "yuv422p10le";
    const FrameDesc stored = FrameDescFor(spec);
    REQUIRE(stored.format == PixelFormat::Yuv422p16);
    for (const DisplayOrientation o : {DisplayOrientation{90, false}, DisplayOrientation{180, false}}) {
        INFO(o.Token());
        spec.orientation = o;
        const auto clip = WriteClip(d / ("clip_" + Name(o) + ".mkv"), spec);
        VideoDecoder probe(clip, VideoDecoder::Probe());
        const VideoStreamInfo& info = probe.Info();
        CHECK(info.decoded.format == (o.SwapsAxes() ? PixelFormat::Yuv444p16 : PixelFormat::Yuv422p16));
        CHECK(info.decoded.bitDepth == 10);
        CHECK(SourceLayout(info) == FrameDesc{info.width, info.height, PixelFormat::Yuv422p16, 10});  // what «auto» follows
        const auto frames = DecodeAll(clip);
        REQUIRE(frames.size() == 3);
        for (int64_t i = 0; i < 3; ++i) {
            CpuFrame want;
            OrientFrame(SyntheticFrame(i, stored), o, want);
            CHECK(Same(frames[static_cast<size_t>(i)], want));
        }
    }
}

TEST_CASE("Passthrough of a turned clip: the result upright and without a display matrix, bit-exact to the shown frames", "[integration][orientation][passthrough]") {
    const auto d = Dir("passthrough");
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d, "pt.mkv");
    o.passthrough = true;
    const ProcessResult r = RunProcess(o);
    CHECK(r.outputFormat == FrameDesc{36, 64, PixelFormat::Yuv420p, 8});
    {
        VideoDecoder out(o.output, VideoDecoder::Probe());
        CHECK(out.Info().orientation.Identity());
        CHECK(out.Info().width == 36);
        CHECK(out.Info().height == 64);
    }
    const auto got = DecodeAll(o.output), want = DecodeAll(clip);
    REQUIRE(got.size() == want.size());
    bool all = true;
    for (size_t i = 0; i < got.size(); ++i) all = all && Same(got[i], want[i]);
    CHECK(all);
    // a turned 4:2:2 stream: decoded as 4:4:4, the result in the stream's own 4:2:2 (the encoder's «auto»), luma exact
    spec.pixFmt = "yuv422p10le";
    const auto clip422 = WriteClip(d / "clip422.mkv", spec);
    ProcessOptions o422 = Options(clip422, d, "pt422.mkv");
    o422.passthrough = true;
    const ProcessResult r422 = RunProcess(o422);
    CHECK(r422.outputFormat == FrameDesc{36, 64, PixelFormat::Yuv422p16, 10});
    CHECK(Summarize(o422.output).pixelFormat == "yuv422p10le");
    const auto got422 = DecodeAll(o422.output), want422 = DecodeAll(clip422);
    REQUIRE(got422.size() == want422.size());
    for (size_t i = 0; i < got422.size(); ++i) {
        const size_t luma = got422[i].desc.PlaneSize(0);
        REQUIRE(want422[i].desc.PlaneSize(0) == luma);
        CHECK(std::equal(got422[i].Plane(0), got422[i].Plane(0) + luma, want422[i].Plane(0)));
    }
}

TEST_CASE("process on a turned clip gives the result of its upright twin; the passes carry the turn, the result no matrix", "[integration][orientation][process]") {
    const FrameDesc stored{64, 36, PixelFormat::Yuv420p, 8};
    for (const DisplayOrientation o : {DisplayOrientation{90, false}, DisplayOrientation{180, false}, DisplayOrientation{0, true}}) {
        INFO(o.Token());
        const auto d = Dir("process_" + Name(o));
        ClipSpec spec;
        spec.frames = 4;
        spec.width = stored.width;
        spec.height = stored.height;
        spec.orientation = o;
        const auto turned = WriteClip(d / "turned.mkv", spec);
        ClipSpec twinSpec = spec;
        twinSpec.orientation = {};
        twinSpec.width = o.SwapsAxes() ? stored.height : stored.width;
        twinSpec.height = o.SwapsAxes() ? stored.width : stored.height;
        const auto twin = WriteClipWith(d / "twin.mkv", twinSpec, [&](int64_t i) { return Shown(i, stored, o); });
        const auto td = d / "t", wd = d / "w";
        std::filesystem::create_directories(td);
        std::filesystem::create_directories(wd);
        const ProcessResult tr = RunProcess(Options(turned, td));
        const ProcessResult wr = RunProcess(Options(twin, wd));
        CHECK(tr.outputFormat == wr.outputFormat);
        const auto a = DecodeAll(td / "result.mkv"), b = DecodeAll(wd / "result.mkv");
        REQUIRE(a.size() == b.size());
        REQUIRE(!a.empty());
        bool all = true;
        for (size_t i = 0; i < a.size(); ++i) all = all && Same(a[i], b[i]);
        CHECK(all);
        CHECK(Manifest::Load(td / "passes" / "color_sr").sourceFormat == "yuv420p+" + o.Token());
        CHECK(Manifest::Load(wd / "passes" / "color_sr").sourceFormat == "yuv420p");
        VideoDecoder out(td / "result.mkv", VideoDecoder::Probe());
        CHECK(out.Info().orientation.Identity());
        // the rerun reuses every pass
        for (const auto& s : PlanProcess(Options(turned, td)).stages) {
            INFO(s.name);
            CHECK(s.action == "reuse");
        }
    }
}

TEST_CASE("A turned source's passes made before the turn was applied are recomputed, and the plan says why", "[integration][orientation][process]") {
    const auto d = Dir("before");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {180, false};  // the same size either way: only the label tells
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    RunProcess(o);
    // depth_dlss as the build before made it: no turn in its label nor its fingerprint
    Manifest legacy = Manifest::Load(d / "passes" / "depth_dlss");
    REQUIRE(legacy.sourceFormat == "yuv420p+r180");
    legacy.sourceFormat = "yuv420p";
    legacy.fingerprint = ComputeFingerprint("depth", legacy.sourceHash, legacy.paramsCanonical, legacy.inputs, ToolInfo::FromJson(legacy.tool), "yuv420p").value;
    legacy.Save(d / "passes" / "depth_dlss");
    const StageDecision depth = Decision(PlanProcess(o), "depth");
    CHECK(depth.action == "run");
    CHECK(depth.kind == "source_changed");
    CHECK_THAT(depth.detail, ContainsSubstring("yuv420p (now yuv420p+r180)"));
    REQUIRE(depth.diff.contains("source_format"));
    CHECK(depth.diff["source_format"] == nlohmann::json::array({"yuv420p", "yuv420p+r180"}));
}

TEST_CASE("The result check refuses a file that carries a display orientation: the frames were written upright", "[integration][orientation][result]") {
    const auto d = Dir("check");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto turned = WriteClip(d / "turned.mkv", spec);
    CHECK_THAT(CheckEncodedResult(turned, ResultExpectation{64, 36, 3, false}), ContainsSubstring("display orientation (r90)"));
    spec.orientation = {};
    const auto plain = WriteClip(d / "plain.mkv", spec);
    CHECK(CheckEncodedResult(plain, ResultExpectation{64, 36, 3, false}).empty());
}

TEST_CASE("A colour pass of a turned source's stored frame is not encoded: the result would be sideways", "[integration][orientation][process][regression]") {
    const auto d = Dir("sideways");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "upscale";
    RunProcess(o);
    const auto sr = d / "passes" / "color_sr";
    REQUIRE(Manifest::Load(sr).sourceFormat == "yuv420p+r90");
    REQUIRE(Manifest::Load(sr).width == 72);
    for (auto& s : o.stages) s.enabled = false;  // no stage remakes it: the pass on disk is encoded
    o.output = d / "upright.mkv";
    CHECK_NOTHROW(RunProcess(o));  // made from the shown frame
    // labelled as made before the turn was applied
    Manifest m = Manifest::Load(sr);
    m.sourceFormat = "yuv420p";
    m.Save(sr);
    o.output = d / "r_label.mkv";
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("Результат взялся бы из пасса color_sr"));  // refused before the stages (review of TASK-0072)
    CHECK_FALSE(std::filesystem::exists(o.output));
    EncodeSettings es;
    es.codec = "ffv1";
    CHECK_THROWS_WITH(EncodePassToVideo(sr, clip, d / "direct_label.mkv", es), ContainsSubstring("made from the stored frame"));  // the encode's own guard
    // no label (a standalone command's pass): its proportions tell - the stored frame's are refused, the shown frame's taken
    std::filesystem::remove_all(sr);
    WritePass(sr, PassKind::ColorSr, 128, 72, 3);
    o.output = d / "r_stored.mkv";
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("Результат взялся бы из пасса color_sr"));
    CHECK_THROWS_WITH(EncodePassToVideo(sr, clip, d / "direct_stored.mkv", es), ContainsSubstring("made from the stored frame"));
    std::filesystem::remove_all(sr);
    WritePass(sr, PassKind::ColorSr, 72, 128, 3);
    o.output = d / "r_shown.mkv";
    CHECK_NOTHROW(RunProcess(o));
    CHECK_NOTHROW(EncodePassToVideo(sr, clip, d / "direct_shown.mkv", es));
    // the result's settings note says the source's turn although the unlabelled pass does not (review of TASK-0072)
    const auto tags = ReadContainerTags(o.output);
    REQUIRE(tags.count("comment"));
    CHECK(tags.at("comment").find("| source: rotate=90 |") != std::string::npos);
}

TEST_CASE("A proxy of a turned clip holds its frames as shown; one made before the turn was applied is not used", "[integration][orientation][viewport][proxy]") {
    const auto d = Dir("proxy");
    const FrameDesc stored{64, 36, PixelFormat::Yuv420p, 8};
    ClipSpec spec;
    spec.frames = 8;
    spec.width = stored.width;
    spec.height = stored.height;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    const auto passes = d / "passes";
    const auto proxy = BuildProxy(clip, passes, "source");
    REQUIRE(ValidProxy(clip, passes, "source") == proxy);
    const auto side = ProxyDir(passes) / "source.proxy.json";
    nlohmann::json j;
    std::ifstream(side) >> j;
    CHECK(j["orientation"] == "r90");
    const auto frames = DecodeAll(proxy);
    REQUIRE(frames.size() == 8);
    CHECK(Same(frames[3], Shown(3, stored, {90, false})));
    j.erase("orientation");  // as the builds before wrote it: made from the stored frame
    std::ofstream(side) << j.dump(2);
    CHECK_FALSE(ValidProxy(clip, passes, "source"));
    // an unturned clip's proxy without the field stays valid: no rebuild for it
    spec.orientation = {};
    const auto plain = WriteClip(d / "plain.mkv", spec);
    const auto plainPasses = d / "plain_passes";
    BuildProxy(plain, plainPasses, "source");
    const auto plainSide = ProxyDir(plainPasses) / "source.proxy.json";
    nlohmann::json pj;
    std::ifstream(plainSide) >> pj;
    CHECK(pj["orientation"] == "");
    pj.erase("orientation");
    std::ofstream(plainSide) << pj.dump(2);
    CHECK(ValidProxy(plain, plainPasses, "source"));
}

TEST_CASE("An untagged turned 960x540 clip stays BT.601 in the result: the matrix is guessed by the stored height", "[integration][orientation][color]") {
    const auto d = Dir("color");
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 960;
    spec.height = 540;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d, "pt.mkv");
    o.passthrough = true;
    RunProcess(o);
    CHECK(Summarize(o.output).colorSpace == 6);  // SMPTE 170M (BT.601), as the unturned clip; by the shown 960 high it was BT.709
}

TEST_CASE("The plan warns of masks drawn over a turned source's stored frame and of a target of its proportions", "[integration][orientation][process]") {
    const auto d = Dir("masks");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "nr" || s.name == "upscale";
    auto nrWarnings = [&] {
        std::string all;
        const StageDecision nr = Decision(PlanProcess(o), "nr");  // a copy: the plan is a temporary
        for (const auto& w : nr.warnings) all += w + "\n";
        return all;
    };
    WritePass(d / "passes" / "mask_ui", PassKind::Mask, 64, 36, 3, FileFormat::Png);  // drawn over the stored frame
    CHECK_THAT(nrWarnings(), ContainsSubstring("Сделана по боковому кадру: маска ui. Источник повёрнут на 90° — маска ляжет неправильно"));
    WritePass(d / "passes" / "mask_skin", PassKind::Mask, 64, 36, 3, FileFormat::Png);  // two: the plural (review of TASK-0072)
    CHECK_THAT(nrWarnings(), ContainsSubstring("Сделаны по боковому кадру: маска ui, маска skin. Источник повёрнут на 90° — маски лягут неправильно"));
    std::filesystem::remove_all(d / "passes" / "mask_skin");
    std::filesystem::remove_all(d / "passes" / "mask_ui");
    WritePass(d / "passes" / "mask_ui", PassKind::Mask, 36, 64, 3, FileFormat::Png);  // over the shown one
    CHECK_THAT(nrWarnings(), !ContainsSubstring("по боковому кадру"));
    CHECK_THAT(nrWarnings(), !ContainsSubstring("до учёта поворота"));
    // an unlabelled color_sr of the shown frame's proportions (a standalone command of this build): no word of a project from before
    WritePass(d / "passes" / "color_sr", PassKind::ColorSr, 72, 128, 3);
    CHECK_THAT(nrWarnings(), !ContainsSubstring("до учёта поворота"));
    std::filesystem::remove_all(d / "passes" / "color_sr");
    // 180 deg keeps the proportions: a project processed before the turn was applied is what tells
    spec.orientation = {180, false};
    const auto clip180 = WriteClip(d / "clip180.mkv", spec);
    const auto d180 = d / "p180";
    std::filesystem::create_directories(d180);
    ProcessOptions o180 = Options(clip180, d180);
    for (auto& s : o180.stages) s.enabled = s.name == "nr";
    WritePass(d180 / "passes" / "mask_ui", PassKind::Mask, 64, 36, 3, FileFormat::Png);
    RunProcess(o180);
    auto warnings180 = [&] {
        std::string all;
        const StageDecision nr180 = Decision(PlanProcess(o180), "nr");
        for (const auto& w : nr180.warnings) all += w + "\n";
        return all;
    };
    CHECK_THAT(warnings180(), !ContainsSubstring("до учёта поворота"));
    Manifest nr = Manifest::Load(d180 / "passes" / "color_nr");
    nr.sourceFormat = "yuv420p";
    nr.Save(d180 / "passes" / "color_nr");
    CHECK_THAT(warnings180(), ContainsSubstring("Проект обработан до учёта поворота источника (источник повёрнут на 180°)"));
    CHECK_THAT(warnings180(), ContainsSubstring("проверьте их по повёрнутому кадру"));
    // an explicit target of the stored frame's proportions
    ApplyParamSpec(o.stages, "upscale.target_width=128");
    ApplyParamSpec(o.stages, "upscale.target_height=72");
    std::string up;
    const StageDecision stored = Decision(PlanProcess(o), "upscale");
    for (const auto& w : stored.warnings) up += w;
    CHECK_THAT(up, ContainsSubstring("пропорции бокового кадра"));
    ApplyParamSpec(o.stages, "upscale.artifact_reduction_only=true");  // x1: the target is not used - nothing stretches (review of TASK-0072)
    const StageDecision unused = Decision(PlanProcess(o), "upscale");
    for (const auto& w : unused.warnings) CHECK(w.find("пропорции бокового кадра") == std::string::npos);
    ApplyParamSpec(o.stages, "upscale.artifact_reduction_only=false");
    ApplyParamSpec(o.stages, "upscale.target_width=72");
    ApplyParamSpec(o.stages, "upscale.target_height=128");
    up.clear();
    const StageDecision shown = Decision(PlanProcess(o), "upscale");
    for (const auto& w : shown.warnings) up += w;
    CHECK(up.find("пропорции бокового кадра") == std::string::npos);
}

TEST_CASE("The result of a turned 4:2:2 clip through the stages keeps the stream's 4:2:2, not the 4:4:4 it decodes as", "[integration][orientation][formats]") {
    const auto d = Dir("chroma_process");
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 64;
    spec.height = 36;
    spec.pixFmt = "yuv422p10le";
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "upscale";
    const ProcessResult r = RunProcess(o);
    CHECK(r.outputFormat == FrameDesc{72, 128, PixelFormat::Yuv422p16, 10});
    CHECK(Summarize(o.output).pixelFormat == "yuv422p10le");
}

TEST_CASE("The control mask advice measures a turned source's shown frame, not a sideways color_sr left from before", "[integration][orientation][mask]") {
    const auto d = Dir("control_mask");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "nr";
    {  // the colour pass of the build before: the stored frame, x2
        Manifest m = Manifest::ForPass(PassKind::ColorSr, 128, 72, FileFormat::Exr);
        m.sourceFormat = "yuv420p";
        PassWriter wr(o.passesRoot / "color_sr", m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorSr, 128, 72));
        wr.Finish();
    }
    PassImage mask;  // drawn over the stored frame's proportions
    mask.Allocate(64, 36, PixelType::U16, {"A"});
    for (uint32_t y = 0; y < 36; ++y)
        for (uint32_t x = 0; x < 64; ++x) mask.Set(x, y, 0, 30000.f);
    WritePng(d / "mask.png", mask);
    ImportControlMask(d / "mask.png", o.passesRoot);
    std::string all;
    const StageDecision nr = Decision(PlanProcess(o), "nr");
    for (const auto& w : nr.warnings) all += w + "\n";
    CHECK_THAT(all, ContainsSubstring("64×36 растянется на кадр 36×64"));  // the frame NR is to work at: the shown one
    CHECK_THAT(nr.error, ContainsSubstring("Пасс color_sr (стадия «Апскейл») сделан по боковому кадру"));  // the sideways color_sr is no input of it
    // an unlabelled color_sr of the shown frame (a standalone upscale with a 3:4 target): NR's frame, as the plan takes it (review of TASK-0072)
    std::filesystem::remove_all(o.passesRoot / "color_sr");
    WritePass(o.passesRoot / "color_sr", PassKind::ColorSr, 48, 64, 3);
    PassImage fits;
    fits.Allocate(48, 64, PixelType::U16, {"A"});
    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 48; ++x) fits.Set(x, y, 0, 30000.f);
    WritePng(d / "fits.png", fits);
    ImportControlMask(d / "fits.png", o.passesRoot);
    const StageDecision taken = Decision(PlanProcess(o), "nr");
    CHECK(taken.error.empty());
    std::string advice;
    for (const auto& w : taken.warnings) advice += w + "\n";
    CHECK(advice.find("растянется") == std::string::npos);  // measured on color_sr 48x64, not the source's 36x64
}

TEST_CASE("A stage refuses an input made from a turned source's stored frame that the run does not remake", "[integration][orientation][process][regression]") {
    // review of TASK-0072: NR alone over the color_sr of the build before read the sideways picture, its color_nr got the turn's label and the
    // encode took it - a sideways result
    const auto d = Dir("stored_input");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    auto pass = [&](const char* name, PassKind kind, uint32_t w, uint32_t h, const std::string& label) {
        std::filesystem::remove_all(o.passesRoot / name);
        Manifest m = Manifest::ForPass(kind, w, h, FileFormat::Exr);
        m.sourceFormat = label;
        PassWriter wr(o.passesRoot / name, m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(kind, w, h));
        wr.Finish();
    };
    for (auto& s : o.stages) s.enabled = s.name == "nr";  // upscale off: color_sr is read from the disk
    pass("color_sr", PassKind::ColorSr, 128, 72, "yuv420p");  // the build before: the stored frame x2
    CHECK_THAT(Decision(PlanProcess(o), "nr").error,
               ContainsSubstring("Пасс color_sr (стадия «Апскейл») сделан по боковому кадру, а источник повёрнут на 90°: включите эту стадию или удалите пасс"));
    o.output = d / "r_labelled.mkv";
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("Пасс color_sr (стадия «Апскейл») сделан по боковому кадру"));
    CHECK_FALSE(std::filesystem::exists(o.output));
    CHECK_FALSE(std::filesystem::exists(o.passesRoot / "color_nr"));  // refused before anything ran
    pass("color_sr", PassKind::ColorSr, 128, 72, "");  // a standalone command's: the stored frame's proportions
    CHECK_FALSE(Decision(PlanProcess(o), "nr").error.empty());
    pass("color_sr", PassKind::ColorSr, 72, 128, "");  // ... the shown frame's: taken
    CHECK(Decision(PlanProcess(o), "nr").error.empty());
    pass("color_sr", PassKind::ColorSr, 72, 128, "yuv420p+r90");
    CHECK(Decision(PlanProcess(o), "nr").error.empty());
    o.output = d / "r_upright.mkv";
    CHECK_NOTHROW(RunProcess(o));
    // the upscale on: it remakes color_sr in this run, the old one is no matter
    pass("color_sr", PassKind::ColorSr, 128, 72, "yuv420p");
    for (auto& s : o.stages) s.enabled = s.name == "nr" || s.name == "upscale";
    CHECK(Decision(PlanProcess(o), "nr").error.empty());
    // fg alone over the color_nr of the build before
    for (auto& s : o.stages) s.enabled = s.name == "fg";
    pass("color_nr", PassKind::ColorNr, 128, 72, "yuv420p");
    CHECK_THAT(Decision(PlanProcess(o), "fg").error, ContainsSubstring("Пасс color_nr (стадия «Улучшение») сделан по боковому кадру"));
    // turned by 180: not sideways - made without the turn (review of TASK-0072)
    spec.orientation = {180, false};
    const auto clip180 = WriteClip(d / "clip180.mkv", spec);
    ProcessOptions o180 = Options(clip180, d / "p180");
    for (auto& s : o180.stages) s.enabled = s.name == "nr";
    std::filesystem::create_directories(d / "p180");
    {
        Manifest m = Manifest::ForPass(PassKind::ColorSr, 128, 72, FileFormat::Exr);
        m.sourceFormat = "yuv420p";
        PassWriter wr(o180.passesRoot / "color_sr", m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorSr, 128, 72));
        wr.Finish();
    }
    CHECK_THAT(Decision(PlanProcess(o180), "nr").error,
               ContainsSubstring("Пасс color_sr (стадия «Апскейл») сделан без учёта поворота источника, а источник повёрнут на 180°"));
}

TEST_CASE("The result is not taken from a pass of a stage that is off made from a turned source's stored frame: the plan says so, the run refuses at once", "[integration][orientation][process][regression]") {
    // review of TASK-0072: the plan was silent, the run remade every stage and only the encode refused - with advice (--force) that could not help
    const auto d = Dir("stored_result");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "upscale" || s.name == "nr";  // fg off: its old color_fg would be the result
    {
        Manifest m = Manifest::ForPass(PassKind::ColorFg, 128, 72, FileFormat::Exr);
        m.sourceFormat = "yuv420p";  // the build before: the stored frame
        PassWriter wr(o.passesRoot / "color_fg", m);
        for (int64_t f = 0; f < 5; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorFg, 128, 72));
        wr.Finish();
    }
    const ProcessPlan plan = PlanProcess(o);
    CHECK(plan.error == "Результат взялся бы из пасса color_fg (стадия «Генерация кадров»): он сделан по боковому кадру, а источник повёрнут на 90°. "
                        "Включите эту стадию или удалите пасс");
    CHECK(plan.ToJson()["error"] == plan.error);
    for (const auto& s : plan.stages) CHECK(s.error.empty());  // no stage's
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("Результат взялся бы из пасса color_fg"));
    CHECK_FALSE(std::filesystem::exists(o.passesRoot / "color_sr"));  // before any stage ran
    CHECK_FALSE(std::filesystem::exists(o.output));
    // `process --plan` says it and exits 1
    const auto out = d / "plan.txt";
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" process -i \"" + clip.string() + "\" -o \"" + o.output.string() + "\" --passes \"" + o.passesRoot.string() +
                            "\" --warp --stages upscale,nr --param upscale.backend=nis --param nr.backend=stub --plan >\"" + out.string() + "\" 2>&1\"";
    CHECK(std::system(cmd.c_str()) == 1);
    std::ifstream in(out);
    std::stringstream text;
    text << in.rdbuf();
    CHECK_THAT(text.str(), ContainsSubstring("error: Результат взялся бы из пасса color_fg"));
    // a run that encodes nothing (the stages before NR for the preview, --no-encode) takes no result: no refusal (review of TASK-0072)
    o.encode = false;
    CHECK(PlanProcess(o).error.empty());
    CHECK_NOTHROW(RunProcess(o));
    CHECK(std::filesystem::exists(o.passesRoot / "color_sr"));
    o.encode = true;
    // fg on: it remakes its pass - no error
    for (auto& s : o.stages) s.enabled = s.name == "upscale" || s.name == "nr" || s.name == "fg";
    CHECK(PlanProcess(o).error.empty());
    // nr and fg off: color_nr of the build before would be the result
    std::filesystem::remove_all(o.passesRoot / "color_fg");
    for (auto& s : o.stages) s.enabled = s.name == "upscale";
    {
        Manifest m = Manifest::ForPass(PassKind::ColorNr, 72, 128, FileFormat::Exr);  // unlabelled, the shown frame's proportions: taken
        PassWriter wr(o.passesRoot / "color_nr", m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorNr, 72, 128));
        wr.Finish();
    }
    CHECK(PlanProcess(o).error.empty());
}

TEST_CASE("Frame generation without DLSS-G is not refused over stored-frame guide passes it does not read", "[integration][orientation][process][regression]") {
    // review of TASK-0072: fg lists depth_dlss / mv_dlss whatever its backend; blend and RIFE never read them
    const auto d = Dir("fg_guides");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "fg";
    for (const auto& [pass, kind] : {std::pair<const char*, PassKind>{"depth_dlss", PassKind::DepthDlss}, {"mv_dlss", PassKind::MvDlss}}) {
        Manifest m = Manifest::ForPass(kind, 64, 36, FileFormat::Exr);  // the build before: the stored frame
        m.sourceFormat = "yuv420p";
        PassWriter wr(o.passesRoot / pass, m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(kind, 64, 36));
        wr.Finish();
    }
    ApplyParamSpec(o.stages, "fg.backend=blend");
    CHECK(Decision(PlanProcess(o), "fg").error.empty());
    ApplyParamSpec(o.stages, "fg.backend=dlssg");  // DLSS-G reads the guides: refused
    CHECK_THAT(Decision(PlanProcess(o), "fg").error, ContainsSubstring("сделан по боковому кадру"));
}

TEST_CASE("A standalone upscale pass made under the 4K cap by the axes is not adopted under the long-side cap", "[integration][orientation][upscale][regression]") {
    // review of TASK-0072: a portrait clip's pass of 2160 high (a standalone dlssvid upscale of the build before) was adopted and stamped
    // with the +sr.cap fingerprint of the 2400-high frame: the plan said 1350x2400, the result was 1216x2160 for good
    const auto d = Dir("legacy_cap");
    ClipSpec spec;
    spec.frames = 1;
    spec.width = 720;
    spec.height = 1280;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "upscale";
    ApplyParamSpec(o.stages, "upscale.target_height=2400");
    auto legacy = [&](uint32_t w, uint32_t h) {
        std::filesystem::remove_all(o.passesRoot / "color_sr");
        Manifest m = Manifest::ForPass(PassKind::ColorSr, w, h, FileFormat::Exr);
        m.sourceHash = "sha256:" + Sha256FileCached(clip);  // made from this clip, without a fingerprint
        PassWriter wr(o.passesRoot / "color_sr", m);
        wr.WriteFrame(0, MakePassImage(PassKind::ColorSr, w, h));
        wr.Finish();
    };
    legacy(1216, 2160);
    const StageDecision old = Decision(PlanProcess(o), "upscale");
    CHECK(old.action == "run");
    CHECK(old.kind == "tool_changed");
    CHECK_THAT(old.detail, ContainsSubstring("4K cap by the axes"));
    legacy(1350, 2400);  // the size of the long-side cap: adopted
    const StageDecision now = Decision(PlanProcess(o), "upscale");
    CHECK(now.action == "reuse");
    CHECK(now.kind == "legacy");
}

TEST_CASE("dlssvid scenes labels a turned clip's list as process does: the plan reuses it", "[integration][orientation][cli][scenes]") {
    const auto d = Dir("scenes_cli");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    const auto root = d / "passes";
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" scenes -i \"" + clip.string() + "\" -o \"" + root.string() + "\" >nul 2>&1\"";
    REQUIRE(std::system(cmd.c_str()) == 0);
    CHECK(Manifest::Load(root / "scenes").sourceFormat == "yuv420p+r90");
    ProcessOptions o = Options(clip, d);
    for (auto& s : o.stages) s.enabled = s.name == "scenes";
    CHECK(Decision(PlanProcess(o), "scenes").action == "reuse");
}

TEST_CASE("dlssvid info names the turn of a turned clip; an unturned clip's info is as before", "[integration][orientation][cli]") {
    const auto d = Dir("info");
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 64;
    spec.height = 36;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    auto info = [&](const std::filesystem::path& file) {
        const auto out = d / "info.txt";
        const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" info \"" + file.string() + "\" >\"" + out.string() + "\" 2>&1\"";
        REQUIRE(std::system(cmd.c_str()) == 0);
        std::ifstream in(out);
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    };
    const std::string turned = info(clip);
    CHECK_THAT(turned, ContainsSubstring("ffv1 36x64"));
    CHECK_THAT(turned, ContainsSubstring("orientation: rotated 90 deg clockwise (display matrix), stored 64x36"));
    spec.orientation = {};
    const std::string plain = info(WriteClip(d / "plain.mkv", spec));
    CHECK_THAT(plain, ContainsSubstring("ffv1 64x36"));
    CHECK(plain.find("orientation:") == std::string::npos);
}

// TASK-0072: a turned phone video is a portrait frame for the NVIDIA features - DLSS SR, Neural Rendering and DLSS Frame Generation take it
// (one feature each, the normal path): no fallback, no stage skipped, the passes and the result portrait.
TEST_CASE("DLSS SR, Neural Rendering and Frame Generation take the portrait frame of a turned clip", "[integration][orientation][gpu][ngx][dlssg]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("needs an NVIDIA GPU");
    if (const auto a = UpscalerAvailable("dlss"); !a.available) SKIP(a.reason);
    if (const auto a = NrAvailable("ngx"); !a.available) SKIP(a.reason);
    if (const auto a = FgAvailable("dlssg"); !a.available) SKIP(a.reason);
    const NvidiaDriverVersion drv = NvidiaDriverFromUmd(dev.UmdDriverVersion());
    if (drv.valid && !drv.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor)) SKIP("driver " + drv.ToString() + " is older than 616.56");
    const auto d = Dir("ngx");
    ClipSpec spec;
    spec.frames = 6;
    spec.width = 640;
    spec.height = 360;
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ProcessOptions o = Options(clip, d);
    o.warp = false;
    for (auto& s : o.stages) s.enabled = s.name == "upscale" || s.name == "nr" || s.name == "fg";
    for (const char* p : {"upscale.backend=dlss", "upscale.no_fallback=true", "nr.backend=ngx", "fg.backend=dlssg", "fg.multiplier=2"}) ApplyParamSpec(o.stages, p);
    ProcessResult r;
    try {
        r = RunProcess(o);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("FAIL_FeatureNotSupported") != std::string::npos) SKIP("the NR DLL refuses this GPU - patch it with `dlssvid nr-patch` (" + what + ")");
        if (what.find("not available on this GPU") != std::string::npos) SKIP(what);
        throw;
    }
    CHECK(Report(r, "upscale").status == "ran");
    CHECK_FALSE(Report(r, "upscale").details.value("fallback", false));
    CHECK(Report(r, "nr").status == "ran");
    CHECK(Report(r, "fg").status == "ran");
    CHECK(r.warnings.empty());
    for (const char* pass : {"color_sr", "color_nr", "color_fg"}) {
        INFO(pass);
        const Manifest m = Manifest::Load(d / "passes" / pass);
        CHECK(m.width == 720);
        CHECK(m.height == 1280);
    }
    CHECK(r.outputFormat.width == 720);
    CHECK(r.outputFormat.height == 1280);
}
