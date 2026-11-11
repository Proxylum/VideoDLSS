// TASK-0073: a source of non-square pixels (DVD 720x480 at 32:27 is shown 16:9) - the decoder reads the proportions, the stages work on the
// stored grid, `process` stretches its result to square pixels, passthrough keeps the grid and writes the proportions, the targets and the
// 4K cap are of the frame as shown, the viewport shows it stretched. Clips: 54x36 at 32:27 (shown 64x36), 72x32 at 8:9 (shown 72x36).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
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
#include "io/VideoEncoder.h"
#include "passes/Manifest.h"
#include "passes/PassImage.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "pipeline/ControlMask.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/ProcessRunner.h"
#include "stages/fg/IFrameGenerator.h"
#include "util/Sha256.h"
#include "util/Subprocess.h"
#include "viewport/FrameStore.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

const PixelAspect kDvdWide{32, 27};

std::filesystem::path Dir(const std::string& name) {
    const auto d = TempDir() / "aspect" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// A dark frame with a bright rectangle of rw x rh stored samples, centred: 27 x 32 of a 54x36 grid at 32:27 is a 32 x 32 square as shown
CpuFrame Box(int64_t index, uint32_t w, uint32_t h, uint32_t rw, uint32_t rh) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p});
    f.index = index;
    f.pts = index;
    const uint32_t x0 = (w - rw) / 2, y0 = (h - rh) / 2;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) f.Plane(0)[y * w + x] = x >= x0 && x < x0 + rw && y >= y0 && y < y0 + rh ? 235 : 16;
    std::fill(f.Plane(1), f.Plane(1) + f.desc.PlaneWidth(1) * f.desc.PlaneHeight(1), static_cast<uint8_t>(128));
    std::fill(f.Plane(2), f.Plane(2) + f.desc.PlaneWidth(2) * f.desc.PlaneHeight(2), static_cast<uint8_t>(128));
    return f;
}

// The bright run through the middle row / column of a decoded frame: the rectangle's size in it
std::pair<uint32_t, uint32_t> BrightSize(const CpuFrame& f) {
    const uint32_t w = f.desc.width, h = f.desc.height;
    const size_t pitch = f.desc.PlaneWidth(0);
    auto luma = [&](uint32_t x, uint32_t y) -> int {
        return f.desc.HighBitDepth() ? f.Plane16(0)[y * f.desc.PlaneCols(0) + x] >> (f.desc.bitDepth - 8) : f.Plane(0)[y * pitch + x];
    };
    uint32_t bw = 0, bh = 0;
    for (uint32_t x = 0; x < w; ++x) bw += luma(x, h / 2) > 125 ? 1 : 0;
    for (uint32_t y = 0; y < h; ++y) bh += luma(w / 2, y) > 125 ? 1 : 0;
    return {bw, bh};
}

ClipSpec Anamorphic(uint32_t w = 54, uint32_t h = 36, PixelAspect sar = kDvdWide, int frames = 3) {
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    spec.sar = sar;
    return spec;
}

std::vector<ProcessStage> StubStages() {
    std::vector<ProcessStage> s = DefaultProcessStages();
    for (const char* spec : {"depth.backend=stub", "flow.backend=stub", "upscale.backend=nis", "upscale.scale=2", "nr.backend=stub", "fg.backend=blend",
                             "fg.multiplier=2"})
        ApplyParamSpec(s, spec);
    return s;
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

void Only(ProcessOptions& o, std::initializer_list<const char*> names) {
    for (auto& s : o.stages) {
        s.enabled = false;
        for (const char* n : names) s.enabled = s.enabled || s.name == n;
    }
}

StageDecision Decision(const ProcessPlan& p, const std::string& name) {
    for (const auto& s : p.stages)
        if (s.name == name) return s;
    throw std::runtime_error("no decision for " + name);
}

std::string App(const StageDecision& d) { return d.tool.is_object() ? d.tool.value("app", std::string()) : std::string(); }

std::string Warnings(const StageDecision& d) {
    std::string all;
    for (const auto& w : d.warnings) all += w + "\n";
    return all;
}

bool Same(const CpuFrame& a, const CpuFrame& b) { return a.desc == b.desc && a.data == b.data; }

}  // namespace

TEST_CASE("VideoDecoder reads the pixels' proportions: the container's, else the bitstream's; the frames stay the stored grid", "[integration][aspect]") {
    const auto d = Dir("decoder");
    ClipSpec spec = Anamorphic();
    const auto all = WriteClip(d / "all.mkv", spec);
    {
        VideoDecoder dec(all);
        const VideoStreamInfo& i = dec.Info();
        CHECK(i.width == 54);
        CHECK(i.height == 36);
        CHECK(i.codedSar == kDvdWide);
        CHECK(i.sar == kDvdWide);
        CHECK(i.sarFrom == "container");
        CpuFrame f;
        for (int64_t n = 0; n < 3; ++n) {
            REQUIRE(dec.NextFrame(f));
            CHECK(Same(f, SyntheticFrame(n, 54, 36)));  // the grid as stored: nothing resampled
        }
    }
    spec.sarIn = "container";  // MKV's DisplayWidth only
    {
        VideoDecoder dec(WriteClip(d / "container.mkv", spec), VideoDecoder::Probe());
        CHECK(dec.Info().sar == kDvdWide);
        CHECK(dec.Info().sarFrom == "container");
    }
    spec.sarIn = "all";
    spec.sar = {};  // square: as before
    {
        VideoDecoder dec(WriteClip(d / "square.mkv", spec), VideoDecoder::Probe());
        CHECK(dec.Info().sar.Square());
        CHECK(dec.Info().codedSar.Square());
        CHECK(dec.Info().sarFrom.empty());
    }
    spec.sar = {1000, 1};  // a broken file: not plausible - square
    spec.sarIn = "container";
    {
        VideoDecoder dec(WriteClip(d / "broken.mkv", spec), VideoDecoder::Probe());
        CHECK(dec.Info().sar.Square());
    }
    // the bitstream alone (a raw MPEG-4 stream: no container field)
    if (VideoEncoder::EncoderAvailable("mpeg4")) {
        ClipSpec raw = Anamorphic(64, 48);
        raw.codec = "mpeg4";
        raw.sarIn = "codec";
        raw.format = "m4v";  // the raw muxer: no container field at all (by the extension FFmpeg writes an MP4 with pasp)
        const auto file = WriteClip(d / "raw.m4v", raw);
        CHECK(Summarize(file).streamSar.Square());
        VideoDecoder dec(file, VideoDecoder::Probe());
        CHECK(dec.Info().sar == kDvdWide);
        CHECK(dec.Info().sarFrom == "bitstream");
    }
    // MOV's pasp
    if (VideoEncoder::EncoderAvailable("prores_ks")) {
        ClipSpec mov = Anamorphic(64, 36);
        mov.codec = "prores_ks";
        mov.pixFmt = "yuv422p10le";
        VideoDecoder dec(WriteClip(d / "pasp.mov", mov), VideoDecoder::Probe());
        CHECK(dec.Info().sar == kDvdWide);
        CHECK(dec.Info().sarFrom == "container");
    }
}

TEST_CASE("process stretches an anamorphic clip's result to square pixels: a square stays square", "[integration][aspect][process][regression]") {
    // before TASK-0073: 54x36 at 32:27 x2 gave 108x72 without the proportions - players showed it 3:2, the square 54 wide
    const auto d = Dir("process");
    const ClipSpec spec = Anamorphic();
    const auto clip = WriteClipWith(d / "clip.mkv", spec, [](int64_t i) { return Box(i, 54, 36, 27, 32); });
    const ProcessResult r = RunProcess(Options(clip, d));
    CHECK(r.outputFormat.width == 128);
    CHECK(r.outputFormat.height == 72);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 128);
    CHECK(s.height == 72);
    CHECK(s.sar.Square());  // square pixels: written as such
    CHECK(CheckEncodedResult(d / "result.mkv", ResultExpectation{128, 72, -1, false}).empty());
    const auto frames = DecodeAll(d / "result.mkv");
    REQUIRE(!frames.empty());
    const auto [bw, bh] = BrightSize(frames.front());
    CHECK(bw >= 62);
    CHECK(bw <= 66);
    CHECK(bh >= 62);
    CHECK(bh <= 66);
    // the stages made the grid: scale x2 of the stored pixels
    const Manifest sr = Manifest::Load(d / "passes" / "color_sr");
    CHECK(sr.width == 108);
    CHECK(sr.height == 72);
    CHECK(sr.stageParams.value("pixel_aspect", std::string()) == "32:27");
    // the scale alone keeps the fingerprint of the build before: no +sr.sar, the rerun reuses every pass
    for (const auto& st : PlanProcess(Options(clip, d)).stages) {
        INFO(st.name);
        CHECK(st.action == "reuse");
        CHECK(App(st).find("sr.sar") == std::string::npos);
    }
    // the settings note says the source's pixels
    CHECK_THAT(s.comment, ContainsSubstring("| source: sar=32:27 |"));
}

TEST_CASE("process on narrower pixels makes the frame higher, nothing narrowed: 72x32 at 8:9 x2 is 144x72", "[integration][aspect][process]") {
    const auto d = Dir("narrow");
    const ClipSpec spec = Anamorphic(72, 32, PixelAspect{8, 9});
    const auto clip = WriteClipWith(d / "clip.mkv", spec, [](int64_t i) { return Box(i, 72, 32, 32, 16); });  // 32 x 18 as shown: 16:9
    ProcessOptions o = Options(clip, d);
    Only(o, {"upscale"});
    RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 144);
    CHECK(s.height == 72);
    CHECK(s.sar.Square());
    const auto frames = DecodeAll(d / "result.mkv");
    REQUIRE(!frames.empty());
    const auto [bw, bh] = BrightSize(frames.front());
    CHECK(bw >= 62);
    CHECK(bw <= 66);
    CHECK(bh >= 34);
    CHECK(bh <= 38);
    const Manifest sr = Manifest::Load(d / "passes" / "color_sr");
    CHECK(sr.width == 144);
    CHECK(sr.height == 64);
}

TEST_CASE("A colour pass already of the shown proportions is not stretched again when encoded", "[integration][aspect][process]") {
    // an explicit target of the build before was taken in square pixels: color_sr 128x72 of a 54x36 clip at 32:27 - stretched it would be 152x72
    const auto d = Dir("shown_pass");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    ProcessOptions o = Options(clip, d);
    Only(o, {"depth"});  // the colour stages off: the result is taken from the pass on disk
    Manifest m = Manifest::ForPass(PassKind::ColorSr, 128, 72, FileFormat::Exr);
    m.sourceHash = "sha256:" + Sha256FileCached(clip);
    PassWriter wr(o.passesRoot / "color_sr", m);
    for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorSr, 128, 72));
    wr.Finish();
    RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 128);
    CHECK(s.height == 72);
    CHECK(s.sar.Square());
    // a pass of the grid's proportions is stretched
    std::filesystem::remove_all(o.passesRoot / "color_sr");
    Manifest g = Manifest::ForPass(PassKind::ColorSr, 108, 72, FileFormat::Exr);
    g.sourceHash = m.sourceHash;
    PassWriter gw(o.passesRoot / "color_sr", g);
    for (int64_t f = 0; f < 3; ++f) gw.WriteFrame(f, MakePassImage(PassKind::ColorSr, 108, 72));
    gw.Finish();
    RunProcess(o);
    CHECK(Summarize(d / "result.mkv").width == 128);
}

TEST_CASE("An anamorphic stretch beyond 4K is fit into 3840x2160: 1000x540 at 4:1 through NR alone is 3840x518", "[integration][aspect][process][regression]") {
    // review of TASK-0073: a grid with no upscale to cap it was stretched past 3840 (an anamorphic 4K at 4:3 to 5120x2160 - H.264 NVENC
    // refuses it after the whole run)
    const auto d = Dir("stretch_cap");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic(1000, 540, PixelAspect{4, 1}, 2));
    ProcessOptions o = Options(clip, d);
    Only(o, {"nr"});
    const ProcessResult r = RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 3840);
    CHECK(s.height == 518);
    CHECK(s.sar.Square());
    CHECK(r.outputFormat.width == 3840);
}

TEST_CASE("An anamorphic upscale grid of the source's shown proportions is still stretched: the pass says its pixels", "[integration][aspect][process][regression]") {
    // review of TASK-0073: HDV-like 72x54 at 4:3 (shown 96x54) with the target 128x54 makes the grid 96x54 - the proportions of the source as
    // shown, which the guess for the passes of the build before took for square pixels: 96x54 was written, not 128x54
    const auto d = Dir("stamped");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic(72, 54, PixelAspect{4, 3}, 2));
    ProcessOptions o = Options(clip, d);
    Only(o, {"upscale", "nr"});
    ApplyParamSpec(o.stages, "upscale.target_width=128");
    ApplyParamSpec(o.stages, "upscale.target_height=54");
    RunProcess(o);
    CHECK(Manifest::Load(d / "passes" / "color_sr").width == 96);
    CHECK(Manifest::Load(d / "passes" / "color_nr").stageParams.value("pixel_aspect", std::string()) == "4:3");
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 128);
    CHECK(s.height == 54);
}

TEST_CASE("NR and FG of an anamorphic source take the pixels of the colour they read: a square pass of the build before stays square", "[integration][aspect][process][regression]") {
    // review of TASK-0073: FG took the source's pixels for a color_sr that an explicit target of the build before had stretched already -
    // DLSS-G's camera stretched it a second time and +fg.sar recomputed a right pass into a wrong one
    const auto d = Dir("colour_pixels");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    ProcessOptions o = Options(clip, d);
    Only(o, {"nr", "fg"});  // fg: blend
    auto colourPass = [&](uint32_t w, uint32_t h, const nlohmann::json& stageParams) {
        std::filesystem::remove_all(o.passesRoot / "color_sr");
        Manifest m = Manifest::ForPass(PassKind::ColorSr, w, h, FileFormat::Exr);
        m.sourceHash = "sha256:" + Sha256FileCached(clip);
        m.stageParams = stageParams;
        PassWriter wr(o.passesRoot / "color_sr", m);
        for (int64_t f = 0; f < 3; ++f) wr.WriteFrame(f, MakePassImage(PassKind::ColorSr, w, h));
        wr.Finish();
    };
    auto recorded = [&](const char* pass) { return Manifest::Load(o.passesRoot / pass).stageParams.value("pixel_aspect", std::string()); };
    ProcessOptions g = Options(clip, d);
    Only(g, {"fg"});
    ApplyParamSpec(g.stages, "fg.backend=dlssg");
    colourPass(128, 72, {{"backend", "nis"}});  // an explicit target of the build before: square pixels, no record
    RunProcess(o);
    CHECK(recorded("color_nr") == "1:1");
    CHECK(recorded("color_fg") == "1:1");
    CHECK(Summarize(d / "result.mkv").width == 128);
    CHECK(App(Decision(PlanProcess(g), "fg")).find("fg.sar") == std::string::npos);  // DLSS-G on square pixels: its camera as before
    // ... with color_nr remade in the same run from that square color_sr: square too, not the grid (review of TASK-0073)
    ProcessOptions remade = o;
    ApplyParamSpec(remade.stages, "fg.backend=dlssg");
    remade.forceStages = {"nr"};
    CHECK(App(Decision(PlanProcess(remade), "fg")).find("fg.sar") == std::string::npos);
    // ... but with color_sr remade too (upscale on): the grid
    ProcessOptions all = remade;
    for (auto& s : all.stages) s.enabled = s.enabled || s.name == "upscale";
    all.forceStages = {"upscale", "nr"};
    CHECK_THAT(App(Decision(PlanProcess(all), "fg")), ContainsSubstring("+fg.sar"));
    colourPass(108, 72, {{"backend", "nis"}, {"pixel_aspect", "32:27"}});  // the grid of this build
    // planned with NR remade: FG's colour is the pass this run makes (the grid), not the square color_nr still on disk
    ProcessOptions both = o;
    ApplyParamSpec(both.stages, "fg.backend=dlssg");
    both.forceStages = {"nr"};
    CHECK_THAT(App(Decision(PlanProcess(both), "fg")), ContainsSubstring("+fg.sar"));
    o.forceStages = {"nr", "fg"};  // both colour passes are hand-made without fingerprints: the same pseudo-fingerprint to the plan
    RunProcess(o);
    CHECK(recorded("color_nr") == "32:27");
    CHECK(recorded("color_fg") == "32:27");
    CHECK(Summarize(d / "result.mkv").width == 128);
    CHECK_THAT(App(Decision(PlanProcess(g), "fg")), ContainsSubstring("+fg.sar"));
}

TEST_CASE("The stages' preview videos of an anamorphic clip carry its pixels' proportions", "[integration][aspect][cli][regression]") {
    // review of TASK-0073: dlssvid nr --video wrote the grid without them - shown squeezed, while upscale's and fg's played 16:9
    const auto d = Dir("previews");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    const auto passes = d / "passes";
    auto cli = [&](const std::string& stage, const std::string& backend, const char* video) {
        std::string out;
        return Subprocess::Run({DLSSVID_CLI_PATH, stage, "-i", clip.string(), "-o", passes.string(), "--backend", backend, "--warp", "--video", (d / video).string(),
                                "--codec", "ffv1"},
                               &out);
    };
    REQUIRE(cli("upscale", "nis", "sr.mkv") == 0);
    REQUIRE(cli("nr", "stub", "nr.mkv") == 0);  // on color_sr
    REQUIRE(cli("fg", "blend", "fg.mkv") == 0);  // on color_nr
    for (const char* video : {"sr.mkv", "nr.mkv", "fg.mkv"}) {
        INFO(video);
        CHECK(Summarize(d / video).sar == kDvdWide);
    }
    CHECK(Manifest::Load(passes / "color_nr").stageParams.value("pixel_aspect", std::string()) == "32:27");
}

TEST_CASE("Passthrough keeps an anamorphic clip's grid bit for bit and writes its pixels' proportions", "[integration][aspect][passthrough][regression]") {
    // before TASK-0073: the result had none - shown 3:2 instead of 16:9
    const auto d = Dir("passthrough");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    ProcessOptions o = Options(clip, d);
    o.passthrough = true;
    RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 54);
    CHECK(s.height == 36);
    CHECK(s.streamSar == kDvdWide);
    CHECK(s.sar == kDvdWide);
    const auto a = DecodeAll(d / "result.mkv"), b = DecodeAll(clip);
    REQUIRE(a.size() == b.size());
    bool all = true;
    for (size_t i = 0; i < a.size(); ++i) all = all && Same(a[i], b[i]);
    CHECK(all);
    // MOV: pasp
    if (VideoEncoder::EncoderAvailable("prores_ks")) {
        ProcessOptions m = Options(clip, d, "result.mov");
        m.passthrough = true;
        m.codec = "prores_ks";
        RunProcess(m);
        CHECK(Summarize(d / "result.mov").sar == kDvdWide);
    }
    // no colour pass at all (every colour stage off): the source passes through, its proportions with it
    const auto e = Dir("passthrough_fallback");
    {
        ProcessOptions none = Options(clip, e);
        Only(none, {"depth"});
        CHECK(PlanProcess(none).finalPass.empty());
    }
    ProcessOptions f = Options(clip, e);
    Only(f, {"depth"});
    RunProcess(f);
    CHECK(Summarize(e / "result.mkv").sar == kDvdWide);
    CHECK(Summarize(e / "result.mkv").width == 54);
}

TEST_CASE("The result check compares the pixels' proportions written with those in the file", "[integration][aspect][result][regression]") {
    const auto d = Dir("check");
    const auto wide = WriteClip(d / "wide.mkv", Anamorphic());
    ClipSpec plain = Anamorphic();
    plain.sar = {};
    const auto square = WriteClip(d / "square.mkv", plain);
    CHECK_THAT(CheckEncodedResult(wide, ResultExpectation{54, 36, 3, false}), ContainsSubstring("pixel aspect of 32:27 (square pixels were written)"));
    CHECK_THAT(CheckEncodedResult(square, ResultExpectation{54, 36, 3, false, kDvdWide}), ContainsSubstring("its pixels are square, 32:27 was written"));
    CHECK(CheckEncodedResult(wide, ResultExpectation{54, 36, 3, false, kDvdWide}).empty());
    CHECK(CheckEncodedResult(square, ResultExpectation{54, 36, 3, false}).empty());
    // a muxer that keeps the frame as shown in whole pixels: WebM's DisplayWidth 853 of 720 at 32:27 reads back 853:720 - the same picture;
    // 856:720 is 3 pixels off (review of TASK-0073: the passthrough of a .webm failed at the end)
    ClipSpec rounded = Anamorphic(720, 48, PixelAspect{853, 720});
    rounded.sarIn = "container";
    const auto webmLike = WriteClip(d / "rounded.mkv", rounded);
    REQUIRE(Summarize(webmLike).sar == PixelAspect{853, 720});
    CHECK(CheckEncodedResult(webmLike, ResultExpectation{720, 48, 3, false, kDvdWide}).empty());
    rounded.sar = PixelAspect{856, 720};
    CHECK_THAT(CheckEncodedResult(WriteClip(d / "off.mkv", rounded), ResultExpectation{720, 48, 3, false, kDvdWide}),
               ContainsSubstring("its pixels are 107:90, 32:27 was written"));
}

TEST_CASE("A turned anamorphic clip: the proportions turn with the frame; passthrough and process as shown", "[integration][aspect][orientation]") {
    const auto d = Dir("turned");
    ClipSpec spec = Anamorphic();
    spec.orientation = {90, false};
    const auto clip = WriteClip(d / "clip.mkv", spec);
    {
        VideoDecoder dec(clip, VideoDecoder::Probe());
        CHECK(dec.Info().width == 36);
        CHECK(dec.Info().height == 54);
        CHECK(dec.Info().codedSar == kDvdWide);
        CHECK(dec.Info().sar == PixelAspect{27, 32});
    }
    ProcessOptions p = Options(clip, d, "pass.mkv");
    p.passthrough = true;
    RunProcess(p);
    const StreamSummary ps = Summarize(d / "pass.mkv");
    CHECK(ps.width == 36);
    CHECK(ps.height == 54);
    CHECK(ps.sar == PixelAspect{27, 32});
    CHECK(VideoDecoder(d / "pass.mkv", VideoDecoder::Probe()).Info().orientation.Identity());
    ProcessOptions o = Options(clip, d);
    Only(o, {"upscale"});
    RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mkv");
    CHECK(s.width == 72);  // 36 x 64 as shown, x2
    CHECK(s.height == 128);
    CHECK(s.sar.Square());
    CHECK_THAT(s.comment, ContainsSubstring("| source: rotate=90 sar=32:27 |"));
}

TEST_CASE("An anamorphic source's target and cap are of the frame as shown: only the passes that change are recomputed", "[integration][aspect][process][fingerprint]") {
    const auto d = Dir("fingerprints");
    const ClipSpec spec = Anamorphic();
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ClipSpec plain = spec;
    plain.sar = {};
    const auto twin = WriteClip(d / "twin.mkv", plain);
    // the passes of the build before: the same frames read as square ones, under the anamorphic clip's hash
    ProcessOptions before = Options(twin, d);
    before.sourceHash = "sha256:" + Sha256FileCached(clip);
    Only(before, {"upscale"});
    RunProcess(before);
    ProcessOptions o = Options(clip, d);
    Only(o, {"upscale"});
    const StageDecision byScale = Decision(PlanProcess(o), "upscale");
    CHECK(byScale.action == "reuse");  // its pixels are the same
    CHECK(App(byScale).find("sr.sar") == std::string::npos);
    ApplyParamSpec(o.stages, "upscale.target_width=128");
    ApplyParamSpec(o.stages, "upscale.target_height=72");
    ApplyParamSpec(before.stages, "upscale.target_width=128");
    ApplyParamSpec(before.stages, "upscale.target_height=72");
    RunProcess(before);
    const StageDecision byTarget = Decision(PlanProcess(o), "upscale");
    CHECK(byTarget.action == "run");
    CHECK(byTarget.kind == "tool_changed");
    CHECK_THAT(App(byTarget), ContainsSubstring("+sr.sar"));
    CHECK(App(Decision(PlanProcess(before), "upscale")).find("sr.sar") == std::string::npos);  // a square source: as before
    RunProcess(o);
    CHECK(Manifest::Load(d / "passes" / "color_sr").width == 108);  // 128x72 as shown: 108x72 of the stored pixels
    CHECK(Summarize(d / "result.mkv").width == 128);
    // DLSS-G's camera takes the shown proportions: another picture
    ProcessOptions g = Options(clip, d);
    Only(g, {"fg"});
    ApplyParamSpec(g.stages, "fg.backend=dlssg");
    CHECK_THAT(App(Decision(PlanProcess(g), "fg")), ContainsSubstring("+fg.sar"));
    ProcessOptions gs = Options(twin, d);
    Only(gs, {"fg"});
    ApplyParamSpec(gs.stages, "fg.backend=dlssg");
    CHECK(App(Decision(PlanProcess(gs), "fg")).find("fg.sar") == std::string::npos);
    ApplyParamSpec(g.stages, "fg.backend=blend");  // no camera
    CHECK(App(Decision(PlanProcess(g), "fg")).find("fg.sar") == std::string::npos);
}

TEST_CASE("A standalone upscale or DLSS-G pass made before the pixels' proportions were applied is not adopted", "[integration][aspect][process][regression]") {
    const auto d = Dir("legacy");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    ProcessOptions o = Options(clip, d);
    Only(o, {"upscale"});
    ApplyParamSpec(o.stages, "upscale.target_width=128");
    ApplyParamSpec(o.stages, "upscale.target_height=72");
    auto legacy = [&](const char* pass, PassKind kind, uint32_t w, uint32_t h, int64_t frames, const nlohmann::json& stageParams) {
        std::filesystem::remove_all(o.passesRoot / pass);
        Manifest m = Manifest::ForPass(kind, w, h, FileFormat::Exr);
        m.sourceHash = "sha256:" + Sha256FileCached(clip);  // made from this clip, without a fingerprint
        m.stageParams = stageParams;
        PassWriter wr(o.passesRoot / pass, m);
        for (int64_t f = 0; f < frames; ++f) wr.WriteFrame(f, MakePassImage(kind, w, h));
        wr.Finish();
    };
    legacy("color_sr", PassKind::ColorSr, 128, 72, 3, {{"backend", "nis"}});  // the target in the stored pixels
    const StageDecision old = Decision(PlanProcess(o), "upscale");
    CHECK(old.action == "run");
    CHECK(old.kind == "tool_changed");
    CHECK_THAT(old.detail, ContainsSubstring("target in the source's pixels, before TASK-0073"));
    legacy("color_sr", PassKind::ColorSr, 108, 72, 3, {{"backend", "nis"}, {"pixel_aspect", "32:27"}});  // a standalone pass of this build
    const StageDecision now = Decision(PlanProcess(o), "upscale");
    CHECK(now.action == "reuse");
    CHECK(now.kind == "legacy");
    // DLSS-G
    ProcessOptions g = Options(clip, d);
    Only(g, {"fg"});
    ApplyParamSpec(g.stages, "fg.backend=dlssg");
    legacy("color_fg", PassKind::ColorFg, 54, 36, FgFrameCount(3, 2), {{"backend", "dlssg"}});
    const StageDecision fgOld = Decision(PlanProcess(g), "fg");
    CHECK(fgOld.kind == "tool_changed");
    CHECK_THAT(fgOld.detail, ContainsSubstring("dlssg made with the camera of the stored grid's proportions"));
    legacy("color_fg", PassKind::ColorFg, 54, 36, FgFrameCount(3, 2), {{"backend", "dlssg"}, {"pixel_aspect", "32:27"}});
    CHECK(Decision(PlanProcess(g), "fg").kind != "tool_changed");
    legacy("color_fg", PassKind::ColorFg, 54, 36, FgFrameCount(3, 2), {{"backend", "rife"}});  // no camera: not this case
    CHECK(Decision(PlanProcess(g), "fg").detail.find("camera") == std::string::npos);
}

TEST_CASE("The plan's advice on an anamorphic source: masks of the shown frame fit, a target of the grid's proportions distorts", "[integration][aspect][process][mask]") {
    const auto d = Dir("advice");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    ProcessOptions o = Options(clip, d);
    Only(o, {"nr"});
    auto mask = [&](uint32_t w, uint32_t h) {
        PassImage m;
        m.Allocate(w, h, PixelType::U16, {"A"});
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) m.Set(x, y, 0, 30000.f);
        const auto file = d / ("mask_" + std::to_string(w) + "x" + std::to_string(h) + ".png");
        WritePng(file, m);
        ImportControlMask(file, o.passesRoot);
        return Warnings(Decision(PlanProcess(o), "nr"));
    };
    CHECK(mask(64, 36).find("растянется") == std::string::npos);  // the frame as shown: before TASK-0073 a warning
    CHECK(mask(54, 36).find("растянется") == std::string::npos);  // the grid
    CHECK_THAT(mask(100, 36), ContainsSubstring("100×36 растянется на кадр 64×36"));
    ProcessOptions u = Options(clip, d);
    Only(u, {"upscale"});
    ApplyParamSpec(u.stages, "upscale.target_width=108");
    ApplyParamSpec(u.stages, "upscale.target_height=72");
    CHECK_THAT(Warnings(Decision(PlanProcess(u), "upscale")), ContainsSubstring("пропорции кадра в пикселях источника, а у источника пиксели 32:27"));
    ApplyParamSpec(u.stages, "upscale.target_width=128");
    CHECK(Warnings(Decision(PlanProcess(u), "upscale")).find("пикселях источника") == std::string::npos);
    // turned by 90: a target of the stored frame as players show it unturned (64x36 of 54x36 at 32:27) squeezes the portrait (review)
    ClipSpec turnedSpec = Anamorphic();
    turnedSpec.orientation = {90, false};
    const auto turned = WriteClip(d / "turned.mkv", turnedSpec);
    std::filesystem::create_directories(d / "t");
    ProcessOptions t = Options(turned, d / "t");
    Only(t, {"upscale"});
    ApplyParamSpec(t.stages, "upscale.target_width=128");
    ApplyParamSpec(t.stages, "upscale.target_height=72");
    CHECK_THAT(Warnings(Decision(PlanProcess(t), "upscale")), ContainsSubstring("128×72 — пропорции бокового кадра, а источник показывается повёрнутым (36×64)"));
    ApplyParamSpec(t.stages, "upscale.target_width=72");  // the portrait as shown
    ApplyParamSpec(t.stages, "upscale.target_height=128");
    CHECK(Warnings(Decision(PlanProcess(t), "upscale")).find("бокового") == std::string::npos);
}

TEST_CASE("The viewport shows an anamorphic clip stretched: the reference image is the frame as shown", "[integration][aspect][viewport]") {
    const auto d = Dir("viewport");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic());
    const auto src = FrameStore::VideoSource("source", clip);
    REQUIRE(src);
    CHECK(src->width == 54);
    CHECK(src->sar == kDvdWide);
    D3D12Device dev({true, false});
    FrameStore store(dev);
    store.SetSources({*src});
    CHECK(store.ImageWidth() == 64);  // before TASK-0073: 54
    CHECK(store.ImageHeight() == 36);
    // the pass folders carry the pixels of their grid: a colour pass by its record, depth by its proportions, a mask drawn over the frame
    // as shown square (review of TASK-0073: their labels named the grid next to the source as shown)
    auto pass = [&](const char* name, PassKind kind, uint32_t w, uint32_t h, const nlohmann::json& stageParams) {
        Manifest m = Manifest::ForPass(kind, w, h, kind == PassKind::Mask ? FileFormat::Png : FileFormat::Exr);
        m.stageParams = stageParams;
        PassWriter wr(d / "passes" / name, m);
        wr.WriteFrame(0, MakePassImage(kind, w, h, kind == PassKind::Mask ? std::optional<PixelType>(PixelType::U8) : std::nullopt));
        wr.Finish();
        const auto s = FrameStore::PassFolderSource(d / "passes" / name, name, name);
        REQUIRE(s);
        return *s;
    };
    const ViewportSource sr = pass("color_sr", PassKind::ColorSr, 108, 72, {{"pixel_aspect", "32:27"}});
    const ViewportSource depth = pass("depth_raw", PassKind::DepthRaw, 54, 36, nlohmann::json::object());
    const ViewportSource mask = pass("mask_ui", PassKind::Mask, 64, 36, nlohmann::json::object());
    store.SetSources({*src, sr, depth, mask});
    CHECK(store.FindSource("color_sr")->sar == kDvdWide);
    CHECK(store.FindSource("depth_raw")->sar == kDvdWide);
    CHECK(store.FindSource("mask_ui")->sar.num == 1);  // square, known
    CHECK(store.FindSource("mask_ui")->sar.Square());
    // no source video: the first pass as shown by its record
    store.SetSources({sr, depth});
    CHECK(store.ImageWidth() == 128);
    CHECK(store.ImageHeight() == 72);
}

TEST_CASE("dlssvid info names an anamorphic clip's pixels; a square clip's info is as before", "[integration][aspect][cli]") {
    const auto d = Dir("info");
    auto info = [&](const std::filesystem::path& file) {
        const auto out = d / "info.txt";
        const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" info \"" + file.string() + "\" >\"" + out.string() + "\" 2>&1\"";
        REQUIRE(std::system(cmd.c_str()) == 0);
        std::ifstream in(out);
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    };
    CHECK_THAT(info(WriteClip(d / "wide.mkv", Anamorphic())), ContainsSubstring("pixels:      32:27 (container), shown 64x36"));
    ClipSpec turned = Anamorphic();
    turned.orientation = {90, false};
    CHECK_THAT(info(WriteClip(d / "turned.mkv", turned)), ContainsSubstring("pixels:      32:27 (container), shown 36x64"));  // stored axes, as ffprobe
    ClipSpec plain = Anamorphic();
    plain.sar = {};
    CHECK(info(WriteClip(d / "plain.mkv", plain)).find("pixels:") == std::string::npos);
}

// TASK-0073 on the GPU: NVENC writes the proportions (MP4 pasp and the VUI) in passthrough, and the result of process is square.
TEST_CASE("hevc_nvenc keeps an anamorphic clip's proportions in passthrough; process writes square pixels", "[integration][aspect][gpu]") {
    D3D12Device dev({false, false});
    if (dev.IsWarp() || !dev.IsNvidia() || !VideoEncoder::EncoderAvailable("hevc_nvenc")) SKIP("no NVIDIA GPU / hevc_nvenc");
    const auto d = Dir("nvenc");
    const auto clip = WriteClip(d / "clip.mkv", Anamorphic(256, 144, kDvdWide, 3));
    ProcessOptions p = Options(clip, d, "pass.mp4");
    p.passthrough = true;
    p.codec = "hevc_nvenc";
    RunProcess(p);
    const StreamSummary ps = Summarize(d / "pass.mp4");
    CHECK(ps.streamSar == kDvdWide);  // pasp
    CHECK(ps.codecSar == kDvdWide);   // the VUI
    ProcessOptions o = Options(clip, d, "result.mp4");
    Only(o, {"upscale"});
    o.codec = "hevc_nvenc";
    RunProcess(o);
    const StreamSummary s = Summarize(d / "result.mp4");
    CHECK(s.width == 606);  // 256x144 at 32:27 is 304x144 as shown, x2 (the grid 512x288)
    CHECK(s.height == 288);
    CHECK(s.sar.Square());
    // DLSS-G takes the camera of the frame as shown and says so in its pass
    if (const auto a = FgAvailable("dlssg"); !a.available) SKIP(a.reason);
    ProcessOptions g = Options(clip, d, "fg.mkv");
    Only(g, {"fg"});
    ApplyParamSpec(g.stages, "fg.backend=dlssg");
    g.warp = false;
    RunProcess(g);
    const nlohmann::json fg = Manifest::Load(d / "passes" / "color_fg").stageParams;
    CHECK(fg.value("pixel_aspect", std::string()) == "32:27");
    CHECK(std::fabs(fg.value("camera_aspect", 0.0) / (304.0 / 144.0) - 1.0) < 0.01);  // 16:9-ish as shown, not 16:9 of the grid (1.78)
}
