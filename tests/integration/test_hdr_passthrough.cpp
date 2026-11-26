// TASK-0079 (HDR step TB-2): an HDR / wide-gamut source passed through as it is - its effective tags (filled where unsaid), its HDR
// records within the container, notes of what is lost -, its processing stopped before anything is hashed, a part of other colour
// warned of, an SDR source on the legacy path as before. CPU cases on FFV1 / ProRes (WARP for the passthrough's GPU stage).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "TestClips.h"
#include "convert/ColorPlan.h"
#include "gpu/D3D12Device.h"
#include "io/ResultFile.h"
#include "io/SourceColor.h"
#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "pipeline/ProcessRunner.h"
#include "pipeline/ResultColour.h"
#include "viewport/Project.h"
#include "viewport/ViewportProxy.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "hdr_pt" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

struct CliRun {
    int code = 0;
    std::string out;
};

CliRun RunAll(const std::vector<std::string>& args, const std::filesystem::path& log) {
    std::string cmd = "\"\"" DLSSVID_CLI_PATH "\"";
    for (const auto& a : args) cmd += " \"" + a + "\"";
    cmd += " > \"" + log.string() + "\" 2>&1\"";
    CliRun r;
    r.code = std::system(cmd.c_str());
    std::ifstream in(log, std::ios::binary);
    r.out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return r;
}

RawTags Tags(int m, int p, int t, int range = 1) { return RawTags{m, p, t, range, 0}; }

Mdcv Mastering1000() {
    Mdcv m;
    m.primaries[0][0] = {35400, 50000};
    m.primaries[0][1] = {14600, 50000};
    m.primaries[1][0] = {8500, 50000};
    m.primaries[1][1] = {39850, 50000};
    m.primaries[2][0] = {6550, 50000};
    m.primaries[2][1] = {2300, 50000};
    m.white[0] = {15635, 50000};
    m.white[1] = {16450, 50000};
    m.minLum = {1, 10000};
    m.maxLum = {10000000, 10000};
    m.hasPrimaries = m.hasLuminance = true;
    return m;
}

ClipSpec Small(const char* codec = "ffv1", const char* pixFmt = "yuv420p10le", int frames = 4) {
    ClipSpec s;
    s.width = 64;
    s.height = 48;
    s.frames = frames;
    s.codec = codec;
    s.pixFmt = pixFmt;
    return s;
}

std::filesystem::path PqFfv1(const std::filesystem::path& dir, const char* name, ClipHdr hdr = {}, RawTags tags = Tags(9, 9, 16)) {
    ClipSpec s = Small();
    s.color = tags;
    s.hdr = hdr;
    return WriteClip(dir / name, s);
}

std::filesystem::path HlgProRes(const std::filesystem::path& dir, const char* name) {
    ClipSpec s = Small("prores_ks", "yuv422p10le");
    s.color = Tags(9, 9, 14);
    s.frameColor = [](int64_t) { return Tags(9, 9, 18); };
    s.hdr.mastering = Mastering1000();
    s.hdr.light = Cll{1000, 400};
    s.hdr.ambient = Amve{{3140000, 10000}, {15635, 50000}, {16450, 50000}};
    return WriteClip(dir / name, s);
}

ProcessOptions Passthrough(const std::filesystem::path& in, const std::filesystem::path& out, const char* codec) {
    ProcessOptions o;
    o.input = in;
    o.output = out;
    o.codec = codec;
    o.passthrough = true;
    o.warp = true;
    o.settingsNote = false;
    return o;
}

bool Has(const std::vector<std::string>& v, const std::string& part) {
    return std::any_of(v.begin(), v.end(), [&](const std::string& s) { return s.find(part) != std::string::npos; });
}

bool HasKey(const std::vector<PlanNote>& v, const std::string& key) {
    return std::any_of(v.begin(), v.end(), [&](const PlanNote& n) { return n.key == key; });
}

}  // namespace

TEST_CASE("Passthrough of PQ FFV1 MKV keeps 9/9/16, the samples bit for bit, MasteringMetadata and MaxCLL", "[integration][hdr][passthrough]") {
    const auto dir = Dir("pq_mkv");
    ClipHdr hdr;
    hdr.mastering = Mastering1000();
    hdr.light = Cll{1000, 400};
    const auto in = PqFfv1(dir, "in.mkv", hdr);
    const auto out = dir / "out.mkv";
    const ProcessResult r = RunProcess(Passthrough(in, out, "ffv1"));
    CHECK(r.framesOut == 4);
    const StreamSummary s = Summarize(out);
    CHECK(s.colorSpace == 9);  // TA wrote 1 (the matrix by height)
    CHECK(s.colorPrimaries == 9);
    CHECK(s.colorTrc == 16);
    CHECK(s.colorRange == 1);
    REQUIRE(s.hdr.mdcv);
    CHECK(s.hdr.mdcv->maxLum.Value() == 1000.0);
    REQUIRE(s.hdr.cll);
    CHECK(*s.hdr.cll == Cll{1000, 400});
    const auto a = DecodeAll(in), b = DecodeAll(out);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(a[i].data == b[i].data);
}

TEST_CASE("Passthrough of HLG ProRes MOV keeps trc 18 in colr and in the frame header, mdcv clli and amve exactly", "[integration][hdr][passthrough]") {
    const auto dir = Dir("hlg_mov");
    const auto in = HlgProRes(dir, "in.mov");
    const auto out = dir / "out.mov";
    const ProcessResult r = RunProcess(Passthrough(in, out, "prores_ks"));
    CHECK(r.framesOut == 4);
    const StreamSummary s = Summarize(out);
    CHECK(s.colorTrc == 18);  // the frame's HLG, not the colr's bt2020-10
    CHECK(s.colorPrimaries == 9);
    CHECK(s.colorSpace == 9);
    const RawTags header = FrameColors(out, 1, true).at(0);  // prores_ks writes the frame header from the same tags
    CHECK(header.trc == 18);
    CHECK(header.primaries == 9);
    CHECK(header.matrix == 9);
    REQUIRE(s.hdr.mdcv);
    CHECK(*s.hdr.mdcv == Mastering1000());
    REQUIRE(s.hdr.cll);
    CHECK(*s.hdr.cll == Cll{1000, 400});
    REQUIRE(s.hdr.amve);
    CHECK(s.hdr.amve->illuminance.Value() == 314.0);
}

TEST_CASE("Passthrough of HLG into MKV notes the missing amve and publishes", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("hlg_mkv");
    const auto in = HlgProRes(dir, "in.mov");
    const auto out = dir / "out.mkv";
    const ProcessResult r = RunProcess(Passthrough(in, out, "ffv1"));
    CHECK(std::filesystem::exists(out));
    CHECK(Has(r.warnings, "ambient viewing environment"));
    const StreamSummary s = Summarize(out);
    CHECK(s.colorTrc == 18);
    CHECK_FALSE(s.hdr.amve);
    CHECK(s.hdr.mdcv);
}

TEST_CASE("MKV keeps no light level with MaxFALL 0 and none is expected back", "[integration][hdr][result]") {
    const auto dir = Dir("mkv_fall0");
    ClipHdr hdr;
    hdr.light = Cll{1000, 0};
    const auto in = PqFfv1(dir, "in.mov", hdr);  // MOV keeps it
    REQUIRE(Summarize(in).hdr.cll);
    const auto out = dir / "out.mkv";
    CHECK_NOTHROW(RunProcess(Passthrough(in, out, "ffv1")));
    CHECK(std::filesystem::exists(out));
    CHECK_FALSE(Summarize(out).hdr.cll);  // matroskadec makes none of MaxCLL alone
}

TEST_CASE("HDR passthrough into AVI or M4V is refused before any file is made; SDR into AVI publishes as before", "[integration][hdr][process]") {
    const auto dir = Dir("no_tags_container");
    const auto pq = PqFfv1(dir, "pq.mkv");
    for (const char* name : {"out.avi", "out.m4v"}) {
        const auto out = dir / name;
        INFO(name);
        try {
            RunProcess(Passthrough(pq, out, "ffv1"));
            FAIL("not refused");
        } catch (const ColourUnsupported& e) {
            CHECK(e.kind == ColourStop::NoContainerTags);
            CHECK_THAT(std::string(e.what()), ContainsSubstring(".mp4, .mov or .mkv"));
        }
        CHECK_FALSE(std::filesystem::exists(out));
        CHECK_FALSE(std::filesystem::exists(PartialResultPath(out)));
    }
    ClipSpec sdr = Small("ffv1", "yuv420p");
    sdr.color = Tags(1, 1, 1);
    const auto in = WriteClip(dir / "sdr.mkv", sdr);
    CHECK_NOTHROW(RunProcess(Passthrough(in, dir / "sdr.avi", "ffv1")));
    CHECK(std::filesystem::exists(dir / "sdr.avi"));
}

TEST_CASE("HDR passthrough into 8 bits warns of banding: h264_nvenc planned, FFV1 at bit depth 8 run", "[integration][hdr][process]") {
    const auto dir = Dir("hdr_8bit");
    const auto pq = PqFfv1(dir, "pq.mkv");
    ProcessOptions plan = Passthrough(pq, dir / "out.mp4", "h264_nvenc");
    CHECK(HasKey(PlanProcess(plan).warnings, "encode.hdr_8bit"));
    plan.codec = "hevc_nvenc";
    CHECK_FALSE(HasKey(PlanProcess(plan).warnings, "encode.hdr_8bit"));
    ProcessOptions run = Passthrough(pq, dir / "out8.mkv", "ffv1");
    run.outputFormat.bitDepth = 8;
    const ProcessResult r = RunProcess(run);
    CHECK(Has(r.warnings, "HDR in 8 bits"));
}

TEST_CASE("Processing a PQ source is refused before hashing: process upscale depth flow fg nr scenes export, and process --project with -i",
          "[integration][hdr][process][cli]") {
    const auto dir = Dir("refused");
    const auto pq = PqFfv1(dir, "pq.mkv");
    const std::string p = pq.string();
    struct Case {
        std::vector<std::string> args;
        std::string where;
    };
    const std::vector<Case> cases = {
        {{"process", "-i", p, "-o", (dir / "r.mkv").string(), "--warp", "--codec", "ffv1", "--stages", "upscale", "--param", "upscale.backend=bicubic"}, "process: "},
        {{"upscale", "-i", p, "-o", (dir / "up").string(), "--warp", "--backend", "bicubic"}, "upscale: "},
        {{"depth", "-i", p, "-o", (dir / "d").string(), "--warp"}, "depth: "},
        {{"flow", "-i", p, "-o", (dir / "f").string(), "--warp"}, "flow: "},
        {{"fg", "-i", p, "-o", (dir / "fg").string(), "--warp"}, "fg: "},
        {{"nr", "-i", p, "-o", (dir / "nr").string(), "--warp"}, "nr: "},
        {{"scenes", "-i", p, "-o", (dir / "sc").string()}, "scenes: "},
        {{"export", "-i", p, "-o", (dir / "ex").string(), "--range", "0-1"}, "export: "},
    };
    for (const Case& c : cases) {
        const auto run = RunAll(c.args, dir / (c.args[0] + ".log"));
        INFO(run.out);
        CHECK(run.code != 0);
        CHECK_THAT(run.out, ContainsSubstring(c.where + p + ": processing of HDR PQ"));
        CHECK_THAT(run.out, ContainsSubstring("passthrough (no stages) writes it as it is"));
    }
    CHECK_FALSE(std::filesystem::exists(dir / "r.mkv"));
    // a project of an SDR video run with -i on the PQ one: refused before -i is hashed («is not the source of the project» is said after it)
    ClipSpec sdr = Small("ffv1", "yuv420p");
    const auto sdrClip = WriteClip(dir / "sdr.mkv", sdr);
    const auto project = dir / "p.dlssvid.json";
    Project::Create(sdrClip, dir / "passes").Save(project);
    const auto withInput = RunAll({"process", "--project", project.string(), "-i", p, "--warp", "--codec", "ffv1", "--stages", "upscale", "--param",
                                   "upscale.backend=bicubic", "-o", (dir / "pr.mkv").string()},
                                  dir / "project.log");
    INFO(withInput.out);
    CHECK(withInput.code != 0);
    CHECK_THAT(withInput.out, ContainsSubstring("processing of HDR PQ"));
    CHECK_THAT(withInput.out, !ContainsSubstring("is not the source of the project"));
    CHECK_FALSE(std::filesystem::exists(dir / "pr.mkv"));
}

TEST_CASE("process --plan --passthrough refuses Dolby Vision 5 and prints the source notes of PQ", "[integration][hdr][process][cli]") {
    const auto dir = Dir("plan_pt");
    ClipSpec dv = Small();
    DoviConf p5;
    p5.profile = 5;
    dv.hdr.dovi = p5;
    const auto ipt = WriteClip(dir / "dv5.mkv", dv);
    const auto refused = RunAll({"process", "-i", ipt.string(), "-o", (dir / "o.mkv").string(), "--passthrough", "--plan", "--codec", "ffv1"}, dir / "dv5.log");
    {
        INFO(refused.out);
        CHECK(refused.code != 0);
        CHECK_THAT(refused.out, ContainsSubstring("a Dolby Vision IPT base layer"));
        CHECK_THAT(refused.out, ContainsSubstring("neither processed nor passed through"));
    }
    const auto pq = PqFfv1(dir, "pq.mkv", {}, RawTags{2, 9, 16, 1, 0});  // no matrix: BT.2020 assumed, no records
    const auto plan = RunAll({"process", "-i", pq.string(), "-o", (dir / "o.mp4").string(), "--passthrough", "--plan", "--codec", "hevc_nvenc"}, dir / "pq.log");
    INFO(plan.out);
    CHECK(plan.code == 0);
    CHECK_THAT(plan.out, ContainsSubstring("note: source: matrix unspecified under smpte2084: BT.2020 assumed"));
    CHECK_THAT(plan.out, ContainsSubstring("note: source: PQ without mastering display or light level metadata"));
    CHECK_THAT(plan.out, ContainsSubstring("note: encode: the result is tagged matrix bt2020nc"));
}

TEST_CASE("CheckEncodedResult names a colour tag or HDR metadata that differs", "[integration][hdr][result]") {
    const auto dir = Dir("check");
    ClipHdr hdr;
    hdr.mastering = Mastering1000();
    hdr.light = Cll{1000, 400};
    const auto file = PqFfv1(dir, "f.mkv", hdr);
    ResultExpectation e{64, 48, -1, false, PixelAspect{}};
    ResultColor c;
    c.matrix = 9;
    c.primaries = 9;
    c.trc = 16;
    c.range = 1;
    c.mdcv = Mastering1000();
    c.cll = Cll{1000, 400};
    e.color = c;
    CHECK(CheckEncodedResult(file, e).empty());
    ResultColor trc = c;
    trc.trc = 18;
    e.color = trc;
    CHECK(CheckEncodedResult(file, e) == "its transfer is smpte2084, arib-std-b67 was written");
    ResultColor range = c;
    range.range = 2;
    e.color = range;
    CHECK_THAT(CheckEncodedResult(file, e), ContainsSubstring("its range is tv"));
    range.checkRange = false;
    e.color = range;
    CHECK(CheckEncodedResult(file, e).empty());
    ResultColor light = c;
    light.cll = Cll{4000, 400};
    e.color = light;
    CHECK_THAT(CheckEncodedResult(file, e), ContainsSubstring("content light level"));
    ResultColor mastering = c;
    mastering.mdcv->maxLum = {4000, 1};
    e.color = mastering;
    CHECK_THAT(CheckEncodedResult(file, e), ContainsSubstring("mastering display"));
    ResultColor amve = c;
    amve.amve = Amve{{314, 1}, {3127, 10000}, {329, 1000}};
    e.color = amve;
    CHECK_THAT(CheckEncodedResult(file, e), ContainsSubstring("ambient viewing environment"));
}

TEST_CASE("Legacy ProRes and FFV1 results in MOV and MKV with tags 1/1/1 2/2/2 matrix only and full range publish as before", "[integration][result][regression]") {
    const auto dir = Dir("legacy");
    int i = 0;
    for (const RawTags tags : {Tags(1, 1, 1), RawTags{2, 2, 2, 0, 0}, RawTags{1, 2, 2, 1, 0}, Tags(6, 6, 6)})
        for (const auto& [codec, ext, pix] : {std::tuple{"prores_ks", ".mov", "yuv422p10le"}, std::tuple{"ffv1", ".mkv", "yuv420p"}, std::tuple{"ffv1", ".mov", "yuv420p"}}) {
            ClipSpec s = Small(codec, pix);
            s.color = tags;
            const auto in = WriteClip(dir / ("in" + std::to_string(i) + ext), s);
            VideoDecoder probe(in, VideoDecoder::Probe());
            REQUIRE(probe.Info().color.legacyPath);
            const ColorTags before = ResultColorTags(ColorInfoFromStream(probe.Info()), probe.Info());  // what TA wrote
            const auto out = dir / ("out" + std::to_string(i++) + ext);
            INFO(codec << ext << " tags " << tags.matrix << "/" << tags.primaries << "/" << tags.trc);
            const ProcessResult r = RunProcess(Passthrough(in, out, codec));
            CHECK(r.warnings.empty());
            const StreamSummary got = Summarize(out);
            if (before.matrix != 2 && before.primaries != 2 && before.transfer != 2) {
                CHECK(got.colorSpace == before.matrix);
                CHECK(got.colorPrimaries == before.primaries);
                CHECK(got.colorTrc == before.transfer);
            }
            CHECK_FALSE(got.hdr.mdcv);
        }
    // full range into a MOV of ProRes (which keeps none): lost without a word, as before TB-2 - no note on the legacy path
    ClipSpec s = Small("ffv1", "yuv420p");
    s.color = Tags(1, 1, 1, 2);
    const auto full = WriteClip(dir / "full.mkv", s);
    VideoDecoder probe(full, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.legacyPath);
    REQUIRE(probe.Info().color.effective.range == 2);
    const ProcessResult r = RunProcess(Passthrough(full, dir / "full_out.mov", "prores_ks"));
    CHECK(r.warnings.empty());
    CHECK(std::filesystem::exists(dir / "full_out.mov"));
}

TEST_CASE("Frames turning PQ under colr 1/1/1 warn at the first PQ frame in passthrough; frames 6/6/6 do not", "[integration][hdr][passthrough]") {
    const auto dir = Dir("class_change");
    ClipSpec s = Small("prores_ks", "yuv422p10le", 10);
    s.color = Tags(1, 1, 1);
    s.frameColor = [](int64_t i) { return i < 6 ? Tags(1, 1, 1) : Tags(9, 9, 16); };
    const auto turning = WriteClip(dir / "turning.mov", s);
    const ProcessResult r = RunProcess(Passthrough(turning, dir / "out.mov", "prores_ks"));
    CHECK(Has(r.warnings, "frame 6 and on are HDR PQ"));
    s.frameColor = [](int64_t i) { return i < 6 ? Tags(1, 1, 1) : Tags(6, 6, 6); };
    const auto sdr = WriteClip(dir / "sdr.mov", s);
    const ProcessResult q = RunProcess(Passthrough(sdr, dir / "out2.mov", "prores_ks"));
    CHECK_FALSE(Has(q.warnings, "and on are"));
}

TEST_CASE("The source passed through with every stage off keeps the passthrough warnings, each once", "[integration][hdr][process][regression]") {
    const auto dir = Dir("stages_off");
    const auto pq = PqFfv1(dir, "pq.mkv", {}, RawTags{2, 9, 16, 1, 0});
    ProcessOptions o;
    o.input = pq;
    o.output = dir / "out.mkv";
    o.codec = "ffv1";
    o.warp = true;
    o.settingsNote = false;
    o.stages = DefaultProcessStages();
    for (auto& st : o.stages) st.enabled = false;
    const ProcessPlan plan = PlanProcess(o);
    CHECK(plan.finalPass.empty());
    CHECK(HasKey(plan.warnings, "source.bt2020_assumed"));
    CHECK(HasKey(plan.warnings, "encode.tags_filled"));
    const ProcessResult r = RunProcess(o);
    CHECK(Has(r.warnings, "BT.2020 assumed"));
    CHECK(Has(r.warnings, "the result is tagged"));
    CHECK(std::count_if(r.warnings.begin(), r.warnings.end(), [](const std::string& w) { return w.find("BT.2020 assumed") != std::string::npos; }) == 1);
    CHECK(Summarize(o.output).colorSpace == 9);  // filled
}

TEST_CASE("Dolby Vision 8.4 FFV1 MKV passes through without its mapping and with a note; 5 is refused", "[integration][hdr][dovi][passthrough]") {
    const auto dir = Dir("dovi");
    auto clip = [&](const char* name, RawTags tags, int profile, int compat) {
        ClipSpec s = Small();
        s.color = tags;
        DoviConf d;
        d.major = 1;
        d.profile = profile;
        d.level = 6;
        d.blCompat = compat;
        d.rpu = d.bl = true;
        s.hdr.dovi = d;
        return WriteClip(dir / name, s);
    };
    const auto hlg = clip("p84.mkv", Tags(9, 9, 18), 8, 4);
    const ProcessResult r = RunProcess(Passthrough(hlg, dir / "out.mkv", "ffv1"));
    CHECK(Has(r.warnings, "the RPU is dropped in every output"));
    CHECK_FALSE(Summarize(dir / "out.mkv").dovi);
    CHECK(Summarize(dir / "out.mkv").colorTrc == 18);
    const auto ipt = clip("p5.mkv", RawTags{2, 2, 2, 2, 0}, 5, 0);
    CHECK_THROWS_AS(RunProcess(Passthrough(ipt, dir / "out5.mkv", "ffv1")), ColourUnsupported);
    CHECK_FALSE(std::filesystem::exists(dir / "out5.mkv"));
}

TEST_CASE("A proxy of an HLG ProRes clip carries the frame's trc 18; an SDR proxy keeps the stream's tags", "[integration][hdr][viewport][proxy]") {
    const auto dir = Dir("proxy");
    const auto hlg = HlgProRes(dir, "hlg.mov");
    const auto proxy = BuildProxy(hlg, dir / "passes", "source");
    const StreamSummary s = Summarize(proxy);
    CHECK(s.colorTrc == 18);
    CHECK(s.colorPrimaries == 9);
    ClipSpec sdr = Small("prores_ks", "yuv422p10le");
    sdr.color = Tags(1, 1, 1);
    const auto sdrClip = WriteClip(dir / "sdr.mov", sdr);
    const auto sdrProxy = BuildProxy(sdrClip, dir / "passes2", "source");
    CHECK(Summarize(sdrProxy).colorTrc == 1);
    CHECK(Summarize(sdrProxy).colorSpace == 1);
}

TEST_CASE("dlssvid info of a PQ video exits 0, says HDR10 records go to the container only and that processing waits", "[integration][hdr][cli]") {
    const auto dir = Dir("info");
    ClipHdr hdr;
    hdr.light = Cll{1000, 400};
    const auto pq = PqFfv1(dir, "pq.mkv", hdr);
    const auto info = RunAll({"info", pq.string()}, dir / "info.log");
    INFO(info.out);
    CHECK(info.code == 0);
    CHECK_THAT(info.out, ContainsSubstring("processing:  not yet - processing of HDR PQ"));
    CHECK_THAT(info.out, ContainsSubstring("hdr note:    the HDR10 records go to the container only"));
    ClipSpec dv = Small();
    DoviConf p5;
    p5.profile = 5;
    dv.hdr.dovi = p5;
    const auto ipt = RunAll({"info", WriteClip(dir / "dv5.mkv", dv).string()}, dir / "dv5.log");
    INFO(ipt.out);
    CHECK(ipt.code == 0);
    CHECK_THAT(ipt.out, ContainsSubstring("processing:  not supported - a Dolby Vision IPT base layer"));
    CHECK_THAT(ipt.out, ContainsSubstring("IPT base - not supported"));
}

TEST_CASE("A processed legacy source whose frames turn PQ warns in the result with the frame number", "[integration][hdr][process]") {
    const auto dir = Dir("process_turn");
    ClipSpec s = Small("prores_ks", "yuv422p10le", 10);
    s.color = Tags(1, 1, 1);
    s.frameColor = [](int64_t i) { return i < 6 ? Tags(1, 1, 1) : Tags(9, 9, 16); };
    const auto turning = WriteClip(dir / "turning.mov", s);
    ProcessOptions o;
    o.input = turning;
    o.output = dir / "out.mkv";
    o.codec = "ffv1";
    o.warp = true;
    o.settingsNote = false;
    o.stages = DefaultProcessStages();
    for (auto& st : o.stages) st.enabled = st.name == "upscale";
    for (auto& st : o.stages)
        if (st.name == "upscale") st.params = {{"backend", "bicubic"}, {"scale", 2.0}};
    const ProcessResult r = RunProcess(o);
    CHECK(Has(r.warnings, "frame 6 and on are HDR PQ"));
}

TEST_CASE("Dolby Vision 8.2 FFV1 MKV processed with bicubic notes the removed mapping in the plan and in the result", "[integration][hdr][dovi][process]") {
    const auto dir = Dir("dovi82");
    auto run = [&](const char* name, bool dovi) {
        ClipSpec s = Small("ffv1", "yuv420p");
        s.color = Tags(1, 1, 1);
        if (dovi) {
            DoviConf d;
            d.profile = 8;
            d.level = 6;
            d.blCompat = 2;
            d.rpu = d.bl = true;
            s.hdr.dovi = d;
        }
        ProcessOptions o;
        o.input = WriteClip(dir / name, s);
        o.output = dir / (std::string(name) + ".out.mkv");
        o.codec = "ffv1";
        o.warp = true;
        o.settingsNote = false;
        o.stages = DefaultProcessStages();
        for (auto& st : o.stages) st.enabled = st.name == "upscale";
        for (auto& st : o.stages)
            if (st.name == "upscale") st.params = {{"backend", "bicubic"}, {"scale", 2.0}};
        return std::pair{PlanProcess(o), RunProcess(o)};
    };
    const auto [plan, result] = run("p82.mkv", true);
    CHECK(HasKey(plan.warnings, "source.dv_removed"));
    CHECK(Has(result.warnings, "the RPU is dropped in every output"));
    const auto [plainPlan, plainResult] = run("sdr.mkv", false);
    CHECK(plainPlan.warnings.empty());
    CHECK(plainResult.warnings.empty());
}

TEST_CASE("ProRes MOV with PQ colr and SDR frame headers is stopped for processing and passes through with the frames' tags", "[integration][hdr][process]") {
    const auto dir = Dir("mixed");
    ClipSpec s = Small("prores_ks", "yuv422p10le");
    s.color = Tags(9, 9, 16);
    s.frameColor = [](int64_t) { return Tags(1, 1, 1); };
    const auto in = WriteClip(dir / "mixed.mov", s);
    ProcessOptions o;
    o.input = in;
    o.output = dir / "out.mkv";
    o.codec = "ffv1";
    o.warp = true;
    o.settingsNote = false;
    o.stages = DefaultProcessStages();
    for (auto& st : o.stages) st.enabled = st.name == "upscale";
    for (auto& st : o.stages)
        if (st.name == "upscale") st.params = {{"backend", "bicubic"}, {"scale", 2.0}};
    try {
        PlanProcess(o);
        FAIL("not stopped");
    } catch (const ColourUnsupported& e) {
        CHECK(e.kind == ColourStop::NotYet);
        CHECK_THAT(std::string(e.what()), ContainsSubstring("the source's layers disagree"));
    }
    const ProcessResult r = RunProcess(Passthrough(in, dir / "pt.mov", "prores_ks"));
    CHECK(Has(r.warnings, "the first frame says"));
    CHECK(Summarize(dir / "pt.mov").colorTrc == 1);  // the frames' tags
}

TEST_CASE("Passthrough notes a mastering display it cannot write and a full range it cannot keep", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("lost");
    ClipHdr lum;
    Mdcv m;
    m.maxLum = {1000, 1};
    m.minLum = {1, 10000};
    m.hasLuminance = true;
    lum.mastering = m;
    const auto partial = PqFfv1(dir, "lum.mkv", lum);
    REQUIRE(Summarize(partial).hdr.mdcv);
    const ProcessResult a = RunProcess(Passthrough(partial, dir / "lum.mov", "prores_ks"));
    CHECK(Has(a.warnings, "the mastering display says only its luminance"));
    CHECK_FALSE(Summarize(dir / "lum.mov").hdr.mdcv);
    const auto full = PqFfv1(dir, "full.mkv", {}, Tags(9, 9, 16, 2));
    const ProcessResult b = RunProcess(Passthrough(full, dir / "full.mov", "ffv1"));
    CHECK(Has(b.warnings, "full range"));
    CHECK(std::filesystem::exists(dir / "full.mov"));
}

TEST_CASE("Container colour caps by muxer: mp4 and mov keep amve, mkv not, mov no range, avi m4v ts nothing", "[integration][hdr][result]") {
    const auto mp4 = VideoEncoder::ContainerColorCapsFor("a.mp4");
    CHECK(mp4.tags);
    CHECK(mp4.allThreeOrNone);
    CHECK(mp4.range);
    CHECK(mp4.mdcvNeedsBoth);
    CHECK(mp4.amve);
    const auto mov = VideoEncoder::ContainerColorCapsFor("a.mov");
    CHECK(mov.tags);
    CHECK_FALSE(mov.range);
    CHECK(mov.amve);
    for (const char* f : {"a.mkv", "a.webm"}) {
        const auto mkv = VideoEncoder::ContainerColorCapsFor(f);
        CHECK(mkv.tags);
        CHECK_FALSE(mkv.allThreeOrNone);
        CHECK(mkv.range);
        CHECK_FALSE(mkv.mdcvNeedsBoth);
        CHECK_FALSE(mkv.amve);
    }
    for (const char* f : {"a.avi", "a.m4v", "a.ts", "a.hevc"}) {
        INFO(f);
        CHECK_FALSE(VideoEncoder::ContainerColorCapsFor(f).tags);
    }
    CHECK(CodecKeepsRange("hevc_nvenc"));
    CHECK_FALSE(CodecKeepsRange("prores_ks"));
    CHECK_FALSE(CodecKeepsRange("ffv1"));
}

// ---- NVENC ----------------------------------------------------------------------------------------------------------------------

namespace {
bool HaveNvenc(const char* codec) {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia() && VideoEncoder::EncoderAvailable(codec);
}
}  // namespace

TEST_CASE("NVENC passthrough of PQ and HLG writes VUI 9/16/9 and 9/18/9, colr mdcv clli and amve in MP4", "[integration][hdr][gpu][formats]") {
    const auto dir = Dir("nvenc");
    for (const char* codec : {"hevc_nvenc", "av1_nvenc"}) {
        if (!HaveNvenc(codec)) SKIP(std::string("no NVIDIA GPU with ") + codec);
        INFO(codec);
        ClipHdr hdr;
        hdr.mastering = Mastering1000();
        hdr.light = Cll{1000, 400};
        ClipSpec s = Small();
        s.width = 320;
        s.height = 192;
        s.color = Tags(9, 9, 16);
        s.hdr = hdr;
        const auto pq = WriteClip(dir / (std::string(codec) + "_pq.mkv"), s);
        const auto out = dir / (std::string(codec) + "_pq.mp4");
        const ProcessResult r = RunProcess(Passthrough(pq, out, codec));
        CHECK(r.framesOut == 4);
        const StreamSummary got = Summarize(out);
        CHECK(got.colorSpace == 9);
        CHECK(got.colorPrimaries == 9);
        CHECK(got.colorTrc == 16);
        REQUIRE(got.hdr.mdcv);
        CHECK(*got.hdr.mdcv == Mastering1000());
        REQUIRE(got.hdr.cll);
        CHECK(*got.hdr.cll == Cll{1000, 400});
    }
    if (!HaveNvenc("hevc_nvenc")) SKIP("no NVIDIA GPU with hevc_nvenc");
    ClipSpec h = Small("prores_ks", "yuv422p10le");
    h.width = 320;
    h.height = 192;
    h.color = Tags(9, 9, 14);
    h.frameColor = [](int64_t) { return Tags(9, 9, 18); };
    h.hdr.ambient = Amve{{3140000, 10000}, {15635, 50000}, {16450, 50000}};
    const auto hlg = WriteClip(dir / "hlg.mov", h);
    const auto out = dir / "hlg.mp4";
    RunProcess(Passthrough(hlg, out, "hevc_nvenc"));
    const StreamSummary got = Summarize(out);
    CHECK(got.colorTrc == 18);
    REQUIRE(got.hdr.amve);
    CHECK(got.hdr.amve->illuminance.Value() == 314.0);
}

TEST_CASE("Legacy h264 hevc and av1 NVENC results in mp4 mov and mkv publish as before", "[integration][result][gpu][regression]") {
    const auto dir = Dir("nvenc_legacy");
    ClipSpec s = Small("ffv1", "yuv420p");
    s.width = 320;
    s.height = 192;
    s.color = Tags(1, 1, 1);
    const auto in = WriteClip(dir / "sdr.mkv", s);
    for (const char* codec : {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}) {
        if (!HaveNvenc(codec)) continue;
        for (const char* ext : {".mp4", ".mov", ".mkv"}) {
            if (std::string(codec) == "av1_nvenc" && std::string(ext) == ".mov") continue;  // MOV takes no AV1
            INFO(codec << ext);
            const auto out = dir / (std::string(codec) + ext);
            const ProcessResult r = RunProcess(Passthrough(in, out, codec));
            CHECK(r.warnings.empty());
            const StreamSummary got = Summarize(out);
            CHECK(got.colorSpace == 1);
            CHECK(got.colorPrimaries == 1);
            CHECK(got.colorTrc == 1);
            CHECK_FALSE(got.hdr.mdcv);
        }
    }
}
