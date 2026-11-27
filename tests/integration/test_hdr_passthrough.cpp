// TASK-0079 (HDR step TB-2): an HDR / wide-gamut source passed through as it is - its effective tags (filled where unsaid), its HDR
// records within the container, notes of what is lost -, its processing stopped before anything is hashed, a part of other colour
// warned of, an SDR source on the legacy path as before. CPU cases on FFV1 / ProRes (WARP for the passthrough's GPU stage).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
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

// every file under `root` (relative), sorted: what a refused run must leave as it was
std::vector<std::string> FilesUnder(const std::filesystem::path& root) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return out;
    for (const auto& e : std::filesystem::recursive_directory_iterator(root))
        if (e.is_regular_file()) out.push_back(std::filesystem::relative(e.path(), root).generic_string());
    std::sort(out.begin(), out.end());
    return out;
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
    CHECK(r.warnings.empty());  // nothing filled, lost or assumed
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
    {  // nothing filled, lost or assumed: no encode note (an HLG source's records are carried: none dropped)
        VideoDecoder probe(in, VideoDecoder::Probe());
        CHECK(PlanPassthroughColour(probe.Info(), "prores_ks", {}, out, "t").encodeNotes.empty());
    }
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
    CHECK(std::llround(s.hdr.amve->lightX.Value() * 50000) == 15635);
    CHECK(std::llround(s.hdr.amve->lightY.Value() * 50000) == 16450);
}

TEST_CASE("Passthrough of HLG into MKV notes the missing amve and publishes", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("hlg_mkv");
    const auto in = HlgProRes(dir, "in.mov");
    const auto out = dir / "out.mkv";
    const ProcessResult r = RunProcess(Passthrough(in, out, "ffv1"));
    CHECK(std::filesystem::exists(out));
    CHECK(Has(r.warnings, "ambient viewing environment"));
    CHECK(HasKey(PlanProcess(Passthrough(in, out, "ffv1")).warnings, "encode.no_amve"));
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
    ProcessResult r;
    CHECK_NOTHROW(r = RunProcess(Passthrough(in, out, "ffv1")));
    CHECK(std::filesystem::exists(out));
    CHECK_FALSE(Summarize(out).hdr.cll);  // matroskadec makes none of MaxCLL alone
    CHECK(Has(r.warnings, "the light level MaxCLL 1000 / MaxFALL 0: matroska's readers keep one only with both values"));
    CHECK(HasKey(PlanProcess(Passthrough(in, out, "ffv1")).warnings, "encode.cll_invalid"));
    // MaxCLL 0 with a MaxFALL: dropped by matroskadec too - noted
    hdr.light = Cll{0, 400};
    CHECK(HasKey(PlanProcess(Passthrough(PqFfv1(dir, "cll0.mov", hdr), dir / "cll0.mkv", "ffv1")).warnings, "encode.cll_invalid"));
    // 0 / 0 (nothing known): no note, published, none in MKV; kept in MOV
    hdr.light = Cll{0, 0};
    const auto zero = PqFfv1(dir, "zero.mov", hdr);
    REQUIRE(Summarize(zero).hdr.cll);
    CHECK_NOTHROW(RunProcess(Passthrough(zero, dir / "zero.mkv", "ffv1")));
    CHECK_FALSE(HasKey(PlanProcess(Passthrough(zero, dir / "zero.mkv", "ffv1")).warnings, "encode.cll_invalid"));
    CHECK_FALSE(Summarize(dir / "zero.mkv").hdr.cll);
    CHECK_NOTHROW(RunProcess(Passthrough(zero, dir / "zero_out.mov", "ffv1")));
    REQUIRE(Summarize(dir / "zero_out.mov").hdr.cll);
    CHECK(*Summarize(dir / "zero_out.mov").hdr.cll == Cll{0, 0});
}

TEST_CASE("Mastering and light level outside MP4 MOV fields are noted and not written; the publish holds", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("box_fields");
    ClipHdr hdr;
    Mdcv units = Mastering1000();
    units.primaries[0][0] = {34000, 1};  // x265's units in Matroska's floats: 34000 for 0.68
    hdr.mastering = units;
    hdr.light = Cll{70000, 400};
    const auto in = PqFfv1(dir, "in.mkv", hdr);
    REQUIRE(Summarize(in).hdr.mdcv);
    REQUIRE(Summarize(in).hdr.cll);
    const ProcessResult r = RunProcess(Passthrough(in, dir / "out.mov", "prores_ks"));
    CHECK(Has(r.warnings, "a value outside mov's mdcv fields"));
    CHECK(Has(r.warnings, "does not fit mov's clli fields"));
    const ProcessPlan plan = PlanProcess(Passthrough(in, dir / "out.mov", "prores_ks"));
    CHECK(HasKey(plan.warnings, "encode.mdcv_unfit"));
    CHECK(HasKey(plan.warnings, "encode.cll_invalid"));
    {  // MP4 (NVENC's default container) the same, planned on the CPU: neither written nor expected
        VideoDecoder probe(in, VideoDecoder::Probe());
        const PassthroughColour mp4 = PlanPassthroughColour(probe.Info(), "hevc_nvenc", {}, "a.mp4", "t");
        CHECK(HasKey(mp4.encodeNotes, "encode.mdcv_unfit"));
        CHECK(HasKey(mp4.encodeNotes, "encode.cll_invalid"));
        CHECK_FALSE(mp4.meta.mdcv);
        CHECK_FALSE(mp4.meta.cll);
        REQUIRE(mp4.expect);
        CHECK_FALSE(mp4.expect->mdcv);
        CHECK_FALSE(mp4.expect->cll);
    }
    {  // clli at its edge: 65535 written and read back, 65536 noted and not written
        for (const auto& [cll, fits] : {std::pair{Cll{65535, 400}, true}, std::pair{Cll{65536, 400}, false}}) {
            INFO("MaxCLL " << cll.maxCll);
            ClipHdr h;
            h.light = cll;
            const auto name = "cll" + std::to_string(cll.maxCll);
            const auto src = PqFfv1(dir, (name + ".mkv").c_str(), h);
            CHECK(HasKey(PlanProcess(Passthrough(src, dir / (name + ".mov"), "prores_ks")).warnings, "encode.cll_invalid") == !fits);
            CHECK_NOTHROW(RunProcess(Passthrough(src, dir / (name + ".mov"), "prores_ks")));
            const StreamSummary back = Summarize(dir / (name + ".mov"));
            CHECK(back.hdr.cll.has_value() == fits);
            if (fits && back.hdr.cll) CHECK(*back.hdr.cll == cll);
        }
    }
    for (int field = 0; field < 8; ++field) {  // each coordinate of the primaries and the white point past 16 bits, alone
        INFO("coordinate " << field);
        Mdcv one = Mastering1000();
        Ratio& r = field < 6 ? one.primaries[field / 2][field % 2] : one.white[field - 6];
        r = {131071, 100000};
        ClipHdr h;
        h.mastering = one;
        VideoDecoder probe(PqFfv1(dir, ("coord" + std::to_string(field) + ".mkv").c_str(), h), VideoDecoder::Probe());
        const PassthroughColour mov = PlanPassthroughColour(probe.Info(), "prores_ks", {}, "a.mov", "t");
        CHECK(HasKey(mov.encodeNotes, "encode.mdcv_unfit"));
        CHECK_FALSE(mov.meta.mdcv);
    }
    {  // a negative minimum luminance (from a MOV: mov.c reads it so) - outside the u32
        ClipSpec neg = Small("prores_ks", "yuv422p10le");
        neg.color = Tags(9, 9, 16);
        Mdcv m = Mastering1000();
        m.minLum = {-1, 10000};
        neg.hdr.mastering = m;
        VideoDecoder probe(WriteClip(dir / "neg.mov", neg), VideoDecoder::Probe());
        REQUIRE(probe.Info().color.hdr.mdcv);
        REQUIRE(probe.Info().color.hdr.mdcv->minLum.Value() < 0);
        CHECK(HasKey(PlanPassthroughColour(probe.Info(), "prores_ks", {}, "a.mov", "t").encodeNotes, "encode.mdcv_unfit"));
    }
    {  // MaxFALL alone past 16 bits
        ClipHdr fall;
        fall.light = Cll{1000, 70000};
        VideoDecoder probe(PqFfv1(dir, "fall.mkv", fall), VideoDecoder::Probe());
        const PassthroughColour mov = PlanPassthroughColour(probe.Info(), "prores_ks", {}, "a.mov", "t");
        CHECK(HasKey(mov.encodeNotes, "encode.cll_invalid"));
        CHECK_FALSE(mov.meta.cll);
    }
    const StreamSummary out = Summarize(dir / "out.mov");
    CHECK_FALSE(out.hdr.mdcv);
    CHECK_FALSE(out.hdr.cll);
    CHECK(out.colorTrc == 16);
    // the edge as movenc rounds it: 1.31071 is 65535.5 units, written as 65536 - its low 16 bits 0
    Mdcv edge = Mastering1000();
    edge.primaries[0][0] = {131071, 100000};
    ClipHdr edgeHdr;
    edgeHdr.mastering = edge;
    const auto edgeIn = PqFfv1(dir, "edge.mkv", edgeHdr);
    const ProcessResult e = RunProcess(Passthrough(edgeIn, dir / "edge.mov", "prores_ks"));
    CHECK(Has(e.warnings, "a value outside mov's mdcv fields"));
    CHECK_FALSE(Summarize(dir / "edge.mov").hdr.mdcv);
    // under the edge: 1.310707 is 65535.35 units - written as 65535 and read back
    edge.primaries[0][0] = {1310707, 1000000};
    edgeHdr.mastering = edge;
    const ProcessResult under = RunProcess(Passthrough(PqFfv1(dir, "under.mkv", edgeHdr), dir / "under.mov", "prores_ks"));
    CHECK_FALSE(Has(under.warnings, "mdcv fields"));
    CHECK(Summarize(dir / "under.mov").hdr.mdcv);
    // the luminance: up to INT32_MAX units mov.c reads back positive (214748 cd/m2), above it negative; x265's units far above
    {  // a minimum of 0 (ST 2086 allows it): kept in MOV and in Matroska, no note
        Mdcv black = Mastering1000();
        black.minLum = {0, 10000};
        ClipHdr h;
        h.mastering = black;
        const auto src = PqFfv1(dir, "black.mkv", h);
        for (const auto& [ext, codec] : {std::pair{".mov", "prores_ks"}, std::pair{".mkv", "ffv1"}}) {
            INFO(ext);
            const auto out = dir / (std::string("black") + ext + ext);
            const ProcessPlan plan = PlanProcess(Passthrough(src, out, codec));
            CHECK_FALSE(HasKey(plan.warnings, "encode.mdcv_unfit"));
            CHECK_FALSE(HasKey(plan.warnings, "encode.mdcv_invalid"));
            CHECK_NOTHROW(RunProcess(Passthrough(src, out, codec)));
            const StreamSummary back = Summarize(out);
            REQUIRE(back.hdr.mdcv);
            CHECK(back.hdr.mdcv->hasLuminance);
            CHECK(back.hdr.mdcv->minLum.Value() == 0.0);
        }
    }
    for (const auto& [lum, fits] : {std::pair{214748, true}, std::pair{214749, false}, std::pair{10000000, false}}) {
        INFO("max luminance " << lum);
        Mdcv bright = Mastering1000();
        bright.maxLum = {lum, 1};
        ClipHdr h;
        h.mastering = bright;
        const auto name = "lum" + std::to_string(lum);
        const ProcessResult b = RunProcess(Passthrough(PqFfv1(dir, (name + ".mkv").c_str(), h), dir / (name + ".mov"), "prores_ks"));
        CHECK(Has(b.warnings, "a value outside mov's mdcv fields") == !fits);
        CHECK(Summarize(dir / (name + ".mov")).hdr.mdcv.has_value() == fits);
    }
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
    // a scenes pass of an SDR clip in the root the PQ run is pointed at: kept where it is (not retired to the history)
    ClipSpec sdr = Small("ffv1", "yuv420p");
    const auto sdrClip = WriteClip(dir / "sdr.mkv", sdr);
    REQUIRE(RunAll({"scenes", "-i", sdrClip.string(), "-o", (dir / "sc").string()}, dir / "scenes_sdr.log").code == 0);
    const std::vector<std::string> scenesBefore = FilesUnder(dir / "sc");
    REQUIRE_FALSE(scenesBefore.empty());
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
        // from a colour pass (--start past the warm-up): the frames would be processed in that colour all the same
        {{"nr", "-i", p, "-o", (dir / "nr2").string(), "--warp", "--color-dir", (dir / "no_pass").string(), "--start", "1"}, "nr: "},
        {{"scenes", "-i", p, "-o", (dir / "sc").string()}, "scenes: "},
        {{"export", "-i", p, "-o", (dir / "ex").string(), "--range", "0-1"}, "export: "},
        // through Prepare (no CLI check of its own): refused before the hash too
        {{"bench", "-i", p, "--warp", "--frames", "2", "--codec", "ffv1", "--stages", "upscale", "--param", "upscale.backend=bicubic"}, "process: "},
    };
    for (const Case& c : cases) {
        std::vector<std::string> verbose = c.args;
        verbose.insert(verbose.begin(), "-v");  // the debug log says each hash: none before the refusal
        const auto run = RunAll(verbose, dir / (c.args[0] + std::to_string(&c - cases.data()) + ".log"));
        INFO(run.out);
        CHECK_THAT(run.out, !ContainsSubstring("hashing"));
        CHECK(run.code != 0);
        CHECK_THAT(run.out, ContainsSubstring(c.where + p + ": processing of HDR PQ"));
        CHECK_THAT(run.out, ContainsSubstring("passthrough (no stages) writes it as it is"));
        if (c.args[0] != "scenes" && c.args[0] != "process" && c.args[0] != "bench") CHECK(FilesUnder(c.args[4]).empty());  // no pass, no part file
    }
    CHECK_FALSE(std::filesystem::exists(dir / "r.mkv"));
    CHECK(FilesUnder(dir / "sc") == scenesBefore);
    // scenes --threshold 0 reads no frame (the pass says «off»): not refused
    const auto off = RunAll({"scenes", "-i", p, "-o", (dir / "sc0").string(), "--threshold", "0"}, dir / "scenes_off.log");
    {
        INFO(off.out);
        CHECK(off.code == 0);
        CHECK_THAT(off.out, !ContainsSubstring("processing of HDR"));
    }
    // a project of an SDR video run with -i on the PQ one: refused before -i is hashed («is not the source of the project» is said after it)
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

TEST_CASE("process --project of a PQ source refuses before the project's source is hashed; an SDR one is hashed", "[integration][hdr][process][cli]") {
    const auto dir = Dir("project_hash");
    const auto pq = PqFfv1(dir, "pq.mkv");
    Project::Create(pq, dir / "passes").Save(dir / "pq.dlssvid.json");  // no hash stored, the default stages on
    const auto refused = RunAll({"-v", "process", "--project", (dir / "pq.dlssvid.json").string(), "--warp", "--stages", "upscale", "--param",
                                 "upscale.backend=bicubic"},
                                dir / "pq.log");
    {
        INFO(refused.out);
        CHECK(refused.code != 0);
        CHECK_THAT(refused.out, ContainsSubstring("processing of HDR PQ"));
        CHECK_THAT(refused.out, !ContainsSubstring("hashing"));
    }
    ClipSpec s = Small("ffv1", "yuv420p");
    Project::Create(WriteClip(dir / "sdr.mkv", s), dir / "passes2").Save(dir / "sdr.dlssvid.json");
    const auto sdr = RunAll({"-v", "process", "--project", (dir / "sdr.dlssvid.json").string(), "--warp", "--plan", "--stages", "upscale", "--param",
                             "upscale.backend=bicubic"},
                            dir / "sdr.log");
    INFO(sdr.out);
    CHECK_THAT(sdr.out, ContainsSubstring("hashing"));  // the log line the check above looks for
}

TEST_CASE("process --project of a PQ source with every stage off passes it through, as the GUI's only-encoding run does", "[integration][hdr][process][cli]") {
    const auto dir = Dir("project_off");
    const auto pq = PqFfv1(dir, "pq.mkv", {}, RawTags{2, 9, 16, 1, 0});  // no matrix: filled, an encode note
    Project p = Project::Create(pq, dir / "passes");
    for (auto& s : p.stages) s.enabled = false;
    p.resultVideo = dir / "result.mkv";
    p.Save(dir / "p.dlssvid.json");
    const auto plan = RunAll({"process", "--project", (dir / "p.dlssvid.json").string(), "--warp", "--codec", "ffv1", "--plan"}, dir / "plan.log");
    {
        INFO(plan.out);
        CHECK(plan.code == 0);
        CHECK_THAT(plan.out, ContainsSubstring("note: encode: the result is tagged matrix bt2020nc"));
        CHECK_THAT(plan.out, !ContainsSubstring("processing of HDR"));
    }
    const auto run = RunAll({"process", "--project", (dir / "p.dlssvid.json").string(), "--warp", "--codec", "ffv1", "--disable-unavailable"}, dir / "run.log");
    INFO(run.out);
    CHECK(run.code == 0);
    const StreamSummary s = Summarize(dir / "result.mkv");
    CHECK(s.colorSpace == 9);
    CHECK(s.colorTrc == 16);
}

TEST_CASE("A transfer H.273 leaves undefined (19 in Matroska) reads as unspecified and the passthrough publishes", "[integration][hdr][passthrough][regression]") {
    const auto dir = Dir("unnamed");
    // matroskaenc writes no transfer past its table: an 18 written, its byte made 19 (TransferCharacteristics 0x55BA, one byte)
    const auto written = PqFfv1(dir, "hlg.mkv", {}, Tags(9, 9, 18));
    std::string bytes;
    {
        std::ifstream f(written, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    const std::string element("\x55\xBA\x81\x12", 4);
    const size_t at = bytes.find(element);
    REQUIRE(at != std::string::npos);
    REQUIRE(bytes.find(element, at + 1) == std::string::npos);
    bytes[at + 3] = 19;
    const auto in = dir / "t19.mkv";
    std::ofstream(in, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(Summarize(in).colorTrc == 19);  // matroskadec keeps it
    VideoDecoder probe(in, VideoDecoder::Probe());
    CHECK(probe.Info().color.effective.trc == 2);
    CHECK(probe.Info().color.cls == SourceClass::Wide);  // BT.2020 without a curve
    for (const char* ext : {".mkv", ".mov"}) {
        INFO(ext);
        const auto out = dir / (std::string("out") + ext);
        ProcessResult r;
        CHECK_NOTHROW(r = RunProcess(Passthrough(in, out, "ffv1")));
        CHECK(Summarize(out).colorTrc == 1);  // the wide class's curve filled
        CHECK_FALSE(Has(r.warnings, "from the first one"));  // the frames' 19 is unspecified too: no class change
        CHECK_FALSE(Has(r.warnings, "and on are"));
    }
    CHECK_FALSE(probe.Info().color.Note("source.layers_differ"));
}

TEST_CASE("An encoder outside the table writes the colour as planned, checks nothing back and publishes: utvideo of full-range PQ", "[integration][hdr][passthrough][result][regression]") {
    if (!VideoEncoder::EncoderAvailable("utvideo")) SKIP("no utvideo encoder in this FFmpeg");
    const auto dir = Dir("outside_table");
    const auto in = PqFfv1(dir, "full.mkv", {}, Tags(9, 9, 16, 2));
    VideoDecoder probe(in, VideoDecoder::Probe());
    const PassthroughColour pc = PlanPassthroughColour(probe.Info(), "utvideo", {}, "a.mkv", "t");
    CHECK(HasKey(pc.encodeNotes, "encode.unchecked"));
    REQUIRE(pc.expect);
    CHECK_FALSE(pc.expect->checkRange);
    CHECK(pc.expect->trc == 2);
    CHECK_FALSE(HasKey(PlanPassthroughColour(probe.Info(), "ffv1", {}, "a.mkv", "t").encodeNotes, "encode.unchecked"));
    // its decoder reads the matrix its own way (BT.601 / BT.709 only): a check of the tags would fail after the whole encode
    CHECK_NOTHROW(RunProcess(Passthrough(in, dir / "out.mkv", "utvideo")));
    CHECK(std::filesystem::exists(dir / "out.mkv"));
}

TEST_CASE("Frames saying no colour under a PQ colr are the container's PQ to the class observer: no warning", "[integration][hdr][passthrough]") {
    const auto dir = Dir("frames_bare");
    ClipSpec s = Small("prores_ks", "yuv422p10le");
    s.color = Tags(9, 9, 16);
    s.frameColor = [](int64_t) { return RawTags{0, 0, 0, 1, 0}; };  // prores_ks writes 0 into the header: unspecified for ProRes
    const auto in = WriteClip(dir / "bare_frames.mov", s);
    REQUIRE(FrameColors(in, 1, true).at(0).trc == 0);
    VideoDecoder probe(in, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Pq);
    for (const auto& [ext, codec] : {std::pair{".mov", "prores_ks"}, std::pair{".mkv", "ffv1"}}) {
        INFO(ext);
        const ProcessResult r = RunProcess(Passthrough(in, dir / (std::string("out") + ext), codec));
        CHECK_FALSE(Has(r.warnings, "from the first one"));
        CHECK_FALSE(Has(r.warnings, "and on are"));
    }
}

TEST_CASE("A MOV of a source with an unspecified tag left writes no colr and expects none; it publishes", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("no_colr");
    ClipSpec s = Small("prores_ks", "yuv422p10le");
    s.color = RawTags{2, 2, 16, 1, 0};
    s.frameColor = [](int64_t) { return RawTags{2, 2, 1, 1, 0}; };
    const auto in = WriteClip(dir / "mixed.mkv", s);
    VideoDecoder probe(in, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.Note("source.layers_differ"));
    const PassthroughColour pc = PlanPassthroughColour(probe.Info(), "prores_ks", {}, "a.mov", "t");
    REQUIRE(pc.expect);
    CHECK(pc.expect->matrix == 2);
    CHECK(pc.expect->primaries == 2);
    CHECK(pc.expect->trc == 2);
    CHECK_FALSE(pc.expect->checkRange);
    {  // NVENC writes its VUI where MP4 writes no colr: the tags said are expected back, those unsaid not
        const PassthroughColour nv = PlanPassthroughColour(probe.Info(), "hevc_nvenc", {}, "a.mp4", "t");
        REQUIRE(nv.expect);
        CHECK(nv.expect->trc == 1);
        CHECK(nv.expect->matrix == 2);
        CHECK(nv.expect->primaries == 2);
    }
    CHECK_NOTHROW(RunProcess(Passthrough(in, dir / "pt.mov", "prores_ks")));
    CHECK(std::filesystem::exists(dir / "pt.mov"));
}

TEST_CASE("Wide gamut SDR and a log curve pass through with their tags and without stale HDR records; into AVI they are refused", "[integration][hdr][passthrough]") {
    const auto dir = Dir("wide_log");
    ClipHdr stale;  // a container's records of an HDR master the SDR grade kept
    stale.mastering = Mastering1000();
    stale.light = Cll{1000, 400};
    for (const RawTags tags : {Tags(9, 9, 14), Tags(9, 9, 8)}) {
        INFO("tags " << tags.matrix << "/" << tags.primaries << "/" << tags.trc);
        const auto in = PqFfv1(dir, (std::to_string(tags.trc) + ".mkv").c_str(), stale, tags);
        REQUIRE(Summarize(in).hdr.mdcv);
        VideoDecoder probe(in, VideoDecoder::Probe());
        REQUIRE_FALSE(probe.Info().color.Hdr());
        REQUIRE_FALSE(probe.Info().color.legacyPath);
        CHECK(HasKey(PlanPassthroughColour(probe.Info(), "ffv1", {}, "a.mkv", "t").encodeNotes, "encode.records_dropped"));
        const auto out = dir / (std::to_string(tags.trc) + "_out.mkv");
        CHECK_FALSE(HasKey(PlanProcess(Passthrough(in, dir / "o.mp4", "h264_nvenc")).warnings, "encode.hdr_8bit"));  // no HDR to band
        CHECK_NOTHROW(RunProcess(Passthrough(in, out, "ffv1")));
        const StreamSummary s = Summarize(out);
        CHECK(s.colorSpace == 9);
        CHECK(s.colorTrc == tags.trc);
        CHECK_FALSE(s.hdr.mdcv);
        CHECK_FALSE(s.hdr.cll);
        try {
            RunProcess(Passthrough(in, dir / (std::to_string(tags.trc) + ".avi"), "ffv1"));
            FAIL("not refused");
        } catch (const ColourUnsupported& e) {
            CHECK(e.kind == ColourStop::NoContainerTags);
        }
    }
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
    const auto json = dir / "plan.json";
    const auto plan = RunAll({"process", "-i", pq.string(), "-o", (dir / "o.mp4").string(), "--passthrough", "--plan", "--codec", "hevc_nvenc", "--json",
                              json.string()},
                             dir / "pq.log");
    INFO(plan.out);
    CHECK(plan.code == 0);
    {  // the same notes in the plan's JSON: key, level, scope, text
        std::ifstream f(json);
        const nlohmann::json j = nlohmann::json::parse(f);
        REQUIRE(j.contains("warnings"));
        bool source = false, encode = false;
        for (const auto& w : j["warnings"]) {
            source = source || (w["key"] == "source.bt2020_assumed" && w["level"] == "note" && w["scope"] == "source" && !w["text"].get<std::string>().empty());
            encode = encode || (w["key"] == "encode.tags_filled" && w["scope"] == "encode");
        }
        CHECK(source);
        CHECK(encode);
    }
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
    {  // a file without them
        const auto bare = PqFfv1(dir, "bare.mkv");
        ResultColor want;
        want.mdcv = Mastering1000();
        e.color = want;
        CHECK(CheckEncodedResult(bare, e) == "it has no mastering display metadata, it was written");
        want.mdcv.reset();
        want.cll = Cll{1000, 400};
        e.color = want;
        CHECK(CheckEncodedResult(bare, e) == "it has no content light level, MaxCLL 1000 was written");
    }
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
    for (int which = 0; which < 6; ++which) {  // a primary, the white point (y, x), the halves said, the minimum luminance
        ResultColor m = c;
        if (which == 0) m.mdcv->primaries[1][0].num += 100;
        if (which == 1) m.mdcv->white[1].num += 100;
        if (which == 2) m.mdcv->hasPrimaries = false;
        if (which == 3) m.mdcv->white[0].num += 100;
        if (which == 4) m.mdcv->hasLuminance = false;
        if (which == 5) m.mdcv->minLum = {1, 100};  // 0.01 for 0.0001
        e.color = m;
        INFO("case " << which);
        CHECK(CheckEncodedResult(file, e) == "its mastering display metadata differs from what was written");
    }
    ResultColor amve = c;
    amve.amve = Amve{{314, 1}, {3127, 10000}, {329, 1000}};
    e.color = amve;
    CHECK_THAT(CheckEncodedResult(file, e), ContainsSubstring("ambient viewing environment"));
    ResultColor matrix = c;
    matrix.matrix = 1;
    e.color = matrix;
    CHECK(CheckEncodedResult(file, e) == "its matrix is bt2020nc, bt709 was written");
    // a MOV with its amve: one value off is named
    const auto hlg = HlgProRes(dir, "hlg.mov");
    ResultColor ambient;
    ambient.matrix = ambient.primaries = 9;
    ambient.trc = 14;  // the colr's
    ambient.amve = Amve{{3140000, 10000}, {15635, 50000}, {16450, 50000}};
    ResultExpectation m{64, 48, -1, false, PixelAspect{}};
    m.color = ambient;
    CHECK(CheckEncodedResult(hlg, m).empty());
    for (int field = 0; field < 3; ++field) {
        ResultColor off = ambient;
        (field == 0 ? off.amve->illuminance : field == 1 ? off.amve->lightX : off.amve->lightY).num += 100;
        m.color = off;
        INFO("field " << field);
        CHECK(CheckEncodedResult(hlg, m) == "its ambient viewing environment differs from what was written");
    }
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
    // HLG headers turning to BT.2020's SDR curve (ProRes keeps its tags in every frame: a real change, not H.264's lost SEI)
    {
        ClipSpec h = s;
        h.color = Tags(9, 9, 14);
        h.frameColor = [](int64_t i) { return i < 6 ? Tags(9, 9, 18) : Tags(9, 9, 14); };
        const auto hlg = WriteClip(dir / "hlg_turning.mov", h);
        VideoDecoder probe(hlg, VideoDecoder::Probe());
        REQUIRE(probe.Info().color.cls == SourceClass::Hlg);
        CHECK(Has(RunProcess(Passthrough(hlg, dir / "hlg_out.mov", "prores_ks")).warnings, "frame 6 and on are wide gamut SDR"));
    }
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
    // no encode (--no-encode): no container to refuse, no encode notes - as the run
    ProcessOptions noEncode = o;
    noEncode.output = dir / "none.avi";
    noEncode.encode = false;
    const ProcessPlan quiet = PlanProcess(noEncode);
    CHECK_FALSE(HasKey(quiet.warnings, "encode.tags_filled"));
    CHECK(HasKey(quiet.warnings, "source.bt2020_assumed"));
    // into AVI: refused in Prepare, before the passes' folder is made (bench's case in «refused before hashing» sees no hash)
    o.output = dir / "avi.avi";
    CHECK_THROWS_AS(RunProcess(o), ColourUnsupported);
    CHECK_FALSE(std::filesystem::exists(dir / "avi_passes"));
    CHECK_FALSE(std::filesystem::exists(dir / "avi.avi"));
}

TEST_CASE("The passthrough result check is wired: a codec option that rewrites the primaries fails the publish", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("wired");
    const auto pq = PqFfv1(dir, "pq.mkv");
    ProcessOptions o = Passthrough(pq, dir / "out.mkv", "ffv1");
    o.codecOptions = {{"color_primaries", "bt709"}};  // an AVCodecContext option: avcodec_open2 sets it over the source's
    CHECK_THROWS_WITH(RunProcess(o), ContainsSubstring("its primaries is bt709, bt2020 was written"));
    CHECK_FALSE(std::filesystem::exists(dir / "out.mkv"));
}

TEST_CASE("What a passthrough expects back per codec and container: tags, range, mastering halves, light level, nothing for SDR", "[integration][hdr][result]") {
    const auto dir = Dir("expect");
    ClipHdr hdr;
    hdr.mastering = Mastering1000();
    hdr.light = Cll{1000, 400};
    VideoDecoder probe(PqFfv1(dir, "pq.mkv", hdr), VideoDecoder::Probe());
    const VideoStreamInfo& info = probe.Info();
    const PassthroughColour mp4 = PlanPassthroughColour(info, "hevc_nvenc", {}, "a.mp4", "t");
    REQUIRE(mp4.expect);
    CHECK(mp4.expect->matrix == 9);
    CHECK(mp4.expect->primaries == 9);
    CHECK(mp4.expect->trc == 16);
    CHECK(mp4.expect->checkRange);
    REQUIRE(mp4.expect->mdcv);
    CHECK(mp4.expect->mdcv->hasPrimaries);  // (read back from Matroska's floats: not the ratios written)
    CHECK(std::llround(mp4.expect->mdcv->maxLum.Value()) == 1000);
    REQUIRE(mp4.expect->cll);
    CHECK(*mp4.expect->cll == Cll{1000, 400});
    const PassthroughColour mov = PlanPassthroughColour(info, "prores_ks", {}, "a.mov", "t");
    REQUIRE(mov.expect);
    CHECK_FALSE(mov.expect->checkRange);  // neither MOV nor ProRes keeps a range
    CHECK(mov.expect->mdcv);
    CHECK(mov.expect->cll);
    CHECK_FALSE(PlanPassthroughColour(info, "prores_aw", {}, "a.mov", "t").expect->checkRange);  // an encoder outside the table: no range
    VideoDecoder full(PqFfv1(dir, "full.mkv", hdr, Tags(9, 9, 16, 2)), VideoDecoder::Probe());
    for (const char* nvenc : {"hevc_nvenc", "h264_nvenc"}) {  // MOV keeps no range, the codec's VUI does
        INFO(nvenc);
        const PassthroughColour nv = PlanPassthroughColour(full.Info(), nvenc, {}, "a.mov", "t");
        REQUIRE(nv.expect);
        CHECK(nv.expect->checkRange);
        CHECK(nv.expect->range == 2);
        CHECK(nv.expect->trc == 16);
        CHECK_FALSE(HasKey(nv.encodeNotes, "encode.range_lost"));
    }
    CHECK(PlanPassthroughColour(info, "ffv1", {}, "a.mkv", "t").expect->checkRange);              // Matroska's Colour keeps it
    CHECK_FALSE(PlanPassthroughColour(info, "ffv1", {}, "a.mkv", "t").expect->amve);
    CHECK_FALSE(PlanPassthroughColour(info, "ffv1", {}, "", "t").expect);  // no output named yet: nothing to check
    {  // the ambient viewing environment: expected back in MOV and MP4 (Matroska has none)
        VideoDecoder hlg(HlgProRes(dir, "hlg.mov"), VideoDecoder::Probe());
        for (const auto& [codec, out] : {std::pair{"prores_ks", "a.mov"}, std::pair{"hevc_nvenc", "a.mp4"}}) {
            INFO(codec);
            const PassthroughColour h = PlanPassthroughColour(hlg.Info(), codec, {}, out, "t");
            REQUIRE(h.expect);
            REQUIRE(h.expect->amve);
            CHECK(h.expect->amve->illuminance.Value() == 314.0);
            CHECK(std::llround(h.expect->amve->lightX.Value() * 50000) == 15635);
            CHECK(std::llround(h.expect->amve->lightY.Value() * 50000) == 16450);
        }
        CHECK_FALSE(PlanPassthroughColour(hlg.Info(), "ffv1", {}, "a.mkv", "t").expect->amve);
    }
    VideoDecoder sdr(WriteClip(dir / "sdr.mkv", Small()), VideoDecoder::Probe());
    CHECK_FALSE(PlanPassthroughColour(sdr.Info(), "ffv1", {}, "a.mkv", "t").expect);  // SDR: checked as before
}

TEST_CASE("A mastering display Matroska's reader would drop is noted and not expected back; ProRes notes a full range in MKV", "[integration][hdr][passthrough][result]") {
    const auto dir = Dir("mdcv_invalid");
    ClipSpec s = Small("prores_ks", "yuv422p10le");
    s.color = Tags(9, 9, 16);
    Mdcv m = Mastering1000();
    m.maxLum = {0, 1};  // max not above min: MP4 / MOV keep it, matroskadec drops the luminance
    s.hdr.mastering = m;
    const auto in = WriteClip(dir / "in.mov", s);
    REQUIRE(Summarize(in).hdr.mdcv);
    REQUIRE(Summarize(in).hdr.mdcv->hasLuminance);
    const ProcessResult r = RunProcess(Passthrough(in, dir / "out.mkv", "ffv1"));
    CHECK(Has(r.warnings, "luminance does not hold"));
    CHECK(HasKey(PlanProcess(Passthrough(in, dir / "out.mkv", "ffv1")).warnings, "encode.mdcv_invalid"));
    const StreamSummary out = Summarize(dir / "out.mkv");
    REQUIRE(out.hdr.mdcv);
    CHECK(out.hdr.mdcv->hasPrimaries);
    CHECK_FALSE(out.hdr.mdcv->hasLuminance);
    // a zero coordinate: the primaries dropped, the luminance kept; both at once
    Mdcv zero = Mastering1000();
    zero.primaries[1][0] = {0, 50000};
    s.hdr.mastering = zero;
    const ProcessResult z = RunProcess(Passthrough(WriteClip(dir / "zero.mov", s), dir / "zero.mkv", "ffv1"));
    CHECK(Has(z.warnings, "primaries do not hold"));
    REQUIRE(Summarize(dir / "zero.mkv").hdr.mdcv);
    CHECK_FALSE(Summarize(dir / "zero.mkv").hdr.mdcv->hasPrimaries);
    CHECK(Summarize(dir / "zero.mkv").hdr.mdcv->hasLuminance);
    for (int field = 0; field < 8; ++field) {  // each coordinate 0, alone: the primaries' half dropped
        INFO("coordinate " << field);
        Mdcv one = Mastering1000();
        (field < 6 ? one.primaries[field / 2][field % 2] : one.white[field - 6]) = {0, 50000};
        s.hdr.mastering = one;
        VideoDecoder probe(WriteClip(dir / ("zero" + std::to_string(field) + ".mov"), s), VideoDecoder::Probe());
        const PassthroughColour mkv = PlanPassthroughColour(probe.Info(), "ffv1", {}, "a.mkv", "t");
        CHECK(HasKey(mkv.encodeNotes, "encode.mdcv_invalid"));
        REQUIRE(mkv.expect);
        REQUIRE(mkv.expect->mdcv);
        CHECK_FALSE(mkv.expect->mdcv->hasPrimaries);
        CHECK(mkv.expect->mdcv->hasLuminance);
    }
    // the luminance's other two edges: a minimum below 0 (as mov.c reads a u32 past INT32_MAX) and max equal to min
    for (int which = 0; which < 2; ++which) {
        INFO("case " << which);
        Mdcv lum = Mastering1000();
        if (which == 0) lum.minLum = {-1, 10000};
        if (which == 1) lum.maxLum = lum.minLum = {1, 10000};
        s.hdr.mastering = lum;
        const auto src = WriteClip(dir / ("lum" + std::to_string(which) + ".mov"), s);
        VideoDecoder probe(src, VideoDecoder::Probe());
        REQUIRE(probe.Info().color.hdr.mdcv);
        if (which == 0) REQUIRE(probe.Info().color.hdr.mdcv->minLum.Value() < 0);
        const auto out = dir / ("lum" + std::to_string(which) + ".mkv");
        CHECK(HasKey(PlanProcess(Passthrough(src, out, "ffv1")).warnings, "encode.mdcv_invalid"));
        ProcessResult lr;
        CHECK_NOTHROW(lr = RunProcess(Passthrough(src, out, "ffv1")));
        CHECK(Has(lr.warnings, "luminance does not hold (a coordinate 0, a minimum below 0 or max not above min)"));
        REQUIRE(Summarize(out).hdr.mdcv);
        CHECK(Summarize(out).hdr.mdcv->hasPrimaries);
        CHECK_FALSE(Summarize(out).hdr.mdcv->hasLuminance);
    }
    zero.maxLum = {0, 1};
    s.hdr.mastering = zero;
    const ProcessResult both = RunProcess(Passthrough(WriteClip(dir / "both.mov", s), dir / "both.mkv", "ffv1"));
    CHECK(Has(both.warnings, "primaries and luminance do not hold"));
    CHECK_FALSE(Summarize(dir / "both.mkv").hdr.mdcv);
    // a luminance past INT_MAX in Matroska's floats: {1, 0} from av_d2q, written back as inf - kept by matroskadec, expected, published
    Mdcv huge = Mastering1000();
    huge.maxLum = {1, 0};
    ClipHdr hugeHdr;
    hugeHdr.mastering = huge;
    const auto hugeIn = PqFfv1(dir, "huge.mkv", hugeHdr);
    REQUIRE(Summarize(hugeIn).hdr.mdcv);
    REQUIRE(Summarize(hugeIn).hdr.mdcv->hasLuminance);
    ProcessResult hr;
    CHECK_NOTHROW(hr = RunProcess(Passthrough(hugeIn, dir / "huge_out.mkv", "ffv1")));
    CHECK_FALSE(Has(hr.warnings, "does not hold"));
    // full range into ProRes in MKV: the container says full, FFmpeg's ProRes decoder says limited
    const auto full = PqFfv1(dir, "full.mkv", {}, Tags(9, 9, 16, 2));
    const ProcessResult p = RunProcess(Passthrough(full, dir / "full_prores.mkv", "prores_ks"));
    CHECK(Has(p.warnings, "ProRes keeps no range"));
}

TEST_CASE("Passthrough of ICtCp PQ FFV1 MKV keeps matrix 14; processing it is not supported", "[integration][hdr][passthrough]") {
    const auto dir = Dir("ictcp");
    const auto in = PqFfv1(dir, "ictcp.mkv", {}, Tags(14, 9, 16));
    VideoDecoder probe(in, VideoDecoder::Probe());
    REQUIRE(ResolveColor(probe.Info()).process == ColourStop::Matrix);
    RunProcess(Passthrough(in, dir / "out.mkv", "ffv1"));
    const StreamSummary s = Summarize(dir / "out.mkv");
    CHECK(s.colorSpace == 14);
    CHECK(s.colorTrc == 16);
    const auto info = RunAll({"info", in.string()}, dir / "info.log");
    INFO(info.out);
    CHECK_THAT(info.out, ContainsSubstring("processing:  not supported - matrix ictcp"));
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
    ClipSpec sdr = Small("ffv1", "yuv420p");
    sdr.color = RawTags{2, 1, 1, 1, 0};
    const auto sdrClip = WriteClip(dir / "sdr.mkv", sdr);
    const auto sdrProxy = BuildProxy(sdrClip, dir / "passes2", "source");
    CHECK(Summarize(sdrProxy).colorTrc == 1);
    CHECK(Summarize(sdrProxy).colorSpace == 2);  // not ResultColorTags' BT.601 by the height
    // SDR frame headers under a PQ colr (class SDR, off the legacy path): the frames' tags, not the colr's
    ClipSpec mixed = Small("prores_ks", "yuv422p10le");
    mixed.color = Tags(9, 9, 16);
    mixed.frameColor = [](int64_t) { return Tags(1, 1, 1); };
    const auto mixedProxy = BuildProxy(WriteClip(dir / "mixed.mov", mixed), dir / "passes3", "source");
    CHECK(Summarize(mixedProxy).colorTrc == 1);
    CHECK(Summarize(mixedProxy).colorSpace == 1);
    CHECK(Summarize(mixedProxy).colorPrimaries == 1);
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
    const auto compare = RunAll({"compare", "--ref", pq.string(), "--test", pq.string(), "--frames", "2"}, dir / "compare.log");
    {
        INFO(compare.out);
        CHECK(compare.code == 0);
        CHECK_THAT(compare.out, !ContainsSubstring("processing of HDR"));
    }
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
    const auto rgb = RunAll({"info", PqFfv1(dir, "rgb_matrix.mkv", hdr, RawTags{0, 9, 16, 1, 0}).string()}, dir / "rgb.log");
    INFO(rgb.out);
    CHECK_THAT(rgb.out, ContainsSubstring("processing:  not supported - the RGB matrix on a Y'CbCr stream"));
    CHECK_THAT(rgb.out, !ContainsSubstring("hdr note:"));
    const auto bare = RunAll({"info", PqFfv1(dir, "bare.mkv").string()}, dir / "bare.log");
    CHECK_THAT(bare.out, ContainsSubstring("processing:  not yet"));
    CHECK_THAT(bare.out, !ContainsSubstring("hdr note:"));
    ClipHdr stale;
    stale.mastering = Mastering1000();
    stale.light = Cll{1000, 400};
    const auto wide = RunAll({"info", PqFfv1(dir, "wide.mkv", stale, Tags(9, 9, 14)).string()}, dir / "wide.log");
    INFO(wide.out);
    CHECK_THAT(wide.out, ContainsSubstring("processing:  not yet"));
    CHECK_THAT(wide.out, !ContainsSubstring("hdr note:"));
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
    // a pass shorter than the source (an MKV has no frame count: the pass of a --frames run is taken whole), the change past its end:
    // the encode decodes the source on for its audio, but the change is not in the result - no warning
    const auto mkv = WriteClip(dir / "turning.mkv", s);
    ProcessOptions part = o;
    part.input = mkv;
    part.output = dir / "part.mkv";
    part.frames = 4;
    REQUIRE(RunProcess(part).framesOut == 4);
    ProcessOptions whole = part;
    whole.frames = -1;
    const ProcessResult w = RunProcess(whole);
    REQUIRE(w.framesOut == 4);  // the short pass reused
    CHECK_FALSE(Has(w.warnings, "and on are"));
    for (const auto& [frames, warns] : {std::pair{6, false}, std::pair{7, true}}) {  // the change at source frame 6: held from 7 frames on
        INFO("frames " << frames);
        ProcessOptions p = part;
        p.output = dir / ("part" + std::to_string(frames) + ".mkv");
        p.frames = frames;
        REQUIRE(RunProcess(p).framesOut == frames);
        ProcessOptions all = p;
        all.frames = -1;
        const ProcessResult a = RunProcess(all);
        REQUIRE(a.framesOut == frames);
        CHECK(Has(a.warnings, "frame 6 and on are HDR PQ") == warns);
    }
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
    s.hdr.mastering = Mastering1000();  // a stale colr's records: not written with SDR tags
    s.hdr.light = Cll{1000, 400};
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
    CHECK_FALSE(Summarize(dir / "pt.mov").hdr.mdcv);
    CHECK_FALSE(Summarize(dir / "pt.mov").hdr.cll);
    CHECK_NOTHROW(RunProcess(Passthrough(in, dir / "pt.avi", "ffv1")));  // SDR frames: an AVI as any SDR video
    CHECK(std::filesystem::exists(dir / "pt.avi"));
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
    CHECK(HasKey(PlanProcess(Passthrough(partial, dir / "lum.mov", "prores_ks")).warnings, "encode.mdcv_lost"));
    CHECK_FALSE(Summarize(dir / "lum.mov").hdr.mdcv);
    const auto full = PqFfv1(dir, "full.mkv", {}, Tags(9, 9, 16, 2));
    const ProcessResult b = RunProcess(Passthrough(full, dir / "full.mov", "ffv1"));
    CHECK(Has(b.warnings, "full range"));
    CHECK(HasKey(PlanProcess(Passthrough(full, dir / "full.mov", "ffv1")).warnings, "encode.range_lost"));
    CHECK(std::filesystem::exists(dir / "full.mov"));
}

TEST_CASE("Container colour caps by muxer: mp4 and mov keep amve, mkv not, mov no range, avi m4v ts nothing", "[integration][hdr][result]") {
    const auto mp4 = VideoEncoder::ContainerColorCapsFor("a.mp4");
    CHECK(mp4.tags);
    CHECK(mp4.allThreeOrNone);
    CHECK(mp4.range);
    CHECK(mp4.mdcvNeedsBoth);
    CHECK(mp4.amve);
    CHECK(mp4.mdcv);
    CHECK(mp4.cll);
    const auto mov = VideoEncoder::ContainerColorCapsFor("a.mov");
    CHECK(mov.tags);
    CHECK_FALSE(mov.range);
    CHECK(mov.amve);
    CHECK(mov.mdcv);
    CHECK(mov.cll);
    for (const char* f : {"a.mkv", "a.webm"}) {
        const auto mkv = VideoEncoder::ContainerColorCapsFor(f);
        CHECK(mkv.tags);
        CHECK_FALSE(mkv.allThreeOrNone);
        CHECK(mkv.range);
        CHECK_FALSE(mkv.mdcvNeedsBoth);
        CHECK_FALSE(mkv.amve);
        CHECK(mkv.mdcv);
        CHECK(mkv.cll);
    }
    for (const char* f : {"a.avi", "a.m4v", "a.ts", "a.hevc"}) {
        INFO(f);
        const auto none = VideoEncoder::ContainerColorCapsFor(f);
        CHECK_FALSE(none.tags);
        CHECK_FALSE(none.range);
        CHECK_FALSE(none.mdcv);
        CHECK_FALSE(none.cll);
    }
    CHECK(CodecKeepsRange("hevc_nvenc"));
    CHECK(CodecKeepsRange("av1_nvenc"));
    CHECK(CodecKeepsRange("h264_nvenc"));
    CHECK_FALSE(CodecKeepsRange("prores_ks"));
    CHECK_FALSE(CodecKeepsRange("prores_aw"));  // an encoder outside the table: its range is not known
    CHECK_FALSE(CodecKeepsRange("ffv1"));
}

// ---- NVENC ----------------------------------------------------------------------------------------------------------------------

namespace {
bool HaveNvenc(const char* codec) {
    D3D12Device dev;
    return !dev.IsWarp() && dev.IsNvidia() && VideoEncoder::EncoderAvailable(codec);
}

// RunProcess; nullopt when NVENC says this GPU lacks the encoder (av1_nvenc before Ada) - that codec skipped, any other failure fails
std::optional<ProcessResult> RunIfSupported(const ProcessOptions& o) {
    FfmpegLogCapture log;
    try {
        return RunProcess(o);
    } catch (const std::exception& e) {
        if (!SaysUnsupported(log.Text()) && !SaysUnsupported(e.what())) throw;
        WARN(o.codec << " is not supported on this GPU: " << e.what());
        return std::nullopt;
    }
}

// an NVENC clip of 320x192 (NVENC's minimum is above 64x48); nullopt when NVENC refuses it here
std::optional<std::filesystem::path> NvencClip(const std::filesystem::path& file, ClipSpec spec, std::string* why) {
    spec.codec = "hevc_nvenc";
    spec.width = 320;
    spec.height = 192;
    return WriteClipIfSupported(file, spec, why);
}
}  // namespace

TEST_CASE("NVENC passthrough of PQ and HLG writes VUI 9/16/9 and 9/18/9, colr mdcv clli and amve in MP4", "[integration][hdr][gpu][formats]") {
    const auto dir = Dir("nvenc");
    for (const char* codec : {"hevc_nvenc", "av1_nvenc"}) {
        if (!HaveNvenc(codec)) continue;
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
        const std::optional<ProcessResult> r = RunIfSupported(Passthrough(pq, out, codec));
        if (!r) continue;
        CHECK(r->framesOut == 4);
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
    CHECK(std::llround(got.hdr.amve->lightX.Value() * 50000) == 15635);
    CHECK(std::llround(got.hdr.amve->lightY.Value() * 50000) == 16450);
}

TEST_CASE("HEVC HLG with the transfer in an SEI passes through tagged trc 18; under a BT.709 VUI it warns from the first frame", "[integration][hdr][gpu][passthrough]") {
    if (!HaveNvenc("hevc_nvenc")) SKIP("no NVIDIA GPU with hevc_nvenc");
    const auto dir = Dir("hevc_sei");
    ClipSpec spec;
    spec.frames = 4;
    spec.pixFmt = "p010le";
    spec.color = Tags(9, 9, 14);
    spec.videoPacket = [](int64_t, const std::string& au) { return WithNalBeforeSlice(au, AlternativeTransferSei(18)); };
    std::string why;
    const auto hlg = NvencClip(dir / "hlg_sei.mp4", spec, &why);
    if (!hlg) SKIP("NVENC does not write the clip here: " << why);
    VideoDecoder probe(*hlg, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Hlg);
    const ProcessResult r = RunProcess(Passthrough(*hlg, dir / "out.mkv", "ffv1"));
    CHECK(Summarize(dir / "out.mkv").colorTrc == 18);
    CHECK_FALSE(Has(r.warnings, "from the first one"));
    // 8-bit, VUI 1/1/1: nothing in the tags says HDR, no first frame is looked at - the frames are HLG from the start
    spec.pixFmt = "yuv420p";
    spec.color = Tags(1, 1, 1);
    const auto sdrVui = NvencClip(dir / "sdr_vui.mp4", spec, &why);
    if (!sdrVui) SKIP("NVENC does not write the clip here: " << why);
    VideoDecoder p709(*sdrVui, VideoDecoder::Probe());
    REQUIRE(p709.Info().color.cls == SourceClass::Legacy);
    const ProcessResult w = RunProcess(Passthrough(*sdrVui, dir / "out709.mkv", "ffv1"));
    CHECK(Has(w.warnings, "the frames are HDR HLG"));
    CHECK(Has(w.warnings, "from the first one"));
    CHECK_FALSE(Has(w.warnings, "cut the video"));
}

TEST_CASE("H.264 HLG with the alternative transfer SEI on its key frame only passes through without a class change in frame threads", "[integration][hdr][gpu][regression]") {
    if (!HaveNvenc("h264_nvenc")) SKIP("no NVIDIA GPU with h264_nvenc");
    const auto dir = Dir("h264_sei_key");
    ClipSpec spec;
    spec.codec = "h264_nvenc";
    spec.width = 320;
    spec.height = 192;
    spec.frames = 12;
    spec.color = Tags(9, 9, 14);
    // as x264 writes it: the SEI on the key frame only (the stream's one IDR at frame 0)
    spec.videoPacket = [](int64_t i, const std::string& au) { return i == 0 ? WithNalBeforeH264Slice(au, H264AlternativeTransferSei(18)) : au; };
    std::string why;
    const auto clip = WriteClipIfSupported(dir / "hlg_key.mkv", spec, &why);
    if (!clip) SKIP("NVENC does not write the clip here: " << why);
    VideoDecoder probe(*clip, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Hlg);  // (the stream's 18 already: find_stream_info decodes a frame of H.264)
    for (const char* hw : {"none", "cuda"}) {
        INFO("hwaccel " << hw);
        ProcessOptions o = Passthrough(*clip, dir / (std::string("out_") + hw + ".mkv"), "ffv1");
        o.hwaccel = hw;
        const ProcessResult r = RunProcess(o);
        CHECK(r.framesOut == 12);
        CHECK_FALSE(Has(r.warnings, "and on are"));
        CHECK_FALSE(Has(r.warnings, "the frames are"));
        CHECK(Summarize(o.output).colorTrc == 18);
    }
}

TEST_CASE("Raw H.264 of HLG by an SEI then BT.709 warns of the change in software decoding", "[integration][hdr][gpu][regression]") {
    if (!HaveNvenc("h264_nvenc")) SKIP("no NVIDIA GPU with h264_nvenc");
    const auto dir = Dir("h264_hlg_then_709");
    ClipSpec spec;
    spec.codec = "h264_nvenc";
    spec.width = 320;
    spec.height = 192;
    spec.frames = 8;
    spec.color = Tags(9, 9, 14);
    spec.videoPacket = [](int64_t i, const std::string& au) { return i == 0 ? WithNalBeforeH264Slice(au, H264AlternativeTransferSei(18)) : au; };
    std::string why;
    const auto hlg = WriteClipIfSupported(dir / "hlg.h264", spec, &why);
    if (!hlg) SKIP("NVENC does not write the clip here: " << why);
    spec.color = Tags(1, 1, 1);
    spec.videoPacket = {};
    const auto sdr = WriteClipIfSupported(dir / "sdr.h264", spec, &why);
    REQUIRE(sdr);
    const auto joined = ConcatFiles(*hlg, *sdr, dir / "joined.h264");
    VideoDecoder probe(joined, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Hlg);
    // software, frame threads: the BT.709 part's frames from a thread that never parsed the SEI read 1/1/1 - not taken for a lost SEI
    // (its primaries and matrix differ). The frame number depends on the threads; by NVDEC (one thread) FFmpeg keeps the SEI's 18 on: none
    const ProcessResult r = RunProcess(Passthrough(joined, dir / "out.mkv", "ffv1"));
    CHECK(Has(r.warnings, "and on are SDR"));
}

TEST_CASE("HEVC MP4 with a PQ colr and no colour in its VUI passes through without a class change, in software and by NVDEC", "[integration][hdr][gpu][nvdec]") {
    if (!HaveNvenc("hevc_nvenc")) SKIP("no NVIDIA GPU with hevc_nvenc");
    const auto dir = Dir("hevc_colr_only");
    ClipSpec spec;
    spec.frames = 4;
    spec.pixFmt = "p010le";
    spec.containerColor = Tags(9, 9, 16);
    std::string why;
    const auto clip = NvencClip(dir / "pq_colr.mp4", spec, &why);
    if (!clip) SKIP("NVENC does not write the clip here: " << why);
    REQUIRE(Summarize(*clip).colorTrc == 2);  // the VUI says none; the colr says PQ
    VideoDecoder probe(*clip, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Pq);
    for (const char* hw : {"none", "cuda"}) {
        INFO("hwaccel " << hw);
        ProcessOptions o = Passthrough(*clip, dir / (std::string("out_") + hw + ".mkv"), "ffv1");
        o.hwaccel = hw;
        const ProcessResult r = RunProcess(o);
        CHECK_FALSE(Has(r.warnings, "from the first one"));
        CHECK_FALSE(Has(r.warnings, "and on are"));
        CHECK(Summarize(o.output).colorTrc == 16);
    }
}

TEST_CASE("Raw HEVC of SDR then PQ segments warns at the first PQ frame, decoded in software and by NVDEC; 1/1/1 then 6/6/6 does not", "[integration][hdr][gpu][nvdec]") {
    if (!HaveNvenc("hevc_nvenc")) SKIP("no NVIDIA GPU with hevc_nvenc");
    const auto dir = Dir("raw_segments");
    ClipSpec spec;
    spec.frames = 6;
    spec.pixFmt = "p010le";  // one shape: a new SPS of the same size and depth (NVDEC keeps its decoder)
    std::string why;
    spec.color = Tags(1, 1, 1);
    const auto sdr = NvencClip(dir / "sdr.hevc", spec, &why);
    if (!sdr) SKIP("NVENC does not write the clip here: " << why);
    spec.color = Tags(9, 9, 16);
    const auto pq = NvencClip(dir / "pq.hevc", spec, &why);
    spec.color = Tags(6, 6, 6);
    const auto bt601 = NvencClip(dir / "601.hevc", spec, &why);
    REQUIRE(pq);
    REQUIRE(bt601);
    const auto turning = ConcatFiles(*sdr, *pq, dir / "turning.hevc");
    const auto sdrOnly = ConcatFiles(*sdr, *bt601, dir / "sdr_only.hevc");
    for (const char* hw : {"none", "cuda"}) {
        INFO("hwaccel " << hw);
        ProcessOptions o = Passthrough(turning, dir / (std::string("turning_") + hw + ".mkv"), "ffv1");
        o.hwaccel = hw;
        const ProcessResult r = RunProcess(o);
        CHECK(r.framesOut == 12);
        CHECK(Has(r.warnings, "frame 6 and on are HDR PQ"));
        ProcessOptions q = Passthrough(sdrOnly, dir / (std::string("sdr_") + hw + ".mkv"), "ffv1");
        q.hwaccel = hw;
        CHECK_FALSE(Has(RunProcess(q).warnings, "and on are"));
    }
    // PQ then a part whose SPS has no colour description (x265's default): its frames say nothing - read as SDR, not as the first part's PQ
    spec.color = RawTags{2, 2, 2, 1, 0};
    const auto bare = NvencClip(dir / "bare.hevc", spec, &why);
    REQUIRE(bare);
    const auto pqThenBare = ConcatFiles(*pq, *bare, dir / "pq_bare.hevc");
    VideoDecoder probe(pqThenBare, VideoDecoder::Probe());
    REQUIRE(probe.Info().color.cls == SourceClass::Pq);
    for (const char* hw : {"none", "cuda"}) {
        INFO("hwaccel " << hw);
        ProcessOptions o = Passthrough(pqThenBare, dir / (std::string("pq_bare_") + hw + ".mkv"), "ffv1");
        o.hwaccel = hw;
        const ProcessResult r = RunProcess(o);
        CHECK(Has(r.warnings, "frame 6 and on are SDR"));
        CHECK_FALSE(Has(r.warnings, "range unknown"));  // a frame that says no range: the source's
    }
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
            const std::optional<ProcessResult> r = RunIfSupported(Passthrough(in, out, codec));
            if (!r) break;  // this GPU lacks the encoder
            CHECK(r->warnings.empty());
            const StreamSummary got = Summarize(out);
            CHECK(got.colorSpace == 1);
            CHECK(got.colorPrimaries == 1);
            CHECK(got.colorTrc == 1);
            CHECK_FALSE(got.hdr.mdcv);
        }
    }
}
