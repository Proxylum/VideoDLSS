// TASK-0080 (HDR step TC-1): guards of the SDR path. TC-2 reads HDR sources into SDR and labels their passes with colour tokens; an SDR
// source (the legacy path of ColorPlan) must keep what it writes. These tests hold what main writes today, before anything changes, and
// stay unchanged through TC-2 and TC-3: the export and NR manifests and the SettingsNote as literals, the previews as the pass frame put
// back by the source's decode matrix, info / --plan / the run report without colour lines, compare's warning and JSON keys.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "passes/PassSequence.h"
#include "pipeline/SettingsNote.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const std::string& name) {
    const auto d = TempDir() / "sdr_guards" / name;
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

nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream f(p);
    return nlohmann::json::parse(f);
}

ClipSpec Sdr(const char* pixFmt, int frames = 3) {
    ClipSpec s;
    s.width = 64;
    s.height = 48;
    s.frames = frames;
    s.pixFmt = pixFmt;
    return s;
}

bool SamePlanes(const CpuFrame& a, const CpuFrame& b) {
    if (a.desc.width != b.desc.width || a.desc.height != b.desc.height || a.desc.format != b.desc.format) return false;
    for (int p = 0; p < 3; ++p)
        if (!std::equal(a.Plane(p), a.Plane(p) + a.desc.PlaneSize(p), b.Plane(p))) return false;
    return true;
}

}  // namespace

TEST_CASE("upscale, nr and fg --video of an SDR clip write untagged YUV of the decode matrix as before", "[integration][hdr][sdr][regression]") {
    struct Cmd {
        const char* stage;
        std::vector<std::string> extra;
        const char* pass;
    };
    const std::vector<Cmd> cmds = {{"upscale", {"--backend", "bicubic"}, "color_sr"},
                                   {"nr", {"--backend", "stub", "--intensity", "0"}, "color_nr"},
                                   {"fg", {"--backend", "blend", "--multiplier", "2"}, "color_fg"}};
    struct Clip {
        const char* pixFmt;
        bool tagged;  // BT.709 tags: the decode matrix by the tags, where the height rule (48 lines) would take BT.601
    };
    for (const Clip cl : {Clip{"yuv420p", false}, Clip{"yuv420p10le", false}, Clip{"yuv420p", true}}) {
        const char* pixFmt = cl.pixFmt;
        const auto dir = Dir(std::string("video_") + pixFmt + (cl.tagged ? "_709" : ""));
        ClipSpec spec = Sdr(pixFmt);
        if (cl.tagged) spec.color = RawTags{1, 1, 1, 1, 0};
        const auto clip = WriteClip(dir / "clip.mkv", spec);
        const ColorInfo ci = ColorInfoFromStream(VideoDecoder(clip).Info());
        CHECK(ci.matrix == (cl.tagged ? ColorMatrix::Bt709 : ColorMatrix::Bt601));
        for (const Cmd& c : cmds) {
            INFO(pixFmt << (cl.tagged ? " bt709" : "") << " " << c.stage);
            const auto passes = dir / (std::string(c.stage) + "_passes");
            const auto video = dir / (std::string(c.stage) + ".mkv");
            std::vector<std::string> args{c.stage, "-i", clip.string(), "-o", passes.string(), "--warp", "--video", video.string(), "--codec", "ffv1"};
            args.insert(args.end(), c.extra.begin(), c.extra.end());
            const CliRun r = RunAll(args, dir / (std::string(c.stage) + ".log"));
            INFO(r.out);
            REQUIRE(r.code == 0);
            // the preview says no colour: players read it by their defaults, as before
            const std::vector<RawTags> tags = FrameColors(video);
            REQUIRE(tags.size() == 1);
            CHECK(tags[0] == RawTags{2, 2, 2, 0, 0});
            // its Y'CbCr is the pass frame put back by the source's decode matrix
            const PassReader rd = PassReader::Open(passes / c.pass);
            const std::vector<CpuFrame> frames = DecodeAll(video);
            REQUIRE(static_cast<int64_t>(frames.size()) == rd.Man().frameCount);
            for (size_t i = 0; i < frames.size(); ++i) {
                CpuFrame expected;
                RgbToYuv420p(rd.ReadFrame(rd.Man().firstFrame + static_cast<int64_t>(i)), ci, expected);
                INFO("frame " << i);
                CHECK(SamePlanes(frames[i], expected));
            }
        }
    }
}

TEST_CASE("SDR info, process --plan and the report carry no colour lines", "[integration][hdr][sdr][regression]") {
    const std::vector<std::string> keys{"processing:", "colour:", "hdr note", "convert.", "encode.as_is", "encode.untagged_matrix"};
    for (const char* pixFmt : {"yuv420p", "yuv420p10le"}) {
        const auto dir = Dir(std::string("lines_") + pixFmt);
        ClipSpec spec = Sdr(pixFmt);
        const auto clip = WriteClip(dir / "clip.mkv", spec);
        spec.color = RawTags{1, 1, 1, 1, 0};
        const auto tagged = WriteClip(dir / "tagged.mkv", spec);
        for (const auto& c : {clip, tagged}) {
            INFO(pixFmt << " " << c.filename().string());
            const CliRun info = RunAll({"info", c.string()}, dir / "info.log");
            INFO(info.out);
            REQUIRE(info.code == 0);
            for (const auto& k : keys) CHECK_FALSE(info.out.find(k) != std::string::npos);
            // a colour stage and a guide stage alone (TC-2's «as it is» note is for non-SDR sources only)
            for (const char* stages : {"upscale", "depth"}) {
                INFO(stages);
                const auto json = dir / "plan.json";
                const CliRun plan = RunAll({"process", "-i", c.string(), "-o", (dir / "out.mkv").string(), "--passes", (dir / "passes").string(), "--warp", "--plan",
                                            "--stages", stages, "--param", "upscale.backend=bicubic", "--param", "depth.backend=stub", "--codec", "ffv1",
                                            "--json", json.string()},
                                           dir / "plan.log");
                INFO(plan.out);
                REQUIRE(plan.code == 0);
                for (const auto& k : keys) CHECK_FALSE(plan.out.find(k) != std::string::npos);
                const nlohmann::json j = ReadJson(json);
                CHECK_FALSE(j.contains("colour"));
                for (const auto& w : j.value("warnings", nlohmann::json::array())) {
                    INFO(w.dump());
                    for (const auto& k : keys) CHECK_FALSE(w.value("key", "").rfind(k, 0) == 0);
                }
            }
        }
        // the run's report
        const auto json = dir / "report.json";
        const CliRun run = RunAll({"process", "-i", clip.string(), "-o", (dir / "out.mkv").string(), "--passes", (dir / "run_passes").string(), "--warp", "--stages",
                                   "upscale", "--param", "upscale.backend=bicubic", "--codec", "ffv1", "--json", json.string()},
                                  dir / "run.log");
        INFO(run.out);
        REQUIRE(run.code == 0);
        for (const auto& k : keys) CHECK_FALSE(run.out.find(k) != std::string::npos);
        const nlohmann::json j = ReadJson(json);
        CHECK_FALSE(j.contains("colour"));
        CHECK(j.value("warnings", nlohmann::json::array()) == nlohmann::json::array());  // the report's warnings are texts: none here
    }
}

TEST_CASE("SDR export keeps its manifest", "[integration][hdr][sdr][regression]") {
    const auto dir = Dir("export");
    const auto clip = WriteClip(dir / "clip.mkv", Sdr("yuv420p10le", 2));
    const CliRun r = RunAll({"export", "-i", clip.string(), "-o", (dir / "exported").string()}, dir / "export.log");
    INFO(r.out);
    REQUIRE(r.code == 0);
    nlohmann::json j = ReadJson(dir / "exported" / Manifest::kFileName);
    j["source"].erase("hash");  // the clip's bytes
    CHECK(j.dump(2) == R"({
  "channels": [
    "R",
    "G",
    "B"
  ],
  "colorspace": "srgb",
  "convention": "raw",
  "depth": {
    "far": 1000.0,
    "max": 0.0,
    "min": 0.0,
    "near": 0.10000000149011612,
    "relative": false,
    "units": "meters"
  },
  "file_pattern": "color_source_%06d.exr",
  "format": "exr",
  "fps": {
    "den": 1,
    "num": 24
  },
  "frame_count": 2,
  "frame_range": {
    "first": 0,
    "last": 1
  },
  "height": 48,
  "model": {
    "name": "",
    "version": ""
  },
  "mv": {
    "direction": "forward",
    "ref_height": 0,
    "ref_width": 0,
    "y_up": false
  },
  "pass": "color_source",
  "pixel_type": "f16",
  "schema_version": 1,
  "source": {
    "file": "clip.mkv"
  },
  "stage_params": {
    "matrix": "bt601",
    "range": "limited"
  },
  "width": 64
})");
    // from a pass folder
    const CliRun f = RunAll({"export", "--from-dir", (dir / "exported").string(), "-o", (dir / "png").string(), "--format", "png16"}, dir / "from.log");
    INFO(f.out);
    REQUIRE(f.code == 0);
    nlohmann::json p = ReadJson(dir / "png" / Manifest::kFileName);
    p["source"].erase("hash");
    CHECK(p.dump(2) == R"({
  "channels": [
    "R",
    "G",
    "B"
  ],
  "colorspace": "srgb",
  "convention": "raw",
  "depth": {
    "far": 1000.0,
    "max": 0.0,
    "min": 0.0,
    "near": 0.10000000149011612,
    "relative": false,
    "units": "meters"
  },
  "file_pattern": "color_source_%06d.png",
  "format": "png16",
  "fps": {
    "den": 1,
    "num": 24
  },
  "frame_count": 2,
  "frame_range": {
    "first": 0,
    "last": 1
  },
  "height": 48,
  "model": {
    "name": "",
    "version": ""
  },
  "mv": {
    "direction": "forward",
    "ref_height": 0,
    "ref_width": 0,
    "y_up": false
  },
  "pass": "color_source",
  "pixel_type": "u16",
  "schema_version": 1,
  "source": {
    "file": "clip.mkv"
  },
  "stage_params": {
    "matrix": "bt601",
    "range": "limited"
  },
  "width": 64
})");
}

TEST_CASE("NR with input transfer pq on an SDR clip keeps its manifest and note", "[integration][hdr][sdr][regression]") {
    const auto dir = Dir("nr_pq");
    const auto clip = WriteClip(dir / "clip.mkv", Sdr("yuv420p10le", 2));
    const auto passes = dir / "passes";
    const CliRun r = RunAll({"process", "-i", clip.string(), "-o", (dir / "out.mkv").string(), "--passes", passes.string(), "--warp", "--stages", "nr", "--param",
                             "nr.backend=stub", "--param", "nr.intensity=0", "--param", "nr.input_transfer=pq", "--codec", "ffv1"},
                            dir / "run.log");
    INFO(r.out);
    REQUIRE(r.code == 0);
    const nlohmann::json j = ReadJson(passes / "color_nr" / Manifest::kFileName);
    CHECK(j["stage_params"]["tonemap"].dump() == R"({"curve":"passthrough","exposure_ev":0.0,"input_transfer":"pq"})");
    CHECK_FALSE(j["stage_params"].contains("input_transfer_asked"));
    CHECK_FALSE(j["stage_params"].contains("input_transfer_note"));
    const std::string note = BuildSettingsNote(passes, "color_nr");  // after the app's name and version
    CHECK(note.rfind("VideoDLSS ", 0) == 0);
    REQUIRE(note.find(" | ") != std::string::npos);
    CHECK(note.substr(note.find(" | ")) == " | nr: stub (no model) intensity=0 input_transfer=pq");
}

TEST_CASE("compare of two SDR videos keeps its warning and JSON", "[integration][hdr][sdr][regression]") {
    const auto dir = Dir("compare");
    ClipSpec spec = Sdr("yuv420p", 2);
    spec.color = RawTags{1, 1, 1, 1, 0};
    const auto a = WriteClip(dir / "a709.mkv", spec);
    spec.color = RawTags{6, 6, 6, 1, 0};
    const auto b = WriteClip(dir / "b601.mkv", spec);
    const auto json = dir / "compare.json";
    const CliRun r = RunAll({"compare", "--ref", a.string(), "--test", b.string(), "-q", "--json", json.string()}, dir / "compare.log");
    INFO(r.out);
    REQUIRE(r.code == 0);
    CHECK_THAT(r.out, ContainsSubstring("compare: the videos decode with different colour matrices/ranges (bt709 limited vs bt601 limited) — tag both with the same "
                                        "colorspace, otherwise PSNR measures the mismatch"));
    const nlohmann::json j = ReadJson(json);
    std::vector<std::string> top, summary;
    for (auto it = j.begin(); it != j.end(); ++it) top.push_back(it.key());
    for (auto it = j["summary"].begin(); it != j["summary"].end(); ++it) summary.push_back(it.key());
    CHECK(top == std::vector<std::string>{"frames", "reference", "summary", "test"});
    CHECK(summary == std::vector<std::string>{"frames", "mae_rgb_mean", "mae_y_mean", "psnr_rgb_mean", "psnr_y_mean", "psnr_y_min", "size", "ssim_y_mean", "ssim_y_min"});
    for (const auto& f : j["frames"]) CHECK_FALSE(f.contains("domain"));
}
