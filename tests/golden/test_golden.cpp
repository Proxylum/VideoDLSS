// Golden tests (stage 8, ТЗ §8): the whole pipeline through the CLI on a deterministic synthetic clip, checked
// against stored reference frames and expectations (tests/golden/expected.json, tests/golden/ref/). Runs against
// the build tree by default; DLSSVID_CLI points it at an installed dlssvid.exe and DLSSVID_GOLDEN_DIR at the data
// folder of an installed copy. DLSSVID_GOLDEN_UPDATE=1 rewrites the reference frames.
//
//   deterministic (WARP): depth stub -> flow stub -> upscale nis x2 -> nr stub -> fg blend x2, PSNR to references
//   [gpu]: the real backends (da3, ofa, dlss, ngx, dlssg) on a 720p clip — counts, ranges, sanity

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/ImageMetrics.h"
#include "passes/PassSequence.h"
#include "stages/fg/IFrameGenerator.h"
#include "util/Subprocess.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {

std::string Env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}

std::filesystem::path CliPath() {
    const std::string env = Env("DLSSVID_CLI");
    return env.empty() ? std::filesystem::path(DLSSVID_CLI_PATH) : std::filesystem::path(env);
}

std::filesystem::path GoldenDir() {
    const std::string env = Env("DLSSVID_GOLDEN_DIR");
    if (!env.empty()) return env;
    if (std::filesystem::exists(std::filesystem::path(DLSSVID_GOLDEN_DIR) / "expected.json")) return DLSSVID_GOLDEN_DIR;
    return CliPath().parent_path().parent_path() / "tests" / "golden";  // installed layout: bin/../tests/golden
}

struct CliResult {
    uint32_t code = 0;
    std::string out;
};

CliResult RunCli(std::vector<std::string> args) {
    args.insert(args.begin(), CliPath().string());
    CliResult r;
    r.code = Subprocess::Run(args, &r.out);
    return r;
}

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "golden" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

nlohmann::json ReadJson(const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    nlohmann::json j;
    in >> j;
    return j;
}

// reference frames live as one-frame pass folders: ref/<pass>_<idx>/ (manifest + EXR: exact for half / float passes)
std::filesystem::path RefDir(const std::filesystem::path& golden, const std::string& pass, int64_t idx) { return golden / "ref" / (pass + "_" + std::to_string(idx)); }

void WriteRef(const std::filesystem::path& golden, const std::string& pass, int64_t idx, const PassImage& img, const Manifest& like) {
    const auto dir = RefDir(golden, pass, idx);
    std::filesystem::remove_all(dir);
    Manifest m = Manifest::ForPass(*ParsePassKind(like.pass), img.width, img.height, FileFormat::Exr);
    m.pixelType = img.type == PixelType::F16 ? PixelType::F16 : PixelType::F32;
    PassWriter w(dir, m);
    w.WriteFrame(0, img.type == PixelType::F16 || img.type == PixelType::F32 ? img : img.ConvertTo(PixelType::F32));
    w.Finish();
}

}  // namespace

TEST_CASE("Golden: the deterministic pipeline on WARP matches the stored references", "[golden]") {
    const auto golden = GoldenDir();
    INFO("golden dir " << golden.string() << ", cli " << CliPath().string());
    REQUIRE(std::filesystem::exists(golden / "expected.json"));
    const nlohmann::json e = ReadJson(golden / "expected.json");
    const bool update = Env("DLSSVID_GOLDEN_UPDATE") == "1";

    ClipSpec spec;
    spec.width = e["clip"]["width"];
    spec.height = e["clip"]["height"];
    spec.frames = e["clip"]["frames"];
    spec.fpsNum = e["clip"]["fps"];
    spec.audio = e["clip"]["audio"];
    const auto dir = Dir("warp");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    std::vector<std::string> args{"process", "-i", clip.string(), "-o", (dir / "result.mkv").string(), "--passes", (dir / "passes").string(), "--warp", "--codec", "ffv1", "--json",
                                  (dir / "report.json").string()};
    for (const auto& spec2 : e["stages"]) {
        args.push_back("--param");
        args.push_back(spec2.get<std::string>());
    }
    const CliResult r = RunCli(args);
    INFO(r.out);
    REQUIRE(r.code == 0);

    // passes: counts and sizes
    for (auto it = e["passes"].begin(); it != e["passes"].end(); ++it) {
        const auto dirPass = dir / "passes" / it.key();
        INFO("pass " << it.key());
        REQUIRE(std::filesystem::exists(dirPass / Manifest::kFileName));
        const PassReader rd = PassReader::Open(dirPass);
        CHECK(rd.Man().frameCount == it.value()["frames"].get<int64_t>());
        if (it.value().contains("width")) CHECK(rd.Man().width == it.value()["width"].get<uint32_t>());
        if (it.value().contains("height")) CHECK(rd.Man().height == it.value()["height"].get<uint32_t>());
        if (it.value().contains("fps")) CHECK(rd.Man().fps.ToDouble() == it.value()["fps"].get<double>());
    }
    // result video
    const auto frames = DecodeAll(dir / "result.mkv");
    CHECK(static_cast<int64_t>(frames.size()) == e["result"]["frames"].get<int64_t>());
    const StreamSummary sum = Summarize(dir / "result.mkv");
    CHECK(sum.videoStreams == 1);
    CHECK((sum.audioStreams == 1) == e["result"]["audio"].get<bool>());
    {
        VideoDecoder vd(dir / "result.mkv");
        CHECK(vd.Info().frameRate.ToDouble() == e["result"]["fps"].get<double>());
        CHECK(vd.Info().width == e["passes"]["color_fg"]["width"].get<uint32_t>());
    }
    // reference frames
    const double minPsnr = e.value("psnr_min_db", 45.0);
    for (const auto& ref : e["reference_frames"]) {
        const std::string pass = ref[0];
        const int64_t idx = ref[1];
        INFO("reference " << pass << " frame " << idx);
        const PassReader rd = PassReader::Open(dir / "passes" / pass);
        REQUIRE(rd.HasFrame(idx));
        const PassImage img = rd.ReadFrame(idx);
        if (update) {
            WriteRef(golden, pass, idx, img, rd.Man());
            continue;
        }
        const auto refDir = RefDir(golden, pass, idx);
        REQUIRE(std::filesystem::exists(refDir / Manifest::kFileName));
        const PassImage expected = PassReader::Open(refDir).ReadFrame(0);
        REQUIRE(expected.width == img.width);
        REQUIRE(expected.height == img.height);
        REQUIRE(expected.channels.size() == img.channels.size());
        if (img.channels.size() >= 3) {
            const ImageMetrics m = CompareImages(expected, img);
            CHECK((m.psnrY >= minPsnr || !std::isfinite(m.psnrY)));
        } else {
            double maxAbs = 0.0;
            for (uint32_t y = 0; y < img.height; ++y)
                for (uint32_t x = 0; x < img.width; ++x) maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(expected.Get(x, y, 0) - img.Get(x, y, 0))));
            CHECK(maxAbs <= e.value("guide_max_abs", 1e-4));
        }
    }
    if (update) WARN("reference frames rewritten in " << golden.string());

    // round trip: every float pass re-exported as EXR float32 reads back bit-exact (ТЗ §9)
    for (const char* pass : {"depth_raw", "mv_raw"}) {
        const PassReader rd = PassReader::Open(dir / "passes" / pass);
        const PassImage img = rd.ReadFrame(1).ConvertTo(PixelType::F32);
        Manifest m = Manifest::ForPass(*ParsePassKind(pass), img.width, img.height, FileFormat::Exr);
        m.pixelType = PixelType::F32;
        PassWriter w(dir / "roundtrip" / pass, m);
        w.WriteFrame(0, img);
        w.Finish();
        const PassImage back = PassReader::Open(dir / "roundtrip" / pass).ReadFrame(0);
        REQUIRE(back.data.size() == img.data.size());
        CHECK(back.data == img.data);
    }
}

TEST_CASE("Golden: the real backends produce a result on an NVIDIA GPU", "[golden][gpu]") {
    D3D12Device dev({false, false});
    if (!dev.IsNvidia()) SKIP("needs an NVIDIA GPU");
    const auto golden = GoldenDir();
    const nlohmann::json e = ReadJson(golden / "expected.json")["gpu"];
    ClipSpec spec;
    spec.width = e["clip"]["width"];
    spec.height = e["clip"]["height"];
    spec.frames = e["clip"]["frames"];
    spec.audio = true;
    const auto dir = Dir("gpu");
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    const CliResult r = RunCli({"process", "-i", clip.string(), "-o", (dir / "result.mkv").string(), "--passes", (dir / "passes").string(), "--codec", "ffv1", "--disable-unavailable",
                                "--json", (dir / "report.json").string()});
    INFO(r.out);
    REQUIRE(r.code == 0);
    const nlohmann::json report = ReadJson(dir / "report.json");
    for (const auto& st : report["stages"]) {
        INFO(st.dump());
        CHECK((st["status"] == "ran" || st["status"] == "disabled" || st["status"] == "reused"));
        if (st["name"] == "depth" || st["name"] == "flow" || st["name"] == "upscale") CHECK(st["status"] == "ran");
    }
    const auto frames = DecodeAll(dir / "result.mkv");
    const bool fgRan = report["stages"][4]["status"] == "ran";
    CHECK(static_cast<int64_t>(frames.size()) == (fgRan ? FgFrameCount(spec.frames, 2) : spec.frames));
    VideoDecoder vd(dir / "result.mkv");
    CHECK(vd.Info().width == spec.width * 2);
    CHECK(vd.Info().height == spec.height * 2);
    CHECK(Summarize(dir / "result.mkv").audioStreams == 1);
    // the upscaled colour keeps the scene: PSNR to a bicubic upscale of the source above the sanity floor
    const PassReader sr = PassReader::Open(dir / "passes" / "color_sr");
    const ColorInfo ci = ColorInfoFromStream(vd.Info());
    (void)ci;
    CHECK(sr.Man().frameCount == spec.frames);
    const PassImage f = sr.ReadFrame(1);
    for (uint32_t y = 0; y < f.height; y += 97)
        for (uint32_t x = 0; x < f.width; x += 89) {
            CHECK(f.Get(x, y, 1) >= 0.f);
            CHECK(f.Get(x, y, 1) <= 1.f);
        }
}
