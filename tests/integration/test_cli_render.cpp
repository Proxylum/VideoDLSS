// CLI parity of the viewport (ТЗ §6/§7): `dlssvid project init` + `dlssvid render` as processes,
// and FrameStore random access into a video (seek) against a software decode.

#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "TestClips.h"
#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/PassSequence.h"
#include "passes/formats/PngIO.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"
#include "viewport/ViewportRenderer.h"

using namespace dlssvid;
using namespace dlssvid::test;

namespace {
int Run(const std::string& args) {
    const std::string cmd = "\"\"" DLSSVID_CLI_PATH "\" " + args + " >nul 2>&1\"";
    return std::system(cmd.c_str());
}
std::string Q(const std::filesystem::path& p) { return "\"" + p.string() + "\""; }

void WriteDepth(const std::filesystem::path& dir, uint32_t w, uint32_t h, int frames) {
    PassWriter writer(dir, Manifest::ForPass(PassKind::DepthRaw, w, h, FileFormat::Exr));
    for (int f = 0; f < frames; ++f) {
        PassImage img = MakePassImage(PassKind::DepthRaw, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) img.Set(x, y, 0, static_cast<float>(x + f));
        writer.WriteFrame(f, img);
    }
    writer.Finish();
}

struct Stats {
    int dark = 0, bright = 0, colored = 0;
};
Stats Analyse(const PassImage& img) {
    Stats s;
    for (size_t i = 0; i + 3 < img.data.size(); i += 4) {
        const int r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
        if (r + g + b < 90) ++s.dark;
        if (r + g + b > 600) ++s.bright;
        if (std::abs(r - g) > 40 || std::abs(g - b) > 40) ++s.colored;
    }
    return s;
}
}  // namespace

TEST_CASE("dlssvid project init + render write a project and PNG frames", "[integration][cli][viewport]") {
    ClipSpec spec;
    spec.frames = 8;
    spec.width = 64;
    spec.height = 36;
    const auto dir = TempDir() / "cli_render";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    WriteDepth(dir / "passes" / "depth_raw", 64, 36, 8);
    const auto proj = dir / "clip.dlssvid.json";

    REQUIRE(Run("project init -i " + Q(clip) + " --passes " + Q(dir / "passes") + " -o " + Q(proj)) == 0);
    REQUIRE(std::filesystem::exists(proj));
    const Project p = Project::Load(proj);
    CHECK(p.sourceVideo == std::filesystem::absolute(clip));
    CHECK(p.Sources().size() == 2);
    REQUIRE(Run("project show --project " + Q(proj)) == 0);

    // single view of the video at 2x
    const auto png = dir / "frame3.png";
    REQUIRE(Run("render --warp --project " + Q(proj) + " --frame 3 --size 128x72 -o " + Q(png)) == 0);
    PassImage img = ReadPng(png);
    CHECK(img.width == 128);
    CHECK(img.height == 72);
    CHECK(img.channels.size() == 4);
    Stats s = Analyse(img);
    CHECK(s.bright + s.dark < static_cast<int>(img.width * img.height));  // a real picture, not a flat fill

    // depth pass with turbo, and the 2x2 grid
    REQUIRE(Run("render --warp --project " + Q(proj) + " --frame 0 --source depth_raw --display turbo -o " + Q(dir / "depth.png")) == 0);
    img = ReadPng(dir / "depth.png");
    CHECK(img.width == 64);
    CHECK(Analyse(img).colored > 1000);
    REQUIRE(Run("render --warp --project " + Q(proj) + " --frame 2 --mode grid --sources source,depth_raw,source,depth_raw --size 128x72 -o " + Q(dir / "grid.png")) == 0);
    CHECK(ReadPng(dir / "grid.png").width == 128);
    REQUIRE(Run("render --warp --project " + Q(proj) + " --frame 2 --layers source,depth_raw:viridis:0.5:multiply --wipe v:0.5 -o " + Q(dir / "overlay.png")) == 0);
    CHECK(ReadPng(dir / "overlay.png").height == 36);

    // --save-state writes the viewport state back into the project
    REQUIRE(Run("render --warp --project " + Q(proj) + " --frame 5 --source depth_raw --zoom 2 --save-state -o " + Q(dir / "s.png")) == 0);
    const Project q = Project::Load(proj);
    CHECK(std::abs(q.viewport.time - 5.0 / 24.0) < 1e-9);  // --frame counts at the base rate, the project stores a time
    CHECK(q.viewport.layers[0].source == "depth_raw");
    CHECK(q.viewport.view.zoom == 2.f);

    // without a project: --input + --passes; benchmark mode
    REQUIRE(Run("render --warp -i " + Q(clip) + " --passes " + Q(dir / "passes") + " --frame 1 -o " + Q(dir / "noproj.png")) == 0);
    REQUIRE(Run("render --warp --project " + Q(proj) + " --bench 4") == 0);
    // --time in seconds (wins over --frame); the seventh frame at 24 fps is 0.25 s
    REQUIRE(Run("render --warp --project " + Q(proj) + " --time 0.25 --frame 1 --save-state -o " + Q(dir / "time.png")) == 0);
    CHECK(ReadPng(dir / "time.png").width == 64);
    CHECK(std::abs(Project::Load(proj).viewport.time - 0.25) < 1e-9);
    CHECK(Run("render --warp --project " + Q(proj) + " --time 9 -o " + Q(dir / "x.png")) != 0);

    // errors: frame out of range, unknown mode, missing inputs
    CHECK(Run("render --warp --project " + Q(proj) + " --frame 99 -o " + Q(dir / "x.png")) != 0);
    CHECK(Run("render --warp --project " + Q(proj) + " --mode weird -o " + Q(dir / "x.png")) != 0);
    CHECK(Run("render --warp -o " + Q(dir / "x.png")) != 0);
    CHECK(Run("project init -i " + Q(dir / "missing.mkv")) != 0);
}

TEST_CASE("FrameStore seeks into a video and matches a software decode", "[integration][viewport][gpu]") {
    ClipSpec spec;
    spec.frames = 12;
    spec.width = 48;
    spec.height = 32;
    const auto dir = TempDir() / "store_video";
    std::filesystem::create_directories(dir);
    const auto clip = WriteClip(dir / "seek.mkv", spec);
    const std::vector<CpuFrame> ref = DecodeAll(clip);
    REQUIRE(ref.size() == 12);
    VideoDecoder probe(clip);
    const ColorInfo ci = ColorInfoFromStream(probe.Info());

    D3D12Device dev({true, false});
    FrameStore::Options opt;
    opt.prefetch = 1;
    FrameStore store(dev, opt);
    store.SetSources(Project::Create(clip, dir / "none").Sources());
    REQUIRE(store.Sources().size() == 1);
    CHECK(store.FrameCount() == 12);
    CHECK(store.Fps().num == 24);
    ViewportRenderer r(dev);

    auto check = [&](int64_t f) {
        store.SetCurrentFrame(f, {"source"});
        store.WaitForCurrent();
        REQUIRE(store.GetStatus(f, "source") == FrameStore::Status::Ready);
        const FrameTextures tex = store.Textures(f);
        REQUIRE(tex.count("source") == 1);
        const LayerTexture& t = tex.at("source");
        CHECK(t.kind == TextureKind::Color);
        CHECK(t.width == 48);
        const PassImage rgb = Yuv420pToRgb(ref[static_cast<size_t>(f)], ci, PixelType::F16);
        for (auto [x, y] : {std::pair<uint32_t, uint32_t>{5, 7}, {30, 20}, {47, 31}}) {
            const auto v = r.ReadTexel(t, x, y);
            CHECK(v[0] == rgb.Get(x, y, 0));
            CHECK(v[1] == rgb.Get(x, y, 1));
            CHECK(v[2] == rgb.Get(x, y, 2));
        }
    };
    check(7);   // seek forward
    check(2);   // seek backward
    check(11);  // last frame
    check(0);   // back to the start
    store.SetCurrentFrame(12, {"source"});
    store.WaitForCurrent();
    CHECK(store.GetStatus(12, "source") == FrameStore::Status::Missing);
}
