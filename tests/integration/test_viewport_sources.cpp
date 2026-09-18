// Regression: vectors of ViewportSource holding a video entry and pass entries are copied, moved
// and destroyed in every combination the viewport uses (GUI project reload, CLI render, tests).
// Stage 4 hit a crash here that turned out to be a stale object file after the struct changed
// (two layouts linked together); the test guards the struct's copy/move/destroy paths.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>

#include "TestClips.h"
#include "passes/PassSequence.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"

using namespace dlssvid;
using namespace dlssvid::test;

TEST_CASE("ViewportSource vectors with video and pass entries survive copies, moves and destruction", "[integration][viewport][regression]") {
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 32;
    spec.height = 16;
    const auto dir = TempDir() / "viewport_sources";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto clip = WriteClip(dir / "clip.mkv", spec);
    for (const char* name : {"depth_raw", "mv_raw"}) {
        const PassKind kind = std::string(name) == "depth_raw" ? PassKind::DepthRaw : PassKind::MvRaw;
        PassWriter w(dir / "passes" / name, Manifest::ForPass(kind, 32, 16, FileFormat::Exr));
        w.WriteFrame(0, MakePassImage(kind, 32, 16));
        w.Finish();
    }
    const auto vs = FrameStore::VideoSource("source", clip);
    REQUIRE(vs.has_value());
    CHECK(vs->isVideo);
    CHECK(!vs->manifest);

    // reallocation with a video element in front, then moved-in pass elements
    for (int reserve = 0; reserve < 2; ++reserve) {
        std::vector<ViewportSource> v;
        if (reserve) v.reserve(8);
        v.push_back(*vs);
        v.push_back(*vs);
        for (auto& p : FrameStore::DiscoverPasses(dir / "passes")) v.push_back(std::move(p));
        auto passes = FrameStore::DiscoverPasses(dir / "passes");
        v.push_back(passes[0]);
        v.push_back(std::move(passes[1]));
        REQUIRE(v.size() == 6);
        CHECK(v[2].manifest->pass == "depth_raw");
        CHECK(v[5].manifest->pass == "mv_raw");
        std::vector<ViewportSource> copy = v;
        std::vector<ViewportSource> moved = std::move(copy);
        CHECK(moved.size() == 6);
        CHECK(moved[4].manifest->pass == "depth_raw");
    }
    // the project path used by the GUI and the CLI: temporary vector into the store
    {
        auto sources = Project::Create(clip, dir / "passes").Sources();
        CHECK(sources.size() == 3);
    }
    {
        D3D12Device dev({true, false});
        FrameStore store(dev);
        store.SetSources(Project::Create(clip, dir / "passes").Sources());
        CHECK(store.Sources().size() == 3);
        store.SetSources(Project::Create(clip, dir / "none").Sources());
        CHECK(store.Sources().size() == 1);
        store.SetSources({});
        CHECK(store.Sources().empty());
    }
}
