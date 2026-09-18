// FrameStore: pass discovery, statuses, prefetch window and LRU eviction within a VRAM budget.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <thread>

#include "gpu/D3D12Device.h"
#include "passes/PassSequence.h"
#include "viewport/FrameStore.h"

using namespace dlssvid;

namespace {

std::filesystem::path Root(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "frame_store" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

void WritePass(const std::filesystem::path& dir, PassKind kind, uint32_t w, uint32_t h, int64_t first, int64_t last, FileFormat format = FileFormat::Exr) {
    PassWriter writer(dir, Manifest::ForPass(kind, w, h, format));
    for (int64_t f = first; f <= last; ++f) {
        PassImage img = MakePassImage(kind, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (size_t c = 0; c < img.channels.size(); ++c) img.Set(x, y, c, static_cast<float>(f * 10 + x + c));
        writer.WriteFrame(f, img);
    }
    writer.Finish();
}

bool WaitUntil(const std::function<bool()>& pred, int ms = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

}  // namespace

TEST_CASE("FrameStore discovers pass folders and their kinds", "[viewport][store]") {
    const auto root = Root("discover");
    WritePass(root / "depth_raw", PassKind::DepthRaw, 8, 4, 0, 4);
    WritePass(root / "mv_raw", PassKind::MvRaw, 8, 4, 0, 3);
    WritePass(root / "mask", PassKind::Mask, 8, 4, 2, 4, FileFormat::Png);
    WritePass(root / "color_sr", PassKind::ColorSr, 16, 8, 0, 4, FileFormat::Png);
    std::filesystem::create_directories(root / "not_a_pass");
    const auto sources = FrameStore::DiscoverPasses(root);
    REQUIRE(sources.size() == 4);  // sorted by name
    CHECK(sources[0].name == "color_sr");
    CHECK(sources[0].kind == TextureKind::Color);
    CHECK(sources[0].width == 16);
    CHECK(sources[1].name == "depth_raw");
    CHECK(sources[1].kind == TextureKind::Scalar);
    CHECK(sources[1].lastFrame == 4);
    CHECK(sources[2].name == "mask");
    CHECK(sources[2].kind == TextureKind::Mask);
    CHECK(sources[2].firstFrame == 2);
    CHECK(sources[3].name == "mv_raw");
    CHECK(sources[3].kind == TextureKind::Mv);
    CHECK(sources[3].HasFrame(3));
    CHECK(!sources[3].HasFrame(4));
    CHECK(FrameStore::DiscoverPasses(root / "nope").empty());
    CHECK(!FrameStore::VideoSource("source", root / "missing.mp4"));
}

TEST_CASE("FrameStore loads the current frame, reports statuses and prefetches", "[viewport][store][gpu]") {
    const auto root = Root("load");
    WritePass(root / "depth_raw", PassKind::DepthRaw, 8, 4, 0, 9);
    WritePass(root / "mask", PassKind::Mask, 8, 4, 0, 9, FileFormat::Png);
    D3D12Device dev({true, false});
    FrameStore::Options opt;
    opt.prefetch = 2;
    FrameStore store(dev, opt);
    store.SetSources(FrameStore::DiscoverPasses(root));
    CHECK(store.ImageWidth() == 8);
    CHECK(store.ImageHeight() == 4);
    CHECK(store.FrameCount() == 10);

    store.SetCurrentFrame(5, {"depth_raw", "mask"});
    CHECK(store.GetStatus(5, "depth_raw") != FrameStore::Status::Missing);
    store.WaitForCurrent();
    CHECK(store.GetStatus(5, "depth_raw") == FrameStore::Status::Ready);
    CHECK(store.GetStatus(5, "mask") == FrameStore::Status::Ready);
    CHECK(store.GetStatus(5, "nope") == FrameStore::Status::Missing);
    CHECK(store.GetStatus(42, "depth_raw") == FrameStore::Status::Missing);
    FrameTextures tex = store.Textures(5);
    REQUIRE(tex.count("depth_raw") == 1);
    REQUIRE(tex.count("mask") == 1);
    CHECK(tex["depth_raw"].kind == TextureKind::Scalar);
    CHECK(tex["depth_raw"].width == 8);
    CHECK(tex["depth_raw"].minValue == 50.f);  // frame 5: values 50..57
    CHECK(tex["depth_raw"].maxValue == 57.f);
    CHECK(tex["mask"].kind == TextureKind::Mask);
    CHECK(store.Textures(6).empty() == (store.GetStatus(6, "depth_raw") != FrameStore::Status::Ready));

    // prefetch window: frames 3..7 of both sources become resident
    REQUIRE(WaitUntil([&] {
        store.Update();
        return store.ResidentCount() >= 10;
    }));
    CHECK(store.GetStatus(3, "depth_raw") == FrameStore::Status::Ready);
    CHECK(store.GetStatus(7, "mask") == FrameStore::Status::Ready);
    CHECK(store.GetStatus(8, "depth_raw") == FrameStore::Status::Queued);  // loadable, outside the window
    CHECK(store.ResidentCount() == 10);
    CHECK(store.ResidentBytes() == 5 * (8 * 4 * 4 + 8 * 4));

    // a frame that does not exist in a partial pass is Missing, not an error
    store.SetSources(FrameStore::DiscoverPasses(root));
    CHECK(store.ResidentCount() == 0);
}

TEST_CASE("FrameStore evicts least recently used frames outside the window within the budget", "[viewport][store][gpu]") {
    const auto root = Root("lru");
    WritePass(root / "depth_raw", PassKind::DepthRaw, 16, 8, 0, 19);
    D3D12Device dev({true, false});
    const size_t frameBytes = 16 * 8 * 4;
    FrameStore::Options opt;
    opt.prefetch = 1;
    opt.vramBudgetBytes = frameBytes * 3;
    FrameStore store(dev, opt);
    store.SetSources(FrameStore::DiscoverPasses(root));
    for (int64_t f = 0; f < 12; ++f) {
        store.SetCurrentFrame(f, {"depth_raw"});
        store.WaitForCurrent();
        CHECK(store.GetStatus(f, "depth_raw") == FrameStore::Status::Ready);
        CHECK(store.ResidentBytes() <= opt.vramBudgetBytes);
    }
    // settle the outstanding prefetch uploads
    WaitUntil([&] {
        store.Update();
        return store.GetStatus(12, "depth_raw") == FrameStore::Status::Ready && store.GetStatus(10, "depth_raw") == FrameStore::Status::Ready;
    });
    CHECK(store.ResidentBytes() <= opt.vramBudgetBytes);
    CHECK(store.ResidentCount() <= 3);
    CHECK(store.GetStatus(11, "depth_raw") == FrameStore::Status::Ready);
    CHECK(store.GetStatus(0, "depth_raw") == FrameStore::Status::Queued);  // evicted: loadable again
    // going back reloads it
    store.SetCurrentFrame(0, {"depth_raw"});
    store.WaitForCurrent();
    CHECK(store.GetStatus(0, "depth_raw") == FrameStore::Status::Ready);
}
