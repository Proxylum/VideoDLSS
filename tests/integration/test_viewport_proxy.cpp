// Scrubbing (TASK-0040): keyframe scan, the bit-exact proxy and its validity, and the FrameStore loaders - a proxy
// gives the same pixels, quick back-and-forth jumps lose no frame and mark none missing.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>

#include "TestClips.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "viewport/FrameStore.h"
#include "viewport/ViewportProxy.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path Dir(const char* name) {
    const auto d = TempDir() / "viewport_proxy" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::filesystem::path Clip(const std::filesystem::path& dir, int frames, int gop, const std::string& pixFmt = "yuv420p", uint32_t w = 96, uint32_t h = 64) {
    ClipSpec spec;
    spec.frames = frames;
    spec.width = w;
    spec.height = h;
    spec.gop = gop;
    spec.pixFmt = pixFmt;
    if (pixFmt == "yuv420p" || pixFmt == "p010le" || pixFmt == "yuv444p16le") return WriteClip(dir / "clip.mkv", spec);
    const FrameDesc desc = FrameDescFor(spec);
    return WriteClipWith(dir / "clip.mkv", spec, [&](int64_t i) { return SyntheticFrame(i, desc); });
}

// every frame of the proxy, decoded as the viewport does (slice threads, the original's rate for the indices)
std::vector<CpuFrame> DecodeProxy(const std::filesystem::path& proxy, Rational rate) {
    VideoDecoder::Options o;
    o.sliceThreadsOnly = true;
    o.indexRate = rate;
    VideoDecoder dec(proxy, o);
    REQUIRE(dec.SeekToKeyframeBefore(0));
    std::vector<CpuFrame> out;
    CpuFrame f;
    while (dec.NextFrame(f)) out.push_back(f);
    return out;
}

std::vector<uint8_t> Pixels(D3D12Device& dev, FrameStore& store, double t, const std::string& source) {
    store.SetCurrentTime(t, {source});
    store.WaitForCurrent(20000);
    const FrameTextures tex = store.TexturesAt(t);
    const auto it = tex.find(source);
    REQUIRE(it != tex.end());
    size_t pitch = 0;
    return dev.ReadbackTexture2D(it->second.texture, pitch);
}

}  // namespace

TEST_CASE("ScanKeyframes: keyframes and the longest run with its tail; PlanProxy wants a proxy for long intervals only", "[integration][viewport][proxy]") {
    const auto d = Dir("scan");
    const KeyframeScan g12 = ScanKeyframes(Clip(d / "g12", 40, 12));
    CHECK(g12.frames == 40);
    CHECK(g12.keyframes == 4);       // 0, 12, 24, 36
    CHECK(g12.maxInterval == 12);    // the tail after 36 is 4
    CHECK_FALSE(g12.intraOnly());
    const KeyframeScan intra = ScanKeyframes(Clip(d / "g1", 20, 1));
    CHECK(intra.intraOnly());
    CHECK(intra.maxInterval == 1);
    const KeyframeScan one = ScanKeyframes(Clip(d / "g100", 30, 100));
    CHECK(one.keyframes == 1);
    CHECK(one.maxInterval == 30);    // one keyframe: the whole clip is its run
    CHECK(PlanProxy(d / "g12" / "clip.mkv").needed);
    CHECK_FALSE(PlanProxy(d / "g1" / "clip.mkv").needed);
    const ProxyPlan p = PlanProxy(d / "g100" / "clip.mkv");
    CHECK(p.needed);
    CHECK(p.frames == 30);
    CHECK(p.estimatedBytes > 0);
}

TEST_CASE("BuildProxy: the same frames bit for bit under the same indices, for 8, 10 and 12 bits", "[integration][viewport][proxy]") {
    struct Case {
        const char* name;
        const char* pixFmt;
        const char* codec;
    };
    for (const Case c : {Case{"p420", "yuv420p", "magicyuv"}, Case{"p422_10", "yuv422p10le", "ffvhuff"}, Case{"p444", "yuv444p", "magicyuv"},
                         Case{"p444_12", "yuv444p12le", "ffvhuff"}}) {
        DYNAMIC_SECTION(c.name) {
            const auto d = Dir(c.name);
            const auto clip = Clip(d, 30, 12, c.pixFmt, 98, 66);  // sizes not a multiple of 4 or 16
            const auto passes = d / "passes";
            const auto proxy = BuildProxy(clip, passes, "source");
            CHECK(proxy == ProxyFile(passes, "source"));
            nlohmann::json side;
            std::ifstream(ProxyDir(passes) / "source.proxy.json") >> side;
            CHECK(side["codec"] == c.codec);
            CHECK(side["frames"] == 30);
            const std::vector<CpuFrame> want = DecodeAll(clip);
            const Rational rate = VideoDecoder(clip, VideoDecoder::Probe()).Info().frameRate;
            const std::vector<CpuFrame> got = DecodeProxy(proxy, rate);
            REQUIRE(got.size() == want.size());
            for (size_t i = 0; i < got.size(); ++i) {
                INFO("frame " << i);
                CHECK(got[i].index == static_cast<int64_t>(i));
                CHECK(got[i].desc == want[i].desc);
                CHECK(got[i].data == want[i].data);
            }
            CHECK(ValidProxy(clip, passes, "source") == proxy);
            CHECK_FALSE(ValidProxy(clip, passes, "result"));
            CHECK(std::filesystem::is_empty(ProxyDir(passes)) == false);
        }
    }
}

TEST_CASE("A proxy follows its source: rewritten, cancelled, removed", "[integration][viewport][proxy][regression]") {
    const auto d = Dir("valid");
    const auto clip = Clip(d, 20, 12);
    const auto passes = d / "passes";
    BuildProxy(clip, passes, "source");
    REQUIRE(ValidProxy(clip, passes, "source"));
    // the source rewritten in place (a stage writing the result again): the proxy shows other frames - not valid any more
    ClipSpec spec;
    spec.frames = 21;
    spec.width = 96;
    spec.height = 64;
    WriteClip(clip, spec);
    CHECK_FALSE(ValidProxy(clip, passes, "source"));
    // a cancelled build leaves nothing behind
    std::atomic<bool> cancel{true};
    ProxyBuildOptions o;
    o.cancel = &cancel;
    CHECK_THROWS_WITH(BuildProxy(clip, passes, "result", o), ContainsSubstring("cancelled"));
    CHECK_FALSE(std::filesystem::exists(ProxyFile(passes, "result")));
    CHECK_FALSE(std::filesystem::exists(ProxyDir(passes) / "result.building.mkv"));
    BuildProxy(clip, passes, "source");
    REQUIRE(ValidProxy(clip, passes, "source"));
    RemoveProxy(passes, "source");
    CHECK_FALSE(ValidProxy(clip, passes, "source"));
    CHECK_FALSE(std::filesystem::exists(ProxyFile(passes, "source")));
}

TEST_CASE("FrameStore with a proxy shows the same pixels as the original and serves jumps from it", "[integration][viewport][proxy][gpu]") {
    const auto d = Dir("store");
    const auto clip = Clip(d, 48, 24);
    const auto passes = d / "passes";
    BuildProxy(clip, passes, "source");
    D3D12Device dev({true, false});
    auto src = FrameStore::VideoSource("source", clip);
    REQUIRE(src);
    FrameStore plain(dev), viaProxy(dev);
    plain.SetSources({*src});
    ViewportSource withProxy = *src;
    withProxy.proxy = ProxyFile(passes, "source");
    viaProxy.SetSources({withProxy});
    const double rate = src->fps.ToDouble();
    for (const int64_t f : {40, 3, 27, 26, 47, 0, 13}) {
        INFO("frame " << f);
        CHECK(Pixels(dev, viaProxy, static_cast<double>(f) / rate, "source") == Pixels(dev, plain, static_cast<double>(f) / rate, "source"));
    }
    CHECK(viaProxy.Stats("source").proxyLoads > 0);
    CHECK(plain.Stats("source").proxyLoads == 0);
}

// Review of the scrubbing design (TASK-0040): a load dropped because the time moved on, while SetCurrentTime had
// skipped it as in flight, was lost when the time came back - WaitForCurrent timed out after 60 s. And a dropped load
// must not mark its frame missing (missing_ is sticky until SetSources).
TEST_CASE("FrameStore: quick jumps back and forth lose no frame and mark none missing", "[integration][viewport][proxy][gpu][regression]") {
    const auto d = Dir("race");
    const auto clip = Clip(d, 240, 240, "yuv420p", 320, 180);  // one keyframe: every jump walks from frame 0
    D3D12Device dev({true, false});
    auto src = FrameStore::VideoSource("source", clip);
    REQUIRE(src);
    FrameStore store(dev);
    store.SetSources({*src});
    const double rate = src->fps.ToDouble();
    const std::vector<std::string> needed{"source"};
    for (int round = 0; round < 6; ++round) {
        const int64_t a = 200 - round * 7, b = 60 + round * 5;
        store.SetCurrentTime(static_cast<double>(a) / rate, needed);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        store.SetCurrentTime(static_cast<double>(b) / rate, needed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        store.SetCurrentTime(static_cast<double>(a) / rate, needed);
        const auto t0 = std::chrono::steady_clock::now();
        store.WaitForCurrent(20000);  // before: a lost key, timed out
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10));
        CHECK(store.GetStatus(a, "source") == FrameStore::Status::Ready);
        CHECK(store.GetStatus(b, "source") != FrameStore::Status::Missing);
    }
    store.SetCurrentTime(60.0 / rate, needed);
    store.WaitForCurrent(20000);
    CHECK(store.GetStatus(60, "source") == FrameStore::Status::Ready);  // a frame dropped once loads when wanted again
}
