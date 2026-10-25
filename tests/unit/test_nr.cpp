// Neural Rendering building blocks (stage 6): driver version decoding, architecture names, the patcher
// command line, the forwarder module, masks and the resolve kernel on WARP.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "GpuTextures.h"
#include "gpu/D3D12Device.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrCompose.h"
#include "stages/nr/NrPatch.h"
#include "stages/nr/NrStage.h"
#include "stages/nr/StubNrBackend.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {
uint64_t Umd(uint32_t product, uint32_t version, uint32_t subversion, uint32_t build) {
    return (static_cast<uint64_t>((product << 16) | version) << 32) | ((subversion << 16) | build);
}

std::filesystem::path Dir(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "nr" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// fake snippet entry points the forwarder is pointed at
int __cdecl FakeCreate(void*, int featureId, void*, void** handle) {
    *handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
    return featureId == 18 ? 1 : 0xBAD00001;
}
int __cdecl FakeEvaluate(void*, void* handle, void*, void*) { return handle ? 1 : 0xBAD00005; }
int __cdecl FakeRelease(void* handle) { return handle ? 1 : 0; }
int g_infoPos = 0;
int __cdecl FakeInitInfoVersion(unsigned long long app, const wchar_t*, void*, const void* info, int version) {
    g_infoPos = (info != nullptr && version == 0x15 && app == 7) ? 4 : -1;
    return 1;
}
int __cdecl FakeInitVersionInfo(unsigned long long app, const wchar_t*, void*, int version, const void* info) {
    g_infoPos = (info != nullptr && version == 0x15 && app == 7) ? 5 : -1;
    return 1;
}
}  // namespace

TEST_CASE("NvidiaDriverFromUmd decodes the DXGI UMD version", "[nr][unit]") {
    const NvidiaDriverVersion v = NvidiaDriverFromUmd(Umd(32, 0, 15, 9186));
    CHECK(v.valid);
    CHECK(v.major == 591);
    CHECK(v.minor == 86);
    CHECK(v.ToString() == "591.86");
    CHECK(!v.AtLeast(kNrMinDriverMajor, kNrMinDriverMinor));
    const NvidiaDriverVersion w = NvidiaDriverFromUmd(Umd(32, 0, 16, 1656));
    CHECK(w.ToString() == "616.56");
    CHECK(w.AtLeast(616, 56));
    CHECK(w.AtLeast(591, 86));
    CHECK(!w.AtLeast(616, 57));
    CHECK(NvidiaDriverFromUmd(Umd(32, 0, 16, 1700)).ToString() == "617.00");
    const NvidiaDriverVersion none = NvidiaDriverFromUmd(0);
    CHECK(!none.valid);
    CHECK(none.ToString() == "?");
    CHECK(!none.AtLeast(0, 0));
}

TEST_CASE("GPU architecture from the adapter name and the compute capability", "[nr][unit]") {
    CHECK(GpuArchitectureFromName("NVIDIA GeForce RTX 4070 Ti SUPER") == "Ada");
    CHECK(GpuArchitectureFromName("NVIDIA GeForce RTX 5090") == "Blackwell");
    CHECK(GpuArchitectureFromName("NVIDIA GeForce RTX 3080") == "Ampere");
    CHECK(GpuArchitectureFromName("NVIDIA GeForce RTX 2070") == "Turing");
    CHECK(GpuArchitectureFromName("NVIDIA GeForce GTX 1660 SUPER") == "Turing");
    CHECK(GpuArchitectureFromName("NVIDIA RTX 6000 Ada Generation") == "Ada");
    CHECK(GpuArchitectureFromName("Microsoft Basic Render Driver").empty());
    CHECK(GpuArchitectureFromComputeCapability(8, 9) == "Ada");
    CHECK(GpuArchitectureFromComputeCapability(8, 6) == "Ampere");
    CHECK(GpuArchitectureFromComputeCapability(7, 5) == "Turing");
    CHECK(GpuArchitectureFromComputeCapability(12, 0) == "Blackwell");
    CHECK(GpuArchitectureFromComputeCapability(6, 1).empty());
}

TEST_CASE("NR styles, backends and availability messages", "[nr][unit]") {
    CHECK(ParseNrStyle("natural") == 1);
    CHECK(ParseNrStyle("cinematic") == 2);
    CHECK(ParseNrStyle("default") == 0);
    CHECK_THROWS(ParseNrStyle("4"));  // PR-2 E7: 3..6 gave the cinematic picture - not styles of the schema any more
    CHECK_THROWS(ParseNrStyle("6"));
    CHECK(ParseNrStyle("Natural") == 1);
    CHECK_THROWS(ParseNrStyle("vivid"));
    CHECK_THROWS(ParseNrStyle("99"));
    CHECK_THROWS(ParseNrStyle("7"));  // TASK-0044 / PR-2 E7: 0..2 only (3..6 gave the cinematic picture); 7..15 were taken by the CLI alone
    CHECK(NrBackends() == std::vector<std::string>{"ngx", "stub"});
    CHECK(NrAvailable("stub").available);
    const NrAvailability bogus = NrAvailable("bogus");
    CHECK(!bogus.available);
    CHECK(bogus.reason.find("unknown") != std::string::npos);
    CHECK_THROWS(CreateNrBackend("bogus"));
    const NrAvailability ngx = NrAvailable("ngx");
    if (!ngx.available) {
        // no DLL on this machine: the reason is the instruction the user needs
        CHECK((ngx.reason.find(kNrDllName) != std::string::npos || ngx.reason.find("DLSS SDK") != std::string::npos || ngx.reason.find("forwarder") != std::string::npos));
        CHECK(ngx.reason.find("docs/dll-setup.md") != std::string::npos);
    }
    CHECK(NrDllInstruction().find("nr-patch") != std::string::npos);
}

TEST_CASE("dlssnr-patcher command line and script lookup", "[nr][unit]") {
    NrPatchOptions o;
    o.input = "C:/dlls/nvngx_dlssnr.dll";
    o.python = "py";
    o.cudaBin = "C:/cuda13/bin";
    o.archs = {"ada"};
    const std::vector<std::string> cmd = BuildPatchCommand(o, "C:/p/dlssnr_patcher.py", "C:/out/nvngx_dlssnr.dll");
    CHECK(cmd == std::vector<std::string>{"py", "C:/p/dlssnr_patcher.py", "--ada", "-o", "C:/out/nvngx_dlssnr.dll", "--force", "--cuda-bin", "C:/cuda13/bin",
                                          "C:/dlls/nvngx_dlssnr.dll"});
    o.archs = {"ampere", "turing", "blackwell"};
    o.dryRun = true;
    o.cudaBin.clear();
    const std::vector<std::string> dry = BuildPatchCommand(o, "p.py", "out.dll");
    CHECK(dry == std::vector<std::string>{"py", "p.py", "--ampere", "--turing", "--blackwell", "--dry-run", "C:/dlls/nvngx_dlssnr.dll"});
    o.archs = {"all"};
    o.dryRun = false;
    o.force = false;
    CHECK(BuildPatchCommand(o, "p.py", "out.dll") == std::vector<std::string>{"py", "p.py", "-o", "out.dll", "C:/dlls/nvngx_dlssnr.dll"});
    o.archs = {"hopper"};
    CHECK_THROWS(BuildPatchCommand(o, "p.py", "out.dll"));

    const auto d = Dir("patcher");
    CHECK(FindPatcherScript(d).empty());
    std::ofstream(d / "dlssnr_patcher.py") << "print('fake')\n";
    CHECK(FindPatcherScript(d) == d / "dlssnr_patcher.py");
    CHECK(FindPatcherScript(d / "dlssnr_patcher.py") == d / "dlssnr_patcher.py");
    CHECK(FindPatcherScript(d / "missing").empty());
    CHECK(DefaultNrDllOutput().filename() == "nvngx_dlssnr.dll");
    CHECK(DefaultNrDllOutput().parent_path().filename() == "nvidia");
    CHECK(!DefaultPatchPython().empty());
    NrPatchOptions bad;
    bad.input = d / "nope.dll";
    CHECK_THROWS(RunNrPatch(bad));
    bad.input = d / "dlssnr_patcher.py";  // exists; the patcher is looked up from a folder without one
    bad.patcher = d / "missing";
    CHECK_THROWS(RunNrPatch(bad));
}

TEST_CASE("The forwarder module is built next to the executable, is named nvngx.dll_*, and passes calls through", "[nr][unit]") {
    const std::filesystem::path fwd = FindNrForwarder();
    REQUIRE(!fwd.empty());
    CHECK(fwd.filename().string().find("nvngx.dll") != std::string::npos);  // the name the snippet checks its caller against
    HMODULE mod = LoadLibraryW(fwd.wstring().c_str());
    REQUIRE(mod != nullptr);
    using InitFn = int(__cdecl*)(void*, int, unsigned long long, const wchar_t*, void*, int, const void*);
    using CreateFn = int(__cdecl*)(void*, void*, int, void*, void**);
    using EvalFn = int(__cdecl*)(void*, void*, void*, void*, void*);
    using ReleaseFn = int(__cdecl*)(void*, void*);
    using VersionFn = const char*(__cdecl*)();
    auto init = reinterpret_cast<InitFn>(GetProcAddress(mod, "dlssvid_nr_call_init"));
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(mod, "dlssvid_nr_call_create"));
    auto eval = reinterpret_cast<EvalFn>(GetProcAddress(mod, "dlssvid_nr_call_evaluate"));
    auto release = reinterpret_cast<ReleaseFn>(GetProcAddress(mod, "dlssvid_nr_call_release"));
    auto version = reinterpret_cast<VersionFn>(GetProcAddress(mod, "dlssvid_nr_forwarder_version"));
    REQUIRE(init);
    REQUIRE(create);
    REQUIRE(eval);
    REQUIRE(release);
    REQUIRE(version);
    CHECK(std::string(version()).find("dlssvid-nr-forwarder") != std::string::npos);
    int marker = 0;
    CHECK(init(reinterpret_cast<void*>(&FakeInitInfoVersion), 0, 7, L"path", nullptr, 0x15, &marker) == 1);
    CHECK(g_infoPos == 4);
    CHECK(init(reinterpret_cast<void*>(&FakeInitVersionInfo), 1, 7, L"path", nullptr, 0x15, &marker) == 1);
    CHECK(g_infoPos == 5);
    void* handle = nullptr;
    CHECK(create(reinterpret_cast<void*>(&FakeCreate), nullptr, 18, nullptr, &handle) == 1);
    CHECK(handle == reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234)));
    CHECK(create(reinterpret_cast<void*>(&FakeCreate), nullptr, 1, nullptr, &handle) == 0xBAD00001);
    CHECK(eval(reinterpret_cast<void*>(&FakeEvaluate), nullptr, handle, nullptr, nullptr) == 1);
    CHECK(eval(reinterpret_cast<void*>(&FakeEvaluate), nullptr, nullptr, nullptr, nullptr) == 0xBAD00005);
    CHECK(release(reinterpret_cast<void*>(&FakeRelease), handle) == 1);
    CHECK(create(nullptr, nullptr, 18, nullptr, &handle) == 0);  // null function pointer is not called
    FreeLibrary(mod);
}

TEST_CASE("Masks: 8-bit to float and per-pixel max", "[nr][unit]") {
    PassImage m;
    m.Allocate(4, 1, PixelType::U8, {"A"});
    m.As<uint8_t>()[0] = 0;
    m.As<uint8_t>()[1] = 255;
    m.As<uint8_t>()[2] = 128;
    m.As<uint8_t>()[3] = 51;
    const PassImage f = MaskToFloat(m);
    CHECK(f.type == PixelType::F32);
    CHECK(f.Get(0, 0, 0) == 0.f);
    CHECK(f.Get(1, 0, 0) == 1.f);
    CHECK(f.Get(2, 0, 0) == Approx(128.f / 255.f));
    CHECK(f.Get(3, 0, 0) == Approx(0.2f));
    PassImage g = f;
    g.Set(0, 0, 0, 0.7f);
    g.Set(1, 0, 0, 0.2f);
    const PassImage mx = MaskMax(f, g);
    CHECK(mx.Get(0, 0, 0) == Approx(0.7f));
    CHECK(mx.Get(1, 0, 0) == 1.f);
    PassImage other;
    other.Allocate(2, 1, PixelType::F32, {"A"});
    CHECK_THROWS(MaskMax(f, other));
}

TEST_CASE("NrCompose resolve on WARP: direct mode, masks, ratio transfer and the temporal filter", "[nr][unit][gpu]") {
    D3D12Device dev({true, false});
    NrCompose compose(dev);
    const uint32_t w = 8, h = 4;
    auto original = UploadRgba16fConst(dev, w, h, 0.25f, 0.5f, 0.75f);
    auto model = UploadRgba16fConst(dev, w, h, 0.5f, 0.25f, 0.5f);
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    NrResolveParams p;
    p.width = w;
    p.height = h;
    p.workWidth = w;
    p.workHeight = h;

    SECTION("direct mode returns the model output") {
        NrResolveInputs in;
        in.original = original.Get();
        in.model = model.Get();
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        const std::vector<float> rgb = ReadbackRgb(dev, out.Get(), w, h);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            CHECK(rgb[i * 3] == 0.5f);
            CHECK(rgb[i * 3 + 1] == 0.25f);
            CHECK(rgb[i * 3 + 2] == 0.5f);
        }
    }
    SECTION("protect mask keeps the original, skin mask scales the edit") {
        std::vector<float> protect(static_cast<size_t>(w) * h, 0.f), skin(static_cast<size_t>(w) * h, 0.f);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                if (x < w / 2) protect[y * w + x] = 1.f;  // left half protected (ui / ignore)
                else skin[y * w + x] = 1.f;               // right half skin
            }
        auto protectTex = UploadR32f(dev, w, h, protect);
        auto skinTex = UploadR32f(dev, w, h, skin);
        NrResolveInputs in;
        in.original = original.Get();
        in.model = model.Get();
        in.protect = protectTex.Get();
        in.skin = skinTex.Get();
        p.skinBlend = 0.5f;
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        const std::vector<float> rgb = ReadbackRgb(dev, out.Get(), w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3;
                if (x < w / 2) {
                    CHECK(rgb[i] == Approx(0.25f).margin(1e-3));  // original
                    CHECK(rgb[i + 1] == Approx(0.5f).margin(1e-3));
                } else {
                    CHECK(rgb[i] == Approx(0.375f).margin(2e-3));  // half way between original and model
                    CHECK(rgb[i + 1] == Approx(0.375f).margin(2e-3));
                    CHECK(rgb[i + 2] == Approx(0.625f).margin(2e-3));
                }
            }
    }
    SECTION("ratio transfer carries a reduced-resolution edit onto the full frame") {
        auto proxy = UploadRgba16fConst(dev, w / 2, h / 2, 0.25f, 0.5f, 0.75f);     // what the model saw
        auto smallModel = UploadRgba16fConst(dev, w / 2, h / 2, 0.5f, 0.5f, 0.75f / 3.f);  // model: red x2, green x1, blue /3
        NrResolveInputs in;
        in.original = original.Get();
        in.model = smallModel.Get();
        in.proxy = proxy.Get();
        p.workWidth = w / 2;
        p.workHeight = h / 2;
        p.maxRatio = 4.f;
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        std::vector<float> rgb = ReadbackRgb(dev, out.Get(), w, h);
        CHECK(rgb[0] == Approx(0.5f).margin(2e-3));
        CHECK(rgb[1] == Approx(0.5f).margin(2e-3));
        CHECK(rgb[2] == Approx(0.25f).margin(2e-3));
        p.maxRatio = 1.5f;  // the guard limits the ratio
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        rgb = ReadbackRgb(dev, out.Get(), w, h);
        CHECK(rgb[0] == Approx(0.375f).margin(2e-3));
        CHECK(rgb[2] == Approx(0.5f).margin(2e-3));
        p.maxRatio = 4.f;
        p.transfer = 0.5f;  // half the edit
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        rgb = ReadbackRgb(dev, out.Get(), w, h);
        CHECK(rgb[0] == Approx(0.375f).margin(2e-3));
    }
    SECTION("temporal filter blends with the previous frame under the threshold and follows the motion vectors") {
        auto prev = UploadRgba16fConst(dev, w, h, 0.52f, 0.25f, 0.5f);  // close to the model (red +0.02)
        NrResolveInputs in;
        in.original = original.Get();
        in.model = model.Get();
        in.prev = prev.Get();
        p.temporal = 0.5f;
        p.temporalThreshold = 0.1f;
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        std::vector<float> rgb = ReadbackRgb(dev, out.Get(), w, h);
        const float gate = 1.f - 0.02f / 0.1f;  // |res - prev| = 0.02
        CHECK(rgb[0] == Approx(0.5f + 0.5f * gate * 0.02f).margin(2e-3));
        CHECK(rgb[1] == Approx(0.25f).margin(2e-3));
        auto farPrev = UploadRgba16fConst(dev, w, h, 0.9f, 0.25f, 0.5f);  // beyond the threshold: rejected
        in.prev = farPrev.Get();
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        rgb = ReadbackRgb(dev, out.Get(), w, h);
        CHECK(rgb[0] == Approx(0.5f).margin(2e-3));
        // motion vectors: prev has a different colour in its right half; a vector of +4 px moves the sample there
        std::vector<float> half(static_cast<size_t>(w) * h * 3);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3;
                half[i] = x < w / 2 ? 0.5f : 0.54f;
                half[i + 1] = 0.25f;
                half[i + 2] = 0.5f;
            }
        auto prevHalf = UploadRgba16f(dev, w, h, half);
        std::vector<float> mv(static_cast<size_t>(w) * h * 2, 0.f);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) mv[i * 2] = 4.f;  // p(t-1) = p(t) + 4 px in x
        auto mvTex = UploadRg32f(dev, w, h, mv);
        in.prev = prevHalf.Get();
        in.mv = mvTex.Get();
        p.mvWidth = w;
        p.mvHeight = h;
        p.temporal = 1.f;
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        rgb = ReadbackRgb(dev, out.Get(), w, h);
        const float gate4 = 1.f - 0.04f / 0.1f;
        CHECK(rgb[0] == Approx(0.5f + gate4 * 0.04f).margin(3e-3));            // x = 0 samples prev at x = 4 (0.54)
        CHECK(rgb[(w - 1) * 3] == Approx(0.5f).margin(3e-3));                 // x = 7 + 4 is outside: no blend
    }
}

// PR-7: NrCompose applies the edit strength after the model; the model (and the stub) edit at their own strength (NrResolveParams::gain).
TEST_CASE("NrCompose edit strength: 1 is the resolve before PR-7, 2 doubles the edit, 0 is the original, masks scale it", "[nr][unit][gpu]") {
    D3D12Device dev({true, false});
    NrCompose compose(dev);
    const uint32_t w = 8, h = 4;
    auto original = UploadRgba16fConst(dev, w, h, 0.25f, 0.5f, 0.75f);
    auto model = UploadRgba16fConst(dev, w, h, 0.5f, 0.25f, 0.5f);  // the edit: (+0.25, -0.25, -0.25)
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    NrResolveParams p;
    p.width = w;
    p.height = h;
    p.workWidth = w;
    p.workHeight = h;
    NrResolveInputs in;
    in.original = original.Get();
    in.model = model.Get();
    auto resolve = [&](float gain) {
        p.gain = gain;
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        return ReadbackRgb(dev, out.Get(), w, h);
    };
    auto expect = [&](const std::vector<float>& rgb, float r, float g, float b) {
        CHECK(rgb[0] == Approx(r).margin(2e-3));
        CHECK(rgb[1] == Approx(g).margin(2e-3));
        CHECK(rgb[2] == Approx(b).margin(2e-3));
    };
    expect(resolve(1.f), 0.5f, 0.25f, 0.5f);      // the model's picture
    expect(resolve(2.f), 0.75f, 0.f, 0.25f);      // twice the edit
    expect(resolve(0.5f), 0.375f, 0.375f, 0.625f);
    expect(resolve(2.5f), 0.875f, 0.f, 0.125f);   // saturate last: green -0.125 clamps to 0
    CHECK_THROWS(resolve(-0.5f));
    p.gain = 1.f;

    // 0 is the original exactly, the flicker filter and the masks included (the filter fades with the edit below 1)
    auto prev = UploadRgba16fConst(dev, w, h, 0.3f, 0.45f, 0.7f);
    std::vector<float> half(static_cast<size_t>(w) * h, 0.f);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = w / 2; x < w; ++x) half[y * w + x] = 1.f;
    auto skinTex = UploadR32f(dev, w, h, half);
    in.prev = prev.Get();
    in.skin = skinTex.Get();
    p.temporal = 1.f;
    p.temporalThreshold = 1.f;
    p.skinBlend = 0.5f;
    const std::vector<float> zero = resolve(0.f);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        CHECK(zero[i * 3] == 0.25f);
        CHECK(zero[i * 3 + 1] == 0.5f);
        CHECK(zero[i * 3 + 2] == 0.75f);
    }
    in.prev = nullptr;
    p.temporal = 0.f;

    // the skin mask scales the edit, the strength scales it again: skinBlend 0.5 x gain 2 = the model's picture on skin
    const std::vector<float> skin = resolve(2.f);
    expect(skin, 0.75f, 0.f, 0.25f);                                   // left: no skin, twice the edit
    const size_t right = (w - 1) * 3;
    CHECK(skin[right] == Approx(0.5f).margin(2e-3));                   // right: w 0.5 x 2 = 1
    CHECK(skin[right + 1] == Approx(0.25f).margin(2e-3));
    in.skin = nullptr;
    auto protectTex = UploadR32f(dev, w, h, half);
    in.protect = protectTex.Get();
    const std::vector<float> prot = resolve(2.f);
    CHECK(prot[right] == Approx(0.25f).margin(2e-3));                  // protected: the original whatever the strength
    expect(prot, 0.75f, 0.f, 0.25f);
}

// spec §2b (PR-7): the strength is applied once - after the ratio transfer and its clamp, with the masks, before the flicker filter
// and saturate. Each case below gives another number when the gain sits elsewhere.
TEST_CASE("NrCompose order of the strength controls: after the ratio transfer, before the flicker filter", "[nr][unit][gpu][regression]") {
    D3D12Device dev({true, false});
    NrCompose compose(dev);
    const uint32_t w = 8, h = 4;
    auto original = UploadRgba16fConst(dev, w, h, 0.25f, 0.5f, 0.75f);
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    NrResolveParams p;
    p.width = w;
    p.height = h;
    auto run = [&](const NrResolveInputs& in) {
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { compose.Resolve(cl, in, out.Get(), p); });
        return ReadbackRgb(dev, out.Get(), w, h);
    };
    {  // the ratio transfer: the clamp of the ratio first, the strength on the edit it gives (before the clamp: 0.375 red)
        auto proxy = UploadRgba16fConst(dev, w / 2, h / 2, 0.25f, 0.5f, 0.75f);
        auto smallModel = UploadRgba16fConst(dev, w / 2, h / 2, 0.5f, 0.5f, 0.25f);  // ratio (2, 1, 1/3)
        NrResolveInputs in;
        in.original = original.Get();
        in.model = smallModel.Get();
        in.proxy = proxy.Get();
        p.workWidth = w / 2;
        p.workHeight = h / 2;
        p.maxRatio = 1.5f;  // clamped to (1.5, 1, 2/3): edited (0.375, 0.5, 0.5)
        p.gain = 2.f;
        std::vector<float> rgb = run(in);
        CHECK(rgb[0] == Approx(0.5f).margin(2e-3));
        CHECK(rgb[1] == Approx(0.5f).margin(2e-3));
        CHECK(rgb[2] == Approx(0.25f).margin(2e-3));
        p.maxRatio = 4.f;
        p.transfer = 0.5f;  // half the transferred edit, twice the strength: the same picture
        rgb = run(in);
        CHECK(rgb[0] == Approx(0.5f).margin(2e-3));
        CHECK(rgb[2] == Approx(0.25f).margin(2e-3));
        p.transfer = 1.f;
    }
    p.workWidth = w;
    p.workHeight = h;
    auto model = UploadRgba16fConst(dev, w, h, 0.5f, 0.25f, 0.5f);
    {  // the flicker filter after the strength: gain 2 gives 0.75 red, the previous frame 0.77 is within the threshold
        auto prev = UploadRgba16fConst(dev, w, h, 0.77f, 0.f, 0.25f);
        NrResolveInputs in;
        in.original = original.Get();
        in.model = model.Get();
        in.prev = prev.Get();
        p.gain = 2.f;
        p.temporal = 0.5f;
        p.temporalThreshold = 0.1f;
        const std::vector<float> rgb = run(in);
        // gate = 1 - 0.02 / 0.1 = 0.8, k = 0.5 x 0.8 = 0.4: 0.75 + 0.4 x 0.02; the strength after the filter would give 0.75
        CHECK(rgb[0] == Approx(0.758f).margin(2e-3));
    }
    {  // below 1 the filter fades with the edit: k = 1 x 0.9 x 0.5 (without the fade 0.42)
        auto prev = UploadRgba16fConst(dev, w, h, 0.425f, 0.375f, 0.625f);
        NrResolveInputs in;
        in.original = original.Get();
        in.model = model.Get();
        in.prev = prev.Get();
        p.gain = 0.5f;
        p.temporal = 1.f;
        p.temporalThreshold = 0.5f;
        const std::vector<float> rgb = run(in);
        CHECK(rgb[0] == Approx(0.3975f).margin(2e-3));
    }
}

TEST_CASE("StubNrBackend edits deterministically at its own strength; local_structure scales the detail", "[nr][unit][gpu]") {
    D3D12Device dev({true, false});
    const uint32_t w = 16, h = 8;
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = ((x / 2 + y) % 2) ? 0.7f : 0.3f;  // checker: local contrast to enhance
    auto src = UploadRgba16f(dev, w, h, rgb);
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto run = [&](float structure) {
        StubNrBackend stub;
        NrConfig cfg;
        cfg.width = w;
        cfg.height = h;
        cfg.localStructure = structure;
        cfg.localTone = 0.f;
        stub.Init(dev, cfg);
        CHECK_FALSE(stub.Describe().contains("intensity"));  // PR-7: the strength is NrCompose's, not the model's
        NrInputs in;
        in.color = src.Get();
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { stub.Evaluate(cl, in, out.Get()); });
        CHECK(stub.Calls() == 1);
        CHECK(stub.Diagnostics().ok);
        return ReadbackRgb(dev, out.Get(), w, h);
    };
    const std::vector<float> a = run(1.f), b = run(1.f), c = run(2.f);
    CHECK(a == b);  // deterministic
    double diffA = 0, diffC = 0;
    for (size_t i = 0; i < rgb.size(); ++i) {
        diffA += std::fabs(a[i] - rgb[i]);
        diffC += std::fabs(c[i] - rgb[i]);
        CHECK(a[i] >= 0.f);
        CHECK(a[i] <= 1.f);
    }
    CHECK(diffA > 0.0);
    CHECK(diffC > diffA);  // more structure, more edit
}

TEST_CASE("NrBypassParams and the A/B switches at intensity 0", "[nr][unit]") {
    CHECK(NrBypassParams({{"intensity", 0}}));
    CHECK(NrBypassParams({{"intensity", 0.0}}));
    CHECK_FALSE(NrBypassParams({{"intensity", 0.05}}));
    CHECK_FALSE(NrBypassParams(nlohmann::json::object()));  // absent = 1
    CHECK_FALSE(NrBypassParams({{"intensity", "0"}}));
    NrStageOptions o;
    o.intensity = 0.f;
    CHECK(NrAbProblems(o).empty());
    o.createWithMv = true;  // switches of the model's guides: the model is not called at 0
    CHECK_FALSE(NrAbProblems(o).empty());
}

// TASK-0048 (spec §2e, E3): DLSSNR.MVecScale - "unit" passes mv_dlss pixels of its own resolution as they are (the sub-rect
// carries the size), "auto" rescales them to the model's working pixels; no vectors, nothing to scale.
TEST_CASE("NrMvScale: unit passes mv_dlss pixels as is, auto rescales them to the model size", "[nr][unit]") {
    using P = std::pair<float, float>;
    CHECK(NrMvScale("unit", 2560, 1440, 1280, 720) == P{1.f, 1.f});
    CHECK(NrMvScale("auto", 2560, 1440, 1280, 720) == P{2.f, 2.f});
    CHECK(NrMvScale("auto", 1280, 720, 1280, 720) == P{1.f, 1.f});  // the same size: auto changes nothing (E3 measures nothing)
    CHECK(NrMvScale("auto", 2560, 1440, 0, 0) == P{1.f, 1.f});
    const auto [x, y] = NrMvScale("auto", 1920, 1080, 960, 544);  // each axis on its own
    CHECK(x == Approx(2.0));
    CHECK(y == Approx(1080.0 / 544.0));
    CHECK_THROWS_WITH(NrMvScale("pixels", 1, 1, 1, 1), Catch::Matchers::ContainsSubstring("unit | auto"));
}

// TASK-0048 (spec E6, E8): the cost of creating the feature - null in JSON until measured (never -1 read as a time); the stub
// measures its own init on WARP so `nr --check` prints the same lines for either backend.
TEST_CASE("NR diagnostics: init / create times and VRAM are null until measured; the stub measures them on WARP", "[nr][unit][gpu]") {
    const nlohmann::json j = NrDiagnostics{}.ToJson();
    for (const char* k : {"init_ms", "create_ms", "vram_init_bytes", "vram_feature_bytes", "vram_budget_bytes"}) {
        INFO(k);
        CHECK(j.at(k).is_null());
    }
    D3D12Device dev({true, false});
    StubNrBackend stub;
    NrConfig cfg;
    cfg.backend = "stub";
    cfg.width = 16;
    cfg.height = 16;
    stub.Init(dev, cfg);
    CHECK(stub.Diagnostics().initMs >= 0.0);
    CHECK(stub.Diagnostics().createMs == 0.0);
    CHECK((stub.Diagnostics().vramFeatureBytes == 0 || stub.Diagnostics().vramFeatureBytes == -1));  // WARP may not report
    CHECK(stub.Diagnostics().ToJson()["init_ms"].is_number());
    stub.Shutdown();
    cfg.mvScale = "pixels";
    CHECK_THROWS(stub.Init(dev, cfg));
}
