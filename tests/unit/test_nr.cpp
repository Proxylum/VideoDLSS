// Neural Rendering building blocks (stage 6): driver version decoding, architecture names, the patcher
// command line, the forwarder module, masks and the resolve kernel on WARP.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
    CHECK(ParseNrStyle("4") == 4);
    CHECK_THROWS(ParseNrStyle("vivid"));
    CHECK_THROWS(ParseNrStyle("99"));
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

TEST_CASE("StubNrBackend edits deterministically and scales with intensity", "[nr][unit][gpu]") {
    D3D12Device dev({true, false});
    const uint32_t w = 16, h = 8;
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) rgb[(static_cast<size_t>(y) * w + x) * 3 + c] = ((x / 2 + y) % 2) ? 0.7f : 0.3f;  // checker: local contrast to enhance
    auto src = UploadRgba16f(dev, w, h, rgb);
    auto out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto run = [&](float intensity) {
        StubNrBackend stub;
        NrConfig cfg;
        cfg.width = w;
        cfg.height = h;
        cfg.intensity = intensity;
        stub.Init(dev, cfg);
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
    CHECK(diffC > diffA);  // more intensity, more edit
}
