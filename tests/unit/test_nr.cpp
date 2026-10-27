// Neural Rendering building blocks (stage 6): driver version decoding, architecture names, the patcher
// command line, the forwarder module, masks, the resolve kernel on WARP and the CPU low band of the control mask (PR-8).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

#include "GpuTextures.h"
#include "frozen/NrComposePr7Dispatch.h"
#include "gpu/D3D12Device.h"
#include "stages/nr/INrBackend.h"
#include "stages/nr/NrCompose.h"
#include "stages/nr/NrLowpass.h"
#include "stages/nr/NrPatch.h"
#include "stages/nr/NrStage.h"
#include "stages/nr/StubNrBackend.h"
#include "stages/nr/shaders/NrCompose_cs.h"

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

// PR-8: d of the control mask's G / B split is one share of the output frame's short side, at least 2 px.
TEST_CASE("NrLowpassDivisor: d from the short side, at least 2, an override wins", "[nr][unit]") {
    CHECK(NrLowpassDivisor(1280, 720) == 4);
    CHECK(NrLowpassDivisor(1920, 1080) == 6);
    CHECK(NrLowpassDivisor(2560, 1440) == 8);
    CHECK(NrLowpassDivisor(3840, 2160) == 12);
    CHECK(NrLowpassDivisor(720, 1280) == 4);  // the short side, whichever it is
    CHECK(NrLowpassDivisor(8, 4) == 2);       // never below 2
    CHECK(NrLowpassDivisor(1920, 1080, 5) == 5);
    const NrLowGrid g = NrLowpassGrid(2560, 1440, 8);
    CHECK(g.width == 320);
    CHECK(g.height == 180);
    CHECK(g.div == 8);
    const NrLowGrid odd = NrLowpassGrid(61, 37, 4);  // the last cell of a row / column is partial
    CHECK(odd.width == 16);
    CHECK(odd.height == 10);
    CHECK_THROWS(NrLowpassGrid(8, 8, 0));
}

namespace {
PassImage Plane(uint32_t w, uint32_t h, size_t channels, const std::function<float(uint32_t, uint32_t, size_t)>& f) {
    PassImage img;
    img.Allocate(w, h, PixelType::F32, channels == 1 ? std::vector<std::string>{"Y"} : std::vector<std::string>{"R", "G", "B"});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < channels; ++c) img.Set(x, y, c, f(x, y, c));
    return img;
}
}  // namespace

// The CPU twin of CSEditBox / CSEditGauss / LowUp: what the kernel tests and E10 measure against.
TEST_CASE("NrLowpassCpu: a constant passes, a 2 px checkerboard has no low band, edge cells average only existing pixels, the upsample is centred", "[nr][unit]") {
    // a constant at sides not divisible by d: the partial edge cells average the pixels they have (a d x d divisor would darken them)
    const PassImage flat = Plane(61, 37, 3, [](uint32_t, uint32_t, size_t c) { return 0.25f + 0.1f * static_cast<float>(c); });
    const PassImage flatLow = NrLowpassCpu(flat, 4);
    REQUIRE(flatLow.width == 61);
    REQUIRE(flatLow.height == 37);
    REQUIRE(flatLow.channels.size() == 3);
    double worst = 0.0;
    for (uint32_t y = 0; y < 37; ++y)
        for (uint32_t x = 0; x < 61; ++x)
            for (size_t c = 0; c < 3; ++c) worst = std::max(worst, std::fabs(static_cast<double>(flatLow.Get(x, y, c)) - flat.Get(x, y, c)));
    CHECK(worst < 1e-6);

    // a 2 px checkerboard: every cell of an even d has as many +1 as -1 - no low band at all
    const PassImage checker = Plane(64, 32, 1, [](uint32_t x, uint32_t y, size_t) { return ((x + y) % 2) ? 1.f : -1.f; });
    const PassImage checkerLow = NrLowpassCpu(checker, 4);
    for (uint32_t y = 0; y < 32; ++y)
        for (uint32_t x = 0; x < 64; ++x) CHECK(checkerLow.Get(x, y, 0) == 0.f);
    CHECK(NrLowShare(checker, 4) == 0.0);

    // a ramp: box means sit at the cell centres (i + 0.5) d, so away from the clamped edges the band reproduces the ramp; centres a half
    // pixel off would shift it by 0.005
    const PassImage ramp = Plane(64, 32, 1, [](uint32_t x, uint32_t y, size_t) { return 0.01f * static_cast<float>(x) + 0.002f * static_cast<float>(y); });
    const PassImage rampLow = NrLowpassCpu(ramp, 4);
    double inner = 0.0;
    for (uint32_t y = 8; y < 24; ++y)
        for (uint32_t x = 8; x < 56; ++x) inner = std::max(inner, std::fabs(static_cast<double>(rampLow.Get(x, y, 0)) - ramp.Get(x, y, 0)));
    CHECK(inner < 1e-5);

    // a non-finite sample counts as 0 in its cell: the band stays finite everywhere
    PassImage spot = Plane(16, 16, 1, [](uint32_t, uint32_t, size_t) { return 1.f; });
    spot.Set(5, 5, 0, std::nanf(""));
    const PassImage spotLow = NrLowpassCpu(spot, 4);
    for (uint32_t y = 0; y < 16; ++y)
        for (uint32_t x = 0; x < 16; ++x) CHECK(std::isfinite(spotLow.Get(x, y, 0)));
    CHECK(spotLow.Get(15, 15, 0) == Catch::Approx(1.0).margin(1e-6));  // far from it the constant passes
    CHECK(spotLow.Get(5, 5, 0) < 1.f);

    // the low-band share: a smooth wave is almost all low band, nothing is nothing
    const PassImage wave = Plane(256, 32, 3, [](uint32_t x, uint32_t, size_t) { return std::sin(static_cast<float>(x) * 6.2831853f / 128.f); });
    CHECK(NrLowShare(wave, 4) > 0.9);
    CHECK(NrLowShare(Plane(8, 8, 1, [](uint32_t, uint32_t, size_t) { return 0.f; }), 2) == 0.0);
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

// PR-8: without a control mask NrCompose runs the resolve of PR-7 bit for bit. The default variant of the shader compiles from text
// that the preprocessor turns into the PR-7 file token for token; the bytecode proves it (no debug info: line numbers do not count).
TEST_CASE("NrCompose default variant compiles to the frozen PR-7 DXIL", "[nr][unit][regression]") {
    REQUIRE(sizeof(g_NrComposeCS) == sizeof(g_NrComposePr7CS));
    CHECK(std::memcmp(g_NrComposeCS, g_NrComposePr7CS, sizeof(g_NrComposeCS)) == 0);
}

namespace {
float Noise(uint32_t x, uint32_t y, uint32_t salt) {
    uint32_t h = x * 73856093u ^ y * 19349663u ^ (salt + 1) * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return static_cast<float>(h & 0xFFFF) / 65535.f;
}
}  // namespace

// The same resolve through today's C++ (constants, flags, the list of SRVs) and through the frozen copy of PR-7's: the raw half words
// of every output agree over sizes, patterns, both modes, masks, the flicker filter and the edit strength.
TEST_CASE("NrCompose without a control mask is bit-identical to the frozen PR-7 resolve (WARP battery)", "[nr][unit][gpu][regression]") {
    D3D12Device dev({true, false});
    NrCompose now(dev);
    test::frozen::NrComposePr7 pr7(dev);
    int configs = 0, mismatches = 0;
    for (const auto& [w, h] : {std::pair<uint32_t, uint32_t>{61, 37}, std::pair<uint32_t, uint32_t>{96, 64}}) {
        const uint32_t ww = w / 2 + 1, wh = h / 2 + 1;  // mode 1: a reduced model, odd sides for 61x37
        const auto outNow = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto outPr7 = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for (int pattern = 0; pattern < 3; ++pattern) {
            auto value = [&](uint32_t x, uint32_t y, uint32_t c) {
                if (pattern == 0) return 0.05f + 0.9f * static_cast<float>(x + 2 * y + c) / static_cast<float>(w + 2 * h + 2);  // gradient
                if (pattern == 1) return ((x / 3 + y / 3 + c) % 2) ? 0.7f : 0.3f;                                               // checker
                return Noise(x, y, c);                                                                                          // noise
            };
            auto image = [&](uint32_t tw, uint32_t th, uint32_t step, uint32_t salt, float edit) {
                std::vector<float> v(static_cast<size_t>(tw) * th * 3);
                for (uint32_t y = 0; y < th; ++y)
                    for (uint32_t x = 0; x < tw; ++x)
                        for (uint32_t c = 0; c < 3; ++c)
                            v[(static_cast<size_t>(y) * tw + x) * 3 + c] =
                                std::clamp(value(std::min(x * step, w - 1), std::min(y * step, h - 1), c) + edit * (Noise(x, y, c + salt) - 0.5f), 0.f, 1.f);
                return v;
            };
            auto plane = [](uint32_t tw, uint32_t th, uint32_t salt, float lo, float hi) {
                std::vector<float> v(static_cast<size_t>(tw) * th);
                for (uint32_t y = 0; y < th; ++y)
                    for (uint32_t x = 0; x < tw; ++x) v[static_cast<size_t>(y) * tw + x] = lo + (hi - lo) * Noise(x, y, salt);
                return v;
            };
            const auto orig = UploadRgba16f(dev, w, h, image(w, h, 1, 0, 0.f));
            const auto modelFull = UploadRgba16f(dev, w, h, image(w, h, 1, 10, 0.3f));
            const auto proxy = UploadRgba16f(dev, ww, wh, image(ww, wh, 2, 0, 0.f));
            const auto modelWork = UploadRgba16f(dev, ww, wh, image(ww, wh, 2, 20, 0.3f));
            const auto prev = UploadRgba16f(dev, w, h, image(w, h, 1, 30, 0.2f));
            std::vector<float> mvData(static_cast<size_t>(ww) * wh * 2);
            for (size_t i = 0; i < mvData.size(); ++i) mvData[i] = 4.f * (Noise(static_cast<uint32_t>(i), 7, 40) - 0.5f);
            const auto mv = UploadRg32f(dev, ww, wh, mvData);
            const auto protect = UploadR32f(dev, w, h, plane(w, h, 50, -0.2f, 1.2f));  // outside [0, 1]: saturate in the shader
            const auto skin = UploadR32f(dev, w, h, plane(w, h, 60, 0.f, 1.f));
            for (int mode = 0; mode < 3; ++mode)
                for (int masks = 0; masks < 2; ++masks)
                    for (int temporal = 0; temporal < 3; ++temporal)
                        for (const float gain : {0.f, 0.5f, 1.f, 2.f}) {
                            NrResolveParams rp;
                            rp.width = w;
                            rp.height = h;
                            rp.workWidth = mode ? ww : w;
                            rp.workHeight = mode ? wh : h;
                            rp.transfer = mode == 1 ? 0.5f : 1.f;
                            rp.maxRatio = mode == 1 ? 1.5f : 4.f;
                            rp.temporal = temporal ? 0.5f : 0.f;
                            rp.temporalThreshold = temporal == 2 ? 1.f : 0.1f;
                            rp.skinBlend = 0.5f;
                            rp.gain = gain;
                            rp.mvWidth = ww;
                            rp.mvHeight = wh;
                            NrResolveInputs ri;
                            ri.original = orig.Get();
                            ri.model = mode ? modelWork.Get() : modelFull.Get();
                            ri.proxy = mode ? proxy.Get() : nullptr;
                            ri.prev = temporal ? prev.Get() : nullptr;
                            ri.mv = temporal ? mv.Get() : nullptr;
                            ri.protect = masks ? protect.Get() : nullptr;
                            ri.skin = masks ? skin.Get() : nullptr;
                            dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
                                now.Resolve(cl, ri, outNow.Get(), rp);
                                pr7.Resolve(cl, ri, outPr7.Get(), rp);
                            });
                            ++configs;
                            if (ReadbackHalfBits(dev, outNow.Get()) != ReadbackHalfBits(dev, outPr7.Get())) {
                                ++mismatches;
                                UNSCOPED_INFO(w << "x" << h << " pattern " << pattern << " mode " << mode << " masks " << masks << " temporal " << temporal
                                                << " gain " << gain);
                            }
                        }
        }
    }
    CHECK(configs == 2 * 3 * 3 * 2 * 3 * 4);
    CHECK(mismatches == 0);
}

namespace {
// The resolve's inputs at w x h for the control mask tests: a reduced model for mode 1, a previous frame, vectors and protect / skin
// masks. Values stay inside [0.2, 0.8] and the edits small, so no saturate clips what the tests compare.
struct ResolveBench {
    D3D12Device& dev;
    uint32_t w, h, ww, wh;
    std::vector<float> orig, model;  // as uploaded: RGB floats of the full-size frame and the mode 0 model
    ComPtr<ID3D12Resource> origTex, modelTex, proxyTex, modelWorkTex, prevTex, mvTex, protectTex, skinTex, out;
    ResolveBench(D3D12Device& d, uint32_t width, uint32_t height) : dev(d), w(width), h(height), ww(width / 2 + 1), wh(height / 2 + 1) {
        auto image = [](uint32_t tw, uint32_t th, uint32_t salt, const std::function<float(float, float)>& f) {
            std::vector<float> v(static_cast<size_t>(tw) * th * 3);
            for (uint32_t y = 0; y < th; ++y)
                for (uint32_t x = 0; x < tw; ++x)
                    for (uint32_t c = 0; c < 3; ++c) v[(static_cast<size_t>(y) * tw + x) * 3 + c] = f(Noise(x, y, c + salt), Noise(x, y, c + salt + 100));
            return v;
        };
        orig = image(w, h, 0, [](float a, float) { return 0.3f + 0.3f * a; });
        model = orig;
        for (size_t i = 0; i < model.size(); ++i) model[i] += 0.2f * (Noise(static_cast<uint32_t>(i), 1, 7) - 0.5f);
        const std::vector<float> proxy = image(ww, wh, 20, [](float a, float) { return 0.3f + 0.3f * a; });
        std::vector<float> modelWork = proxy;
        for (size_t i = 0; i < modelWork.size(); ++i) modelWork[i] *= 1.f + 0.3f * (Noise(static_cast<uint32_t>(i), 2, 9) - 0.5f);  // ratio 0.85..1.15
        origTex = UploadRgba16f(dev, w, h, orig);
        modelTex = UploadRgba16f(dev, w, h, model);
        proxyTex = UploadRgba16f(dev, ww, wh, proxy);
        modelWorkTex = UploadRgba16f(dev, ww, wh, modelWork);
        prevTex = UploadRgba16f(dev, w, h, image(w, h, 40, [](float a, float) { return 0.3f + 0.3f * a; }));
        std::vector<float> mv(static_cast<size_t>(ww) * wh * 2), protect(static_cast<size_t>(w) * h), skin(protect.size());
        for (size_t i = 0; i < mv.size(); ++i) mv[i] = 3.f * (Noise(static_cast<uint32_t>(i), 3, 11) - 0.5f);
        for (size_t i = 0; i < protect.size(); ++i) {
            protect[i] = Noise(static_cast<uint32_t>(i), 4, 13);
            skin[i] = Noise(static_cast<uint32_t>(i), 5, 17);
        }
        mvTex = UploadRg32f(dev, ww, wh, mv);
        protectTex = UploadR32f(dev, w, h, protect);
        skinTex = UploadR32f(dev, w, h, skin);
        out = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    NrResolveParams Params(int mode, float gain, bool temporal) const {
        NrResolveParams p;
        p.width = w;
        p.height = h;
        p.workWidth = mode ? ww : w;
        p.workHeight = mode ? wh : h;
        p.maxRatio = 1.5f;
        p.temporal = temporal ? 0.5f : 0.f;
        p.temporalThreshold = 0.3f;
        p.skinBlend = 0.5f;
        p.gain = gain;
        p.mvWidth = ww;
        p.mvHeight = wh;
        return p;
    }
    NrResolveInputs Inputs(int mode, bool masks, bool temporal) const {
        NrResolveInputs in;
        in.original = origTex.Get();
        in.model = mode ? modelWorkTex.Get() : modelTex.Get();
        in.proxy = mode ? proxyTex.Get() : nullptr;
        in.prev = temporal ? prevTex.Get() : nullptr;
        in.mv = temporal ? mvTex.Get() : nullptr;
        in.protect = masks ? protectTex.Get() : nullptr;
        in.skin = masks ? skinTex.Get() : nullptr;
        return in;
    }
    std::vector<uint16_t> Run(NrCompose& nc, const NrResolveInputs& in, const NrResolveParams& p) {
        dev.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) { nc.Resolve(cl, in, out.Get(), p); });
        return ReadbackHalfBits(dev, out.Get());
    }
    ComPtr<ID3D12Resource> Grey(uint16_t v) const { return UploadR16Unorm(dev, w, h, std::vector<uint16_t>(static_cast<size_t>(w) * h, v)); }
    ComPtr<ID3D12Resource> Rgb(uint16_t r, uint16_t g, uint16_t b) const {
        std::vector<uint16_t> v(static_cast<size_t>(w) * h * 4);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            v[i * 4] = r;
            v[i * 4 + 1] = g;
            v[i * 4 + 2] = b;
            v[i * 4 + 3] = 65535;
        }
        return UploadRgba16Unorm(dev, w, h, v);
    }
};

// Colour samples farther apart than 1 F16 ulp (4 steps among the subnormals, |v| < 2^-14); alpha is not compared.
int UlpViolations(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (i % 4 == 3) continue;
        const int tol = (want[i] & 0x7FFF) < 0x0400 ? 4 : 1;
        if (HalfOrderedDistance(got[i], want[i]) > tol) ++bad;
    }
    return bad;
}
}  // namespace

// PR-8: the control mask variant against the default one. White and grey (v, 1, 1) are the same arithmetic as no mask and as the
// strength v - two bytecodes may fuse orig + t * e differently, hence 1 ulp; R = 0 and G = B = 0 are no edit and no filter.
TEST_CASE("NrCompose control mask: white is within 1 F16 ulp of no mask (subnormals within 4 steps); grey v at w = 1 is within the same tolerance of intensity v; RGB (v,v,v) is v squared; R = 0 and G = B = 0 give the original and stop the temporal blend", "[nr][unit][gpu][mask]") {
    D3D12Device dev({true, false});
    ResolveBench b(dev, 61, 37);
    NrCompose nc(dev);
    const auto white = b.Rgb(65535, 65535, 65535), whiteGrey = b.Grey(65535);
    for (int mode = 0; mode < 2; ++mode)
        for (int masks = 0; masks < 2; ++masks)
            for (int temporal = 0; temporal < 2; ++temporal)
                for (const float gain : {0.5f, 1.f, 2.f}) {
                    INFO("mode " << mode << " masks " << masks << " temporal " << temporal << " gain " << gain);
                    const NrResolveParams p = b.Params(mode, gain, temporal != 0);
                    const std::vector<uint16_t> ref = b.Run(nc, b.Inputs(mode, masks != 0, temporal != 0), p);
                    NrResolveInputs in = b.Inputs(mode, masks != 0, temporal != 0);
                    in.control = white.Get();
                    in.controlRgb = true;
                    CHECK(UlpViolations(b.Run(nc, in, p), ref) == 0);
                    in.control = whiteGrey.Get();
                    in.controlRgb = false;
                    CHECK(UlpViolations(b.Run(nc, in, p), ref) == 0);
                }
    const std::vector<uint16_t> original = ReadbackHalfBits(dev, b.origTex.Get());
    for (const uint16_t code : {uint16_t{16384}, uint16_t{32768}, uint16_t{49152}}) {
        const float v = static_cast<float>(code) / 65535.f;
        const auto grey = b.Grey(code), rgb = b.Rgb(code, code, code);
        for (int mode = 0; mode < 2; ++mode) {
            INFO("code " << code << " mode " << mode);
            for (int temporal = 0; temporal < 2; ++temporal) {  // w = 1: no protect / skin masks
                NrResolveInputs in = b.Inputs(mode, false, temporal != 0);
                in.control = grey.Get();
                CHECK(UlpViolations(b.Run(nc, in, b.Params(mode, 1.f, temporal != 0)), b.Run(nc, b.Inputs(mode, false, temporal != 0), b.Params(mode, v, temporal != 0))) == 0);
            }
            NrResolveInputs in = b.Inputs(mode, false, false);  // (v, v, v): the edit x v^2 (without the filter, whose weight rounds otherwise)
            in.control = rgb.Get();
            in.controlRgb = true;
            CHECK(UlpViolations(b.Run(nc, in, b.Params(mode, 1.f, false)), b.Run(nc, b.Inputs(mode, false, false), b.Params(mode, v * v, false))) == 0);
        }
    }
    // no edit: R = 0, or G = B = 0 - the original exactly, though the previous frame differs and the filter is on
    for (const auto& none : {b.Rgb(0, 65535, 65535), b.Rgb(65535, 0, 0), b.Grey(0)})
        for (int mode = 0; mode < 2; ++mode)
            for (const float gain : {1.f, 2.f}) {
                NrResolveInputs in = b.Inputs(mode, true, true);
                in.control = none.Get();
                in.controlRgb = none->GetDesc().Format == DXGI_FORMAT_R16G16B16A16_UNORM;
                CHECK(b.Run(nc, in, b.Params(mode, gain, true)) == original);
            }
}

// The G / B split against the CPU twin: G = 1, B = 0 adds only the low band of the edit, G = 0, B = 1 only the rest. The edited
// colour comes from the default variant at strength 1 (no saturate clips it), the low band from NrLowpassCpu of edited - orig.
TEST_CASE("NrCompose control mask: G = 1, B = 0 keeps the CPU lowpass of the edit, G = 0, B = 1 the rest, in modes 0 and 1; sides not divisible by d (61x37, d 4)", "[nr][unit][gpu][mask]") {
    D3D12Device dev({true, false});
    ResolveBench b(dev, 61, 37);
    NrCompose nc(dev);
    const auto lowOnly = b.Rgb(65535, 65535, 0), highOnly = b.Rgb(65535, 0, 65535);
    const std::vector<uint16_t> original = ReadbackHalfBits(dev, b.origTex.Get());
    for (int mode = 0; mode < 2; ++mode) {
        INFO("mode " << mode);
        NrResolveParams p = b.Params(mode, 1.f, false);
        p.lowpassDivisor = 4;
        const std::vector<uint16_t> edited = b.Run(nc, b.Inputs(mode, false, false), p);
        PassImage e;
        e.Allocate(b.w, b.h, PixelType::F32, {"R", "G", "B"});
        for (uint32_t i = 0; i < b.w * b.h; ++i)
            for (uint32_t c = 0; c < 3; ++c) e.As<float>()[i * 3 + c] = HalfToFloat(edited[i * 4 + c]) - HalfToFloat(original[i * 4 + c]);
        const PassImage low = NrLowpassCpu(e, 4);
        NrResolveInputs in = b.Inputs(mode, false, false);
        in.controlRgb = in.controlSplit = true;
        in.control = lowOnly.Get();
        const std::vector<uint16_t> lo = b.Run(nc, in, p);
        in.control = highOnly.Get();
        const std::vector<uint16_t> hi = b.Run(nc, in, p);
        double worstLo = 0.0, worstHi = 0.0, lowSize = 0.0;
        for (uint32_t i = 0; i < b.w * b.h; ++i)
            for (uint32_t c = 0; c < 3; ++c) {
                const double o = HalfToFloat(original[i * 4 + c]), l = low.As<float>()[i * 3 + c];
                worstLo = std::max(worstLo, std::fabs(HalfToFloat(lo[i * 4 + c]) - (o + l)));
                worstHi = std::max(worstHi, std::fabs(HalfToFloat(hi[i * 4 + c]) - (HalfToFloat(edited[i * 4 + c]) - l)));
                lowSize = std::max(lowSize, std::fabs(l));
            }
        CHECK(worstLo < 1e-3);
        CHECK(worstHi < 1e-3);
        CHECK(lowSize > 0.005);  // the low band is there to compare
    }
}

// The kernels' grid against the CPU's: the same box means (summed in the same order) and the same Gauss. A NaN of the model counts as
// 0 in its cell, so the band stays finite and only that pixel differs from a model without it.
TEST_CASE("NrEditBox and NrEditGauss match NrLowpassCpu; a NaN of the model stays one pixel", "[nr][unit][gpu][mask]") {
    D3D12Device dev({true, false});
    ResolveBench b(dev, 61, 37);
    NrCompose nc(dev);
    const auto split = b.Rgb(65535, 65535, 16384);
    NrResolveParams p = b.Params(0, 1.f, false);
    p.lowpassDivisor = 4;
    NrResolveInputs in = b.Inputs(0, false, false);
    in.control = split.Get();
    in.controlRgb = in.controlSplit = true;
    b.Run(nc, in, p);
    REQUIRE(nc.LowBand());
    const std::vector<float> gpu = ReadbackRgba32f(dev, nc.LowBand());
    const std::vector<uint16_t> original = ReadbackHalfBits(dev, b.origTex.Get()), model = ReadbackHalfBits(dev, b.modelTex.Get());
    PassImage e;
    e.Allocate(b.w, b.h, PixelType::F32, {"R", "G", "B"});
    for (uint32_t i = 0; i < b.w * b.h; ++i)
        for (uint32_t c = 0; c < 3; ++c) e.As<float>()[i * 3 + c] = HalfToFloat(model[i * 4 + c]) - HalfToFloat(original[i * 4 + c]);
    const PassImage cpu = NrLowBandCpu(e, 4);
    REQUIRE(gpu.size() == static_cast<size_t>(cpu.width) * cpu.height * 4);
    double worst = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(cpu.width) * cpu.height; ++i)
        for (size_t c = 0; c < 3; ++c) worst = std::max(worst, std::fabs(static_cast<double>(gpu[i * 4 + c]) - cpu.As<float>()[i * 3 + c]));
    CHECK(worst < 1e-6);

    // a NaN of the model at (20, 15) against the same model with the original there (an edit of 0: the same cell sum)
    std::vector<float> nanModel = b.model, flatModel = b.model;
    const size_t at = (15 * static_cast<size_t>(b.w) + 20) * 3;
    for (size_t c = 0; c < 3; ++c) {
        nanModel[at + c] = std::nanf("");
        flatModel[at + c] = b.orig[at + c];
    }
    const auto nanTex = UploadRgba16f(dev, b.w, b.h, nanModel), flatTex = UploadRgba16f(dev, b.w, b.h, flatModel);
    in.model = nanTex.Get();
    const std::vector<uint16_t> withNan = b.Run(nc, in, p);
    in.model = flatTex.Get();
    const std::vector<uint16_t> without = b.Run(nc, in, p);
    int differ = 0, nonFinite = 0;
    for (size_t i = 0; i < withNan.size(); ++i) {
        if ((withNan[i] & 0x7C00) == 0x7C00) ++nonFinite;
        if (withNan[i] != without[i] && i / 4 != at / 3) ++differ;
    }
    CHECK(nonFinite == 0);  // saturate(NaN) is 0 at the pixel itself, as without a mask
    CHECK(differ == 0);
}

// Only a mask whose G differs from B somewhere runs the low band; a grey mask is one R16 channel and binds no grid.
TEST_CASE("NrCompose control mask: the low band runs only for a split mask; a grey mask is R16 and binds no t8", "[nr][unit][gpu][mask]") {
    D3D12Device dev({true, false});
    ResolveBench b(dev, 61, 37);
    NrCompose nc(dev);
    const NrResolveParams p = b.Params(0, 1.f, false);
    NrResolveInputs in = b.Inputs(0, false, false);
    const auto grey = b.Grey(30000), even = b.Rgb(40000, 20000, 20000), split = b.Rgb(40000, 20000, 50000);
    in.control = grey.Get();
    b.Run(nc, in, p);
    CHECK(nc.LowpassRuns() == 0);
    CHECK(nc.LowBand() == nullptr);
    in.control = even.Get();
    in.controlRgb = true;
    b.Run(nc, in, p);
    CHECK(nc.LowpassRuns() == 0);
    in.control = split.Get();
    in.controlSplit = true;
    b.Run(nc, in, p);
    CHECK(nc.LowpassRuns() == 1);
    REQUIRE(nc.LowBand());
    const NrLowGrid g = NrLowpassGrid(61, 37, NrLowpassDivisor(61, 37));
    CHECK(nc.LowBand()->GetDesc().Width == g.width);
    CHECK(nc.LowBand()->GetDesc().Height == g.height);
    CHECK(nc.LowBand()->GetDesc().Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
    b.Run(nc, in, p);
    CHECK(nc.LowpassRuns() == 2);
}

TEST_CASE("NrCompose refuses a control mask of another size", "[nr][unit][gpu][mask]") {
    D3D12Device dev({true, false});
    ResolveBench b(dev, 61, 37);
    NrCompose nc(dev);
    const auto tiny = UploadR16Unorm(dev, 30, 20, std::vector<uint16_t>(30 * 20, 65535));
    NrResolveInputs in = b.Inputs(0, false, false);
    in.control = tiny.Get();
    CHECK_THROWS_WITH(b.Run(nc, in, b.Params(0, 1.f, false)), Catch::Matchers::ContainsSubstring("frame size"));
    const auto grey = b.Grey(65535);
    in.control = grey.Get();
    in.controlSplit = true;  // a split needs the colour mask
    CHECK_THROWS_WITH(b.Run(nc, in, b.Params(0, 1.f, false)), Catch::Matchers::ContainsSubstring("colour control mask"));
    in.controlSplit = false;
    CHECK_NOTHROW(b.Run(nc, in, b.Params(0, 1.f, false)));  // the device still works after a refusal
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
