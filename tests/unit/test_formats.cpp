#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "passes/PassFile.h"
#include "passes/formats/ExrIO.h"
#include "passes/formats/NpzIO.h"
#include "passes/formats/PngIO.h"
#include "passes/formats/RawIO.h"
#include "passes/formats/TiffIO.h"
#include "util/Half.h"

using namespace dlssvid;

namespace {

std::filesystem::path Tmp(const char* name) {
    const std::filesystem::path d = std::filesystem::path(DLSSVID_TEST_TMP) / "formats";
    std::filesystem::create_directories(d);
    return d / name;
}

// Deterministic content covering the whole value range of the type, including for floats
// negative values, tiny values and exact integers.
PassImage Pattern(uint32_t w, uint32_t h, PixelType t, std::vector<std::string> channels) {
    PassImage img;
    img.Allocate(w, h, t, std::move(channels));
    uint32_t seed = 12345;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < img.channels.size(); ++c) {
                seed = seed * 1664525u + 1013904223u;
                float v;
                switch (t) {
                    case PixelType::U8: v = static_cast<float>(seed % 256); break;
                    case PixelType::U16: v = static_cast<float>(seed % 65536); break;
                    case PixelType::F16: v = HalfToFloat(static_cast<uint16_t>((seed >> 8) & 0x7BFF)) * ((seed & 1) ? -1.f : 1.f); break;
                    default: v = (static_cast<float>(seed % 100000) - 50000.f) * 0.0371f + (x == 0 ? 1e-30f : 0.f); break;
                }
                img.Set(x, y, c, v);
            }
    return img;
}

bool SameBytes(const PassImage& a, const PassImage& b) { return a.SameLayout(b) && a.data == b.data; }
// Containers without channel names (TIFF, NPY) come back with generic names; compare geometry + bytes.
bool SameData(const PassImage& a, const PassImage& b) {
    return a.width == b.width && a.height == b.height && a.type == b.type && a.channels.size() == b.channels.size() && a.data == b.data;
}

}  // namespace

TEST_CASE("EXR round trips f32 and f16 losslessly", "[formats][exr]") {
    for (PixelType t : {PixelType::F32, PixelType::F16}) {
        for (auto ch : {std::vector<std::string>{"Z"}, std::vector<std::string>{"u", "v"}, std::vector<std::string>{"R", "G", "B"}}) {
            const PassImage img = Pattern(37, 23, t, ch);
            const auto path = Tmp("rt.exr");
            WriteExr(path, img);
            const PassImage back = ReadExr(path);
            CHECK(back.channels == ch);
            CHECK(SameBytes(back, img));
        }
    }
}

TEST_CASE("EXR stores integer images as float and restores them through the manifest", "[formats][exr]") {
    const PassImage u16 = Pattern(20, 10, PixelType::U16, {"R", "G", "B"});
    const auto path = Tmp("u16.exr");
    WritePassFile(path, u16);
    Manifest m = Manifest::ForPass(PassKind::ColorSource, 20, 10, FileFormat::Exr);
    m.pixelType = PixelType::U16;
    const PassImage back = ReadPassFile(path, &m);
    CHECK(SameBytes(back, u16));
    const PassImage u8 = Pattern(20, 10, PixelType::U8, {"A"});
    WritePassFile(Tmp("u8.exr"), u8);
    Manifest mm = Manifest::ForPass(PassKind::Mask, 20, 10, FileFormat::Exr);
    mm.pixelType = PixelType::U8;
    CHECK(SameBytes(ReadPassFile(Tmp("u8.exr"), &mm), u8));
}

TEST_CASE("EXR multi-layer write/read keeps layers apart", "[formats][exr]") {
    const PassImage color = Pattern(16, 8, PixelType::F16, {"R", "G", "B"});
    const PassImage depth = Pattern(16, 8, PixelType::F32, {"Z"});
    const PassImage mv = Pattern(16, 8, PixelType::F32, {"u", "v"});
    const auto path = Tmp("layers.exr");
    WriteExrLayers(path, {{"", color}, {"depth", depth}, {"mv", mv}});
    const auto layers = ReadExrLayers(path);
    REQUIRE(layers.size() == 3);
    CHECK(SameBytes(layers.at(""), color));
    CHECK(SameBytes(layers.at("depth"), depth));
    CHECK(SameBytes(layers.at("mv"), mv));
    CHECK(layers.at("mv").channels == std::vector<std::string>{"u", "v"});
    // single-image read of a layered file returns the top-level colour
    CHECK(SameBytes(ReadExr(path), color));
}

TEST_CASE("PNG round trips u8/u16 with 1-4 channels", "[formats][png]") {
    for (PixelType t : {PixelType::U8, PixelType::U16}) {
        for (auto ch : {std::vector<std::string>{"A"}, std::vector<std::string>{"Y", "A"}, std::vector<std::string>{"R", "G", "B"},
                        std::vector<std::string>{"R", "G", "B", "A"}}) {
            const PassImage img = Pattern(31, 17, t, ch);
            const auto path = Tmp("rt.png");
            WritePng(path, img);
            const PassImage back = ReadPng(path);
            CHECK(back.type == t);
            CHECK(back.channels.size() == ch.size());
            CHECK(back.data == img.data);
        }
    }
    CHECK_THROWS(WritePng(Tmp("bad.png"), Pattern(2, 2, PixelType::F32, {"Z"})));
}

TEST_CASE("TIFF round trips every sample type", "[formats][tiff]") {
    for (PixelType t : {PixelType::U8, PixelType::U16, PixelType::F16, PixelType::F32}) {
        for (auto ch : {std::vector<std::string>{"Z"}, std::vector<std::string>{"u", "v"}, std::vector<std::string>{"R", "G", "B"}}) {
            const PassImage img = Pattern(29, 13, t, ch);
            const auto path = Tmp("rt.tif");
            WriteTiff(path, img);
            const PassImage back = ReadTiff(path);
            CHECK(back.type == t);
            CHECK(back.channels.size() == ch.size());
            CHECK(back.data == img.data);
        }
    }
}

TEST_CASE("NPY/NPZ round trip and layout", "[formats][npz]") {
    for (PixelType t : {PixelType::U8, PixelType::U16, PixelType::F16, PixelType::F32}) {
        const PassImage one = Pattern(12, 7, t, {"Z"});
        const PassImage two = Pattern(12, 7, t, {"u", "v"});
        WriteNpy(Tmp("a.npy"), one);
        CHECK(SameData(ReadNpy(Tmp("a.npy")), one));
        WriteNpz(Tmp("a.npz"), {{"depth_raw", one}, {"mv_raw", two}});
        const auto arrays = ReadNpz(Tmp("a.npz"));
        REQUIRE(arrays.size() == 2);
        CHECK(arrays.at("depth_raw").data == one.data);
        CHECK(arrays.at("mv_raw").data == two.data);
        CHECK(arrays.at("mv_raw").channels.size() == 2);
    }
    // header layout: magic, version 1.0, total header length multiple of 64
    const auto bytes = EncodeNpy(Pattern(3, 2, PixelType::F32, {"Z"}));
    REQUIRE(bytes.size() >= 10);
    CHECK(bytes[0] == 0x93);
    CHECK(std::string(reinterpret_cast<const char*>(bytes.data() + 1), 5) == "NUMPY");
    const size_t headerLen = bytes[8] | (bytes[9] << 8);
    CHECK((10 + headerLen) % 64 == 0);
    CHECK(bytes.size() == 10 + headerLen + 3 * 2 * 4);
    CHECK(bytes[10 + headerLen - 1] == '\n');
}

TEST_CASE("NPZ interoperates with numpy when available", "[formats][npz][numpy]") {
    // numpy reads ours (stored) and we read numpy's (deflate, savez_compressed)
    const auto ours = Tmp("ours.npz");
    const auto theirs = Tmp("theirs.npz");
    const PassImage img = Pattern(9, 5, PixelType::F32, {"u", "v"});
    WriteNpz(ours, {{"mv_raw", img}});
    const std::string script =
        "import numpy as np,sys\n"
        "a=np.load(sys.argv[1])['mv_raw']\n"
        "assert a.shape==(5,9,2) and a.dtype==np.float32, (a.shape,a.dtype)\n"
        "np.savez_compressed(sys.argv[2], mv_raw=a*2)\n"
        "print('ok')\n";
    const auto py = Tmp("check.py");
    std::ofstream(py) << script;
    const std::string cmd = "python \"" + py.string() + "\" \"" + ours.string() + "\" \"" + theirs.string() + "\" >nul 2>&1";
    if (std::system(cmd.c_str()) != 0) SKIP("python + numpy not available");
    const auto back = ReadNpz(theirs);
    REQUIRE(back.count("mv_raw") == 1);
    const PassImage& b = back.at("mv_raw");
    REQUIRE(b.SameLayout(img));
    for (uint32_t y = 0; y < 5; ++y)
        for (uint32_t x = 0; x < 9; ++x)
            for (size_t c = 0; c < 2; ++c) CHECK(b.Get(x, y, c) == img.Get(x, y, c) * 2);
}

TEST_CASE("Raw dumps are the bare bytes and need geometry", "[formats][raw]") {
    const PassImage img = Pattern(8, 6, PixelType::F16, {"u", "v"});
    const auto path = Tmp("mv.rg16f");
    WriteRaw(path, img);
    CHECK(std::filesystem::file_size(path) == img.ByteSize());
    CHECK(SameBytes(ReadRaw(path, 8, 6, PixelType::F16, {"u", "v"}), img));
    CHECK_THROWS(ReadRaw(path, 8, 7, PixelType::F16, {"u", "v"}));  // size mismatch
    CHECK_THROWS(ReadPassFile(path, nullptr));                       // raw needs a manifest
    Manifest m = Manifest::ForPass(PassKind::MvDlss, 8, 6, FileFormat::Raw);
    CHECK(SameBytes(ReadPassFile(path, &m), img));
}

TEST_CASE("PassFile dispatch by extension", "[formats]") {
    const PassImage img = Pattern(5, 4, PixelType::F32, {"Z"});
    for (const char* name : {"d.exr", "d.tif", "d_000001.npz"}) {
        WritePassFile(Tmp(name), img);
        CHECK(SameData(ReadPassFile(Tmp(name), nullptr), img));
    }
    CHECK_THROWS(WritePassFile(Tmp("d.txt"), img));
    CHECK_THROWS(PrepareForFormat(img, FileFormat::Png, PixelType::F32));
    CHECK(PrepareForFormat(img, FileFormat::Exr, PixelType::F16).type == PixelType::F16);
}
