#include <catch2/catch_test_macros.hpp>

#include "passes/PassImage.h"

using namespace dlssvid;

TEST_CASE("PassImage geometry and access", "[passes]") {
    PassImage img;
    img.Allocate(4, 3, PixelType::F32, {"u", "v"});
    CHECK(img.ChannelCount() == 2);
    CHECK(img.PixelBytes() == 8);
    CHECK(img.RowBytes() == 32);
    CHECK(img.ByteSize() == 96);
    CHECK(img.ChannelIndex("v") == 1);
    CHECK(img.ChannelIndex("w") == -1);
    img.Set(3, 2, 1, 7.5f);
    CHECK(img.Get(3, 2, 1) == 7.5f);
    CHECK(img.As<float>()[(2 * 4 + 3) * 2 + 1] == 7.5f);
    CHECK_THROWS(img.Allocate(0, 1, PixelType::U8, {"A"}));
    CHECK_THROWS(img.Allocate(1, 1, PixelType::U8, {}));
}

TEST_CASE("PassImage integer and half access", "[passes]") {
    PassImage u8;
    u8.Allocate(2, 1, PixelType::U8, {"A"});
    u8.Set(0, 0, 0, 300.f);   // clamps
    u8.Set(1, 0, 0, 12.4f);   // rounds
    CHECK(u8.Get(0, 0, 0) == 255.f);
    CHECK(u8.Get(1, 0, 0) == 12.f);

    PassImage u16;
    u16.Allocate(1, 1, PixelType::U16, {"A"});
    u16.Set(0, 0, 0, 65535.f);
    CHECK(u16.Get(0, 0, 0) == 65535.f);

    PassImage f16;
    f16.Allocate(1, 1, PixelType::F16, {"Z"});
    f16.Set(0, 0, 0, 0.25f);
    CHECK(f16.Get(0, 0, 0) == 0.25f);
}

TEST_CASE("PassImage ConvertTo is value preserving where possible", "[passes]") {
    PassImage u16;
    u16.Allocate(3, 2, PixelType::U16, {"R", "G", "B"});
    for (uint32_t y = 0; y < 2; ++y)
        for (uint32_t x = 0; x < 3; ++x)
            for (size_t c = 0; c < 3; ++c) u16.Set(x, y, c, static_cast<float>(1000 * x + 100 * y + c));
    const PassImage f32 = u16.ConvertTo(PixelType::F32);
    CHECK(f32.type == PixelType::F32);
    CHECK(f32.channels == u16.channels);
    CHECK(PassImage::MaxAbsDiff(f32, u16.ConvertTo(PixelType::F32)) == 0.0);
    const PassImage back = f32.ConvertTo(PixelType::U16);
    CHECK(back.data == u16.data);
    // u8 -> f16 -> u8 is exact (all integers up to 2048 are representable in half)
    PassImage u8;
    u8.Allocate(16, 16, PixelType::U8, {"A"});
    for (size_t i = 0; i < 256; ++i) u8.data[i] = static_cast<uint8_t>(i);
    CHECK(u8.ConvertTo(PixelType::F16).ConvertTo(PixelType::U8).data == u8.data);
}

TEST_CASE("Pass kinds have canonical specs", "[passes]") {
    CHECK(ParsePassKind("depth_raw") == PassKind::DepthRaw);
    CHECK(ParsePassKind("nope") == std::nullopt);
    CHECK(ToString(PassKind::MvDlss) == "mv_dlss");
    CHECK(Spec(PassKind::DepthRaw).channels == std::vector<std::string>{"Z"});
    CHECK(Spec(PassKind::MvRaw).channels == std::vector<std::string>{"u", "v"});
    CHECK(Spec(PassKind::ColorSource).isColor);
    CHECK(Spec(PassKind::DepthDlss).convention == Convention::Dlss);
    CHECK(Spec(PassKind::Mask).type == PixelType::U8);
    CHECK(AllPassKinds().size() == 10);
    const PassImage d = MakePassImage(PassKind::MvRaw, 8, 4);
    CHECK(d.type == PixelType::F32);
    CHECK(d.channels.size() == 2);
    CHECK(ParseConvention("dlss") == Convention::Dlss);
    CHECK(ParsePixelType("half") == PixelType::F16);
}
