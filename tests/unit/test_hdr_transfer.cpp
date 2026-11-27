// TASK-0080 (HDR step TC-1): the HDR -> SDR transfer against absolute numbers (Output/hdr-design/tc-plan.md §6, recomputed in double by
// Output/hdr-design/tc-checks/), not against each other: PQ and HLG light, the BT.2390 EETF, the 2020 -> 709 matrix, BT.1886, the grey
// table the scene-cut thumbnails of an HDR source read through.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "convert/ColorConvert.h"
#include "convert/HdrTransfer.h"
#include "convert/SceneCut.h"

using namespace dlssvid;

namespace {
double Code10(double signal) { return 64.0 + 876.0 * signal; }
double Signal10(double code) { return (code - 64.0) / 876.0; }

CpuFrame Luma16(uint32_t w, uint32_t h, int depth, const std::vector<uint16_t>& rowPattern) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, PixelFormat::Yuv420p16, static_cast<uint8_t>(depth)});
    uint16_t* y = f.Plane16(0);
    for (uint32_t r = 0; r < h; ++r)
        for (uint32_t c = 0; c < w; ++c) y[static_cast<size_t>(r) * w + c] = rowPattern[(static_cast<size_t>(r) * w + c) % rowPattern.size()];
    uint16_t* u = f.Plane16(1);
    const uint16_t mid = static_cast<uint16_t>(1u << (depth - 1));
    for (size_t i = 0; i < f.desc.PlaneSize(1) / 2 * 2; ++i) u[i] = mid;  // both chroma planes follow the first
    return f;
}
}  // namespace

TEST_CASE("PQ EOTF and its inverse hit codes 450 509 573 723 855 940", "[unit][hdr][transfer]") {
    struct Row {
        double nits, signal, code;
    };
    for (const Row r : {Row{50, 0.440282, 450}, Row{100, 0.508078, 509}, Row{203, 0.580689, 573}, Row{1000, 0.751827, 723}, Row{4000, 0.902572, 855},
                        Row{10000, 1.0, 940}}) {
        INFO(r.nits << " nits");
        CHECK(std::fabs(hdr::NitsToPq(r.nits) - r.signal) < 1e-6);
        CHECK(std::lround(Code10(hdr::NitsToPq(r.nits))) == static_cast<long>(r.code));
        CHECK(std::fabs(Code10(hdr::NitsToPq(hdr::PqToNits(Signal10(r.code)))) - r.code) < 0.01);
    }
    CHECK(std::fabs(hdr::PqToNits(Signal10(573)) - 203.70) < 0.01);  // the code of 203 nits is 203.70 nits written
    CHECK(hdr::PqToNits(0.0) == 0.0);
    CHECK(hdr::NitsToPq(-5.0) == hdr::NitsToPq(0.0));
}

TEST_CASE("HLG inverse OETF and OOTF at 1000 nits give 203.15 and 50.70 nits", "[unit][hdr][transfer]") {
    auto display = [](double signal) {
        const double e = hdr::HlgToScene(signal);
        return hdr::HlgOotfGain(e) * e;  // a grey: Ys = E
    };
    CHECK(std::fabs(display(Signal10(721)) / 203.15 - 1.0) < 1e-3);
    CHECK(std::fabs(display(0.75) - 203.152) < 0.01);
    CHECK(std::fabs(display(Signal10(502)) / 50.70 - 1.0) < 1e-3);
    CHECK(std::fabs(display(1.0) - 1000.0) < 0.005);  // no light above 1000 nits: HLG needs no «above 1000» note
    CHECK(std::fabs(hdr::HlgToScene(0.5) - 1.0 / 12.0) < 1e-12);  // the two branches meet
    CHECK(std::fabs(hdr::HlgToScene(0.5 + 1e-9) - 1.0 / 12.0) < 1e-8);
    CHECK(hdr::HlgToScene(0.0) == 0.0);
    CHECK(std::fabs(hdr::HlgOotfGain(1.0) - 1000.0) < 1e-9);
}

TEST_CASE("BT.2390 EETF peak 1000 white 203: knee 87.84 nits, monotone, the table", "[unit][hdr][transfer]") {
    CHECK(std::fabs(hdr::EetfKs() - 0.658555) < 1e-6);
    CHECK(std::fabs(hdr::EetfKneeCode() - 0.495120) < 1e-6);
    CHECK(std::fabs(hdr::PqToNits(hdr::EetfKneeCode()) - 87.84) < 0.01);
    struct Row {
        double in, out, signal;
    };
    for (const Row r : {Row{50, 50.00, 0.55776}, Row{100, 99.36, 0.74253}, Row{203, 159.04, 0.90332}, Row{300, 182.49, 0.95659}, Row{500, 198.86, 0.99144},
                        Row{700, 202.43, 0.99882}, Row{1000, 203.0, 1.0}, Row{4000, 203.0, 1.0}, Row{10000, 203.0, 1.0}}) {
        INFO(r.in << " nits");
        const double out = hdr::PqToNits(hdr::Eetf2390(hdr::NitsToPq(r.in)));
        CHECK(std::fabs(out - r.out) < 0.02);
        CHECK(std::fabs(hdr::Bt1886Inverse(out / hdr::kWhiteNits) - r.signal) < 1e-5);
    }
    // below the knee the identity, above it monotone up to the display's peak
    CHECK(hdr::Eetf2390(0.3) == 0.3);
    CHECK(hdr::Eetf2390(hdr::EetfKneeCode() * 0.999) == hdr::EetfKneeCode() * 0.999);
    double prev = -1.0;
    bool monotone = true;
    for (int i = 0; i <= 100000; ++i) {
        const double v = hdr::Eetf2390(i / 100000.0);
        monotone = monotone && v >= prev;
        prev = v;
    }
    CHECK(monotone);
    CHECK(std::fabs(hdr::PqToNits(hdr::Eetf2390(1.0)) - 203.0) < 1e-9);
    // BT.1886 inverse
    CHECK(hdr::Bt1886Inverse(1.0) == 1.0);
    CHECK(hdr::Bt1886Inverse(-0.1) == 0.0);
    CHECK(std::fabs(hdr::Bt1886Inverse(50.0 / 203.0) - 0.55776) < 1e-5);  // sRGB's curve would give 0.53344
}

TEST_CASE("BT.2020 to BT.709 matrix from the primaries round-trips BT.709 red", "[unit][hdr][transfer]") {
    const double expected[3][3] = {{1.660491, -0.587641, -0.072850}, {-0.124550, 1.132900, -0.008349}, {-0.018151, -0.100579, 1.118730}};
    for (int j = 0; j < 3; ++j) {
        double in[3] = {0, 0, 0}, out[3];
        in[j] = 1.0;
        hdr::Bt2020To709(in, out);
        for (int i = 0; i < 3; ++i) CHECK(std::fabs(out[i] - expected[i][j]) < 1e-6);
    }
    const double white[3] = {1, 1, 1};
    double w[3];
    hdr::Bt2020To709(white, w);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(w[i] - 1.0) < 1e-12);  // rows sum to 1: D65 both
    const double red709in2020[3] = {0.627404, 0.069097, 0.016391};
    double red[3];
    hdr::Bt2020To709(red709in2020, red);
    CHECK(std::fabs(red[0] - 1.0) < 1e-5);
    CHECK(std::fabs(red[1]) < 1e-5);
    CHECK(std::fabs(red[2]) < 1e-5);
}

TEST_CASE("RgbToYuv refuses a ColorInfo that converts", "[unit][hdr][transfer][color]") {
    PassImage rgb = MakePassImage(PassKind::ColorSource, 4, 4, PixelType::F32);
    ColorInfo ci{ColorMatrix::Bt2020, ColorRange::Limited};
    CpuFrame out;
    CHECK_NOTHROW(RgbToYuv420p(rgb, ci, out));
    ci.convert = Convert::HdrToSdr1;
    ci.curve = Curve::Pq;
    ci.gamut = Gamut::Bt2020;
    CHECK_THROWS(RgbToYuv420p(rgb, ci, out));
    CHECK_THROWS(RgbToYuv(rgb, ci, FrameDesc{4, 4, PixelFormat::Yuv420p16, 10}, out));
}

TEST_CASE("Grey table: PQ 573 is 0.90390, HLG 721 is 0.90344, 12 bits their own", "[unit][hdr][transfer]") {
    const std::vector<float>& pq = GreyTable(Curve::Pq, 10, ColorRange::Limited);
    REQUIRE(pq.size() == 1024);
    CHECK(std::fabs(pq[573] - 0.90390) < 1e-4);
    CHECK(std::lround(Code10(pq[573])) == 856);
    CHECK(std::fabs(pq[940] - 1.0) < 1e-6);
    CHECK(pq[1023] == pq[940]);  // above white: clamped
    for (int c : {0, 4, 63, 64}) CHECK(pq[c] == 0.f);  // black and below
    CHECK(pq[65] > 0.f);
    const std::vector<float>& hlg = GreyTable(Curve::Hlg, 10, ColorRange::Limited);
    CHECK(std::fabs(hlg[721] - 0.90344) < 1e-4);
    CHECK(std::lround(Code10(hlg[721])) == 855);
    // 12 bits: code 2292 is the 10-bit 573's level
    const std::vector<float>& pq12 = GreyTable(Curve::Pq, 12, ColorRange::Limited);
    REQUIRE(pq12.size() == 4096);
    CHECK(std::fabs(pq12[2292] - pq[573]) < 1e-6);
    // full range: 1023 is white, 0 black, codes by 1023 steps
    const std::vector<float>& full = GreyTable(Curve::Pq, 10, ColorRange::Full);
    CHECK(full[0] == 0.f);
    CHECK(std::fabs(full[1023] - 1.0) < 1e-6);
    CHECK(std::fabs(full[594] - static_cast<float>(hdr::GreyToSdr(Curve::Pq, 594.0 / 1023.0))) < 1e-7);
    // 8 bits
    const std::vector<float>& pq8 = GreyTable(Curve::Pq, 8, ColorRange::Limited);
    REQUIRE(pq8.size() == 256);
    CHECK(std::fabs(pq8[235] - 1.0) < 1e-6);
    CHECK(pq8[16] == 0.f);
    // one table per (curve, depth, range): the same address each time
    CHECK(&GreyTable(Curve::Pq, 10, ColorRange::Limited) == &pq);
    CHECK(&GreyTable(Curve::Hlg, 10, ColorRange::Limited) != &pq);
    CHECK_THROWS(GreyTable(Curve::Sdr, 10, ColorRange::Limited));
    CHECK_THROWS(hdr::GreyToSdr(Curve::Sdr, 0.5));
}

TEST_CASE("MakeLumaThumb maps samples through the grey table before the box mean", "[unit][hdr][transfer][scene_cut]") {
    ColorInfo pq{ColorMatrix::Bt2020, ColorRange::Limited};
    pq.convert = Convert::HdrToSdr1;
    pq.curve = Curve::Pq;
    pq.gamut = Gamut::Bt2020;
    // a flat 573 frame: its SDR grey
    const LumaThumb flat = MakeLumaThumb(Luma16(64, 36, 10, {573}), pq);
    REQUIRE(!flat.Empty());
    CHECK(std::fabs(flat.y[0] - 0.90390) < 1e-4);
    // 1280x720: a box of 2 (SceneCut.cpp) - blocks of two 940 and two 64 are the mean of the SDR signals, 0.5, not the signal of the
    // mean code (0.71962)
    std::vector<uint16_t> pattern(1280 * 2);
    for (size_t i = 0; i < 1280; ++i) pattern[i] = 940;
    for (size_t i = 1280; i < 2560; ++i) pattern[i] = 64;
    const LumaThumb box = MakeLumaThumb(Luma16(1280, 720, 10, pattern), pq);
    REQUIRE(box.width == 640);
    for (const float v : box.y) CHECK(std::fabs(v - 0.5f) < 1e-6f);
    // without the transfer, as before: the codes' normalised mean
    const LumaThumb raw = MakeLumaThumb(Luma16(64, 36, 10, {573}), ColorInfo{ColorMatrix::Bt2020, ColorRange::Limited});
    CHECK(std::fabs(raw.y[0] - (573.0 - 64.0) / 876.0) < 1e-6);
}
