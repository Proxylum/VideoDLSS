// TASK-0080 (HDR step TC-1): the HDR -> SDR transfer against absolute numbers (Output/hdr-design/tc-plan.md §6, §15.7, recomputed in
// double by Output/hdr-design/tc-checks/), not against each other: PQ and HLG light, the BT.2390 EETF, the 2020 -> 709 matrix, BT.1886,
// the grey table the scene-cut thumbnails of an HDR source read through; the float tables a frame goes through against that double path,
// and the AVX2 row against the portable one. The gamut step's own tests: test_hdr_gamut.cpp.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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
    // a converting reading of an SDR curve has no transfer: refused before the bands, on the caller's thread
    CpuFrame frame;
    frame.Allocate(FrameDesc{8, 300, PixelFormat::Yuv420p16, 10});
    ci.curve = Curve::Sdr;
    CHECK_THROWS(YuvToRgb(frame, ci, PixelType::F32, 4));
    CHECK_THROWS(YuvToRgba16f(frame, ci, 4));
    CHECK_THROWS(HdrTransfer::For(Curve::Sdr, Gamut::Bt709));
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
    // the table is the source's: its curve, bit depth and range
    ColorInfo hlg = pq;
    hlg.curve = Curve::Hlg;
    CHECK(std::fabs(MakeLumaThumb(Luma16(64, 36, 10, {721}), hlg).y[0] - 0.90344) < 1e-4);
    CHECK(std::fabs(MakeLumaThumb(Luma16(64, 36, 12, {2292}), pq).y[0] - 0.90390) < 1e-4);  // 12 bits: the 10-bit 573's level
    CHECK(std::fabs(MakeLumaThumb(Luma16(64, 36, 10, {65535}), pq).y[0] - 1.0) < 1e-6);   // a sample above the depth's top reads as white
    // ... also inside a box with black, where the answer is no bound of the final clamp: two 65535 and two 64 are 0.5
    std::vector<uint16_t> above(1280 * 2);
    for (size_t i = 0; i < 1280; ++i) above[i] = 65535;
    for (size_t i = 1280; i < 2560; ++i) above[i] = 64;
    for (const float v : MakeLumaThumb(Luma16(1280, 720, 10, above), pq).y) CHECK(std::fabs(v - 0.5f) < 1e-6f);
    // each box is its own 2x2 of the frame: the top half white and the bottom black, then the left half white and the right black
    std::vector<uint16_t> halves(size_t{1280} * 720, 64);
    std::fill_n(halves.begin(), size_t{1280} * 360, uint16_t{940});
    const LumaThumb topBottom = MakeLumaThumb(Luma16(1280, 720, 10, halves), pq);
    REQUIRE(topBottom.height == 360);
    bool rows = true;
    for (uint32_t ty = 0; ty < 360; ++ty)
        for (uint32_t tx = 0; tx < 640; ++tx) rows = rows && topBottom.y[size_t{ty} * 640 + tx] == (ty < 180 ? 1.f : 0.f);
    CHECK(rows);
    std::vector<uint16_t> leftRight(1280, 64);
    std::fill_n(leftRight.begin(), 640, uint16_t{940});
    const LumaThumb sides = MakeLumaThumb(Luma16(1280, 720, 10, leftRight), pq);
    bool cols = true;
    for (uint32_t ty = 0; ty < 360; ++ty)
        for (uint32_t tx = 0; tx < 640; ++tx) cols = cols && sides.y[size_t{ty} * 640 + tx] == (tx < 320 ? 1.f : 0.f);
    CHECK(cols);
    ColorInfo full = pq;
    full.range = ColorRange::Full;
    CHECK(std::fabs(MakeLumaThumb(Luma16(64, 36, 10, {594}), full).y[0] - static_cast<float>(hdr::GreyToSdr(Curve::Pq, 594.0 / 1023.0))) < 1e-6);
    CpuFrame f8;
    f8.Allocate(FrameDesc{64, 36, PixelFormat::Yuv420p, 8});
    std::fill(f8.data.begin(), f8.data.end(), uint8_t{128});
    std::fill_n(f8.Plane(0), size_t{64} * 36, uint8_t{235});
    CHECK(std::fabs(MakeLumaThumb(f8, pq).y[0] - 1.0) < 1e-6);  // 8 bits: 235 is white
    std::fill_n(f8.Plane(0), size_t{64} * 36, uint8_t{128});
    CHECK(std::fabs(MakeLumaThumb(f8, pq).y[0] - static_cast<float>(hdr::GreyToSdr(Curve::Pq, (128.0 - 16.0) / 219.0))) < 1e-6);
    // 8 bits read by place too: 1280x720, the top half 235 and the bottom 16, then the left half 235 and the right 16
    CpuFrame big8;
    big8.Allocate(FrameDesc{1280, 720, PixelFormat::Yuv420p, 8});
    std::fill(big8.data.begin(), big8.data.end(), uint8_t{128});
    std::fill_n(big8.Plane(0), size_t{1280} * 720, uint8_t{16});
    std::fill_n(big8.Plane(0), size_t{1280} * 360, uint8_t{235});
    const LumaThumb top8 = MakeLumaThumb(big8, pq);
    bool rows8 = top8.height == 360;
    for (uint32_t ty = 0; rows8 && ty < 360; ++ty)
        for (uint32_t tx = 0; tx < 640; ++tx) rows8 = rows8 && top8.y[size_t{ty} * 640 + tx] == (ty < 180 ? 1.f : 0.f);
    CHECK(rows8);
    for (uint32_t r = 0; r < 720; ++r) {
        std::fill_n(big8.Plane(0) + size_t{r} * 1280, 640, uint8_t{235});
        std::fill_n(big8.Plane(0) + size_t{r} * 1280 + 640, 640, uint8_t{16});
    }
    const LumaThumb side8 = MakeLumaThumb(big8, pq);
    bool cols8 = true;
    for (uint32_t ty = 0; ty < 360; ++ty)
        for (uint32_t tx = 0; tx < 640; ++tx) cols8 = cols8 && side8.y[size_t{ty} * 640 + tx] == (tx < 320 ? 1.f : 0.f);
    CHECK(cols8);
}

namespace {

ColorInfo Hdr(Curve curve, Gamut gamut, ColorRange range = ColorRange::Limited) {
    ColorInfo ci{ColorMatrix::Bt2020, range};
    ci.convert = Convert::HdrToSdr1;
    ci.curve = curve;
    ci.gamut = gamut;
    return ci;
}

// A frame of random codes (every plane), `depth` 8 for the 8-bit formats.
CpuFrame RandomFrame(PixelFormat format, int depth, uint32_t w, uint32_t h, uint32_t seed) {
    CpuFrame f;
    f.Allocate(FrameDesc{w, h, format, static_cast<uint8_t>(depth)});
    std::mt19937 rng(seed);
    const uint32_t top = (1u << depth) - 1u;
    for (size_t p = 0; p < 3; ++p) {
        const size_t n = f.desc.PlaneCols(p) * f.desc.PlaneHeight(p);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t v = rng() % (top + 1u);
            if (f.desc.HighBitDepth()) f.Plane16(p)[i] = static_cast<uint16_t>(v);
            else f.Plane(p)[i] = static_cast<uint8_t>(v);
        }
    }
    return f;
}

std::array<double, 9> Inv3(const std::array<double, 9>& m) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return {(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det,
            (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
}

void Ap3(const std::array<double, 9>& m, const double x[3], double y[3]) {
    for (int i = 0; i < 3; ++i) y[i] = m[3 * i] * x[0] + m[3 * i + 1] * x[1] + m[3 * i + 2] * x[2];
}

double DeItp709(const double a709[3], const double b709[3]) {  // BT.2124 of two linear BT.709 colours
    static const std::array<double, 9> inv = Inv3(hdr::Bt2020To709Matrix());
    double a[3], b[3], ia[3], ib[3];
    Ap3(inv, a709, a);
    Ap3(inv, b709, b);
    hdr::Ictcp(a, ia);
    hdr::Ictcp(b, ib);
    return 720.0 * std::sqrt((ia[0] - ib[0]) * (ia[0] - ib[0]) + 0.25 * (ia[1] - ib[1]) * (ia[1] - ib[1]) + (ia[2] - ib[2]) * (ia[2] - ib[2]));
}

constexpr double kCode = 876.0;  // one 10-bit limited code of V'

}  // namespace

TEST_CASE("HDR transfer LUT and the gamut step stay within the bound of the double path, widened by the reference's own hue slope",
          "[unit][hdr][transfer]") {
    std::mt19937 rng(20261127);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const hdr::GamutTables& tables = hdr::Sdr1GamutTables();
    const HdrGamut& gamut = HdrGamut::Get();

    // (a) the chain: BT.709 sources through the whole LUT path, BT.2020 sources up to the gamut step's input
    std::vector<std::array<double, 3>> codes;
    for (int a = 0; a <= 40; ++a)
        for (int b = 0; b <= 40; ++b)
            for (int c = 0; c <= 40; ++c) codes.push_back({a / 40.0, b / 40.0, c / 40.0});
    for (int n = 0; n < 150000; ++n) codes.push_back({uni(rng), uni(rng), uni(rng)});
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        INFO((curve == Curve::Pq ? "PQ" : "HLG"));
        const HdrTransfer& at709 = HdrTransfer::For(curve, Gamut::Bt709);
        const HdrTransfer& at2020 = HdrTransfer::For(curve, Gamut::Bt2020);
        double worst = 0, sum = 0, worstLin = 0;
        for (const auto& c : codes) {
            const float f[3] = {static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2])};
            const double d[3] = {f[0], f[1], f[2]};
            float o[3], l[3];
            double ref[3], lin[3];
            at709.Apply(f[0], f[1], f[2], o);
            hdr::HdrToSdr(curve, Gamut::Bt709, d, ref);
            at2020.Linear(f[0], f[1], f[2], l);
            hdr::LinearOfCodes(curve, Gamut::Bt2020, d, lin);
            for (int k = 0; k < 3; ++k) {
                const double e = std::abs(o[k] - ref[k]) * kCode;
                worst = std::max(worst, e);
                sum += e;
                worstLin = std::max(worstLin, std::abs(hdr::Bt1886Inverse(l[k]) - hdr::Bt1886Inverse(lin[k])) * kCode);
            }
        }
        std::printf("U6 (a) %s chain: BT.709 path %.3f code (mean %.5f), BT.2020 linear %.3f code\n", curve == Curve::Pq ? "PQ" : "HLG", worst,
                    sum / (3.0 * codes.size()), worstLin);
        CHECK(worst <= 0.5);
        CHECK(sum / (3.0 * codes.size()) <= 0.01);
        CHECK(worstLin <= 0.5);
    }

    // (b) the gamut step on the same float input: |dV'| <= 1 code + 2 S where the reference channel >= 1e-4, |d lin| <= 1e-5 + 2 S_lin
    // below; S is the reference's own move when the input's I moves 5e-7 and its hue 0.001 deg (the float step's input error is 2.3e-7
    // and 6.4e-4 deg); outside the steep pixels (S <= 0.25 code) also dE_ITP <= 0.15
    struct Sample {
        float l[3];
        const HdrTransfer* tr;  // codes of the sample, or nullptr: a linear colour straight into the step
        float c[3];
    };
    std::vector<Sample> set;
    const std::array<double, 9> from709 = Inv3(hdr::Bt2020To709Matrix());
    const auto linear = [&](const double l[3]) {
        Sample s{};
        for (int k = 0; k < 3; ++k) s.l[k] = static_cast<float>(std::clamp(l[k], 0.0, 1.0));
        set.push_back(s);
    };
    for (int n = 0; n < 150000; ++n) {  // random BT.2020 colours, uniform in V'
        const double l[3] = {std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4)};
        linear(l);
    }
    for (int n = 0; n < 20000; ++n) {  // the BT.2020 cube's faces
        double l[3] = {std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4)};
        l[n % 3] = (n % 6) >= 3 ? 1.0 : 0.0;
        linear(l);
    }
    for (int n = 0; n < 50000; ++n) {  // BT.709 content in a BT.2020 container
        const double v[3] = {std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4), std::pow(uni(rng), 2.4)};
        double l[3];
        Ap3(from709, v, l);
        linear(l);
    }
    // a few ulp below white: where L' and M' round to Iw, u = 1 and B709 = 0 - the step keeps clamp(T l); the rest sit a few ulp of I
    // below Iw, with a vanishing B709 (the knee or the curve)
    for (int k = 1; k <= 40; ++k) {
        const double e = 1.0 - k * 5.9604644775390625e-8;  // k * 2^-24
        const double a[3] = {1.0, e, 1.0}, b[3] = {e, 1.0, 1.0}, c[3] = {e, e, 1.0}, d[3] = {1.0, e, e}, f[3] = {e, 1.0, e};
        linear(a), linear(b), linear(c), linear(d), linear(f);
    }
    for (int n = 0; n < 20000; ++n) {  // dark (the encoder's branch below 2^-32) and near white
        const double s = std::pow(10.0, -12.0 + 9.0 * uni(rng));
        const double dark[3] = {s * uni(rng), s * uni(rng), s * uni(rng)}, white[3] = {0.9 + 0.1 * uni(rng), 0.9 + 0.1 * uni(rng), 0.9 + 0.1 * uni(rng)};
        linear(dark);
        linear(white);
    }
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {  // code lattices through the chain: the step's input is Linear(codes)
        const HdrTransfer& tr = HdrTransfer::For(curve, Gamut::Bt2020);
        for (int a = 0; a <= 40; ++a)
            for (int b = 0; b <= 40; ++b)
                for (int c = 0; c <= 40; ++c) {
                    Sample s{};
                    s.tr = &tr;
                    s.c[0] = a / 40.f, s.c[1] = b / 40.f, s.c[2] = c / 40.f;
                    tr.Linear(s.c[0], s.c[1], s.c[2], s.l);
                    set.push_back(s);
                }
    }
    constexpr double kDI = 5e-7, kDH = 0.001 * 3.141592653589793 / 180.0;
    long long steep = 0, flagged = 0, broken = 0;
    double worstRatio = 0, worstFlat = 0, sum = 0, worstDeFlat = 0, worstHue = 0;
    for (const Sample& s : set) {
        float out[3];
        if (s.tr) {
            s.tr->Apply(s.c[0], s.c[1], s.c[2], out);
        } else {
            float m[3] = {s.l[0], s.l[1], s.l[2]};
            gamut.Map(m);
            for (int k = 0; k < 3; ++k) out[k] = static_cast<float>(hdr::Bt1886Inverse(m[k]));
        }
        const double l[3] = {s.l[0], s.l[1], s.l[2]};
        double ref[3], refV[3], outLin[3];
        hdr::GamutMap709(l, ref);
        for (int k = 0; k < 3; ++k) refV[k] = hdr::Bt1886Inverse(ref[k]), outLin[k] = std::pow(static_cast<double>(out[k]), 2.4);
        double dv = 0, dl = 0, dvLit = 0;  // dvLit: channels whose reference is >= 1e-4 (below it the rule takes the linear error)
        for (int k = 0; k < 3; ++k) {
            dv = std::max(dv, std::abs(out[k] - refV[k]) * kCode);
            dl = std::max(dl, std::abs(outLin[k] - ref[k]));
            if (ref[k] >= 1e-4) dvLit = std::max(dvLit, std::abs(out[k] - refV[k]) * kCode);
            sum += std::abs(out[k] - refV[k]) * kCode;
        }
        const double de = DeItp709(outLin, ref);
        double sv[3] = {0, 0, 0}, sl[3] = {0, 0, 0};
        if (dv > 0.5 || dl > 5e-6 || de > 0.05) {
            ++flagged;
            double itp[3], base[3];
            hdr::Ictcp(l, itp);
            const double C = std::hypot(itp[1], itp[2]), h = std::atan2(itp[2], itp[1]);
            hdr::GamutMapIctcp(itp, base, tables);
            for (double di : {-kDI, 0.0, kDI})
                for (double dh : {-kDH, 0.0, kDH}) {
                    if (di == 0.0 && dh == 0.0) continue;
                    const double moved[3] = {itp[0] + di, C * std::cos(h + dh), C * std::sin(h + dh)};
                    double o[3];
                    hdr::GamutMapIctcp(moved, o, tables);
                    for (int k = 0; k < 3; ++k) {
                        sv[k] = std::max(sv[k], std::abs(hdr::Bt1886Inverse(o[k]) - hdr::Bt1886Inverse(base[k])) * kCode);
                        sl[k] = std::max(sl[k], std::abs(o[k] - base[k]));
                    }
                }
        }
        const double S = std::max(sv[0], std::max(sv[1], sv[2]));
        if (S > 0.25) ++steep;
        else {
            worstDeFlat = std::max(worstDeFlat, de);
            worstFlat = std::max(worstFlat, dvLit);
        }
        for (int k = 0; k < 3; ++k) {
            double ratio;
            if (ref[k] >= 1e-4) ratio = std::abs(out[k] - refV[k]) * kCode / (1.0 + 2.0 * sv[k]);
            else ratio = std::abs(outLin[k] - ref[k]) / (1e-5 + 2.0 * sl[k]);
            worstRatio = std::max(worstRatio, ratio);
            if (ratio > 1.0) {
                ++broken;
                if (broken <= 5)
                    FAIL_CHECK("channel " << k << " of (" << l[0] << ", " << l[1] << ", " << l[2] << "): " << out[k] << " vs " << refV[k] << ", S "
                                          << sv[k]);
            }
        }
        // the hue of a compressed colour stays: within 0.5 deg of the input where C > 0.02
        double itp[3], back[3], itpOut[3];
        hdr::Ictcp(l, itp);
        const double C = std::hypot(itp[1], itp[2]);
        if (C > 0.02 && C > hdr::kGamutKnee * tables.b709(itp[0], hdr::PseudoHue(itp[1], itp[2]))) {
            Ap3(from709, outLin, back);
            hdr::Ictcp(back, itpOut);
            if (std::hypot(itpOut[1], itpOut[2]) > 0.02) {
                double dh = (std::atan2(itpOut[2], itpOut[1]) - std::atan2(itp[2], itp[1])) * 180.0 / 3.141592653589793;
                dh = std::fmod(std::fmod(dh + 180.0, 360.0) + 360.0, 360.0) - 180.0;
                worstHue = std::max(worstHue, std::abs(dh));
            }
        }
    }
    const double mean = sum / (3.0 * set.size());
    std::printf("U6 (b) gamut step, %zu colours: worst ratio to the bound %.3f; outside the steep pixels %.3f code, dE_ITP %.3f; %lld steep, "
                "%lld flagged; mean %.4f code; hue %.3f deg\n",
                set.size(), worstRatio, worstFlat, worstDeFlat, steep, flagged, mean, worstHue);
    CHECK(broken == 0);
    CHECK(worstDeFlat <= 0.15);
    CHECK(mean <= 0.02);
    CHECK(worstHue <= 0.5);

    // (c) the whole path against hdr::HdrToSdr at BT.2020: printed only - the chain's error times the reference's slope
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        const HdrTransfer& tr = HdrTransfer::For(curve, Gamut::Bt2020);
        double worst = 0;
        long long over = 0;
        for (size_t n = 0; n < codes.size(); n += 3) {
            const float f[3] = {static_cast<float>(codes[n][0]), static_cast<float>(codes[n][1]), static_cast<float>(codes[n][2])};
            const double d[3] = {f[0], f[1], f[2]};
            float o[3];
            double ref[3];
            tr.Apply(f[0], f[1], f[2], o);
            hdr::HdrToSdr(curve, Gamut::Bt2020, d, ref);
            double e = 0;
            for (int k = 0; k < 3; ++k) e = std::max(e, std::abs(o[k] - ref[k]) * kCode);
            worst = std::max(worst, e);
            over += e > 1.0 ? 1 : 0;
        }
        std::printf("U6 (c) %s whole path at BT.2020: %.2f code, %lld pixels above 1 code\n", curve == Curve::Pq ? "PQ" : "HLG", worst, over);
    }
}

TEST_CASE("YuvToRgb with HdrToSdr1 is bit-identical on 1 and 8 threads and to YuvToRgba16f; greys pass the gamut step bit for bit",
          "[unit][hdr][transfer]") {
    const CpuFrame frame = RandomFrame(PixelFormat::Yuv420p16, 10, 517, 300, 7);
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        const ColorInfo ci = Hdr(curve, Gamut::Bt2020);
        for (PixelType type : {PixelType::F32, PixelType::F16, PixelType::U8}) {
            const PassImage one = YuvToRgb(frame, ci, type, 1), eight = YuvToRgb(frame, ci, type, 8);
            CHECK(one.data == eight.data);
        }
        const PassImage f16 = YuvToRgb(frame, ci, PixelType::F16, 8), rgba = YuvToRgba16f(frame, ci, 8);
        bool same = true;
        const uint16_t* a = f16.As<uint16_t>();
        const uint16_t* b = rgba.As<uint16_t>();
        for (size_t i = 0; i < size_t{517} * 300; ++i)
            same = same && a[3 * i] == b[4 * i] && a[3 * i + 1] == b[4 * i + 1] && a[3 * i + 2] == b[4 * i + 2] && b[4 * i + 3] == 0x3C00;
        CHECK(same);
        // and each pixel is the transfer of ForEachRgb's R'G'B': HdrTransfer::Apply, bit for bit
        const PassImage f32 = YuvToRgb(frame, ci, PixelType::F32, 1);
        const HdrTransfer& tr = HdrTransfer::For(curve, Gamut::Bt2020);
        const float kr = 0.2627f, kbl = 0.0593f, kg = 1.f - kr - kbl;
        const float crR = 2.f * (1.f - kr), cbB = 2.f * (1.f - kbl), cbG = -2.f * kbl * (1.f - kbl) / kg, crG = -2.f * kr * (1.f - kr) / kg;
        bool applied = true;
        const uint16_t* Y = frame.Plane16(0);
        const uint16_t* U = frame.Plane16(1);
        const uint16_t* V = frame.Plane16(2);
        const size_t cw = frame.desc.PlaneCols(1);
        for (uint32_t y = 0; y < 300; y += 7)
            for (uint32_t x = 0; x < 517; ++x) {
                const float yy = (static_cast<float>(Y[static_cast<size_t>(y) * 517 + x]) - 64.f) * (1.f / 876.f);
                const float cb = (static_cast<float>(U[(y >> 1) * cw + (x >> 1)]) - 512.f) * (1.f / 896.f);
                const float cr = (static_cast<float>(V[(y >> 1) * cw + (x >> 1)]) - 512.f) * (1.f / 896.f);
                float o[3];
                tr.Apply(std::clamp(yy + crR * cr, 0.f, 1.f), std::clamp(yy + cbG * cb + crG * cr, 0.f, 1.f), std::clamp(yy + cbB * cb, 0.f, 1.f), o);
                const float* got = f32.As<float>() + (static_cast<size_t>(y) * 517 + x) * 3;
                applied = applied && std::memcmp(o, got, sizeof o) == 0;
            }
        CHECK(applied);
    }

    // the wiring of the branch on every layout, gamut and range (the chroma row of 4:2:2, the transfer of the source's gamut): the frozen
    // SDR path gives the R'G'B' the branch sees, the result is their HdrTransfer::Apply bit for bit
    struct Shape {
        PixelFormat format;
        int depth;
    };
    uint32_t seed = 70;
    for (const Shape s : {Shape{PixelFormat::Yuv420p16, 10}, Shape{PixelFormat::Yuv420p, 8}, Shape{PixelFormat::Yuv422p16, 12},
                          Shape{PixelFormat::Yuv422p, 8}, Shape{PixelFormat::Yuv444p16, 10}, Shape{PixelFormat::Yuv444p, 8}})
        for (Curve curve : {Curve::Pq, Curve::Hlg})
            for (Gamut gamut : {Gamut::Bt709, Gamut::Bt2020})
                for (ColorRange range : {ColorRange::Limited, ColorRange::Full}) {
                    const CpuFrame f = RandomFrame(s.format, s.depth, 37, 12, seed++);
                    const PassImage sdr = YuvToRgb(f, ColorInfo{ColorMatrix::Bt2020, range}, PixelType::F32, 1);
                    const PassImage conv = YuvToRgb(f, Hdr(curve, gamut, range), PixelType::F32, 1);
                    const HdrTransfer& tr = HdrTransfer::For(curve, gamut);
                    bool wired = true;
                    const float* in = sdr.As<float>();
                    const float* got = conv.As<float>();
                    for (size_t i = 0; i < size_t{37} * 12; ++i) {
                        float o[3];
                        tr.Apply(in[3 * i], in[3 * i + 1], in[3 * i + 2], o);
                        wired = wired && std::memcmp(o, got + 3 * i, sizeof o) == 0;
                    }
                    INFO(ToString(s.format) << (curve == Curve::Pq ? " PQ" : " HLG") << (gamut == Gamut::Bt2020 ? " 2020" : " 709")
                                            << (range == ColorRange::Full ? " full" : " limited"));
                    CHECK(wired);
                }

    // the branch B709 = 0 of the step (I at Iw: a colour one ulp below white) keeps clamp(T l), finite (tc-plan §15.2 p. 5)
    {
        const HdrGamut& gm = HdrGamut::Get();
        const hdrrow::Step& v = gm.View();
        float l[3] = {1.f, 0.99999994f, 1.f};
        float t[3];
        for (int c = 0; c < 3; ++c) t[c] = (v.t[3 * c] * l[0] + v.t[3 * c + 1] * l[1]) + v.t[3 * c + 2] * l[2];
        REQUIRE_FALSE(gm.Level1(t));
        gm.Map(l);
        for (int c = 0; c < 3; ++c) CHECK(l[c] == std::clamp(t[c], 0.f, 1.f));
    }

    // the gamut step keeps a grey bit for bit: 4097 levels, and every 10-bit code's grey after the chain
    const HdrGamut& g = HdrGamut::Get();
    bool greys = true;
    for (int i = 0; i <= 4096; ++i) {
        const float v = static_cast<float>(i) / 4096.f;
        float l[3] = {v, v, v};
        g.Map(l);
        greys = greys && l[0] == v && l[1] == v && l[2] == v;
    }
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        const HdrTransfer& tr = HdrTransfer::For(curve, Gamut::Bt2020);
        for (int code = 0; code < 1024; ++code) {
            const float c = std::clamp((code - 64.f) / 876.f, 0.f, 1.f);
            float l[3], m[3];
            tr.Linear(c, c, c, l);
            std::copy(l, l + 3, m);
            g.Map(m);
            greys = greys && m[0] == l[0] && m[1] == l[1] && m[2] == l[2];
        }
    }
    CHECK(greys);
    // PQ: a grey of a BT.2020 source is the grey of a BT.709 one bit for bit (F32 and F16); HLG weighs its Ys by its own primaries, so
    // its greys only agree with hdr::GreyToSdr within half a code
    const HdrTransfer& pq709 = HdrTransfer::For(Curve::Pq, Gamut::Bt709);
    const HdrTransfer& pq2020 = HdrTransfer::For(Curve::Pq, Gamut::Bt2020);
    const HdrTransfer& hlg2020 = HdrTransfer::For(Curve::Hlg, Gamut::Bt2020);
    bool pqSame = true;
    double hlgWorst = 0;
    for (int code = 0; code < 1024; ++code) {
        const float c = std::clamp((code - 64.f) / 876.f, 0.f, 1.f);
        float a[3], b[3], h[3];
        pq709.Apply(c, c, c, a);
        pq2020.Apply(c, c, c, b);
        pqSame = pqSame && std::memcmp(a, b, sizeof a) == 0 && a[0] == a[1] && a[1] == a[2];
        hlg2020.Apply(c, c, c, h);
        hlgWorst = std::max(hlgWorst, std::abs(h[0] - hdr::GreyToSdr(Curve::Hlg, c)) * kCode);
    }
    CHECK(pqSame);
    CHECK(hlgWorst <= 0.5);
}

TEST_CASE("The whole HDR to SDR path at BT.709 gamut: PQ red keeps its hue, HLG red is weighted by BT.709", "[unit][hdr][transfer]") {
    struct Row {
        Curve curve;
        Gamut gamut;
        double in[3], out[3];
    };
    // the step only for a BT.2020 source; ictcp_ref.py section D prints the first three rows, the HLG BT.2020 row is its
    // hdr_to_sdr([[0.75, 0, 0]], 'hlg', '2020', Bounds()) (the script does not print it)
    const Row rows[] = {
        {Curve::Pq, Gamut::Bt709, {0.58105, 0, 0}, {0.90390, 0, 0}},
        {Curve::Hlg, Gamut::Bt709, {0.75, 0, 0}, {0.84291, 0, 0}},  // Ys weights of BT.709 from the primaries (2020's: 0.85212)
        {Curve::Pq, Gamut::Bt2020, {0.58105, 0, 0}, {0.98141, 0, 0.15250}},
        {Curve::Hlg, Gamut::Bt2020, {0.75, 0, 0}, {0.92525, 0, 0.14479}},  // Ys weights of BT.2020 (709's would give 0.91525 / 0 / 0.14342)
    };
    for (const Row& r : rows) {
        double out[3];
        hdr::HdrToSdr(r.curve, r.gamut, r.in, out);
        float lut[3];
        HdrTransfer::For(r.curve, r.gamut).Apply(static_cast<float>(r.in[0]), static_cast<float>(r.in[1]), static_cast<float>(r.in[2]), lut);
        for (int k = 0; k < 3; ++k) {
            CHECK(std::abs(out[k] - r.out[k]) <= 1e-4);
            CHECK(std::abs(lut[k] - out[k]) * kCode <= 0.5);
        }
    }
    // the HLG OOTF's Ys weights are the source primaries' Y: BT.2020 (1, 0, 0) at 0.75 is 0.68109 after the EETF, 0.66355 by BT.709's
    double w2020[3], w709[3];
    hdr::LuminanceWeights(Gamut::Bt2020, w2020);
    hdr::LuminanceWeights(Gamut::Bt709, w709);
    const double want2020[3] = {0.262700, 0.677998, 0.059302}, want709[3] = {0.212639, 0.715169, 0.072192};
    for (int k = 0; k < 3; ++k) {
        CHECK(std::abs(w2020[k] - want2020[k]) <= 1e-6);
        CHECK(std::abs(w709[k] - want709[k]) <= 1e-6);
    }
    const double red[3] = {0.75, 0, 0};
    double l2020[3], l709[3];
    hdr::LinearOfCodes(Curve::Hlg, Gamut::Bt2020, red, l2020);
    hdr::LinearOfCodes(Curve::Hlg, Gamut::Bt709, red, l709);
    CHECK(std::abs(l2020[0] - 0.68109) <= 1e-4);
    CHECK(std::abs(l709[0] - 0.66355) <= 1e-4);
}

TEST_CASE("The AVX2 row of the HDR transfer gives the same bits as the portable row", "[unit][hdr][transfer][avx2]") {
    if (!hdrrow::HasAvx2()) {
        WARN("no AVX2 on this CPU: only the portable row runs");
        return;
    }
    struct Shape {
        PixelFormat format;
        int depth;
        uint32_t w, h;
    };
    const Shape shapes[] = {{PixelFormat::Yuv420p16, 10, 1027, 64}, {PixelFormat::Yuv420p, 8, 333, 40},  {PixelFormat::Yuv444p16, 10, 129, 33},
                            {PixelFormat::Yuv422p16, 12, 95, 17},   {PixelFormat::Yuv444p, 8, 7, 9},     {PixelFormat::Yuv420p16, 10, 8, 2},
                            {PixelFormat::Yuv422p, 8, 17, 5}};
    uint32_t seed = 1;
    long long pixels = 0;
    for (const Shape& s : shapes)
        for (Curve curve : {Curve::Pq, Curve::Hlg})
            for (Gamut gamut : {Gamut::Bt709, Gamut::Bt2020})
                for (ColorRange range : {ColorRange::Limited, ColorRange::Full}) {
                    const CpuFrame f = RandomFrame(s.format, s.depth, s.w, s.h, seed++);
                    const ColorInfo ci = Hdr(curve, gamut, range);
                    SetHdrKernel(HdrKernel::Portable);
                    CHECK_FALSE(HdrRowsUseAvx2());
                    const PassImage portable = YuvToRgb(f, ci, PixelType::F32, 1);
                    SetHdrKernel(HdrKernel::Auto);
                    CHECK(HdrRowsUseAvx2());
                    const PassImage avx2 = YuvToRgb(f, ci, PixelType::F32, 1);
                    INFO(ToString(s.format) << " " << s.w << "x" << s.h << (curve == Curve::Pq ? " PQ" : " HLG") << (gamut == Gamut::Bt2020 ? " 2020" : " 709"));
                    CHECK(std::memcmp(portable.data.data(), avx2.data.data(), portable.data.size()) == 0);
                    pixels += static_cast<long long>(s.w) * s.h;
                }
    // a frame of saturated and dark colours as well: every 10-bit corner and edge of the code cube, through the step's slow path
    CpuFrame edges;
    edges.Allocate(FrameDesc{1024, 64, PixelFormat::Yuv444p16, 10});
    std::mt19937 rng(99);
    for (size_t i = 0; i < size_t{1024} * 64; ++i)
        for (size_t p = 0; p < 3; ++p) {
            const uint32_t pick = rng() % 6;
            edges.Plane16(p)[i] = static_cast<uint16_t>(pick == 0 ? 64 : pick == 1 ? 940 : pick == 2 ? 512 : pick == 3 ? 0 : pick == 4 ? 1023 : rng() % 1024);
        }
    CpuFrame calm = RandomFrame(PixelFormat::Yuv420p16, 10, 1024, 64, 98);  // nearly grey: whole vectors done by level 1 and greys
    for (size_t p = 1; p < 3; ++p)
        for (size_t i = 0; i < calm.desc.PlaneCols(p) * calm.desc.PlaneHeight(p); ++i) calm.Plane16(p)[i] = static_cast<uint16_t>(510 + rng() % 5);
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        SetHdrKernel(HdrKernel::Portable);
        const PassImage portable = YuvToRgb(calm, Hdr(curve, Gamut::Bt2020), PixelType::F32, 1);
        SetHdrKernel(HdrKernel::Auto);
        const PassImage avx2 = YuvToRgb(calm, Hdr(curve, Gamut::Bt2020), PixelType::F32, 1);
        CHECK(std::memcmp(portable.data.data(), avx2.data.data(), portable.data.size()) == 0);
        pixels += 1024 * 64;
    }
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        SetHdrKernel(HdrKernel::Portable);
        const PassImage portable = YuvToRgb(edges, Hdr(curve, Gamut::Bt2020), PixelType::F32, 1);
        SetHdrKernel(HdrKernel::Auto);
        const PassImage avx2 = YuvToRgb(edges, Hdr(curve, Gamut::Bt2020), PixelType::F32, 1);
        CHECK(std::memcmp(portable.data.data(), avx2.data.data(), portable.data.size()) == 0);
        pixels += 1024 * 64;
    }
    // the AVX2 row itself, not only behind the dispatch: a 4:2:0 row against the portable arithmetic written out here
    for (Curve curve : {Curve::Pq, Curve::Hlg}) {
        const HdrTransfer& tr = HdrTransfer::For(curve, Gamut::Bt2020);
        constexpr uint32_t kW = 1024;
        std::vector<uint16_t> y(kW), u(kW / 2), v(kW / 2);
        for (auto& x : y) x = static_cast<uint16_t>(rng() % 1024);
        for (auto& x : u) x = static_cast<uint16_t>(rng() % 1024);
        for (auto& x : v) x = static_cast<uint16_t>(rng() % 1024);
        const float kr = 0.2627f, kb = 0.0593f, kg = 1.f - kr - kb;
        const hdrrow::Yuv k{64.f, 1.f / 876.f, 512.f, 1.f / 896.f, 2.f * (1.f - kr), 2.f * (1.f - kb), -2.f * kb * (1.f - kb) / kg, -2.f * kr * (1.f - kr) / kg};
        std::vector<float> r(kW), g(kW), b(kW), scratch(3 * kW);
        std::vector<int32_t> todo(kW);
        REQUIRE(hdrrow::RowAvx2(tr.View(), k, y.data(), u.data(), v.data(), 1, kW, r.data(), g.data(), b.data(), scratch.data(), todo.data()) == kW);
        bool same = true;
        for (uint32_t x = 0; x < kW; ++x) {
            const float yy = (static_cast<float>(y[x]) - k.yOff) * k.yScale;
            const float cb = (static_cast<float>(u[x >> 1]) - k.cOff) * k.cScale, cr = (static_cast<float>(v[x >> 1]) - k.cOff) * k.cScale;
            float o[3];
            tr.Apply(std::clamp(yy + k.crR * cr, 0.f, 1.f), std::clamp(yy + k.cbG * cb + k.crG * cr, 0.f, 1.f), std::clamp(yy + k.cbB * cb, 0.f, 1.f), o);
            same = same && std::memcmp(&o[0], &r[x], 4) == 0 && std::memcmp(&o[1], &g[x], 4) == 0 && std::memcmp(&o[2], &b[x], 4) == 0;
        }
        CHECK(same);
        pixels += kW;
    }
    std::printf("AVX2 row: %lld pixels equal to the portable row bit for bit\n", pixels);
}
