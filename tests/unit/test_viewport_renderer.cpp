// ViewportRenderer on WARP: colour maps, blend modes, opacity, wipe, 2x2 grid, MV / mask displays,
// geometry helpers and the pixel probe (ТЗ §6).

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <set>
#include <vector>

#include "gpu/D3D12Device.h"
#include "util/Half.h"
#include "viewport/ViewportRenderer.h"

using namespace dlssvid;

namespace {

struct Tex {
    ComPtr<ID3D12Resource> res;
    LayerTexture desc;
};

Tex MakeScalar(D3D12Device& dev, uint32_t w, uint32_t h, const std::function<float(uint32_t, uint32_t)>& f, float mn, float mx) {
    std::vector<float> v(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) v[y * w + x] = f(x, y);
    Tex t;
    t.res = dev.CreateTexture2D(w, h, DXGI_FORMAT_R32_FLOAT);
    dev.UploadTexture2D(t.res.Get(), reinterpret_cast<const uint8_t*>(v.data()), w * sizeof(float));
    t.desc = LayerTexture{t.res.Get(), TextureKind::Scalar, w, h, mn, mx, 1.f};
    return t;
}

Tex MakeColor(D3D12Device& dev, uint32_t w, uint32_t h, const std::function<std::array<float, 3>(uint32_t, uint32_t)>& f) {
    std::vector<uint16_t> v(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const auto c = f(x, y);
            uint16_t* p = &v[(y * w + x) * 4];
            p[0] = FloatToHalf(c[0]);
            p[1] = FloatToHalf(c[1]);
            p[2] = FloatToHalf(c[2]);
            p[3] = FloatToHalf(1.f);
        }
    Tex t;
    t.res = dev.CreateTexture2D(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    dev.UploadTexture2D(t.res.Get(), reinterpret_cast<const uint8_t*>(v.data()), w * 8);
    t.desc = LayerTexture{t.res.Get(), TextureKind::Color, w, h, 0.f, 1.f, 1.f};
    return t;
}

Tex MakeMv(D3D12Device& dev, uint32_t w, uint32_t h, const std::function<std::array<float, 2>(uint32_t, uint32_t)>& f, float maxMag) {
    std::vector<float> v(static_cast<size_t>(w) * h * 2);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const auto m = f(x, y);
            v[(y * w + x) * 2] = m[0];
            v[(y * w + x) * 2 + 1] = m[1];
        }
    Tex t;
    t.res = dev.CreateTexture2D(w, h, DXGI_FORMAT_R32G32_FLOAT);
    dev.UploadTexture2D(t.res.Get(), reinterpret_cast<const uint8_t*>(v.data()), w * 8);
    t.desc = LayerTexture{t.res.Get(), TextureKind::Mv, w, h, 0.f, maxMag, maxMag};
    return t;
}

Tex MakeMask(D3D12Device& dev, uint32_t w, uint32_t h, const std::function<uint8_t(uint32_t, uint32_t)>& f) {
    std::vector<uint8_t> v(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) v[y * w + x] = f(x, y);
    Tex t;
    t.res = dev.CreateTexture2D(w, h, DXGI_FORMAT_R8_UNORM);
    dev.UploadTexture2D(t.res.Get(), v.data(), w);
    t.desc = LayerTexture{t.res.Get(), TextureKind::Mask, w, h, 0.f, 1.f, 1.f};
    return t;
}

struct Px {
    int r, g, b, a;
};
Px At(const PassImage& img, uint32_t x, uint32_t y) {
    const uint8_t* p = img.data.data() + (static_cast<size_t>(y) * img.width + x) * 4;
    return {p[0], p[1], p[2], p[3]};
}
bool Near(const Px& p, int r, int g, int b, int tol = 2) { return std::abs(p.r - r) <= tol && std::abs(p.g - g) <= tol && std::abs(p.b - b) <= tol; }
int Lum(const Px& p) { return static_cast<int>(std::lround(0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b)); }

// 1:1 mapping: target = image size, zoom 1 centred.
ViewportState OneToOne(uint32_t w, uint32_t h) {
    ViewportState st;
    st.view.zoom = 1.f;
    st.view.centerX = w * 0.5f;
    st.view.centerY = h * 0.5f;
    return st;
}

}  // namespace

TEST_CASE("ViewportRenderer geometry: cell rects, fit zoom, mapping", "[viewport][renderer]") {
    ViewportState st;
    auto rects = ViewportRenderer::CellRects(st, 200, 100);
    REQUIRE(rects.size() == 1);
    CHECK(rects[0].w == 200.f);
    st.mode = ViewMode::Grid;
    rects = ViewportRenderer::CellRects(st, 200, 100);
    REQUIRE(rects.size() == 4);
    CHECK(rects[1].x == 100.f);
    CHECK(rects[2].y == 50.f);
    CHECK(rects[3].w == 100.f);
    CHECK(ViewportRenderer::CellAt(st, 200, 100, 150.f, 75.f) == 3);
    CHECK(ViewportRenderer::CellAt(st, 200, 100, 10.f, 10.f) == 0);
    st.expandedCell = 2;
    CHECK(ViewportRenderer::CellRects(st, 200, 100).size() == 1);
    CHECK(ViewportRenderer::CellAt(st, 200, 100, 10.f, 10.f) == 2);
    st.mode = ViewMode::Single;
    CHECK(ViewportRenderer::CellAt(st, 200, 100, 10.f, 10.f) == -1);

    const ViewportRenderer::Rect cell{0, 0, 200, 200};
    CHECK(ViewportRenderer::FitZoom(cell, 100, 50) == 2.f);
    ViewTransform fit;
    auto m = ViewportRenderer::MappingFor(fit, cell, 100, 50);
    CHECK(m.zoom == 2.f);
    CHECK(m.offsetX == 0.f);
    CHECK(m.offsetY == 50.f);
    float ix, iy;
    ViewportRenderer::TargetToImage(m, 0.f, 50.f, ix, iy);
    CHECK(ix == 0.f);
    CHECK(iy == 0.f);
    ViewTransform z{4.f, 10.f, 10.f};
    m = ViewportRenderer::MappingFor(z, cell, 100, 50);
    CHECK(m.zoom == 4.f);
    ViewportRenderer::TargetToImage(m, 100.f, 100.f, ix, iy);
    CHECK(ix == 10.f);
    CHECK(iy == 10.f);
    ViewTransform tooBig{100.f, 0.f, 0.f};
    CHECK(ViewportRenderer::MappingFor(tooBig, cell, 100, 50).zoom == kMaxZoom);

    // grid cells are single-layer views of gridSources with the layer settings of a matching layer
    st.mode = ViewMode::Grid;
    st.expandedCell = -1;
    st.gridSources = {"source", "depth_raw", "mv_raw", "result"};
    st.layers[0].source = "depth_raw";
    st.layers[0].display = DisplayMode::Viridis;
    st.layers[0].opacity = 0.3f;
    auto layers = ViewportRenderer::CellLayers(st, 1);
    REQUIRE(layers.size() == 1);
    CHECK(layers[0].display == DisplayMode::Viridis);
    CHECK(layers[0].opacity == 1.f);
    CHECK(ViewportRenderer::CellLayers(st, 2)[0].display == DisplayMode::MvHsv);
}

TEST_CASE("ViewportRenderer grayscale / viridis / turbo colour maps", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 16, h = 2;
    Tex t = MakeScalar(dev, w, h, [](uint32_t x, uint32_t) { return static_cast<float>(x); }, 0.f, 15.f);
    FrameTextures frame{{"depth_raw", t.desc}};
    ViewportState st = OneToOne(w, h);
    st.SetSingleSource("depth_raw");

    st.layers[0].display = DisplayMode::Grayscale;
    PassImage img = r.RenderToImage(st, frame, w, h, w, h);
    REQUIRE(img.width == w);
    REQUIRE(img.height == h);
    REQUIRE(img.channels.size() == 4);
    for (uint32_t x = 0; x < w; ++x) {
        const int expect = static_cast<int>(std::lround(255.0 * x / 15.0));
        const Px p = At(img, x, 0);
        CHECK(Near(p, expect, expect, expect, 1));
        CHECK(p.a == 255);
    }
    // inverted + manual range
    st.layers[0].invert = true;
    st.layers[0].autoRange = false;
    st.layers[0].minValue = 0.f;
    st.layers[0].maxValue = 30.f;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 15, 0), 128, 128, 128, 2));
    CHECK(Near(At(img, 0, 0), 255, 255, 255, 1));
    st.layers[0].invert = false;
    st.layers[0].autoRange = true;

    st.layers[0].display = DisplayMode::Viridis;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 0, 0), 71, 1, 85, 6));
    CHECK(Near(At(img, 15, 0), 252, 231, 34, 8));
    int prev = -1;
    for (uint32_t x = 0; x < w; ++x) {
        const int l = Lum(At(img, x, 0));
        CHECK(l >= prev - 1);  // viridis is monotonic in luminance
        prev = l;
    }

    st.layers[0].display = DisplayMode::Turbo;
    img = r.RenderToImage(st, frame, w, h, w, h);
    const Px lo = At(img, 0, 0), mid = At(img, 8, 0), hi = At(img, 15, 0);
    CHECK((lo.r < 60 && lo.g < 40 && lo.b < 60));
    CHECK(mid.g > 200);
    CHECK((hi.r > 120 && hi.b < 20 && hi.g < 40));
}

TEST_CASE("ViewportRenderer blend modes, opacity, solo and visibility", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 4, h = 4;
    Tex base = MakeColor(dev, w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.5f, 0.5f, 0.5f}; });
    Tex top = MakeColor(dev, w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.25f, 0.75f, 1.f}; });
    FrameTextures frame{{"source", base.desc}, {"result", top.desc}};
    ViewportState st = OneToOne(w, h);
    st.mode = ViewMode::Overlay;
    st.SetSingleSource("source");
    LayerState* l = st.AddOverlayLayer("result");
    REQUIRE(l != nullptr);
    CHECK(st.layers.size() == 2);
    CHECK(l->opacity == 1.f);  // new overlay layers are full (stage 9): comparisons wipe, they do not blend
    l->opacity = 1.f;

    auto render = [&] { return At(r.RenderToImage(st, frame, w, h, w, h), 1, 1); };
    st.layers[1].blend = BlendMode::Normal;
    CHECK(Near(render(), 64, 191, 255));
    st.layers[1].blend = BlendMode::Difference;
    CHECK(Near(render(), 64, 64, 128));
    st.layers[1].blend = BlendMode::Multiply;
    CHECK(Near(render(), 32, 96, 128));
    st.layers[1].blend = BlendMode::Screen;
    CHECK(Near(render(), 159, 223, 255));
    st.layers[1].blend = BlendMode::Normal;
    st.layers[1].opacity = 0.5f;
    CHECK(Near(render(), 96, 159, 191));
    st.layers[1].opacity = 1.f;
    st.layers[1].visible = false;
    CHECK(Near(render(), 128, 128, 128));
    st.layers[1].visible = true;
    st.layers[1].solo = true;
    CHECK(Near(render(), 64, 191, 255));  // solo hides the base
    st.layers[1].solo = false;
    st.layers[1].invert = true;
    st.layers[1].display = DisplayMode::Color;
    CHECK(Near(render(), 191, 64, 0));

    // a source that is not in the frame renders nothing (black), the base stays
    st.layers[1].invert = false;
    st.layers[1].source = "missing";
    CHECK(Near(render(), 128, 128, 128));
    st.layers[0].source = "missing";
    CHECK(Near(render(), 0, 0, 0));
}

TEST_CASE("ViewportRenderer wipe compares two layers side by side", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 64, h = 16;
    Tex a = MakeColor(dev, w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{1.f, 0.f, 0.f}; });
    Tex b = MakeColor(dev, w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.f, 0.f, 1.f}; });
    FrameTextures frame{{"source", a.desc}, {"result", b.desc}};
    ViewportState st = OneToOne(w, h);
    st.mode = ViewMode::Overlay;
    st.SetSingleSource("source");
    st.AddOverlayLayer("result")->opacity = 1.f;
    st.wipe.enabled = true;
    st.wipe.vertical = true;
    st.wipe.position = 0.5f;
    st.wipe.layerA = 0;
    st.wipe.layerB = 1;
    PassImage img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 16, 8), 255, 0, 0));
    CHECK(Near(At(img, 48, 8), 0, 0, 255));
    CHECK(Near(At(img, 32, 8), 255, 255, 255));  // split line
    st.wipe.vertical = false;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 32, 3), 255, 0, 0));
    CHECK(Near(At(img, 32, 12), 0, 0, 255));
    st.wipe.position = 0.25f;
    st.wipe.vertical = true;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 8, 8), 255, 0, 0));
    CHECK(Near(At(img, 24, 8), 0, 0, 255));
    // wipe with the same layer on both sides is ignored
    st.wipe.layerB = 0;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 8, 8), 0, 0, 255));  // plain overlay: result on top
}

TEST_CASE("ViewportRenderer 2x2 grid draws four sources and expands one", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 8, h = 8;
    const std::array<std::array<float, 3>, 4> colors{{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f, 0.f}}};
    std::vector<Tex> texes;
    FrameTextures frame;
    const std::array<std::string, 4> names{"source", "color_sr", "color_nr", "result"};
    for (size_t i = 0; i < 4; ++i) {
        texes.push_back(MakeColor(dev, w, h, [c = colors[i]](uint32_t, uint32_t) { return c; }));
        frame[names[i]] = texes.back().desc;
    }
    ViewportState st;
    st.mode = ViewMode::Grid;
    st.gridSources = names;
    PassImage img = r.RenderToImage(st, frame, w, h, 64, 64);
    REQUIRE(img.width == 64);
    CHECK(Near(At(img, 16, 16), 255, 0, 0));
    CHECK(Near(At(img, 48, 16), 0, 255, 0));
    CHECK(Near(At(img, 16, 48), 0, 0, 255));
    CHECK(Near(At(img, 48, 48), 255, 255, 0));
    st.expandedCell = 2;
    img = r.RenderToImage(st, frame, w, h, 64, 64);
    CHECK(Near(At(img, 16, 16), 0, 0, 255));
    CHECK(Near(At(img, 48, 48), 0, 0, 255));
    // outside the image (fit into a wide target) the checkerboard background shows
    st.expandedCell = -1;
    st.mode = ViewMode::Single;
    img = r.RenderToImage(st, frame, w, h, 64, 16);
    const Px edge = At(img, 1, 8);
    CHECK((edge.r < 80 && edge.g < 80 && edge.b < 80));
    CHECK(Near(At(img, 32, 8), 255, 0, 0));
}

TEST_CASE("ViewportRenderer motion-vector displays (HSV, magnitude, arrows)", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 48, h = 16;
    // three bands: right, zero, left
    Tex mv = MakeMv(
        dev, w, h,
        [](uint32_t x, uint32_t) {
            if (x < 16) return std::array<float, 2>{4.f, 0.f};
            if (x < 32) return std::array<float, 2>{0.f, 0.f};
            return std::array<float, 2>{-4.f, 0.f};
        },
        4.f);
    FrameTextures frame{{"mv_raw", mv.desc}};
    ViewportState st = OneToOne(w, h);
    st.SetSingleSource("mv_raw");
    CHECK(st.layers[0].display == DisplayMode::MvHsv);
    PassImage img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 8, 8), 0, 255, 255, 4));    // right -> cyan
    CHECK(Near(At(img, 24, 8), 255, 255, 255, 2));  // zero -> white
    CHECK(Near(At(img, 40, 8), 255, 0, 0, 4));      // left -> red

    st.layers[0].display = DisplayMode::MvMagnitude;
    st.layers[0].mvScale = 8.f;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 8, 8), 128, 128, 128, 2));
    CHECK(Near(At(img, 24, 8), 0, 0, 0, 1));

    st.layers[0].display = DisplayMode::MvArrows;
    st.layers[0].arrowStep = 8;
    img = r.RenderToImage(st, frame, w, h, w, h);
    int yellow = 0, gray = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const Px p = At(img, x, y);
            if (p.r > 200 && p.g > 180 && p.b < 100) ++yellow;
            if (p.r == p.g && p.g == p.b) ++gray;
        }
    CHECK(yellow > 4);  // arrow lines drawn on top
    CHECK(gray > static_cast<int>(w * h / 2));
}

TEST_CASE("ViewportRenderer mask fill and contour", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 8, h = 4;
    Tex base = MakeColor(dev, w, h, [](uint32_t, uint32_t) { return std::array<float, 3>{0.5f, 0.5f, 0.5f}; });
    Tex mask = MakeMask(dev, w, h, [](uint32_t x, uint32_t) { return x < 4 ? uint8_t{255} : uint8_t{0}; });
    FrameTextures frame{{"source", base.desc}, {"mask", mask.desc}};
    ViewportState st = OneToOne(w, h);
    st.mode = ViewMode::Overlay;
    st.SetSingleSource("source");
    st.AddOverlayLayer("mask")->opacity = 1.f;
    CHECK(st.layers[1].display == DisplayMode::MaskFill);
    PassImage img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 1, 1), 255, 64, 26, 2));     // inside the mask: fill colour
    CHECK(Near(At(img, 6, 1), 128, 128, 128, 2));   // outside: base
    st.layers[1].opacity = 0.5f;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 1, 1), 191, 96, 77, 3));
    st.layers[1].opacity = 1.f;
    st.layers[1].display = DisplayMode::MaskContour;
    img = r.RenderToImage(st, frame, w, h, w, h);
    CHECK(Near(At(img, 3, 1), 255, 230, 26, 3));    // edge texels on both sides of the boundary
    CHECK(Near(At(img, 4, 1), 255, 230, 26, 3));
    CHECK(Near(At(img, 2, 1), 128, 128, 128, 2));   // interior: base
    CHECK(Near(At(img, 6, 1), 128, 128, 128, 2));
}

TEST_CASE("ViewportRenderer pixel probe reads back texels of every kind", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    Tex s = MakeScalar(dev, 4, 4, [](uint32_t x, uint32_t y) { return 10.f * x + y; }, 0.f, 33.f);
    Tex c = MakeColor(dev, 4, 4, [](uint32_t x, uint32_t y) { return std::array<float, 3>{x / 4.f, y / 4.f, 0.75f}; });
    Tex m = MakeMv(dev, 4, 4, [](uint32_t x, uint32_t y) { return std::array<float, 2>{static_cast<float>(x) - 2.f, static_cast<float>(y) * 0.5f}; }, 2.f);
    Tex k = MakeMask(dev, 4, 4, [](uint32_t x, uint32_t) { return static_cast<uint8_t>(x * 85); });
    CHECK(r.ReadTexel(s.desc, 3, 1)[0] == 31.f);
    const auto cc = r.ReadTexel(c.desc, 2, 1);
    CHECK(cc[0] == 0.5f);
    CHECK(cc[1] == 0.25f);
    CHECK(cc[2] == 0.75f);
    const auto mm = r.ReadTexel(m.desc, 0, 3);
    CHECK(mm[0] == -2.f);
    CHECK(mm[1] == 1.5f);
    CHECK(r.ReadTexel(k.desc, 3, 0)[0] == 1.f);
    CHECK(r.ReadTexel(k.desc, 9, 0)[0] == 0.f);  // out of range -> zeros
}

TEST_CASE("ViewportRenderer zoom keeps point sampling and pans to the requested centre", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const uint32_t w = 8, h = 8;
    Tex t = MakeScalar(dev, w, h, [](uint32_t x, uint32_t y) { return (x + y) % 2 ? 1.f : 0.f; }, 0.f, 1.f);
    FrameTextures frame{{"depth_raw", t.desc}};
    ViewportState st;
    st.SetSingleSource("depth_raw");
    st.layers[0].display = DisplayMode::Grayscale;
    // 400 %: image pixel (2,2) at the centre of a 16x16 target -> target (8,8) shows pixel (2,2) = 0
    st.view.zoom = 4.f;
    st.view.centerX = 2.f;
    st.view.centerY = 2.f;
    PassImage img = r.RenderToImage(st, frame, w, h, 16, 16);
    CHECK(Near(At(img, 8, 8), 0, 0, 0, 1));
    CHECK(Near(At(img, 12, 8), 255, 255, 255, 1));  // pixel (3,2) = 1
    CHECK(Near(At(img, 11, 11), 0, 0, 0, 1));       // still pixel (2,2)
    // zoomed out below 100 % the filter switches to linear: a checker averages to grey
    st.view.zoom = 0.5f;
    st.view.centerX = 4.f;
    st.view.centerY = 4.f;
    img = r.RenderToImage(st, frame, w, h, 4, 4);
    const Px p = At(img, 1, 1);
    CHECK((p.r > 60 && p.r < 200));
}

TEST_CASE("ViewportRenderer plates: a missing frame is hatched, a loading one is a plain plate, a ready one is the texture", "[viewport][renderer][gpu]") {
    D3D12Device dev({true, false});
    ViewportRenderer r(dev);
    const ViewportState st = OneToOne(32, 16);
    const FrameTextures none;
    const PassImage empty = r.RenderToImage(st, none, 32, 16, 32, 16);  // no state: the empty background as before
    const FrameStates missing{{"source", FrameState::Missing}}, loading{{"source", FrameState::Loading}};
    const PassImage hatched = r.RenderToImage(st, none, 32, 16, 32, 16, &missing);
    const PassImage plate = r.RenderToImage(st, none, 32, 16, 32, 16, &loading);
    std::set<int> hatchTones, plateTones, emptyTones;
    for (uint32_t x = 0; x < 32; ++x) {
        hatchTones.insert(Lum(At(hatched, x, 8)));
        plateTones.insert(Lum(At(plate, x, 8)));
        emptyTones.insert(Lum(At(empty, x, 8)));
    }
    CHECK(hatchTones.size() == 2);  // diagonal stripes
    CHECK(plateTones.size() == 1);  // one flat tone
    CHECK(*plateTones.begin() < 140);
    CHECK(*hatchTones.rbegin() < 140);
    CHECK(plateTones != emptyTones);
    CHECK(hatchTones != emptyTones);
    CHECK(plateTones != hatchTones);
    // a ready state with a texture renders the texture; a state of another source does not touch this cell
    const Tex tex = MakeColor(dev, 32, 16, [](uint32_t, uint32_t) { return std::array<float, 3>{1.f, 0.f, 0.f}; });
    const FrameTextures frame{{"source", tex.desc}};
    const FrameStates ready{{"source", FrameState::Ready}}, other{{"depth_raw", FrameState::Missing}};
    CHECK(Near(At(r.RenderToImage(st, frame, 32, 16, 32, 16, &ready), 5, 8), 255, 0, 0, 3));
    CHECK(Near(At(r.RenderToImage(st, frame, 32, 16, 32, 16, &other), 5, 8), 255, 0, 0, 3));
    CHECK(Near(At(r.RenderToImage(st, frame, 32, 16, 32, 16), 5, 8), 255, 0, 0, 3));
}
