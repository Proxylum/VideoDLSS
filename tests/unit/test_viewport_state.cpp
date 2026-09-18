// ViewportState: enum names, display defaults, layer stack helpers, JSON round trip.

#include <catch2/catch_test_macros.hpp>

#include "viewport/ViewportState.h"

using namespace dlssvid;

TEST_CASE("ViewportState enums round trip through their names", "[viewport][state]") {
    for (auto m : {ViewMode::Single, ViewMode::Overlay, ViewMode::Grid}) CHECK(ParseViewMode(ToString(m)) == m);
    for (int i = 0; i <= static_cast<int>(DisplayMode::MaskContour); ++i) {
        const auto d = static_cast<DisplayMode>(i);
        CHECK(ParseDisplayMode(ToString(d)) == d);
    }
    for (int i = 0; i <= static_cast<int>(BlendMode::Screen); ++i) {
        const auto b = static_cast<BlendMode>(i);
        CHECK(ParseBlendMode(ToString(b)) == b);
    }
    CHECK(ToString(DisplayMode::MvHsv) == "mv_hsv");
    CHECK(ToString(ViewMode::Grid) == "grid");
    CHECK(!ParseViewMode("nope"));
    CHECK(!ParseDisplayMode(""));
    CHECK(!ParseBlendMode("add"));
}

TEST_CASE("LayerState::DefaultDisplayFor follows the pass kind", "[viewport][state]") {
    CHECK(LayerState::DefaultDisplayFor("source") == DisplayMode::Color);
    CHECK(LayerState::DefaultDisplayFor("result") == DisplayMode::Color);
    CHECK(LayerState::DefaultDisplayFor("color_sr") == DisplayMode::Color);
    CHECK(LayerState::DefaultDisplayFor("color_fg") == DisplayMode::Color);
    CHECK(LayerState::DefaultDisplayFor("depth_raw") == DisplayMode::Turbo);
    CHECK(LayerState::DefaultDisplayFor("depth_dlss") == DisplayMode::Turbo);
    CHECK(LayerState::DefaultDisplayFor("mv_raw") == DisplayMode::MvHsv);
    CHECK(LayerState::DefaultDisplayFor("mv_dlss") == DisplayMode::MvHsv);
    CHECK(LayerState::DefaultDisplayFor("mask") == DisplayMode::MaskFill);
    // custom pass names by prefix
    CHECK(LayerState::DefaultDisplayFor("depth_custom") == DisplayMode::Turbo);
    CHECK(LayerState::DefaultDisplayFor("mv_custom") == DisplayMode::MvHsv);
    CHECK(LayerState::DefaultDisplayFor("mask_sky") == DisplayMode::MaskFill);
    CHECK(LayerState::DefaultDisplayFor("whatever") == DisplayMode::Color);
}

TEST_CASE("ViewportState layer stack helpers", "[viewport][state]") {
    ViewportState s;
    REQUIRE(s.layers.size() == 1);
    CHECK(s.layers[0].source == "source");
    s.SetSingleSource("depth_raw");
    CHECK(s.layers[0].display == DisplayMode::Turbo);
    CHECK(s.layers[0].autoRange);
    s.layers[0].autoRange = false;
    s.layers[0].display = DisplayMode::Viridis;
    s.SetSingleSource("depth_raw");  // same source keeps the settings
    CHECK(!s.layers[0].autoRange);
    CHECK(s.layers[0].display == DisplayMode::Viridis);
    s.SetSingleSource("mv_raw");  // a new source resets them
    CHECK(s.layers[0].autoRange);
    CHECK(s.layers[0].display == DisplayMode::MvHsv);

    for (int i = 1; i < ViewportState::kMaxLayers; ++i) REQUIRE(s.AddOverlayLayer("depth_raw") != nullptr);
    CHECK(s.layers.size() == 5);
    CHECK(s.selectedLayer == 4);
    CHECK(s.AddOverlayLayer("x") == nullptr);
    CHECK(s.layers[1].opacity == 0.5f);
    CHECK(s.layers[1].display == DisplayMode::Turbo);
    CHECK(!s.AnySolo());
    s.layers[3].solo = true;
    CHECK(s.AnySolo());

    s.RemoveLayer(0);  // the base layer stays
    CHECK(s.layers.size() == 5);
    s.RemoveLayer(99);
    CHECK(s.layers.size() == 5);
    s.RemoveLayer(4);
    CHECK(s.layers.size() == 4);
    CHECK(s.selectedLayer == 3);
    s.wipe.layerA = 0;
    s.wipe.layerB = 3;
    s.RemoveLayer(3);
    CHECK(s.layers.size() == 3);
    CHECK(s.wipe.layerB == 1);
    CHECK(s.selectedLayer == 2);
}

TEST_CASE("ViewportState::NeededSources lists what the mode shows", "[viewport][state]") {
    ViewportState s;
    CHECK(s.NeededSources() == std::vector<std::string>{"source"});
    s.mode = ViewMode::Overlay;
    s.AddOverlayLayer("depth_raw");
    s.AddOverlayLayer("source");
    CHECK(s.NeededSources() == std::vector<std::string>{"source", "depth_raw"});
    s.mode = ViewMode::Grid;
    s.gridSources = {"result", "result", "mv_raw", "depth_raw"};
    CHECK(s.NeededSources() == std::vector<std::string>{"result", "mv_raw", "depth_raw"});
}

TEST_CASE("ViewportState JSON round trip and tolerant parsing", "[viewport][state]") {
    ViewportState s;
    s.mode = ViewMode::Overlay;
    s.SetSingleSource("source");
    LayerState* l = s.AddOverlayLayer("depth_raw");
    l->display = DisplayMode::Viridis;
    l->autoRange = false;
    l->minValue = 0.5f;
    l->maxValue = 12.f;
    l->invert = true;
    l->opacity = 0.7f;
    l->blend = BlendMode::Difference;
    l->visible = false;
    l->solo = true;
    l->arrowStep = 24;
    l->mvScale = 3.f;
    s.gridSources = {"a", "b", "c", "d"};
    s.expandedCell = 2;
    s.wipe = WipeState{true, false, 0.3f, 1, 0};
    s.view = ViewTransform{2.5f, 10.f, 20.f};
    s.frame = 123;
    s.selectedLayer = 1;

    const nlohmann::json j = s.ToJson();
    CHECK(j["mode"] == "overlay");
    CHECK(j["layers"].size() == 2);
    CHECK(j["layers"][1]["display"] == "viridis");
    CHECK(j["layers"][1]["blend"] == "difference");
    CHECK(j["wipe"]["position"] == 0.3f);
    CHECK(j["view"]["zoom"] == 2.5f);
    CHECK(j["frame"] == 123);
    CHECK(ViewportState::FromJson(j) == s);
    CHECK(ViewportState::FromJson(nlohmann::json::parse(j.dump())) == s);

    CHECK(ViewportState::FromJson(nlohmann::json::object()) == ViewportState{});

    nlohmann::json bad = {{"mode", "weird"},
                          {"expanded_cell", 9},
                          {"view", {{"zoom", 100.f}}},
                          {"layers", nlohmann::json::array({{{"source", "depth_raw"}, {"opacity", 5.f}, {"display", "zzz"}, {"arrow_step", 1}}})},
                          {"selected_layer", 7}};
    const ViewportState b = ViewportState::FromJson(bad);
    CHECK(b.mode == ViewMode::Single);
    CHECK(b.expandedCell == 3);
    CHECK(b.view.zoom == kMaxZoom);
    REQUIRE(b.layers.size() == 1);
    CHECK(b.layers[0].opacity == 1.f);
    CHECK(b.layers[0].display == DisplayMode::Turbo);
    CHECK(b.layers[0].arrowStep == 4);
    CHECK(b.selectedLayer == 0);

    nlohmann::json many = {{"layers", nlohmann::json::array()}};
    for (int i = 0; i < 8; ++i) many["layers"].push_back({{"source", "source"}});
    CHECK(ViewportState::FromJson(many).layers.size() == static_cast<size_t>(ViewportState::kMaxLayers));
}
