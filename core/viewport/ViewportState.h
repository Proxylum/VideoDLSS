#pragma once

#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace dlssvid {

// ---- viewport state (ТЗ §6), serialisable into the project file ---------------------------

enum class ViewMode : uint8_t { Single, Overlay, Grid };
enum class DisplayMode : uint8_t { Color, Grayscale, Viridis, Turbo, MvHsv, MvArrows, MvMagnitude, MaskFill, MaskContour };
enum class BlendMode : uint8_t { Normal, Difference, Multiply, Screen };

std::string_view ToString(ViewMode m);
std::string_view ToString(DisplayMode m);
std::string_view ToString(BlendMode m);
std::optional<ViewMode> ParseViewMode(std::string_view s);
std::optional<DisplayMode> ParseDisplayMode(std::string_view s);
std::optional<BlendMode> ParseBlendMode(std::string_view s);

// Source names: "source" (decoded video), "result" (result video / pass), or a pass name
// ("depth_raw", "mv_dlss", "color_sr", ...).
struct LayerState {
    std::string source = "source";
    DisplayMode display = DisplayMode::Color;
    bool autoRange = true;   // min/max from the frame's statistics
    float minValue = 0.f;
    float maxValue = 1.f;
    bool invert = false;
    float opacity = 1.f;     // 0..1
    BlendMode blend = BlendMode::Normal;
    bool visible = true;
    bool solo = false;
    int arrowStep = 16;      // px between MV arrows (16..32)
    float mvScale = 0.f;     // 0 = auto (max magnitude of the frame) for HSV/magnitude display

    // Default display for a pass kind (scalar -> turbo, mv -> hsv, mask -> fill, colour -> color).
    static DisplayMode DefaultDisplayFor(const std::string& source);
    nlohmann::json ToJson() const;
    static LayerState FromJson(const nlohmann::json& j);
    bool operator==(const LayerState&) const = default;
};

struct WipeState {
    bool enabled = false;
    bool vertical = true;   // vertical split line (left/right) when true, horizontal otherwise
    float position = 0.5f;  // 0..1 across the image
    int layerA = 0;         // left/top
    int layerB = 1;         // right/bottom
    bool operator==(const WipeState&) const = default;
};

// Image -> screen mapping shared by all cells: `zoom` screen px per image px (0 = fit),
// `center` image point shown at the cell centre.
struct ViewTransform {
    float zoom = 0.f;
    float centerX = 0.f;
    float centerY = 0.f;
    bool operator==(const ViewTransform&) const = default;
    bool IsFit() const { return zoom <= 0.f; }
};

struct ViewportState {
    static constexpr int kMaxLayers = 5;  // base + 4
    ViewMode mode = ViewMode::Single;
    std::vector<LayerState> layers{LayerState{}};        // single: layers[0]; overlay: base + up to 4
    std::array<std::string, 4> gridSources{"source", "depth_raw", "mv_raw", "result"};
    int expandedCell = -1;                                // grid: -1 = 2x2, 0..3 = one cell full size
    WipeState wipe;
    ViewTransform view;
    double time = 0.0;         // seconds on the shared time axis (stage 9): each source shows round(time × its fps)
    int64_t legacyFrame = -1;  // "frame" of a project written before the time axis; resolved by ResolveLegacyFrame()
    int selectedLayer = 0;

    nlohmann::json ToJson() const;
    static ViewportState FromJson(const nlohmann::json& j);
    bool operator==(const ViewportState&) const = default;

    // Helpers used by the UI and the CLI.
    void SetSingleSource(const std::string& source);  // single mode: replace layer 0 keeping display defaults
    LayerState* AddOverlayLayer(const std::string& source);  // nullptr when the stack is full
    void RemoveLayer(int index);
    bool AnySolo() const;
    // Projects before stage 9 stored a frame index: converts it at the base rate once the sources are known.
    void ResolveLegacyFrame(double baseFps);
    // Sources the current mode displays (layers or grid cells), without duplicates - for prefetch.
    std::vector<std::string> NeededSources() const;
};

// "MM:SS.hh" (hours in front from 1 h): the timeline's timecode.
std::string FormatTimecode(double seconds);

// Zoom limits of ТЗ §6 (25 %..800 %).
constexpr float kMinZoom = 0.25f;
constexpr float kMaxZoom = 8.f;

}  // namespace dlssvid
