#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gpu/D3D12Device.h"
#include "passes/PassImage.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

// What a layer source resolves to on the GPU.
enum class TextureKind : uint8_t { None = 0, Color = 1, Scalar = 2, Mv = 3, Mask = 4 };

struct LayerTexture {
    ID3D12Resource* texture = nullptr;
    TextureKind kind = TextureKind::None;
    uint32_t width = 0, height = 0;
    float minValue = 0.f, maxValue = 1.f;  // frame statistics for auto range
    float maxMagnitude = 1.f;              // mv: max |v| of the frame
};

// Textures of one frame by source name ("source", "result", pass names).
using FrameTextures = std::map<std::string, LayerTexture>;

// What the viewport knows about a source's frame at the current time. A cell whose base layer is not ready is
// drawn as a plate — «загрузка…» (plain) or «нет кадра» (hatched) — instead of an empty background; the text
// itself is the UI's (stage 9).
enum class FrameState : uint8_t { Ready = 0, Loading = 1, Missing = 2 };
using FrameStates = std::map<std::string, FrameState>;

struct RenderTargetSize {
    uint32_t width = 0, height = 0;
};

// One D3D12 composite of the viewport (ТЗ §6): the same device as the pipeline, one pixel
// shader for colour maps / overlay / wipe, plus an instanced line pass for MV arrows.
// Renders into a swapchain (window) or into an offscreen texture (screenshots, CLI, tests).
class ViewportRenderer {
public:
    explicit ViewportRenderer(D3D12Device& device);
    ~ViewportRenderer();
    ViewportRenderer(const ViewportRenderer&) = delete;
    ViewportRenderer& operator=(const ViewportRenderer&) = delete;

    // Window output. Resize() must be called when the window size changes.
    void AttachWindow(void* hwnd, uint32_t width, uint32_t height);
    void Resize(uint32_t width, uint32_t height);
    bool HasWindow() const;
    RenderTargetSize WindowSize() const;

    // Image reference size: the "image" all layers are stretched to (usually the source frame size).
    // Renders the state into the window and presents.
    void RenderToWindow(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight,
                        const FrameStates* states = nullptr);
    // Renders into the offscreen target without reading it back (benchmarks, warm-up).
    void RenderOffscreen(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight, uint32_t targetWidth,
                         uint32_t targetHeight, const FrameStates* states = nullptr);
    // Renders into an RGBA8 image (screenshots / CLI / tests).
    PassImage RenderToImage(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight,
                            uint32_t targetWidth, uint32_t targetHeight, const FrameStates* states = nullptr);

    // ---- geometry helpers shared with the UI (cell rects, image<->screen mapping) ----
    struct Rect {
        float x = 0, y = 0, w = 0, h = 0;
        bool Contains(float px, float py) const { return px >= x && py >= y && px < x + w && py < y + h; }
    };
    // Cell rects for the state (single/overlay: one full rect; grid: four or the expanded one).
    static std::vector<Rect> CellRects(const ViewportState& state, uint32_t targetWidth, uint32_t targetHeight);
    // Which cell a target pixel falls into (-1 when none / not grid).
    static int CellAt(const ViewportState& state, uint32_t targetWidth, uint32_t targetHeight, float px, float py);
    // Effective zoom and image-origin offset (target px) for a cell and the shared transform.
    struct Mapping {
        float zoom = 1.f;
        float offsetX = 0.f, offsetY = 0.f;  // target px of image pixel (0,0)
    };
    static Mapping MappingFor(const ViewTransform& view, const Rect& cell, uint32_t imageWidth, uint32_t imageHeight);
    static float FitZoom(const Rect& cell, uint32_t imageWidth, uint32_t imageHeight);
    // Image pixel under a target pixel for the given cell (may be outside the image).
    static void TargetToImage(const Mapping& m, float px, float py, float& imgX, float& imgY);

    // Layers of a cell as the shader sees them (grid cells are single-layer views of gridSources[i]).
    static std::vector<LayerState> CellLayers(const ViewportState& state, int cell);

    // 1x1 readback of a texture (pixel probe). Returns up to 4 components.
    std::array<float, 4> ReadTexel(const LayerTexture& tex, uint32_t x, uint32_t y);

    // Last render statistics.
    double LastRenderMs() const { return lastRenderMs_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    double lastRenderMs_ = 0.0;

    void RenderInto(ID3D12Resource* target, D3D12_CPU_DESCRIPTOR_HANDLE rtv, uint32_t targetWidth, uint32_t targetHeight, const ViewportState& state,
                    const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight, const FrameStates* states,
                    D3D12_RESOURCE_STATES targetStateBefore, D3D12_RESOURCE_STATES targetStateAfter);
};

}  // namespace dlssvid
