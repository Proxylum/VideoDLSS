#include "viewport/ViewportRenderer.h"

#include <directx/d3dx12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "util/Error.h"
#include "util/Half.h"
#include "util/Log.h"

// Shader bytecode compiled by DXC at build time (see core/CMakeLists.txt).
#include "viewport/shaders/Composite_ps.h"
#include "viewport/shaders/Composite_vs.h"
#include "viewport/shaders/Arrows_ps.h"
#include "viewport/shaders/Arrows_vs.h"

namespace dlssvid {

namespace {

constexpr uint32_t kMaxLayers = 5;
constexpr uint32_t kSrvHeapSize = 512;
constexpr uint32_t kConstantSlot = 512;  // bytes per constant slot (>= sizeof(DrawConstantsGpu), 256-aligned)
constexpr uint32_t kConstantSlots = 128;
constexpr DXGI_FORMAT kTargetFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

struct LayerParamsGpu {
    uint32_t display, blend, flags, kind;
    float minValue, maxValue, opacity, mvScale;
    float texW, texH, pad0, pad1;
};
struct DrawConstantsGpu {
    float cell[4];
    float view[4];
    float image[4];
    uint32_t layerCount, anySolo, wipeEnabled, wipeVertical;
    float wipePos;
    uint32_t wipeA, wipeB, cellState;  // cellState: 0 ready, 1 loading, 2 no frame (a plate instead of an empty cell)
    float background[4];
    LayerParamsGpu layers[kMaxLayers];
};
struct ArrowConstantsGpu {
    float cell[4], view[4], image[4], target[4], texSize[4], color[4];
};

uint32_t LayerFlags(const LayerState& l) {
    uint32_t f = 0;
    if (l.visible) f |= 1u;
    if (l.solo) f |= 2u;
    if (l.invert) f |= 4u;
    return f;
}

}  // namespace

struct ViewportRenderer::Impl {
    D3D12Device& device;
    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> compositePso, arrowsPso;
    ComPtr<ID3D12DescriptorHeap> srvHeap, rtvHeap;
    UINT srvInc = 0, rtvInc = 0;
    uint32_t srvCursor = 0;
    ComPtr<ID3D12Resource> constants;  // upload heap ring
    uint8_t* constantsMapped = nullptr;
    uint32_t constantCursor = 0;

    ComPtr<IDXGISwapChain3> swapchain;
    std::array<ComPtr<ID3D12Resource>, 2> backbuffers;
    uint32_t windowW = 0, windowH = 0;

    ComPtr<ID3D12Resource> offscreen;
    uint32_t offscreenW = 0, offscreenH = 0;

    explicit Impl(D3D12Device& d) : device(d) {}

    D3D12_CPU_DESCRIPTOR_HANDLE Rtv(uint32_t index) const {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(rtvHeap->GetCPUDescriptorHandleForHeapStart(), static_cast<INT>(index), rtvInc);
    }

    // Allocates `count` consecutive SRV slots and returns (cpu, gpu) of the first.
    std::pair<D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE> AllocSrvs(uint32_t count) {
        if (srvCursor + count > kSrvHeapSize) srvCursor = 0;
        const uint32_t base = srvCursor;
        srvCursor += count;
        return {CD3DX12_CPU_DESCRIPTOR_HANDLE(srvHeap->GetCPUDescriptorHandleForHeapStart(), static_cast<INT>(base), srvInc),
                CD3DX12_GPU_DESCRIPTOR_HANDLE(srvHeap->GetGPUDescriptorHandleForHeapStart(), static_cast<INT>(base), srvInc)};
    }

    void WriteSrv(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* tex) {
        if (tex) {
            device.Get()->CreateShaderResourceView(tex, nullptr, handle);
        } else {
            D3D12_SHADER_RESOURCE_VIEW_DESC d{};
            d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture2D.MipLevels = 1;
            device.Get()->CreateShaderResourceView(nullptr, &d, handle);
        }
    }

    D3D12_GPU_VIRTUAL_ADDRESS PushConstants(const void* data, size_t size) {
        if (constantCursor >= kConstantSlots) constantCursor = 0;
        const uint32_t slot = constantCursor++;
        std::memcpy(constantsMapped + static_cast<size_t>(slot) * kConstantSlot, data, size);
        return constants->GetGPUVirtualAddress() + static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(slot) * kConstantSlot;
    }
};

ViewportRenderer::ViewportRenderer(D3D12Device& device) : impl_(std::make_unique<Impl>(device)) {
    ID3D12Device* dev = device.Get();
    Impl& im = *impl_;

    // Root signature: b0 root CBV, t0..t4 table, static samplers s0 (point) s1 (linear).
    CD3DX12_DESCRIPTOR_RANGE1 range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kMaxLayers, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    CD3DX12_ROOT_PARAMETER1 params[2];
    params[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
    params[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_ALL);
    CD3DX12_STATIC_SAMPLER_DESC samplers[2];
    samplers[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    samplers[1].Init(1, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rsDesc;
    rsDesc.Init_1_1(2, params, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    ComPtr<ID3DBlob> blob, error;
    HRESULT hr = D3DX12SerializeVersionedRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &blob, &error);
    if (FAILED(hr)) Throw(std::string("root signature: ") + (error ? static_cast<const char*>(error->GetBufferPointer()) : HrToString(hr)));
    CheckHr(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&im.rootSig)), "CreateRootSignature");

    // Composite PSO
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = im.rootSig.Get();
    pso.VS = {g_CompositeVS, sizeof(g_CompositeVS)};
    pso.PS = {g_CompositePS, sizeof(g_CompositePS)};
    pso.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.DepthStencilState.DepthEnable = FALSE;
    pso.DepthStencilState.StencilEnable = FALSE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = kTargetFormat;
    pso.SampleDesc.Count = 1;
    CheckHr(dev->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&im.compositePso)), "CreateGraphicsPipelineState(composite)");

    // Arrows PSO (line list)
    pso.VS = {g_ArrowsVS, sizeof(g_ArrowsVS)};
    pso.PS = {g_ArrowsPS, sizeof(g_ArrowsPS)};
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    CheckHr(dev->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&im.arrowsPso)), "CreateGraphicsPipelineState(arrows)");

    // Heaps
    D3D12_DESCRIPTOR_HEAP_DESC srvDesc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kSrvHeapSize, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    CheckHr(dev->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&im.srvHeap)), "CreateDescriptorHeap(srv)");
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CheckHr(dev->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&im.rtvHeap)), "CreateDescriptorHeap(rtv)");
    im.srvInc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    im.rtvInc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Constants ring (persistently mapped upload heap)
    im.constants = device.CreateBuffer(static_cast<uint64_t>(kConstantSlot) * kConstantSlots, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const CD3DX12_RANGE noRead(0, 0);
    CheckHr(im.constants->Map(0, &noRead, reinterpret_cast<void**>(&im.constantsMapped)), "Map(constants)");
}

ViewportRenderer::~ViewportRenderer() {
    try {
        impl_->device.WaitIdle();
    } catch (...) {
    }
    if (impl_->constants && impl_->constantsMapped) impl_->constants->Unmap(0, nullptr);
}

// ---- window / swapchain ---------------------------------------------------------------------

void ViewportRenderer::AttachWindow(void* hwnd, uint32_t width, uint32_t height) {
    Impl& im = *impl_;
    im.device.WaitIdle();
    im.swapchain.Reset();
    for (auto& b : im.backbuffers) b.Reset();
    ComPtr<IDXGIFactory4> factory;
    CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = std::max(1u, width);
    sd.Height = std::max(1u, height);
    sd.Format = kTargetFormat;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> sc1;
    CheckHr(factory->CreateSwapChainForHwnd(im.device.Queue(), static_cast<HWND>(hwnd), &sd, nullptr, nullptr, &sc1), "CreateSwapChainForHwnd");
    factory->MakeWindowAssociation(static_cast<HWND>(hwnd), DXGI_MWA_NO_ALT_ENTER);
    CheckHr(sc1.As(&im.swapchain), "IDXGISwapChain3");
    im.windowW = sd.Width;
    im.windowH = sd.Height;
    for (uint32_t i = 0; i < 2; ++i) {
        CheckHr(im.swapchain->GetBuffer(i, IID_PPV_ARGS(&im.backbuffers[i])), "GetBuffer");
        im.device.Get()->CreateRenderTargetView(im.backbuffers[i].Get(), nullptr, im.Rtv(i));
    }
}

void ViewportRenderer::Resize(uint32_t width, uint32_t height) {
    Impl& im = *impl_;
    if (!im.swapchain) return;
    width = std::max(1u, width);
    height = std::max(1u, height);
    if (width == im.windowW && height == im.windowH) return;
    im.device.WaitIdle();
    for (auto& b : im.backbuffers) b.Reset();
    CheckHr(im.swapchain->ResizeBuffers(2, width, height, kTargetFormat, 0), "ResizeBuffers");
    im.windowW = width;
    im.windowH = height;
    for (uint32_t i = 0; i < 2; ++i) {
        CheckHr(im.swapchain->GetBuffer(i, IID_PPV_ARGS(&im.backbuffers[i])), "GetBuffer");
        im.device.Get()->CreateRenderTargetView(im.backbuffers[i].Get(), nullptr, im.Rtv(i));
    }
}

bool ViewportRenderer::HasWindow() const { return impl_->swapchain != nullptr; }
RenderTargetSize ViewportRenderer::WindowSize() const { return {impl_->windowW, impl_->windowH}; }

// ---- geometry ---------------------------------------------------------------------------------

std::vector<ViewportRenderer::Rect> ViewportRenderer::CellRects(const ViewportState& state, uint32_t W, uint32_t H) {
    const float w = static_cast<float>(W), h = static_cast<float>(H);
    if (state.mode != ViewMode::Grid || state.expandedCell >= 0) return {Rect{0, 0, w, h}};
    const float hw = std::floor(w / 2), hh = std::floor(h / 2);
    return {Rect{0, 0, hw, hh}, Rect{hw, 0, w - hw, hh}, Rect{0, hh, hw, h - hh}, Rect{hw, hh, w - hw, h - hh}};
}

int ViewportRenderer::CellAt(const ViewportState& state, uint32_t W, uint32_t H, float px, float py) {
    if (state.mode != ViewMode::Grid) return -1;
    if (state.expandedCell >= 0) return state.expandedCell;
    const auto rects = CellRects(state, W, H);
    for (size_t i = 0; i < rects.size(); ++i)
        if (rects[i].Contains(px, py)) return static_cast<int>(i);
    return -1;
}

float ViewportRenderer::FitZoom(const Rect& cell, uint32_t imageWidth, uint32_t imageHeight) {
    if (!imageWidth || !imageHeight) return 1.f;
    return std::min(cell.w / static_cast<float>(imageWidth), cell.h / static_cast<float>(imageHeight));
}

ViewportRenderer::Mapping ViewportRenderer::MappingFor(const ViewTransform& view, const Rect& cell, uint32_t imageWidth, uint32_t imageHeight) {
    Mapping m;
    const bool fit = view.IsFit();
    m.zoom = fit ? FitZoom(cell, imageWidth, imageHeight) : std::clamp(view.zoom, kMinZoom, kMaxZoom);
    const float cx = fit ? static_cast<float>(imageWidth) * 0.5f : view.centerX;
    const float cy = fit ? static_cast<float>(imageHeight) * 0.5f : view.centerY;
    m.offsetX = cell.x + cell.w * 0.5f - cx * m.zoom;
    m.offsetY = cell.y + cell.h * 0.5f - cy * m.zoom;
    return m;
}

void ViewportRenderer::TargetToImage(const Mapping& m, float px, float py, float& imgX, float& imgY) {
    imgX = (px - m.offsetX) / m.zoom;
    imgY = (py - m.offsetY) / m.zoom;
}

std::vector<LayerState> ViewportRenderer::CellLayers(const ViewportState& state, int cell) {
    if (state.mode == ViewMode::Single) return {state.layers.empty() ? LayerState{} : state.layers[0]};
    if (state.mode == ViewMode::Overlay) return state.layers;
    LayerState l;
    const int c = std::clamp(cell, 0, 3);
    l.source = state.gridSources[static_cast<size_t>(c)];
    l.display = LayerState::DefaultDisplayFor(l.source);
    // in grid mode the selected layer's display settings apply to cells showing the same source
    for (const auto& s : state.layers)
        if (s.source == l.source) {
            l = s;
            l.opacity = 1.f;
            l.visible = true;
            l.solo = false;
            break;
        }
    return {l};
}

// ---- rendering ----------------------------------------------------------------------------------

void ViewportRenderer::RenderInto(ID3D12Resource* target, D3D12_CPU_DESCRIPTOR_HANDLE rtv, uint32_t W, uint32_t H, const ViewportState& state,
                                  const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight, const FrameStates* states,
                                  D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    Impl& im = *impl_;
    const auto t0 = std::chrono::steady_clock::now();
    im.srvCursor = 0;
    im.constantCursor = 0;
    if (!imageWidth || !imageHeight) {
        imageWidth = 1;
        imageHeight = 1;
    }
    const auto rects = CellRects(state, W, H);
    const bool grid = state.mode == ViewMode::Grid;

    im.device.ExecuteAndWait([&](ID3D12GraphicsCommandList* cl) {
        if (before != D3D12_RESOURCE_STATE_RENDER_TARGET) {
            const auto b = CD3DX12_RESOURCE_BARRIER::Transition(target, before, D3D12_RESOURCE_STATE_RENDER_TARGET);
            cl->ResourceBarrier(1, &b);
        }
        cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        const float clear[4] = {0.08f, 0.08f, 0.09f, 1.f};
        cl->ClearRenderTargetView(rtv, clear, 0, nullptr);
        ID3D12DescriptorHeap* heaps[] = {im.srvHeap.Get()};
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetGraphicsRootSignature(im.rootSig.Get());

        for (size_t ci = 0; ci < rects.size(); ++ci) {
            const Rect& cell = rects[ci];
            const int cellIndex = grid ? (state.expandedCell >= 0 ? state.expandedCell : static_cast<int>(ci)) : 0;
            const std::vector<LayerState> layers = CellLayers(state, cellIndex);
            const Mapping map = MappingFor(state.view, cell, imageWidth, imageHeight);

            DrawConstantsGpu c{};
            c.cell[0] = cell.x;
            c.cell[1] = cell.y;
            c.cell[2] = cell.w;
            c.cell[3] = cell.h;
            c.view[0] = map.offsetX;
            c.view[1] = map.offsetY;
            c.view[2] = map.zoom;
            c.view[3] = map.zoom < 1.f ? 1.f : 0.f;
            c.image[0] = static_cast<float>(imageWidth);
            c.image[1] = static_cast<float>(imageHeight);
            c.image[2] = 16.f;
            c.layerCount = static_cast<uint32_t>(std::min<size_t>(layers.size(), kMaxLayers));
            c.anySolo = 0;
            for (const auto& l : layers) c.anySolo |= l.solo ? 1u : 0u;
            const bool wipe = !grid && state.wipe.enabled && state.wipe.layerA != state.wipe.layerB && state.wipe.layerA < static_cast<int>(c.layerCount) &&
                              state.wipe.layerB < static_cast<int>(c.layerCount) && state.wipe.layerA >= 0 && state.wipe.layerB >= 0;
            c.wipeEnabled = wipe ? 1u : 0u;
            c.wipeVertical = state.wipe.vertical ? 1u : 0u;
            c.wipePos = std::clamp(state.wipe.position, 0.f, 1.f);
            c.wipeA = static_cast<uint32_t>(std::max(0, state.wipe.layerA));
            c.wipeB = static_cast<uint32_t>(std::max(0, state.wipe.layerB));
            c.cellState = 0;
            if (states && !layers.empty())
                if (const auto it = states->find(layers[0].source); it != states->end()) c.cellState = static_cast<uint32_t>(it->second);
            c.background[0] = c.background[1] = 0.16f;
            c.background[2] = 0.18f;
            c.background[3] = 1.f;

            auto [cpu, gpu] = im.AllocSrvs(kMaxLayers);
            std::vector<std::pair<const LayerState*, const LayerTexture*>> arrowLayers;
            for (uint32_t i = 0; i < kMaxLayers; ++i) {
                LayerParamsGpu& L = c.layers[i];
                const LayerTexture* tex = nullptr;
                if (i < c.layerCount) {
                    const auto it = frame.find(layers[i].source);
                    if (it != frame.end() && it->second.texture) tex = &it->second;
                }
                im.WriteSrv(CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu, static_cast<INT>(i), im.srvInc), tex ? tex->texture : nullptr);
                if (!tex || i >= c.layerCount) {
                    L.kind = 0;
                    continue;
                }
                const LayerState& l = layers[i];
                L.display = static_cast<uint32_t>(l.display);
                L.blend = static_cast<uint32_t>(l.blend);
                L.flags = LayerFlags(l);
                L.kind = static_cast<uint32_t>(tex->kind);
                L.minValue = l.autoRange ? tex->minValue : l.minValue;
                L.maxValue = l.autoRange ? tex->maxValue : l.maxValue;
                if (L.maxValue <= L.minValue) L.maxValue = L.minValue + 1e-6f;
                L.opacity = std::clamp(l.opacity, 0.f, 1.f);
                L.mvScale = l.mvScale > 0.f ? l.mvScale : std::max(tex->maxMagnitude, 1e-3f);
                L.texW = static_cast<float>(tex->width);
                L.texH = static_cast<float>(tex->height);
                if (l.display == DisplayMode::MvArrows && tex->kind == TextureKind::Mv && l.visible) arrowLayers.emplace_back(&l, tex);
            }

            const D3D12_VIEWPORT vp{cell.x, cell.y, cell.w, cell.h, 0.f, 1.f};
            const D3D12_RECT sc{static_cast<LONG>(cell.x), static_cast<LONG>(cell.y), static_cast<LONG>(cell.x + cell.w), static_cast<LONG>(cell.y + cell.h)};
            cl->RSSetViewports(1, &vp);
            cl->RSSetScissorRects(1, &sc);
            cl->SetPipelineState(im.compositePso.Get());
            cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cl->SetGraphicsRootConstantBufferView(0, im.PushConstants(&c, sizeof(c)));
            cl->SetGraphicsRootDescriptorTable(1, gpu);
            cl->DrawInstanced(3, 1, 0, 0);

            for (const auto& [l, tex] : arrowLayers) {
                ArrowConstantsGpu a{};
                std::memcpy(a.cell, c.cell, sizeof(a.cell));
                std::memcpy(a.view, c.view, sizeof(a.view));
                const float step = static_cast<float>(std::clamp(l->arrowStep, 4, 64));
                a.image[0] = static_cast<float>(imageWidth);
                a.image[1] = static_cast<float>(imageHeight);
                a.image[2] = step;
                a.image[3] = 1.f;  // image px of motion -> image px of arrow length
                const uint32_t cols = static_cast<uint32_t>(std::ceil(imageWidth / step)), rows = static_cast<uint32_t>(std::ceil(imageHeight / step));
                a.target[0] = static_cast<float>(W);
                a.target[1] = static_cast<float>(H);
                a.target[2] = static_cast<float>(cols);
                a.target[3] = static_cast<float>(rows);
                a.texSize[0] = static_cast<float>(tex->width);
                a.texSize[1] = static_cast<float>(tex->height);
                a.color[0] = 1.f;
                a.color[1] = 0.95f;
                a.color[2] = 0.2f;
                a.color[3] = 1.f;
                auto [acpu, agpu] = im.AllocSrvs(kMaxLayers);
                for (uint32_t i = 0; i < kMaxLayers; ++i) im.WriteSrv(CD3DX12_CPU_DESCRIPTOR_HANDLE(acpu, static_cast<INT>(i), im.srvInc), i == 0 ? tex->texture : nullptr);
                cl->SetPipelineState(im.arrowsPso.Get());
                cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
                cl->SetGraphicsRootConstantBufferView(0, im.PushConstants(&a, sizeof(a)));
                cl->SetGraphicsRootDescriptorTable(1, agpu);
                cl->DrawInstanced(6, cols * rows, 0, 0);
            }
        }
        if (after != D3D12_RESOURCE_STATE_RENDER_TARGET) {
            const auto b = CD3DX12_RESOURCE_BARRIER::Transition(target, D3D12_RESOURCE_STATE_RENDER_TARGET, after);
            cl->ResourceBarrier(1, &b);
        }
    });
    lastRenderMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ViewportRenderer::RenderToWindow(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight,
                                      const FrameStates* states) {
    Impl& im = *impl_;
    if (!im.swapchain) Throw("ViewportRenderer::RenderToWindow: no window attached");
    const uint32_t index = im.swapchain->GetCurrentBackBufferIndex();
    RenderInto(im.backbuffers[index].Get(), im.Rtv(index), im.windowW, im.windowH, state, frame, imageWidth, imageHeight, states, D3D12_RESOURCE_STATE_PRESENT,
               D3D12_RESOURCE_STATE_PRESENT);
    const HRESULT hr = im.swapchain->Present(1, 0);
    if (FAILED(hr) && hr != DXGI_STATUS_OCCLUDED) CheckHr(hr, "Present");
}

void ViewportRenderer::RenderOffscreen(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight,
                                       uint32_t targetWidth, uint32_t targetHeight, const FrameStates* states) {
    Impl& im = *impl_;
    targetWidth = std::max(1u, targetWidth);
    targetHeight = std::max(1u, targetHeight);
    if (!im.offscreen || im.offscreenW != targetWidth || im.offscreenH != targetHeight) {
        im.device.WaitIdle();
        im.offscreen = im.device.CreateTexture2D(targetWidth, targetHeight, kTargetFormat, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        im.offscreenW = targetWidth;
        im.offscreenH = targetHeight;
        im.device.Get()->CreateRenderTargetView(im.offscreen.Get(), nullptr, im.Rtv(2));
    }
    RenderInto(im.offscreen.Get(), im.Rtv(2), targetWidth, targetHeight, state, frame, imageWidth, imageHeight, states, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COMMON);
}

PassImage ViewportRenderer::RenderToImage(const ViewportState& state, const FrameTextures& frame, uint32_t imageWidth, uint32_t imageHeight,
                                          uint32_t targetWidth, uint32_t targetHeight, const FrameStates* states) {
    RenderOffscreen(state, frame, imageWidth, imageHeight, targetWidth, targetHeight, states);
    Impl& im = *impl_;
    size_t pitch = 0;
    std::vector<uint8_t> bytes = im.device.ReadbackTexture2D(im.offscreen.Get(), pitch);
    PassImage out;
    out.Allocate(im.offscreenW, im.offscreenH, PixelType::U8, {"R", "G", "B", "A"});
    out.data = std::move(bytes);
    return out;
}

std::array<float, 4> ViewportRenderer::ReadTexel(const LayerTexture& tex, uint32_t x, uint32_t y) {
    std::array<float, 4> out{0, 0, 0, 0};
    if (!tex.texture || x >= tex.width || y >= tex.height) return out;
    size_t bytesPerTexel = 0;
    const std::vector<uint8_t> bytes = impl_->device.ReadbackTexel(tex.texture, x, y, bytesPerTexel);
    switch (tex.kind) {
        case TextureKind::Color: {  // RGBA16F
            for (int c = 0; c < 4 && static_cast<size_t>(c) * 2 + 1 < bytes.size(); ++c) {
                uint16_t h;
                std::memcpy(&h, bytes.data() + c * 2, 2);
                out[static_cast<size_t>(c)] = HalfToFloat(h);
            }
            break;
        }
        case TextureKind::Scalar: std::memcpy(&out[0], bytes.data(), std::min<size_t>(4, bytes.size())); break;
        case TextureKind::Mv: std::memcpy(&out[0], bytes.data(), std::min<size_t>(8, bytes.size())); break;
        case TextureKind::Mask: out[0] = bytes.empty() ? 0.f : static_cast<float>(bytes[0]) / 255.f; break;
        default: break;
    }
    return out;
}

}  // namespace dlssvid
