#include "ViewportWindow.h"

#include <QEvent>
#include <QExposeEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "AppModel.h"
#include "util/Log.h"

namespace dlssvid {

ViewportWindow::ViewportWindow(AppModel& model) : model_(model) {
    setSurfaceType(QSurface::Direct3DSurface);
    setMinimumSize(QSize(64, 64));
    pollTimer_.setInterval(16);
    connect(&pollTimer_, &QTimer::timeout, this, [this] {
        if (model_.store().Update()) {
            dirty_ = true;
            model_.notifyFramesUpdated();
        }
        if (dirty_ && isExposed()) render();
    });
    pollTimer_.start();
    connect(&model_, &AppModel::stateChanged, this, [this] { requestRender(); });
    connect(&model_, &AppModel::sourcesChanged, this, [this] { requestRender(); });
}

ViewportWindow::~ViewportWindow() = default;

void ViewportWindow::ensureRenderer() {
    if (renderer_) return;
    renderer_ = std::make_unique<ViewportRenderer>(model_.device());
    const QSize s = size() * devicePixelRatio();
    renderer_->AttachWindow(reinterpret_cast<void*>(winId()), static_cast<uint32_t>(std::max(1, s.width())), static_cast<uint32_t>(std::max(1, s.height())));
}

void ViewportWindow::requestRender() {
    dirty_ = true;
    requestUpdate();
}

void ViewportWindow::render() {
    if (!isExposed()) return;
    ensureRenderer();
    const QSize s = size() * devicePixelRatio();
    renderer_->Resize(static_cast<uint32_t>(std::max(1, s.width())), static_cast<uint32_t>(std::max(1, s.height())));
    try {
        const FrameTextures frame = model_.store().TexturesAt(model_.state().time);
        const FrameStates states = model_.frameStates();  // plates for cells whose frame is loading / missing
        renderer_->RenderToWindow(model_.state(), frame, model_.store().ImageWidth(), model_.store().ImageHeight(), &states);
    } catch (const std::exception& e) {
        Log()->error("viewport render: {}", e.what());
    }
    dirty_ = false;
}

void ViewportWindow::exposeEvent(QExposeEvent*) {
    if (isExposed()) render();
}

void ViewportWindow::resizeEvent(QResizeEvent*) { requestRender(); }

bool ViewportWindow::event(QEvent* e) {
    if (e->type() == QEvent::UpdateRequest) {
        render();
        return true;
    }
    return QWindow::event(e);
}

float ViewportWindow::currentZoom() const {
    const auto rects = ViewportRenderer::CellRects(model_.state(), static_cast<uint32_t>(width()), static_cast<uint32_t>(height()));
    return ViewportRenderer::MappingFor(model_.state().view, rects.front(), model_.store().ImageWidth(), model_.store().ImageHeight()).zoom;
}

ViewportRenderer::Mapping ViewportWindow::mappingAt(const QPointF& pos, int& cell) const {
    const auto& st = model_.state();
    const uint32_t W = static_cast<uint32_t>(std::max(1, width())), H = static_cast<uint32_t>(std::max(1, height()));
    const auto rects = ViewportRenderer::CellRects(st, W, H);
    cell = 0;
    if (st.mode == ViewMode::Grid && st.expandedCell < 0) {
        const int c = ViewportRenderer::CellAt(st, W, H, static_cast<float>(pos.x()), static_cast<float>(pos.y()));
        cell = std::max(0, c);
    } else if (st.mode == ViewMode::Grid) {
        cell = st.expandedCell;
    }
    const size_t ri = st.mode == ViewMode::Grid && st.expandedCell < 0 ? static_cast<size_t>(cell) : 0;
    return ViewportRenderer::MappingFor(st.view, rects[std::min(ri, rects.size() - 1)], model_.store().ImageWidth(), model_.store().ImageHeight());
}

bool ViewportWindow::imageAt(const QPointF& pos, int& cell, float& ix, float& iy) const {
    const auto m = mappingAt(pos, cell);
    ViewportRenderer::TargetToImage(m, static_cast<float>(pos.x()), static_cast<float>(pos.y()), ix, iy);
    return ix >= 0 && iy >= 0 && ix < static_cast<float>(model_.store().ImageWidth()) && iy < static_cast<float>(model_.store().ImageHeight());
}

void ViewportWindow::fitView() {
    model_.state().view = ViewTransform{};
    model_.notifyStateChanged(false);
    emit zoomChanged(currentZoom());
}

void ViewportWindow::zoomTo(float zoom, const QPointF* anchor) {
    zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
    auto& view = model_.state().view;
    int cell = 0;
    const QPointF a = anchor ? *anchor : QPointF(width() / 2.0, height() / 2.0);
    const auto m = mappingAt(a, cell);
    float ix, iy;
    ViewportRenderer::TargetToImage(m, static_cast<float>(a.x()), static_cast<float>(a.y()), ix, iy);
    // keep the image point under the anchor: new center such that the anchor maps to the same image point
    const uint32_t W = static_cast<uint32_t>(std::max(1, width())), H = static_cast<uint32_t>(std::max(1, height()));
    const auto rects = ViewportRenderer::CellRects(model_.state(), W, H);
    const auto& r = rects[std::min<size_t>(static_cast<size_t>(cell), rects.size() - 1)];
    const float cx = r.x + r.w * 0.5f, cy = r.y + r.h * 0.5f;
    view.zoom = zoom;
    view.centerX = ix - (static_cast<float>(a.x()) - cx) / zoom;
    view.centerY = iy - (static_cast<float>(a.y()) - cy) / zoom;
    model_.notifyStateChanged(false);
    emit zoomChanged(zoom);
}

void ViewportWindow::oneToOne() { zoomTo(1.f); }

void ViewportWindow::mousePressEvent(QMouseEvent* e) {
    pressPos_ = lastMouse_ = e->position();
    if (e->button() == Qt::MiddleButton) {
        panning_ = true;
    } else if (e->button() == Qt::LeftButton && model_.state().mode != ViewMode::Grid && model_.state().wipe.enabled) {
        wiping_ = true;
        mouseMoveEvent(e);
    }
}

void ViewportWindow::mouseMoveEvent(QMouseEvent* e) {
    const QPointF pos = e->position();
    if (panning_) {
        auto& view = model_.state().view;
        if (view.IsFit()) {
            // leave fit mode with the current zoom so panning works
            view.zoom = currentZoom();
            view.centerX = model_.store().ImageWidth() * 0.5f;
            view.centerY = model_.store().ImageHeight() * 0.5f;
        }
        const QPointF d = pos - lastMouse_;
        view.centerX -= static_cast<float>(d.x()) / view.zoom;
        view.centerY -= static_cast<float>(d.y()) / view.zoom;
        model_.notifyStateChanged(false);
    } else if (wiping_) {
        int cell;
        float ix, iy;
        imageAt(pos, cell, ix, iy);
        auto& w = model_.state().wipe;
        const float p = w.vertical ? ix / std::max(1u, model_.store().ImageWidth()) : iy / std::max(1u, model_.store().ImageHeight());
        w.position = std::clamp(p, 0.f, 1.f);
        model_.notifyStateChanged(false);
    }
    lastMouse_ = pos;
    int cell;
    float ix, iy;
    const bool inside = imageAt(pos, cell, ix, iy);
    emit probe(cell, ix, iy, inside);
}

void ViewportWindow::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::MiddleButton) panning_ = false;
    if (e->button() == Qt::LeftButton) {
        const bool click = (e->position() - pressPos_).manhattanLength() < 4;
        wiping_ = false;
        if (click && model_.state().mode == ViewMode::Grid) {
            // click on a cell expands it, click again returns to the grid (ТЗ §6)
            auto& st = model_.state();
            if (st.expandedCell >= 0) {
                st.expandedCell = -1;
            } else {
                st.expandedCell = ViewportRenderer::CellAt(st, static_cast<uint32_t>(width()), static_cast<uint32_t>(height()), static_cast<float>(e->position().x()),
                                                           static_cast<float>(e->position().y()));
            }
            model_.notifyStateChanged(false);
        }
    }
}

void ViewportWindow::mouseDoubleClickEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && model_.state().mode != ViewMode::Grid) fitView();
}

void ViewportWindow::wheelEvent(QWheelEvent* e) {
    const float steps = static_cast<float>(e->angleDelta().y()) / 120.f;
    if (steps == 0.f) return;
    const float factor = std::pow(1.25f, steps);
    const QPointF anchor = e->position();
    zoomTo(currentZoom() * factor, &anchor);
}

PassImage ViewportWindow::screenshot() {
    ensureRenderer();
    const QSize s = size() * devicePixelRatio();
    const FrameTextures frame = model_.store().TexturesAt(model_.state().time);
    const FrameStates states = model_.frameStates();
    return renderer_->RenderToImage(model_.state(), frame, model_.store().ImageWidth(), model_.store().ImageHeight(), static_cast<uint32_t>(std::max(1, s.width())),
                                    static_cast<uint32_t>(std::max(1, s.height())), &states);
}

}  // namespace dlssvid
