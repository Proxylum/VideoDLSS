#pragma once

#include <QPointF>
#include <QTimer>
#include <QWindow>
#include <memory>

#include "viewport/ViewportRenderer.h"

namespace dlssvid {

class AppModel;

// Native window with our DXGI swapchain on the shared D3D12 device (ТЗ §6: the viewport is
// rendered by the pipeline's device, passes are never copied to the CPU for display).
class ViewportWindow : public QWindow {
    Q_OBJECT
public:
    explicit ViewportWindow(AppModel& model);
    ~ViewportWindow() override;

    ViewportRenderer& renderer() { return *renderer_; }
    // Image pixel under a window position (cell-aware). Returns false outside the image.
    bool imageAt(const QPointF& pos, int& cell, float& ix, float& iy) const;
    void fitView();
    void zoomTo(float zoom, const QPointF* anchor = nullptr);  // keeps the image point under `anchor` fixed
    void oneToOne();
    void requestRender();
    // Screenshot of the current view (offscreen render at the window size) — ТЗ §6 «скриншот сетки в PNG».
    PassImage screenshot();

signals:
    void probe(int cell, float imageX, float imageY, bool inside);
    void zoomChanged(float zoom);

protected:
    void exposeEvent(QExposeEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    bool event(QEvent* e) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void mouseDoubleClickEvent(QMouseEvent*) override;

private:
    void render();
    void ensureRenderer();
    ViewportRenderer::Mapping mappingAt(const QPointF& pos, int& cell) const;
    float currentZoom() const;

    AppModel& model_;
    std::unique_ptr<ViewportRenderer> renderer_;
    QTimer pollTimer_;  // uploads finished loads and repaints while frames are arriving
    bool dirty_ = true;
    bool panning_ = false, wiping_ = false;
    QPointF lastMouse_;
    QPointF pressPos_;
};

}  // namespace dlssvid
