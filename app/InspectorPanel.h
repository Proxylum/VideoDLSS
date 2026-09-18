#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QListWidget>
#include <QSlider>
#include <QSpinBox>
#include <QWidget>
#include <array>

namespace dlssvid {

class AppModel;
class ViewportWindow;

// Right dock (ТЗ §6): the selected layer's display settings, overlay layer stack, wipe,
// grid cell sources and the pixel probe.
class InspectorPanel : public QWidget {
    Q_OBJECT
public:
    InspectorPanel(AppModel& model, ViewportWindow& viewport, QWidget* parent = nullptr);

public slots:
    void refresh();
    void showProbe(int cell, float imageX, float imageY, bool inside);

private:
    void applyLayer();
    void applyWipe();
    void applyGrid();
    AppModel& model_;
    ViewportWindow& viewport_;
    bool updating_ = false;

    QComboBox* mode_;
    QListWidget* layers_;
    QComboBox* source_;
    QComboBox* display_;
    QCheckBox* autoRange_;
    QDoubleSpinBox* minValue_;
    QDoubleSpinBox* maxValue_;
    QCheckBox* invert_;
    QSlider* opacity_;
    QSpinBox* opacityValue_;
    QComboBox* blend_;
    QCheckBox* visible_;
    QCheckBox* solo_;
    QSpinBox* arrowStep_;
    QDoubleSpinBox* mvScale_;
    QCheckBox* wipeEnabled_;
    QComboBox* wipeAxis_;
    QSlider* wipePos_;
    QComboBox* wipeA_;
    QComboBox* wipeB_;
    std::array<QComboBox*, 4> grid_{};
    QLabel* probe_;
};

}  // namespace dlssvid
