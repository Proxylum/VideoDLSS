#pragma once

#include <QComboBox>
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QWidget>

namespace dlssvid {

class AppModel;

// Scrubber over the time axis (ТЗ §6, stage 9): frames of the timeline rate, play/pause, ±1 frame of the base
// layer, timecode «00:04.98 / 00:09.96», the base layer's frame, and a rate switch when the sources differ
// (24 / 48 fps).
class TimelineWidget : public QWidget {
    Q_OBJECT
public:
    explicit TimelineWidget(AppModel& model, QWidget* parent = nullptr);

private:
    void refresh();
    void refreshRates();
    AppModel& model_;
    QSlider* slider_;
    QSpinBox* frame_;
    QToolButton* play_;
    QToolButton* loop_;
    QLabel* info_;
    QComboBox* fps_;
};

}  // namespace dlssvid
