#pragma once

#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QWidget>

namespace dlssvid {

class AppModel;

// Scrubber, frame number, play/pause and ±1 step (ТЗ §6).
class TimelineWidget : public QWidget {
    Q_OBJECT
public:
    explicit TimelineWidget(AppModel& model, QWidget* parent = nullptr);

private:
    void refresh();
    AppModel& model_;
    QSlider* slider_;
    QSpinBox* frame_;
    QToolButton* play_;
    QLabel* info_;
};

}  // namespace dlssvid
