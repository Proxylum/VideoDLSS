#pragma once

#include <QWidget>
#include <cstdint>

class QHBoxLayout;
class QLabel;
class QToolButton;

namespace dlssvid {

class AppModel;

// The list of the card «Склейки» (PR-13, spec §2d): how many cuts the plan works with, up to eight chips with their time (a click
// shows the cut frame, Alt+click the frame before it, the rest in «ещё N…»), the highest scores just below the threshold - to find a
// missed cut - and what a new threshold changes for the stages below. Read only: the list is the stage's, set by its threshold.
// Hidden while the stage is off or the list is not known yet (it is analysed first).
class SceneCutsView : public QWidget {
    Q_OBJECT
public:
    explicit SceneCutsView(AppModel& model, QWidget* parent = nullptr);

public slots:
    void refresh();

private:
    QToolButton* Chip(int64_t frame, const QString& text, const QString& tip, QWidget* parent);

    AppModel& model_;
    QLabel* summary_;
    QWidget* chips_;
    QHBoxLayout* chipRow_;
    QWidget* near_;
    QHBoxLayout* nearRow_;
    QLabel* change_;
};

}  // namespace dlssvid
