#pragma once

#include <QLabel>
#include <QMainWindow>
#include <array>
#include <string>

#include "AppModel.h"

namespace dlssvid {

class CompareBar;
class ViewportWindow;
class TimelineWidget;
class ProjectPanel;
class InspectorPanel;
class LogPanel;
class TaskQueue;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(bool warp, QWidget* parent = nullptr);
    void openPath(const QString& path);

protected:
    void closeEvent(QCloseEvent*) override;

private:
    void buildMenus();
    void updateCellLabels();
    QString cellLabel(const QString& title, const std::string& source, bool withTime = true);
    void saveScreenshot();
    void selectSource(int hotkey);  // 1..9

    AppModel model_;
    CompareBar* compareBar_;
    ViewportWindow* viewport_;
    QWidget* viewportContainer_;
    TimelineWidget* timeline_;
    ProjectPanel* projectPanel_;
    InspectorPanel* inspector_;
    LogPanel* log_;
    TaskQueue* tasks_;
    std::array<QLabel*, 4> cellLabels_{};
    QLabel* zoomLabel_;
};

}  // namespace dlssvid
