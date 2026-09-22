#pragma once

#include <QByteArray>
#include <QDockWidget>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QStackedWidget>
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
class StartPage;

// The application window: the start page until a project is open, then the work area (compare bar, cell labels,
// viewport, timeline) with the project / inspector / log / task docks. Stage 9 (MR E): the window remembers its
// geometry and dock layout (QSettings), opens maximised the first time, lists recent files, accepts dropped videos and
// projects, brings closed panels back through «Вид», marks unsaved changes with «*» in the title and saves the
// project next to the video on close.
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(bool warp, QWidget* parent = nullptr);
    void openPath(const QString& path);

protected:
    void closeEvent(QCloseEvent*) override;
    void dragEnterEvent(QDragEnterEvent*) override;
    void dropEvent(QDropEvent*) override;

private:
    void buildMenus();
    void updateCellLabels();
    QString cellLabel(const QString& title, const std::string& source, bool withTime = true);
    void saveScreenshot();
    void selectSource(int hotkey);  // 1..9
    void chooseVideo();
    void chooseProject();
    void saveProjectDialog(bool forceDialog);
    void updatePage();   // the start page or the work area; docks hide with the start page
    void updateTitle();  // «dlssvid — face.mp4*»
    void rebuildRecentMenu();
    void saveSettings();

    AppModel model_;
    QStackedWidget* pages_;
    StartPage* startPage_;
    QWidget* workArea_;
    CompareBar* compareBar_;
    ViewportWindow* viewport_;
    QWidget* viewportContainer_;
    TimelineWidget* timeline_;
    ProjectPanel* projectPanel_;
    InspectorPanel* inspector_;
    LogPanel* log_;
    TaskQueue* tasks_;
    QDockWidget* projectDock_;
    QDockWidget* inspectorDock_;
    QDockWidget* logDock_;
    QDockWidget* tasksDock_;
    QMenu* recentMenu_ = nullptr;
    std::array<QLabel*, 4> cellLabels_{};
    QLabel* zoomLabel_;
    QByteArray workState_;  // the dock layout of the work area (kept while the start page hides the docks)
    bool docksSized_ = false;
};

}  // namespace dlssvid
