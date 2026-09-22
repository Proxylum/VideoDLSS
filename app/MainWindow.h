#pragma once

#include <QByteArray>
#include <QDockWidget>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QStackedWidget>
#include <QSystemTrayIcon>
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
class ProcessingPanel;
class ResultBar;

// The application window: the start page until a project is open, then the work area (compare bar, cell labels,
// viewport, timeline) with the project / inspector / log / task docks. Stage 9 (MR E): the window remembers its
// geometry and dock layout (QSettings), opens maximised the first time, lists recent files, accepts dropped videos and
// projects, brings closed panels back through «Вид», marks unsaved changes with «*» in the title and saves the
// project next to the video on close. Stage 9 (MR F): «Обработать» opens the processing page (progress, time left,
// cancel, log tail); a finished run switches the viewport to «До | После» and, when the window is minimised or in
// the background, tells about it with a system notification.
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(bool warp, QWidget* parent = nullptr);
    void openPath(const QString& path);

protected:
    void closeEvent(QCloseEvent*) override;
    bool event(QEvent*) override;
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
    bool settleUnsaved();     // save or ask per AppModel::closeAction(); false when the user cancels
    void closeProject();      // «Другое видео» / Ctrl+W: back to the start page
    void saveResultAs();
    void openResultFolder();
    void exportPasses();      // one `dlssvid export` task per current pass into a chosen folder
    void watchProcess(int taskId, const QString& what);  // the run was queued: show the processing page
    void onTaskFinished(int taskId, bool ok);
    void notify(const QString& title, const QString& text);  // system notification (QSystemTrayIcon)
    bool inBackground() const;  // minimised or not the active window

    AppModel model_;
    QStackedWidget* pages_;
    StartPage* startPage_;
    ProcessingPanel* processing_;
    QWidget* workArea_;
    CompareBar* compareBar_;
    ResultBar* resultBar_;
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
    QSystemTrayIcon* tray_ = nullptr;
    int processTaskId_ = -1;        // the queued `process` run, -1 when none
    bool processingActive_ = false;  // the processing page is the current page
};

}  // namespace dlssvid
