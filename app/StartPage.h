#pragma once

#include <QFrame>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QWidget>

namespace dlssvid {

class AppModel;

// The start screen (stage 9, MR E; the «Старт» mockup): what the tool does, a drop zone for a video, «Открыть
// видео…» / «Открыть проект…», the GPU / driver / DLL readiness line and the recent files with their state.
class StartPage : public QWidget {
    Q_OBJECT
public:
    explicit StartPage(AppModel& model, QWidget* parent = nullptr);

public slots:
    void refresh();

signals:
    void openVideoRequested();
    void openProjectRequested();
    void openPathRequested(const QString& path);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    bool eventFilter(QObject* watched, QEvent* e) override;

private:
    AppModel& model_;
    QFrame* dropZone_;
    QLabel* readiness_;
    QListWidget* recents_;
    QPushButton* clearRecents_;
    QLabel* footer_;
};

// True for the files the window opens by drop: videos and *.dlssvid.json projects.
bool IsOpenablePath(const QString& path);

}  // namespace dlssvid
