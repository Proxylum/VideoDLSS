#pragma once

#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStringList>
#include <QWidget>
#include <vector>

namespace dlssvid {

class AppModel;
class RecentPreviews;
class RecentCard;

// The start screen (stage 9, MR E; laid out as the mockup «1 · Стартовый экран»): two columns centred in the page —
// on the left what the tool does, the drop zone for a video, «Открыть видео…» / «Открыть проект…» and the GPU /
// driver / DLL readiness card; on the right the recent files as cards with a preview frame, «face — результат
// готов» and «1920×800 24 fps → 3840×1600 48 fps · вчера». The footer («Готов · Пассы по умолчанию: рядом с
// видео») is the main window's status bar.
class StartPage : public QWidget {
    Q_OBJECT
public:
    explicit StartPage(AppModel& model, QWidget* parent = nullptr);

    // Tests and screenshots: the cards as shown (titles «face — результат готов», metas «… · вчера 14:02»).
    QStringList cardTitles() const;
    QStringList cardMetas() const;
    bool cardHasPreview(int index) const;
    QString readinessText() const;

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
    void applyPreview(RecentCard* card);

    AppModel& model_;
    RecentPreviews* previews_;
    QFrame* dropZone_;
    QLabel* readinessDot_;
    QLabel* readiness_;
    QPushButton* clearRecents_;
    QWidget* cardsHost_;
    QLabel* empty_;
    std::vector<RecentCard*> cards_;
};

// True for the files the window opens by drop: videos and *.dlssvid.json projects.
bool IsOpenablePath(const QString& path);

}  // namespace dlssvid
