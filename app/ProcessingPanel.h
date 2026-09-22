#pragma once

#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QWidget>
#include <vector>

#include "TaskQueue.h"

namespace dlssvid {

// The processing screen (stage 9, MR F; docs/ux-guidelines.md, principle 6): elapsed time and the time left, the
// overall progress, one row per stage (reused ones are marked at once, the running one shows its percent, the queued
// ones their estimate), «Отменить» (the finished stages stay on disk), «Свернуть в фон» (a notification tells when the
// run ends), the tail of the log with «Показать полный лог», and, when the run is over, «Показать результат» or the way
// back to the project.
class ProcessingPanel : public QWidget {
    Q_OBJECT
public:
    explicit ProcessingPanel(TaskQueue& tasks, QWidget* parent = nullptr);

    void watch(int taskId, const QString& what);  // follow a task; `what`: «face.mp4 → face_result.mp4»
    int taskId() const { return taskId_; }

    // What the screen says (tests): the headline, one line per stage «title · state · time», the log tail.
    QString headline() const { return headline_->text(); }
    QStringList stageLines() const;
    QString logTail() const { return tail_->toPlainText(); }

signals:
    void showResultRequested();  // «Показать результат»
    void backRequested();        // «К проекту» after a cancelled or failed run
    void minimizeRequested();    // «Свернуть в фон»
    void fullLogRequested();     // «Показать полный лог»

private:
    struct Row {
        QString name;
        QLabel* title;
        QProgressBar* bar;
        QLabel* time;
        QLabel* state;
    };
    void render();
    void rebuildRows(const TaskQueue::Progress& p);
    void onProgress(int id, const TaskQueue::Progress& p);
    void onOutput(int id, const QString& line);
    void onFinished(int id, bool ok);

    TaskQueue& tasks_;
    int taskId_ = -1;
    TaskQueue::Progress last_;
    QLabel* headline_;
    QLabel* what_;
    QProgressBar* bar_;
    QWidget* rowsWidget_;
    std::vector<Row> rows_;
    QLabel* note_;
    QPushButton* cancel_;
    QPushButton* minimize_;
    QPushButton* result_;
    QPushButton* back_;
    QPushButton* fullLog_;
    QPlainTextEdit* tail_;
    QTimer tick_;  // the elapsed time keeps counting between progress lines
};

}  // namespace dlssvid
