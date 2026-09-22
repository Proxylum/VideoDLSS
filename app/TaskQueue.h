#pragma once

#include <QElapsedTimer>
#include <QMetaType>
#include <QProcess>
#include <QStringList>
#include <QTableWidget>
#include <QWidget>
#include <memory>
#include <vector>

namespace dlssvid {

// Runs dlssvid CLI commands one after another and shows their progress (ТЗ §6: «очередь задач с
// прогрессом»). Progress is parsed from the CLI's "stage: N/M frames" lines.
//
// Stage 9 (MR F): a task can carry the plan of its stages (from AppModel::outcome): reused stages count as done at
// once, the running stage's ms/frame is measured from its own progress lines (the plan's estimate until a couple of
// frames are in), the stages still queued use their estimates — together the overall fraction and the time left.
// A running task can be cancelled (the child process is killed; the passes finished before it keep their manifests
// and are reused by the next run), a queued one is dropped; both finish with ok = false.
class TaskQueue : public QWidget {
    Q_OBJECT
public:
    struct StagePlan {
        QString name;          // depth | flow | upscale | nr | fg | encode
        QString title;         // «Глубина» …
        double seconds = 0.0;  // the estimate when it runs
        bool reused = false;   // done before the task starts
    };
    struct StageStatus {
        QString name, title;
        QString state;         // reused | queued | running | done | cancelled | failed | skipped (never ran: the task failed before it)
        double seconds = 0.0;  // measured when running / done, the estimate when queued
        int percent = 0;
    };
    struct Progress {
        int id = 0;
        QString stage;          // the stage reporting frames now
        int done = 0, total = 0;
        double elapsed = 0.0;   // seconds since the task started
        double eta = -1.0;      // seconds left; -1 when unknown
        double fraction = 0.0;  // overall 0..1 (by stages when the task has a plan, else by frames)
        int stagesDone = 0, stagesTotal = 0;
        std::vector<StageStatus> stages;
    };

    explicit TaskQueue(QWidget* parent = nullptr);
    ~TaskQueue() override;

    // Queues `program args...`; returns the task id. `plan`: the stages of a `process` task, in run order.
    int enqueue(const QString& title, const QString& program, const QStringList& args, const std::vector<StagePlan>& plan = {});
    // The running task is killed, a queued one is dropped; both finish with ok = false.
    void cancel(int id);
    bool busy() const { return current_ >= 0; }
    int currentId() const { return current_ >= 0 ? tasks_[static_cast<size_t>(current_)].id : -1; }
    Progress progress(int id) const;  // the last progress of a task (empty for an unknown id)
    double elapsed(int id) const;     // seconds the task has run so far (live while it runs)
    bool finished(int id) const;
    bool succeeded(int id) const;
    bool cancelled(int id) const;

signals:
    void taskStarted(int id);
    void taskProgress(int id, const Progress& progress);
    void taskFinished(int id, bool ok);
    void taskOutput(int id, const QString& line);

private:
    struct Task {
        int id = 0;
        QString title, program;
        QStringList args;
        int row = 0;
        bool done = false;
        bool ok = false;
        bool cancelled = false;
        std::vector<StagePlan> plan;
        Progress progress;
        QElapsedTimer timer;
        QString stage;           // the stage the last frames line came from
        qint64 stageStartMs = 0;
        int stageFirstDone = 0;  // frames already done when the stage's first line arrived
    };
    void startNext();
    void onOutput();
    void onFinished(int exitCode, QProcess::ExitStatus status);
    void onError(QProcess::ProcessError error);
    void updateProgress(Task& t, const QString& stage, int done, int total);
    void stageDone(Task& t, const QString& stage, double seconds);
    void stageReused(Task& t, const QString& stage);
    void publish(Task& t);  // the row and the taskProgress signal
    void finishTask(Task& t, bool ok, const QString& status);
    double plannedSeconds(const Task& t, const QString& stage) const;
    const Task* find(int id) const;

    QTableWidget* table_;
    std::vector<Task> tasks_;
    int current_ = -1;
    std::unique_ptr<QProcess> process_;
    QString buffer_;
    int nextId_ = 1;
};

}  // namespace dlssvid

Q_DECLARE_METATYPE(dlssvid::TaskQueue::Progress)
