#pragma once

#include <QElapsedTimer>
#include <QMetaType>
#include <QProcess>
#include <QStringDecoder>
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
//
// PR-10: two lanes on the one process. Main - processing, «Запустить отдельно», exports, the stages before NR: their events are the
// task* signals. Preview - the NR preview (PreviewRunner): its events are the preview* signals only, so what follows a processing
// task (MainWindow::onTaskFinished: the force marks cleared, the sources reloaded) never follows a preview. A Main task preempts
// every preview (the running one is killed and starts after it exits, its files let go); a preview is refused while a Main task
// runs or waits. All previews share one row of the table.
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
        // reused | queued | running | done | cancelled | failed | skipped (never ran: the task failed before it) | disabled (TASK-0071: the
        // runner skipped it as unavailable - no DLL, an old driver, no NVIDIA GPU - and the run went on without it)
        QString state;
        double seconds = 0.0;  // measured when running / done, the estimate when queued
        int percent = 0;
        // what the runner said of it: why it failed or was skipped; a done stage made with another backend than asked for (upscale fell
        // back to nis). Empty when it said nothing
        QString note;
        bool fellBack = false;  // done with another backend
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
        // stages the runner reported failed while the task went on (`process: fg failed - ...`: the result is encoded
        // without it, audit fg-2); their titles, in run order
        QStringList failedStageTitles() const;
        // the stages the result is made without: failed, or skipped as unavailable (`process: nr disabled - ...`, TASK-0071)
        QStringList missingStageTitles() const;
        // the stages made with another backend than asked for (`process: upscale fell back to nis - ...`)
        QStringList fellBackStageTitles() const;
    };

    enum class Lane { Main, Preview };

    explicit TaskQueue(QWidget* parent = nullptr);
    ~TaskQueue() override;

    // Queues `program args...`; returns the task id. `plan`: the stages of a `process` task, in run order. A Main task preempts the
    // previews (previewPreempted, then cancelled); a Preview task while a Main one runs or waits is refused (-1) and drops a preview
    // still waiting.
    int enqueue(const QString& title, const QString& program, const QStringList& args, const std::vector<StagePlan>& plan = {}, Lane lane = Lane::Main);
    // The running task is killed, a queued one is dropped; both finish with ok = false.
    void cancel(int id);
    // Every task: the waiting ones dropped, the running one killed and waited for (up to waitMs), so that the ends of all are handled
    // before the caller goes on - the window closing (TASK-0070: a run killed while it encoded leaves no file behind).
    void stopAll(int waitMs);
    bool busy() const { return current_ >= 0; }  // a process of either lane runs
    bool mainBusy() const;                       // a Main task runs or waits
    bool previewBusy() const;                    // a Preview task runs or waits
    Lane laneOf(int id) const;                   // Main for an unknown id
    int currentId() const { return current_ >= 0 ? tasks_[static_cast<size_t>(current_)].id : -1; }
    QString unterminatedOutput() const { return buffer_.trimmed(); }  // of the process that runs or has just ended: what it wrote since its last line end
    Progress progress(int id) const;  // the last progress of a task (empty for an unknown id)
    double elapsed(int id) const;     // seconds the task has run so far (live while it runs)
    bool finished(int id) const;
    bool succeeded(int id) const;
    bool cancelled(int id) const;
    // Its process crashed or was ended from outside (Task Manager, `taskkill /F`): any exit code but 0, and dlssvid's 1 when its last
    // line was the "[error]" it logs for the error it handles and ends on. Nothing it meant to clean up was (TASK-0070).
    bool endedAbnormally(int id) const;

signals:
    // Main tasks
    void taskQueued(int id);  // from enqueue, before the task can start or end
    void taskStarted(int id);
    void taskProgress(int id, const Progress& progress);
    void taskFinished(int id, bool ok);
    void taskOutput(int id, const QString& line);
    // Preview tasks (PR-10)
    void previewStarted(int id);
    void previewProgress(int id, const Progress& progress);
    void previewFinished(int id, bool ok);
    void previewOutput(int id, const QString& line);
    void previewPreempted(int id);  // a Main task took over: emitted before the preview is cancelled

private:
    struct Task {
        int id = 0;
        Lane lane = Lane::Main;
        QString title, program;
        QStringList args;
        int row = 0;
        bool done = false;
        bool ok = false;
        bool cancelled = false;
        bool abnormal = false;       // endedAbnormally
        bool reportedError = false;  // the last line was an "[error]" of dlssvid's log
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
    void stageFailed(Task& t, const QString& stage, const QString& note = {});
    void stageDisabled(Task& t, const QString& stage, const QString& note);
    void stageFellBack(Task& t, const QString& stage, const QString& note);
    void endStage(Task& t, const QString& stage, const QString& state, const QString& note);
    void publish(Task& t);  // the row and the taskProgress signal
    // the signals of a task's lane
    void emitStarted(const Task& t);
    void emitProgress(const Task& t);
    void emitFinished(const Task& t, bool ok);
    void emitOutput(const Task& t, const QString& line);
    bool ownsRow(const Task& t) const { return t.lane == Lane::Main || previewRowOwner_ == t.id; }  // a late end of a killed preview
    void finishTask(Task& t, bool ok, const QString& status);
    double plannedSeconds(const Task& t, const QString& stage) const;
    const Task* find(int id) const;

    QTableWidget* table_;
    std::vector<Task> tasks_;
    int current_ = -1;
    std::unique_ptr<QProcess> process_;
    QString buffer_;
    // the process's output as UTF-8 across reads: a character split between two of them (the em dash of "stage disabled — ") stays whole
    QStringDecoder decoder_{QStringDecoder::Utf8};
    int nextId_ = 1;
    int previewRow_ = -1;       // the one row of the previews
    int previewRowOwner_ = -1;  // the preview that writes it
};

}  // namespace dlssvid

Q_DECLARE_METATYPE(dlssvid::TaskQueue::Progress)
