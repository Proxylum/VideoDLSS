#pragma once

#include <QElapsedTimer>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QTimer>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "AppModel.h"
#include "pipeline/NrPreview.h"
#include "viewport/ViewportRenderer.h"

namespace dlssvid {

class TaskQueue;

// The NR preview of the card «Улучшение» (PR-10, spec §2f phase 1): a frame (no history, or after 6 frames of it) or a segment of
// 1 / 3 / 5 / 10 s at the cursor, with the card's settings, made by `dlssvid nr` in the Preview lane of the TaskQueue (its own process:
// NGX never runs inside the GUI) into <passes root>/.preview/nr - no pass, no version, the plan and the versions do not change - and
// shown as color_nr@preview on the «after» side. The inputs are exact reuses only («Посчитать этапы до улучшения» computes them with
// `process --no-encode`). A cached preview shows at once; a preview being computed is not killed by a newer request of a frame (the
// last request waits and wins), a segment is. Processing preempts a preview (TaskQueue); another project or passes root drops it.
class PreviewRunner : public QObject {
    Q_OBJECT
public:
    enum class Kind { Frame, FrameHistory, Segment };
    struct Request {
        Kind kind = Kind::Frame;
        int seconds = 0;  // Segment: 1 | 3 | 5 | 10
        bool operator==(const Request&) const = default;
    };
    // What a button would do now: whether it can, the status text and tooltip, the stages to compute first.
    struct Availability {
        bool ok = false;
        QString text, tip;
        QString role = "hint";       // hint | warn | ok (Theme)
        QStringList upstream;        // Upstream / Restore / Pending: the stages «Посчитать этапы до улучшения» runs (titles)
        double upstreamSeconds = 0;  // their estimate
        bool canPrepare = false;     // «Посчитать этапы до улучшения» helps
        double seconds = 0;          // the preview's estimate
        uint64_t bytes = 0;          // its size on disk
        bool cached = false;         // computed already: shows at once
        int64_t frame = 0;           // the source frame at the cursor
        int64_t frames = 0;          // the frames it writes (a segment: cut at the clip's end)
    };

    PreviewRunner(AppModel& model, TaskQueue& tasks, QObject* parent = nullptr);
    ~PreviewRunner() override;

    Availability availability(Request r) const;  // cheap: the plan, the app's blocks and a look in the cache
    bool start(Request r);                        // false: unavailable (statusText says why); a cache hit shows at once, without a task
    void cancel();                                // the running preview and the waiting request
    bool running() const { return taskId_ >= 0; }
    int runningTaskId() const { return taskId_; }
    // «Посчитать этапы до улучшения»: a Main task of `process --stages <enabled of scenes, depth, flow, upscale> --no-encode`; after it
    // the last kind of frame preview runs once. -1 when there is nothing to do or the project cannot be saved.
    int prepareInputs();
    bool isUpstreamTask(int id) const { return id >= 0 && id == upstreamTaskId_; }
    bool upstreamRunning() const { return upstreamTaskId_ >= 0; }
    void cancelUpstream();  // the bar's [Отменить] while «Посчитать этапы до улучшения» runs (upstreamFinished says so)
    void upstreamFinished(int id, bool ok);  // handleTaskFinished, after the sources were reloaded
    void hide();                              // «Убрать»: off the viewport, kept in the cache
    void clearCache();                        // «Очистить превью»: every preview of the project from the disk
    void saveFrame(const QString& file);      // «Сохранить кадр…»: the frame under the cursor (PNG 16 bit, EXR as it is)
    // the same for the preview `dir` and the source `frame` of the click: refused when another preview is shown by now (the dialog)
    void saveFrame(const QString& file, const std::filesystem::path& dir, int64_t frame);
    void setBefore(const QString& which);     // "nr" | "sr": the «before» side
    bool autoUpdate() const { return autoUpdate_; }
    void setAutoUpdate(bool on);              // QSettings card/nr/previewAuto, off by default
    Request lastFrameKind() const { return lastFrameKind_; }
    QString statusText() const { return status_; }
    QString statusTip() const { return tip_; }
    QString statusRole() const { return role_; }
    QString staleReason() const { return stale_; }  // «настройки изменились» ...; empty when the preview shown is current
    NrPreviewUsage usage() const { return usage_; }
    bool housekeeping() const { return housekeeping_ > 0; }

    // tests: whether a mouse button is held (a slider being dragged), the debounce of «обновлять при изменении», the command of a job
    void setPointerHeldForTests(std::function<bool()> held) { pointerHeld_ = std::move(held); }
    void setDebounceMsForTests(int ms);
    void setCommandForTests(std::function<std::pair<QString, QStringList>(const NrPreviewJob&)> command) { command_ = std::move(command); }
    void setFreeSpaceForTests(std::function<uint64_t()> free) { freeSpace_ = std::move(free); }
    void setWatchdogMsForTests(int ms) { watchdogMsForTests_ = ms; }
    void setOnWatchdogForTests(std::function<void()> f) { onWatchdog_ = std::move(f); }  // right after it cancelled the run
    void setPoolGateForTests(std::function<void()> f) { poolGate_ = std::move(f); }  // on the pool's thread before each clean-up
    // MainWindow::onTaskFinished first: the task of «Посчитать этапы до улучшения» - the sources reloaded, the preview after it; true
    // when it was that task (then no force mark is cleared: none was passed, they stay for «Обработать»)
    bool handleTaskFinished(int id, bool ok);

signals:
    void changed();

private:
    NrPreviewEnv env() const;
    uint64_t freeBytes() const;
    NrPreviewMode ModeOf(Request r) const;
    void setStatus(const QString& text, const QString& role = "hint", const QString& tip = {});
    void refuse(const QString& text);  // start()'s refusal: it goes when its condition may have gone (a plan, the mask, a task, the cursor)
    bool liftRefusal();                // that time: the status of the preview shown (or none); false when there was no refusal
    QString blockText(const NrPreviewJob& job, Availability* a) const;  // the gate's refusal in Russian
    // What «Посчитать этапы до улучшения» computes: the enabled stages of scenes, depth, flow, upscale not reused in the preview's plan
    // (their titles), the seconds of those that run, whether any runs (a restore alone takes seconds)
    QStringList upstreamToRun(double* seconds, bool* runs) const;
    QString appBlock(const NrPreviewJob* job) const;                    // the app's: no project, processing, the mask's import, WARP, VFR
    void onStarted(int id);
    void onProgress(int id);
    void onFinished(int id, bool ok);
    void onOutput(int id, const QString& line);
    void runPending();
    void show(const NrPreviewEntry& entry, const NrPreviewJob& job, bool cached, bool seek);
    void updateStale();
    void updateUsage();
    void housekeep(std::function<void()> work, std::function<void()> done);
    void calibrate(const NrPreviewJob& job, double seconds, const NrPreviewEntry& entry);
    void removeWork(const std::filesystem::path& dir);

    AppModel& model_;
    TaskQueue& tasks_;
    // the preview being computed
    std::optional<NrPreviewJob> job_;
    Request jobRequest_;
    int taskId_ = -1;
    uint64_t jobGeneration_ = 0;
    double jobCursor_ = 0.0;  // the time when it was asked: the cursor jumps to its frame only when it has not moved since
    QElapsedTimer jobClock_;
    bool preempted_ = false, watchdogFired_ = false;
    bool jobSuperseded_ = false;  // a newer request was shown from the cache meanwhile: the run fills the cache, it is not shown
    QString lastLine_;  // the CLI's last line of the running preview (its error)
    std::optional<Request> pending_;
    QTimer watchdog_, tick_, throttle_, autoTimer_;
    QElapsedTimer lastStart_;
    // the preview shown
    std::optional<NrPreviewJob> shownJob_;
    std::optional<NrPreviewEntry> shownEntry_;
    Request shownRequest_;
    QString stale_;
    QString uncachedTip_;  // why the preview shown is not in the cache (another DLL, a folder that could not be renamed)
    QString shownStatus_, shownTip_;  // show()'s status: back when the preview is current again
    QString upstreamNote_;  // «Этапы до улучшения посчитаны за …»: said until the preview after them shows
    // the stages before NR
    int upstreamTaskId_ = -1;
    std::optional<Request> afterUpstream_;
    QString upstreamTitles_;
    // state
    Request lastFrameKind_;
    bool everStarted_ = false;  // «обновлять при изменении» follows a preview started by a click
    bool autoUpdate_ = false;
    std::set<std::string> vfrSources_;  // source hashes the CLI refused as variable-frame-rate (this session)
    QString status_, tip_, role_ = "hint";
    bool refused_ = false;
    bool autoStart_ = false;  // start() from «обновлять при изменении»: the preview already on screen is not shown again
    bool pendingAuto_ = false;  // pending_ was parked by an auto start (a clean-up meanwhile): replayed as one
    NrPreviewUsage usage_;
    QThreadPool pool_;  // one thread: GC, «Очистить», «Сохранить кадр»
    int housekeeping_ = 0;  // its jobs queued or running: a new run waits for all of them (a GC removes our work folders not pinned)
    std::function<bool()> pointerHeld_;
    std::function<std::pair<QString, QStringList>(const NrPreviewJob&)> command_;
    std::function<uint64_t()> freeSpace_;
    int watchdogMsForTests_ = 0;
    std::function<void()> onWatchdog_;
    std::function<void()> poolGate_;
};

// What one preview tells of the next ones' estimate: {start of the process up to the first model frame, s; a model frame, ms}. `run` is
// its nr.json: stats.ms_per_frame_wall counts the model's start (diagnostics.init_ms + create_ms) in, so it is taken out; `elapsed` is the
// process's wall time. {-1, -1} without the numbers. With `scanned` (a seek through the video checked first, outside the stage's timer,
// estimated apart by scanBytes) the start is unknown: -1, the frame's time stands.
std::pair<double, double> PreviewCalibration(double elapsed, int64_t modelFrames, const nlohmann::json& run, bool scanned = false);

// «≈ 3 с», «≈ 2 мин», «< 1 с»: FormatDuration with the seconds of a preview (it would say «≈ 0 с»)
QString PreviewDuration(double seconds);

// « · без векторов движения», « · без глубины», « · без векторов движения и глубины»: the guides a frame preview went without (empty: none)
QString NrPreviewDroppedText(const std::vector<std::string>& dropped);

// Why the preview shown (`shown`) is not what the plan would give now (`now`, the same mode and frame): «настройки изменились»,
// «этапы до улучшения нужно пересчитать», «входы пересчитаны», ...; empty when it is current. `planFailed`: the plan has an error and
// no nr decision (an empty plan reads as NR off).
QString NrPreviewStaleReason(const NrPreviewJob& shown, const NrPreviewJob& now, bool planFailed);

// The Disk refusal: over the preview budget (kNrPreviewMaxBytes) or short of free space
QString NrPreviewDiskText(uint64_t estimatedBytes, uint64_t freeBytes);

// The preview's part of a cell label (MainWindow::cellLabel): « · без истории» / « · с историей» / « · отрезок F–L», « · нет кадра превью
// (кадры F–L)» outside its frames, « · устарело», « · без векторов движения» when a frame went without them.
QString NrPreviewCellSuffix(const AppModel::NrPreviewShown& shown, FrameState state, bool stale);

}  // namespace dlssvid
