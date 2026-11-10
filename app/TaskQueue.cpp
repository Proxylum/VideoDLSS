#include "TaskQueue.h"

#include <QHeaderView>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

#include "SourceNames.h"
#include "Theme.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

constexpr int kColTitle = 0, kColProgress = 1, kColStatus = 2, kColCancel = 3;
constexpr int kMeasuredFramesMin = 2;  // frames of a stage measured before its own rate replaces the estimate

TaskQueue::StageStatus* FindStage(TaskQueue::Progress& p, const QString& name) {
    for (auto& s : p.stages)
        if (s.name == name) return &s;
    return nullptr;
}

// Stages done (reused ones count) and the overall fraction: by stages when there is a plan, by frames otherwise.
void Recount(TaskQueue::Progress& p, bool finishedOk) {
    p.stagesTotal = static_cast<int>(p.stages.size());
    p.stagesDone = 0;
    double running = 0.0;
    for (const auto& s : p.stages) {
        if (s.state == "reused" || s.state == "done" || s.state == "disabled") ++p.stagesDone;  // a skipped stage is over
        else if (s.state == "running") running += s.percent / 100.0;
    }
    if (finishedOk) p.fraction = 1.0;
    else if (p.stagesTotal > 0) p.fraction = (p.stagesDone + running) / p.stagesTotal;
    else p.fraction = p.total > 0 ? static_cast<double>(p.done) / p.total : 0.0;
}

// What the runner says after a stage's verb: "— <why>; the result is ..." -> <why>.
QString RunnerNote(const QString& rest) {
    static const QRegularExpression lead(R"(^[\s:\-\x{2014}]+)");
    QString r = rest;
    r.remove(lead);
    if (const qsizetype at = r.indexOf(QLatin1String("; the result")); at >= 0) r.truncate(at);
    return r.trimmed();
}

// The time left: what the running stage still needs (-1: unknown) plus the estimates of the queued stages.
double TimeLeft(const TaskQueue::Progress& p, double runningLeft) {
    bool anyRunning = false;
    double queued = 0.0;
    for (const auto& s : p.stages) {
        if (s.state == "running") anyRunning = true;
        else if (s.state == "queued") queued += s.seconds;
    }
    if (anyRunning && runningLeft < 0.0) return -1.0;
    return std::max(0.0, runningLeft) + queued;
}

}  // namespace

QStringList TaskQueue::Progress::failedStageTitles() const {
    QStringList out;
    for (const auto& s : stages)
        if (s.state == "failed") out << s.title;
    return out;
}

QStringList TaskQueue::Progress::fellBackStageTitles() const {
    QStringList out;
    for (const auto& s : stages)
        if (s.fellBack) out << s.title;
    return out;
}

QStringList TaskQueue::Progress::missingStageTitles() const {
    QStringList out;
    for (const auto& s : stages)
        if (s.state == "failed" || s.state == "disabled") out << s.title;
    return out;
}

TaskQueue::TaskQueue(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    table_ = new QTableWidget(0, 4, this);
    table_->setHorizontalHeaderLabels({tr("Задача"), tr("Прогресс"), tr("Статус"), QString()});
    table_->horizontalHeader()->setSectionResizeMode(kColTitle, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(kColProgress, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(kColStatus, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(kColCancel, QHeaderView::Fixed);  // a cell widget: no item to size the column by
    table_->setColumnWidth(kColCancel, 110);
    table_->verticalHeader()->setVisible(false);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(table_);
}

TaskQueue::~TaskQueue() {
    if (process_ && process_->state() != QProcess::NotRunning) {
        process_->disconnect(this);  // no task events while the queue is going away
        process_->kill();
        process_->waitForFinished(2000);
    }
}

int TaskQueue::enqueue(const QString& title, const QString& program, const QStringList& args, const std::vector<StagePlan>& plan, Lane lane) {
    if (lane == Lane::Preview) {
        if (mainBusy()) return -1;  // processing first: a preview after it is the caller's call
        std::vector<int> waiting;   // ids first: a slot of the finish may queue (the vector grows)
        for (const auto& w : tasks_)
            if (w.lane == Lane::Preview && !w.done && (current_ < 0 || tasks_[static_cast<size_t>(current_)].id != w.id)) waiting.push_back(w.id);
        for (const int id : waiting) cancel(id);
    }
    Task t;
    t.id = nextId_++;
    t.lane = lane;
    t.title = title;
    t.program = program;
    t.args = args;
    t.plan = plan;
    t.progress.id = t.id;
    for (const StagePlan& s : plan) {
        StageStatus st;
        st.name = s.name;
        st.title = s.title;
        st.state = s.reused ? "reused" : "queued";
        st.seconds = s.reused ? 0.0 : s.seconds;
        st.percent = s.reused ? 100 : 0;
        t.progress.stages.push_back(st);
    }
    Recount(t.progress, false);
    t.progress.eta = plan.empty() ? -1.0 : TimeLeft(t.progress, -1.0);
    if (lane == Lane::Preview && previewRow_ >= 0) {  // one row for every preview: a debounced series would fill the table
        t.row = previewRow_;
        table_->item(t.row, kColTitle)->setText(title);
    } else {
        t.row = table_->rowCount();
        table_->insertRow(t.row);
        table_->setItem(t.row, kColTitle, new QTableWidgetItem(title));
        table_->setItem(t.row, kColStatus, new QTableWidgetItem());
        auto* bar = new QProgressBar(table_);
        bar->setRange(0, 100);
        table_->setCellWidget(t.row, kColProgress, bar);
        if (lane == Lane::Preview) previewRow_ = t.row;
    }
    if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, kColProgress))) bar->setValue(static_cast<int>(std::lround(t.progress.fraction * 100)));
    table_->item(t.row, kColStatus)->setText(tr("в очереди"));
    auto* cancel = new QPushButton(tr("Отменить"), table_);  // a new one in the preview's row: it cancels this preview
    SetRole(cancel, "small");
    const int id = t.id;
    connect(cancel, &QPushButton::clicked, this, [this, id] { this->cancel(id); });
    table_->setCellWidget(t.row, kColCancel, cancel);
    if (lane == Lane::Preview) previewRowOwner_ = id;
    tasks_.push_back(t);
    if (lane == Lane::Main) {
        emit taskQueued(id);
        // processing goes first: every preview is preempted - the running one killed (this task starts when it has exited and let its
        // files go), a waiting one dropped
        std::vector<int> previews;
        for (const auto& p : tasks_)
            if (p.lane == Lane::Preview && !p.done) previews.push_back(p.id);
        for (const int p : previews) {
            emit previewPreempted(p);
            this->cancel(p);  // the member, not the row's button above
        }
    }
    if (!busy()) startNext();
    return id;
}

bool TaskQueue::mainBusy() const {
    for (const auto& t : tasks_)
        if (t.lane == Lane::Main && !t.done) return true;
    return false;
}

bool TaskQueue::previewBusy() const {
    for (const auto& t : tasks_)
        if (t.lane == Lane::Preview && !t.done) return true;
    return false;
}

TaskQueue::Lane TaskQueue::laneOf(int id) const {
    const Task* t = find(id);
    return t ? t->lane : Lane::Main;
}

void TaskQueue::emitStarted(const Task& t) {
    const int id = t.id;
    if (t.lane == Lane::Main) emit taskStarted(id);
    else emit previewStarted(id);
}

void TaskQueue::emitProgress(const Task& t) {
    const int id = t.id;
    const Progress copy = t.progress;
    if (t.lane == Lane::Main) emit taskProgress(id, copy);
    else emit previewProgress(id, copy);
}

void TaskQueue::emitFinished(const Task& t, bool ok) {
    const int id = t.id;
    if (t.lane == Lane::Main) emit taskFinished(id, ok);
    else emit previewFinished(id, ok);
}

void TaskQueue::emitOutput(const Task& t, const QString& line) {
    const int id = t.id;
    if (t.lane == Lane::Main) emit taskOutput(id, line);
    else emit previewOutput(id, line);
}

void TaskQueue::cancel(int id) {
    for (size_t i = 0; i < tasks_.size(); ++i) {
        Task& t = tasks_[i];
        if (t.id != id || t.done) continue;
        t.cancelled = true;
        if (static_cast<int>(i) == current_) {
            // a console CLI has no window for terminate() to close: kill it; onFinished completes the task
            Log()->info("task {}: cancelled by the user", id);
            if (process_ && process_->state() != QProcess::NotRunning) process_->kill();
        } else {
            Log()->info("task {}: dropped from the queue", id);
            finishTask(t, false, tr("отменена"));
        }
        return;
    }
}

void TaskQueue::stopAll(int waitMs) {
    std::vector<int> waiting;  // first: none of them starts when the running one has ended
    for (const auto& t : tasks_)
        if (!t.done && t.id != currentId()) waiting.push_back(t.id);
    for (const int id : waiting) cancel(id);
    if (current_ < 0) return;
    QProcess* p = process_.get();  // onFinished hands it to deleteLater: alive until the event loop runs again
    cancel(currentId());
    if (p && p->state() != QProcess::NotRunning) p->waitForFinished(waitMs);  // emits finished from here: the task's end is handled now
}

const TaskQueue::Task* TaskQueue::find(int id) const {
    for (const auto& t : tasks_)
        if (t.id == id) return &t;
    return nullptr;
}

TaskQueue::Progress TaskQueue::progress(int id) const {
    const Task* t = find(id);
    return t ? t->progress : Progress{};
}

double TaskQueue::elapsed(int id) const {
    const Task* t = find(id);
    if (!t || !t->timer.isValid()) return 0.0;
    return t->done ? t->progress.elapsed : static_cast<double>(t->timer.elapsed()) / 1000.0;
}

bool TaskQueue::finished(int id) const {
    const Task* t = find(id);
    return t && t->done;
}

bool TaskQueue::succeeded(int id) const {
    const Task* t = find(id);
    return t && t->done && t->ok;
}

bool TaskQueue::cancelled(int id) const {
    const Task* t = find(id);
    return t && t->cancelled;
}

bool TaskQueue::endedAbnormally(int id) const {
    const Task* t = find(id);
    return t && t->abnormal;
}

double TaskQueue::plannedSeconds(const Task& t, const QString& stage) const {
    for (const auto& s : t.plan)
        if (s.name == stage) return s.seconds;
    return 0.0;
}

void TaskQueue::startNext() {
    if (current_ >= 0) return;
    int next = -1;
    for (const Lane lane : {Lane::Main, Lane::Preview}) {  // processing before any preview
        for (size_t i = 0; i < tasks_.size() && next < 0; ++i)
            if (!tasks_[i].done && tasks_[i].lane == lane) next = static_cast<int>(i);
        if (next >= 0) break;
    }
    if (next < 0) return;
    current_ = next;
    Task& t = tasks_[static_cast<size_t>(current_)];
    if (ownsRow(t)) table_->item(t.row, kColStatus)->setText(tr("выполняется"));
    process_ = std::make_unique<QProcess>(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_.get(), &QProcess::readyRead, this, &TaskQueue::onOutput);
    connect(process_.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, &TaskQueue::onFinished);
    connect(process_.get(), &QProcess::errorOccurred, this, &TaskQueue::onError, Qt::QueuedConnection);
    buffer_.clear();
    decoder_.resetState();
    t.timer.start();
    Log()->info("task {}: {} {}", t.id, t.program.toStdString(), t.args.join(' ').toStdString());
    const Task started = t;  // the signals' slots may queue a task: `t` is not used after them
    emitStarted(started);
    emitProgress(started);  // reused stages are marked before the first line of output
    process_->start(started.program, started.args);
    if (tasks_[static_cast<size_t>(current_)].cancelled) process_->kill();  // cancelled from a taskStarted slot
}

void TaskQueue::onOutput() {
    if (!process_ || current_ < 0) return;
    buffer_ += decoder_.decode(process_->readAll());
    // progress lines look like "\rdepth: 12/90 frames" (process) or "\r12/90 frames" (a single stage); the runner
    // also tells when a stage is done or reused
    static const QRegularExpression progressRe(R"((?:([A-Za-z_]+): )?(\d+)/(\d+) frames)");
    // a stage's line without a total (a source that keeps neither a frame count nor a duration): the whole line, so that a log line
    // such as "process: depth done — 7 frames in 0.3 s" is not one (TASK-0070: the GUI knows the encode step by its progress)
    static const QRegularExpression countRe(R"(^([A-Za-z_]+): (\d+) frames$)");
    static const QRegularExpression doneRe(R"(process: ([A-Za-z_]+) done \S+ (\d+) frames in ([0-9.]+) s)");
    static const QRegularExpression reusedRe(R"(process: ([A-Za-z_]+) \S+ .*(is complete, reused|restored from the history))");
    // a stage that failed while the run went on (fg in its own process, audit fg-2); "process: stage 'fg' failed" - the
    // whole run failing - does not match
    static const QRegularExpression failedRe(R"((?:^|\] )process: (depth|flow|upscale|nr|fg) failed\b(.*)$)");
    // TASK-0071: a stage the run went on without, skipped as unavailable, and a stage made with another backend than asked for; the runner
    // prints these after the stage's progress lines
    static const QRegularExpression disabledRe(R"((?:^|\] )process: (depth|flow|upscale|nr|fg) disabled\b(.*)$)");
    static const QRegularExpression fellBackRe(R"((?:^|\] )process: (upscale) fell back to [a-z]+\b(.*)$)");
    static const QRegularExpression newline("[\r\n]");
    QStringList lines = buffer_.split(newline, Qt::SkipEmptyParts);
    if (!buffer_.endsWith('\n') && !buffer_.endsWith('\r') && !lines.isEmpty()) buffer_ = lines.takeLast();
    else buffer_.clear();
    for (const QString& line : lines) {
        Task& t = tasks_[static_cast<size_t>(current_)];  // re-fetched: a slot may have queued a task (vector growth)
        // dlssvid logs the error it ends on as its last line; an error of a child it went on after (the fg process) is followed by more
        if (!line.trimmed().isEmpty()) t.reportedError = line.contains(QLatin1String("] [error] "));
        if (const auto f = failedRe.match(line); f.hasMatch()) {  // first: a progress update may be glued to this line
            stageFailed(t, f.captured(1), RunnerNote(f.captured(2)));
            emitOutput(tasks_[static_cast<size_t>(current_)], line);
            continue;
        }
        if (const auto f = disabledRe.match(line); f.hasMatch()) {
            stageDisabled(t, f.captured(1), RunnerNote(f.captured(2)));
            emitOutput(tasks_[static_cast<size_t>(current_)], line);
            continue;
        }
        if (const auto f = fellBackRe.match(line); f.hasMatch()) {
            stageFellBack(t, f.captured(1), RunnerNote(f.captured(2)));
            emitOutput(tasks_[static_cast<size_t>(current_)], line);
            continue;
        }
        if (const auto m = progressRe.match(line); m.hasMatch()) {
            updateProgress(t, m.captured(1), m.captured(2).toInt(), m.captured(3).toInt());
            continue;
        }
        if (const auto m = countRe.match(line.trimmed()); m.hasMatch()) {
            updateProgress(t, m.captured(1), m.captured(2).toInt(), 0);
            continue;
        }
        if (line.trimmed().isEmpty()) continue;
        if (const auto d = doneRe.match(line); d.hasMatch()) stageDone(t, d.captured(1), d.captured(3).toDouble());
        else if (const auto r = reusedRe.match(line); r.hasMatch()) stageReused(t, r.captured(1));
        emitOutput(tasks_[static_cast<size_t>(current_)], line);
    }
}

void TaskQueue::updateProgress(Task& t, const QString& stage, int done, int total) {
    Progress& p = t.progress;
    const qint64 now = t.timer.elapsed();
    if (stage != t.stage || !FindStage(p, stage)) {
        if (StageStatus* prev = FindStage(p, t.stage); prev && prev->state == "running") {
            prev->state = "done";
            prev->percent = 100;
            prev->seconds = static_cast<double>(now - t.stageStartMs) / 1000.0;
        }
        t.stage = stage;
        t.stageStartMs = now;
        t.stageFirstDone = done;
        if (!FindStage(p, stage)) {
            StageStatus st;
            st.name = stage;
            st.title = stage.isEmpty() ? t.title : stage;
            p.stages.push_back(st);
        }
        FindStage(p, stage)->state = "running";
    }
    StageStatus* cur = FindStage(p, stage);
    cur->percent = total > 0 ? std::clamp(static_cast<int>(static_cast<int64_t>(done) * 100 / total), 0, 100) : 0;
    const double spent = static_cast<double>(now - t.stageStartMs) / 1000.0;
    cur->seconds = spent;
    p.stage = stage;
    p.done = done;
    p.total = total;
    p.elapsed = static_cast<double>(now) / 1000.0;
    // the running stage: its own rate once a couple of frames are measured, the plan's estimate before that
    double left = -1.0;
    if (total > 0) {
        const int measured = done - t.stageFirstDone;
        const double estimate = plannedSeconds(t, stage);
        if (measured >= kMeasuredFramesMin && spent > 0.0) left = spent / measured * static_cast<double>(total - done);
        else if (estimate > 0.0) left = std::max(0.0, estimate * (1.0 - static_cast<double>(done) / total));
    }
    p.eta = TimeLeft(p, left);
    Recount(p, false);
    publish(t);
}

void TaskQueue::stageDone(Task& t, const QString& stage, double seconds) {
    Progress& p = t.progress;
    if (!FindStage(p, stage)) {
        StageStatus st;
        st.name = stage;
        st.title = stage;
        p.stages.push_back(st);
    }
    StageStatus* s = FindStage(p, stage);
    s->state = "done";
    s->percent = 100;
    s->seconds = seconds;
    if (t.stage == stage) t.stage.clear();
    p.elapsed = static_cast<double>(t.timer.elapsed()) / 1000.0;
    p.eta = TimeLeft(p, -1.0);
    Recount(p, false);
    publish(t);
}

void TaskQueue::stageReused(Task& t, const QString& stage) {
    Progress& p = t.progress;
    if (!FindStage(p, stage)) {
        StageStatus st;
        st.name = stage;
        st.title = stage;
        p.stages.push_back(st);
    }
    StageStatus* s = FindStage(p, stage);
    if (s->state == "queued" || s->state == "running") {
        s->state = "reused";
        s->percent = 100;
        s->seconds = 0.0;
    }
    p.eta = TimeLeft(p, -1.0);
    Recount(p, false);
    publish(t);
}

void TaskQueue::stageFailed(Task& t, const QString& stage, const QString& note) { endStage(t, stage, "failed", note); }

void TaskQueue::stageDisabled(Task& t, const QString& stage, const QString& note) { endStage(t, stage, "disabled", note); }

void TaskQueue::stageFellBack(Task& t, const QString& stage, const QString& note) {
    Progress& p = t.progress;
    if (!FindStage(p, stage)) {
        StageStatus st;
        st.name = stage;
        st.title = stage;
        st.state = "done";  // the line comes when the stage has ended
        p.stages.push_back(st);
    }
    StageStatus* s = FindStage(p, stage);
    s->fellBack = true;
    s->note = note;
    publish(t);
}

void TaskQueue::endStage(Task& t, const QString& stage, const QString& state, const QString& note) {
    Progress& p = t.progress;
    if (!FindStage(p, stage)) {
        StageStatus st;
        st.name = stage;
        st.title = stage;
        p.stages.push_back(st);
    }
    StageStatus* s = FindStage(p, stage);
    if (s->state == "running") s->seconds = static_cast<double>(t.timer.elapsed() - t.stageStartMs) / 1000.0;
    s->state = state;
    s->note = note;
    if (t.stage == stage) t.stage.clear();
    p.elapsed = static_cast<double>(t.timer.elapsed()) / 1000.0;
    p.eta = TimeLeft(p, -1.0);
    Recount(p, false);
    publish(t);
}

void TaskQueue::publish(Task& t) {
    const Progress& p = t.progress;
    if (ownsRow(t)) {
        if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, kColProgress))) bar->setValue(static_cast<int>(std::lround(p.fraction * 100)));
        if (!t.done) {
            QString status = tr("выполняется");
            if (p.eta > 0.0) status += tr(" · осталось %1").arg(FormatDuration(p.eta));
            table_->item(t.row, kColStatus)->setText(status);
        }
    }
    emitProgress(t);
}

void TaskQueue::finishTask(Task& t, bool ok, const QString& status) {
    t.done = true;
    t.ok = ok;
    Progress& p = t.progress;
    if (t.timer.isValid()) p.elapsed = static_cast<double>(t.timer.elapsed()) / 1000.0;
    const QString rest = ok ? "done" : t.cancelled ? "cancelled" : "failed";
    for (auto& s : p.stages) {
        if (s.state == "running") {
            if (ok) s.percent = 100;
            s.state = rest;
        } else if (s.state == "queued") {
            s.state = ok ? "done" : t.cancelled ? "cancelled" : "skipped";  // it never ran: not its failure
            s.percent = ok ? 100 : 0;
            s.seconds = 0.0;  // it never reported a frame: nothing measurable to show (the estimate is moot now)
        }
    }
    p.eta = ok ? 0.0 : -1.0;
    Recount(p, ok);
    if (ownsRow(t)) {  // a preview killed for the next one: the row is the next one's already
        table_->item(t.row, kColStatus)->setText(status);
        if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, kColProgress))) bar->setValue(static_cast<int>(std::lround(p.fraction * 100)));
        if (auto* cancel = table_->cellWidget(t.row, kColCancel)) cancel->setEnabled(false);
    }
    const Task finished = t;  // slots may enqueue: `t` is not touched after this
    emitProgress(finished);
    emitFinished(finished, ok);
}

void TaskQueue::onFinished(int exitCode, QProcess::ExitStatus status) {
    if (current_ < 0) return;
    onOutput();
    if (QProcess* p = process_.release()) p->deleteLater();  // not inside its own signal
    const int index = current_;
    current_ = -1;
    Task& t = tasks_[static_cast<size_t>(index)];
    const bool ok = !t.cancelled && status == QProcess::NormalExit && exitCode == 0;
    // dlssvid's own errors end with 1 after an "[error]" line; Task Manager and `taskkill /F` end a process with 1 too, saying nothing
    t.abnormal = status == QProcess::CrashExit || (exitCode != 0 && !(exitCode == 1 && t.reportedError));
    QString text;
    if (t.cancelled) text = tr("отменена");
    else if (ok) text = tr("готово");
    else if (status == QProcess::CrashExit) text = tr("сбой процесса");
    else text = tr("ошибка (%1)").arg(exitCode);
    Log()->info("task {}: {}", t.id, text.toStdString());
    finishTask(t, ok, text);
    startNext();
}

void TaskQueue::onError(QProcess::ProcessError error) {
    if (error != QProcess::FailedToStart || current_ < 0 || !process_) return;
    const QString why = process_->errorString();
    if (QProcess* p = process_.release()) p->deleteLater();
    const int index = current_;
    current_ = -1;
    Task& t = tasks_[static_cast<size_t>(index)];
    Log()->error("task {}: failed to start: {}", t.id, why.toStdString());
    finishTask(t, false, tr("не запускается: %1").arg(why));
    startNext();
}

}  // namespace dlssvid
