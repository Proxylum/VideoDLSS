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
        if (s.state == "reused" || s.state == "done") ++p.stagesDone;
        else if (s.state == "running") running += s.percent / 100.0;
    }
    if (finishedOk) p.fraction = 1.0;
    else if (p.stagesTotal > 0) p.fraction = (p.stagesDone + running) / p.stagesTotal;
    else p.fraction = p.total > 0 ? static_cast<double>(p.done) / p.total : 0.0;
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

int TaskQueue::enqueue(const QString& title, const QString& program, const QStringList& args, const std::vector<StagePlan>& plan) {
    Task t;
    t.id = nextId_++;
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
    t.row = table_->rowCount();
    table_->insertRow(t.row);
    table_->setItem(t.row, kColTitle, new QTableWidgetItem(title));
    auto* bar = new QProgressBar(table_);
    bar->setRange(0, 100);
    bar->setValue(static_cast<int>(std::lround(t.progress.fraction * 100)));
    table_->setCellWidget(t.row, kColProgress, bar);
    table_->setItem(t.row, kColStatus, new QTableWidgetItem(tr("в очереди")));
    auto* cancel = new QPushButton(tr("Отменить"), table_);
    SetRole(cancel, "small");
    const int id = t.id;
    connect(cancel, &QPushButton::clicked, this, [this, id] { this->cancel(id); });
    table_->setCellWidget(t.row, kColCancel, cancel);
    tasks_.push_back(t);
    if (!busy()) startNext();
    return id;
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

double TaskQueue::plannedSeconds(const Task& t, const QString& stage) const {
    for (const auto& s : t.plan)
        if (s.name == stage) return s.seconds;
    return 0.0;
}

void TaskQueue::startNext() {
    if (current_ >= 0) return;
    int next = -1;
    for (size_t i = 0; i < tasks_.size(); ++i)
        if (!tasks_[i].done) {
            next = static_cast<int>(i);
            break;
        }
    if (next < 0) return;
    current_ = next;
    Task& t = tasks_[static_cast<size_t>(current_)];
    table_->item(t.row, kColStatus)->setText(tr("выполняется"));
    process_ = std::make_unique<QProcess>(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_.get(), &QProcess::readyRead, this, &TaskQueue::onOutput);
    connect(process_.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, &TaskQueue::onFinished);
    connect(process_.get(), &QProcess::errorOccurred, this, &TaskQueue::onError, Qt::QueuedConnection);
    buffer_.clear();
    t.timer.start();
    Log()->info("task {}: {} {}", t.id, t.program.toStdString(), t.args.join(' ').toStdString());
    const int id = t.id;
    const Progress initial = t.progress;
    emit taskStarted(id);
    emit taskProgress(id, initial);  // reused stages are marked before the first line of output
    process_->start(t.program, t.args);
    if (tasks_[static_cast<size_t>(current_)].cancelled) process_->kill();  // cancelled from a taskStarted slot
}

void TaskQueue::onOutput() {
    if (!process_ || current_ < 0) return;
    buffer_ += QString::fromUtf8(process_->readAll());
    // progress lines look like "\rdepth: 12/90 frames" (process) or "\r12/90 frames" (a single stage); the runner
    // also tells when a stage is done or reused
    static const QRegularExpression progressRe(R"((?:([A-Za-z_]+): )?(\d+)/(\d+) frames)");
    static const QRegularExpression doneRe(R"(process: ([A-Za-z_]+) done \S+ (\d+) frames in ([0-9.]+) s)");
    static const QRegularExpression reusedRe(R"(process: ([A-Za-z_]+) \S+ .*(is complete, reused|restored from the history))");
    static const QRegularExpression newline("[\r\n]");
    QStringList lines = buffer_.split(newline, Qt::SkipEmptyParts);
    if (!buffer_.endsWith('\n') && !buffer_.endsWith('\r') && !lines.isEmpty()) buffer_ = lines.takeLast();
    else buffer_.clear();
    for (const QString& line : lines) {
        Task& t = tasks_[static_cast<size_t>(current_)];  // re-fetched: a slot may have queued a task (vector growth)
        if (const auto m = progressRe.match(line); m.hasMatch()) {
            updateProgress(t, m.captured(1), m.captured(2).toInt(), m.captured(3).toInt());
            continue;
        }
        if (line.trimmed().isEmpty()) continue;
        if (const auto d = doneRe.match(line); d.hasMatch()) stageDone(t, d.captured(1), d.captured(3).toDouble());
        else if (const auto r = reusedRe.match(line); r.hasMatch()) stageReused(t, r.captured(1));
        const int id = t.id;
        emit taskOutput(id, line);
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

void TaskQueue::publish(Task& t) {
    const Progress& p = t.progress;
    if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, kColProgress))) bar->setValue(static_cast<int>(std::lround(p.fraction * 100)));
    if (!t.done) {
        QString status = tr("выполняется");
        if (p.eta > 0.0) status += tr(" · осталось %1").arg(FormatDuration(p.eta));
        table_->item(t.row, kColStatus)->setText(status);
    }
    const int id = t.id;
    const Progress copy = p;
    emit taskProgress(id, copy);
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
    table_->item(t.row, kColStatus)->setText(status);
    if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, kColProgress))) bar->setValue(static_cast<int>(std::lround(p.fraction * 100)));
    if (auto* cancel = table_->cellWidget(t.row, kColCancel)) cancel->setEnabled(false);
    const int id = t.id;
    const Progress copy = p;
    emit taskProgress(id, copy);
    emit taskFinished(id, ok);  // slots may enqueue: `t` is not touched after this
}

void TaskQueue::onFinished(int exitCode, QProcess::ExitStatus status) {
    if (current_ < 0) return;
    onOutput();
    if (QProcess* p = process_.release()) p->deleteLater();  // not inside its own signal
    const int index = current_;
    current_ = -1;
    Task& t = tasks_[static_cast<size_t>(index)];
    const bool ok = !t.cancelled && status == QProcess::NormalExit && exitCode == 0;
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
