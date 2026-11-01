#include "PreviewRunner.h"

#include <QCoreApplication>
#include <QGuiApplication>
#include <QLocale>
#include <QSettings>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>

#include "ControlMaskText.h"
#include "SourceNames.h"
#include "TaskQueue.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "stages/nr/INrBackend.h"
#include "pipeline/ProcessRunner.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

namespace fs = std::filesystem;

constexpr int kDebounceMs = 600;        // «обновлять при изменении»: after the last change (a slider: after it is released)
constexpr int kMinStartGapMs = 1500;    // a CLI failing at once does not become a series of create / kill

QString Tr(const char* s) { return QCoreApplication::translate("PreviewRunner", s); }

// «за 2,4 с», «за ≈ 1 мин»: a preview takes seconds - FormatDuration would say «≈ 0 с»
QString Took(double seconds) {
    if (seconds < 60.0) return Tr("за %1 с").arg(QLocale(QLocale::Russian).toString(seconds, 'f', 1));
    return Tr("за %1").arg(FormatDuration(seconds));
}

const StageDecision* Decision(const ProcessPlan& plan, const std::string& stage) {
    for (const auto& d : plan.stages)
        if (d.name == stage) return &d;
    return nullptr;
}

QString StageTitles(AppModel& model, const std::vector<std::string>& stages) {
    QStringList t;
    for (const auto& s : stages) t << model.stageTitle(s);
    return t.join(", ");
}

nlohmann::json ReadJson(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return nullptr;
    try {
        return nlohmann::json::parse(in);
    } catch (const std::exception&) {
        return nullptr;
    }
}

}  // namespace

PreviewRunner::PreviewRunner(AppModel& model, TaskQueue& tasks, QObject* parent) : QObject(parent), model_(model), tasks_(tasks) {
    pool_.setMaxThreadCount(1);
    autoUpdate_ = QSettings().value("card/nr/previewAuto", false).toBool();
    watchdog_.setSingleShot(true);
    tick_.setInterval(1000);
    throttle_.setSingleShot(true);
    autoTimer_.setSingleShot(true);
    autoTimer_.setInterval(kDebounceMs);
    connect(&watchdog_, &QTimer::timeout, this, [this] {  // a run that does not end (a driver hang): stopped
        if (taskId_ < 0) return;
        watchdogFired_ = true;
        pending_.reset();  // asked before the hang; one asked while the killed process exits runs after it
        tasks_.cancel(taskId_);
        if (onWatchdog_) onWatchdog_();
    });
    connect(&tick_, &QTimer::timeout, this, [this] { onProgress(taskId_); });
    connect(&throttle_, &QTimer::timeout, this, &PreviewRunner::runPending);
    connect(&autoTimer_, &QTimer::timeout, this, [this] {
        const bool held = pointerHeld_ ? pointerHeld_() : QGuiApplication::mouseButtons() != Qt::NoButton;
        if (held) {  // a slider being dragged: its release is the change
            autoTimer_.start();
            return;
        }
        if (!autoUpdate_ || !everStarted_ || !model_.hasProject() || tasks_.mainBusy() || model_.controlMaskBusy()) return;  // silent meanwhile
        autoStart_ = true;
        start(lastFrameKind_);
        autoStart_ = false;
    });
    connect(&tasks_, &TaskQueue::previewStarted, this, &PreviewRunner::onStarted);
    connect(&tasks_, &TaskQueue::previewProgress, this, [this](int id, const TaskQueue::Progress&) { onProgress(id); });
    connect(&tasks_, &TaskQueue::previewFinished, this, &PreviewRunner::onFinished);
    connect(&tasks_, &TaskQueue::previewOutput, this, &PreviewRunner::onOutput);
    connect(&tasks_, &TaskQueue::previewPreempted, this, [this](int id) {
        if (id == taskId_) preempted_ = true;
        pending_.reset();
    });
    connect(&tasks_, &TaskQueue::taskProgress, this, [this](int id, const TaskQueue::Progress& p) {
        if (id != upstreamTaskId_) return;
        setStatus(Tr("Этапы до улучшения: %1 % · осталось %2").arg(static_cast<int>(std::lround(p.fraction * 100))).arg(p.eta > 0 ? PreviewDuration(p.eta) : Tr("…")));
    });
    // another project or passes root: the preview in flight is cancelled, its result dropped (the generation), the cache counted again
    connect(&model_, &AppModel::passesContextChanged, this, [this] {
        cancel();
        shownJob_.reset();
        shownEntry_.reset();
        stale_.clear();
        upstreamTaskId_ = -1;
        afterUpstream_.reset();
        everStarted_ = false;  // auto update follows a click in this project (design §3.3): projectChanged comes next
        setStatus({});
        QMetaObject::invokeMethod(this, [this] { updateUsage(); }, Qt::QueuedConnection);  // the new root is set after the signal
    });
    connect(&model_, &AppModel::planChanged, this, &PreviewRunner::updateStale);
    // a refusal («Идёт обработка …», «Загружается маска …», «Курсор за концом клипа») holds while its condition may: then the bar
    // says what a click would do now
    connect(&model_, &AppModel::planChanged, this, &PreviewRunner::liftRefusal);
    connect(&model_, &AppModel::controlMaskChanged, this, &PreviewRunner::liftRefusal);
    connect(&model_, &AppModel::timeChanged, this, &PreviewRunner::liftRefusal);
    connect(&tasks_, &TaskQueue::taskFinished, this, [this](int, bool) {
        if (!liftRefusal()) emit changed();  // processing over: the bar's buttons come back
    });
    connect(&tasks_, &TaskQueue::taskStarted, this, [this](int) { emit changed(); });  // processing: the bar says so, its buttons wait
    auto edited = [this] {
        if (autoUpdate_ && everStarted_) autoTimer_.start();
    };
    connect(&model_, &AppModel::projectChanged, this, edited);
    connect(&model_, &AppModel::controlMaskChanged, this, edited);
    updateUsage();
}

PreviewRunner::~PreviewRunner() { pool_.waitForDone(); }

void PreviewRunner::setDebounceMsForTests(int ms) { autoTimer_.setInterval(ms); }

void PreviewRunner::setAutoUpdate(bool on) {
    autoUpdate_ = on;
    QSettings().setValue("card/nr/previewAuto", on);
    if (!on) autoTimer_.stop();
    emit changed();
}

void PreviewRunner::setStatus(const QString& text, const QString& role, const QString& tip) {
    refused_ = false;
    status_ = text;
    role_ = role;
    tip_ = tip;
    emit changed();
}

void PreviewRunner::refuse(const QString& text) {
    setStatus(text, "warn");
    refused_ = true;
}

bool PreviewRunner::liftRefusal() {
    if (!refused_) return false;
    if (model_.nrPreview() && shownJob_ && !stale_.isEmpty()) setStatus(Tr("Показано превью для прежних настроек — %1").arg(stale_), "warn");
    else if (model_.nrPreview() && shownJob_) setStatus(shownStatus_, "ok", shownTip_);
    else setStatus({});
    return true;
}

NrPreviewMode PreviewRunner::ModeOf(Request r) const {
    return r.kind == Kind::Frame ? NrPreviewMode::Frame : r.kind == Kind::FrameHistory ? NrPreviewMode::History : NrPreviewMode::Segment;
}

NrPreviewEnv PreviewRunner::env() const {
    NrPreviewEnv e;
    const Project& p = model_.project();
    e.source = p.sourceVideo;
    e.cli = std::filesystem::path(model_.cliPath().toStdWString());
    e.warp = model_.device().IsWarp();
    e.freeBytes = freeBytes();
    e.clipFrames = model_.sourceInfo().frames > 0 ? model_.sourceInfo().frames : -1;
    e.fps = model_.sourceInfo().fps > 0 ? model_.sourceInfo().fps : 24.0;
    // the colour NR reads: the result frame's size (the upscale's, else the source's), as the plan's estimate and the calibration key
    e.colorW = model_.outcome().width > 0 ? model_.outcome().width : model_.sourceInfo().width;
    e.colorH = model_.outcome().height > 0 ? model_.outcome().height : model_.sourceInfo().height;
    for (const auto& s : model_.outcome().stages)
        if (s.stage == "nr") e.msPerFrame = s.msPerFrame;
    // the start: the process, the device, NGX and the feature (E8 ≈ 1.6-2 s), the stub's or the bypass's much less; then what the
    // previews of this size measured
    const StageDecision* nr = Decision(model_.previewPlan(), "nr");
    const bool ngx = nr && nr->tool.is_object() && nr->tool.value("backend", std::string()) == "ngx" && !NrBypassParams(nr->params);
    const QString size = QString("%1x%2").arg(e.colorW).arg(e.colorH);
    e.startSeconds = QSettings().value(QString("preview/nr/%1/%2/start_s").arg(ngx ? "ngx" : "other", size), ngx ? 2.4 : 0.5).toDouble();  // benchmarks.md: 2.4 s at 1440p
    if (const double ms = QSettings().value(QString("preview/nr/%1/%2/frame_ms").arg(ngx ? "ngx" : "other", size), 0.0).toDouble(); ms > 0) e.msPerFrame = ms;
    return e;
}

uint64_t PreviewRunner::freeBytes() const { return freeSpace_ ? freeSpace_() : model_.freeSpace(); }

QString PreviewRunner::appBlock(const NrPreviewJob* job) const {
    if (!model_.hasProject()) return Tr("Превью недоступно: нет проекта");
    if (!model_.planError().isEmpty()) return Tr("Превью недоступно: %1").arg(model_.planError());
    if (tasks_.mainBusy()) return Tr("Идёт обработка — превью будет доступно после неё");
    if (model_.controlMaskBusy()) return Tr("Загружается маска управления — подождите");
    const StageDecision* nr = Decision(model_.previewPlan(), "nr");
    if (nr && model_.device().IsWarp() && nr->tool.is_object() && nr->tool.value("backend", std::string()) == "ngx" &&
        !NrBypassParams(nr->params))
        return Tr("Нужна видеокарта NVIDIA RTX: сейчас программный адаптер");
    // a source the CLI refused as variable-frame-rate: no seek into it through the video (the colour pass reads by index)
    if (job && vfrSources_.count(job->sourceHash) && !job->dirs.count("color_sr") && job->first - job->warmup > kSequentialSeekFrames)
        return Tr("У исходника переменная частота кадров: кадр с середины клипа не найти перемоткой. Включите и посчитайте «Апскейл» — "
                  "превью пойдёт по его пассу — или обработайте клип целиком");
    return {};
}

QString PreviewRunner::blockText(const NrPreviewJob& job, Availability* a) const {
    const NrPreviewGate& g = job.gate;
    AppModel& m = model_;
    switch (g.why) {
        case NrPreviewBlock::None: return {};
        case NrPreviewBlock::NrOff: return Tr("Стадия «Улучшение» выключена");
        case NrPreviewBlock::NrError: return Tr("Превью недоступно: %1").arg(QString::fromStdString(g.detail));
        case NrPreviewBlock::NoDll: return Tr("Нет nvngx_dlssnr.dll — превью недоступно (см. строку готовности на стартовом экране)");
        case NrPreviewBlock::ControlMask: return Tr("Маска управления не читается: %1").arg(ControlMaskWhyText(g.detail));
        case NrPreviewBlock::PastEnd: return Tr("Курсор за концом клипа");
        case NrPreviewBlock::Disk: return NrPreviewDiskText(job.estimatedBytes, freeBytes());
        case NrPreviewBlock::Upstream:
        case NrPreviewBlock::Restore:
        case NrPreviewBlock::Pending: {
            // what the click computes - every stage before NR not reused, not only the ones the gate stopped at (a frame drops
            // the depth and the vectors it can go without, the click computes them all the same)
            double seconds = 0;
            bool runs = false;
            QStringList titles = upstreamToRun(&seconds, &runs);
            if (titles.isEmpty())
                for (const auto& s : g.stages) titles << m.stageTitle(s);
            if (a) {
                a->canPrepare = true;
                a->upstreamSeconds = seconds;
                a->upstream = titles;
            }
            if (g.why == NrPreviewBlock::Restore && !runs)
                return Tr("Нужная версия «%1» лежит в истории — «Посчитать этапы до улучшения» вернёт её за секунды").arg(StageTitles(m, g.stages));
            if (g.why == NrPreviewBlock::Pending) return Tr("Склейки ещё не найдены — сначала посчитайте этапы до улучшения");
            return Tr("Сначала посчитайте этапы до улучшения: %1 — по всему клипу, %2").arg(titles.join(", "), PreviewDuration(seconds));
        }
    }
    return {};
}

QStringList PreviewRunner::upstreamToRun(double* seconds, bool* runs) const {
    QStringList titles;
    for (const char* name : {"scenes", "depth", "flow", "upscale"}) {  // as prepareInputs passes them to process --stages
        const StageEntry* e = model_.stageEntry(name);
        const StageDecision* d = Decision(model_.previewPlan(), name);
        if (!e || !e->enabled || !d || d->action == "reuse") continue;
        titles << model_.stageTitle(name);
        if (d->action != "run") continue;
        if (runs) *runs = true;
        for (const auto& s : model_.outcome().stages)
            if (s.stage == name && seconds) *seconds += s.seconds;
    }
    return titles;
}

PreviewRunner::Availability PreviewRunner::availability(Request r) const {
    Availability a;
    if (!model_.hasProject()) {
        a.text = appBlock(nullptr);
        return a;
    }
    a.frame = model_.store().FrameAt("source", model_.time());
    if (const QString b = appBlock(nullptr); !b.isEmpty()) {
        a.text = b;
        a.role = "warn";
        return a;
    }
    const NrPreviewJob job = PrepareNrPreview(model_.previewPlan(), ModeOf(r), a.frame, r.seconds, env());
    a.seconds = job.estimatedSeconds;
    a.bytes = job.estimatedBytes;
    a.frames = job.frames;
    if (!job.gate.ok() && !(job.gate.why == NrPreviewBlock::Disk && FindNrPreview(job))) {  // a cached one needs no room
        a.text = blockText(job, &a);
        a.role = "warn";
        return a;
    }
    if (const QString b = appBlock(&job); !b.isEmpty()) {
        a.text = b;
        a.role = "warn";
        return a;
    }
    a.ok = true;
    a.cached = FindNrPreview(job).has_value();
    a.text = a.cached ? Tr("Кадр %1 · уже посчитан — покажется сразу").arg(a.frame) : Tr("Кадр %1 · %2").arg(a.frame).arg(PreviewDuration(a.seconds));
    return a;
}

bool PreviewRunner::start(Request r) {
    if (r.kind != Kind::Segment) lastFrameKind_ = r;
    everStarted_ = true;
    model_.refreshPlanIfPending();  // an edit 250 ms ago: the preview is of the settings shown
    if (!autoStart_) autoTimer_.stop();  // a click is of every edit so far: their debounced auto request would only replace it
    if (housekeeping_) {  // a clean-up of the cache is moving folders: right after it (an auto request does not replace a click)
        if (!autoStart_ || !pending_ || pendingAuto_) {
            pending_ = r;
            pendingAuto_ = autoStart_;
        }
        return true;
    }
    if (const QString b = appBlock(nullptr); !b.isEmpty()) {
        refuse(b);
        return false;
    }
    const int64_t frame = model_.store().FrameAt("source", model_.time());
    NrPreviewJob job = PrepareNrPreview(model_.previewPlan(), ModeOf(r), frame, r.seconds, env());
    if (!job.gate.ok() && !(job.gate.why == NrPreviewBlock::Disk && FindNrPreview(job))) {  // a cached one needs no room
        refuse(blockText(job, nullptr));
        return false;
    }
    if (const QString b = appBlock(&job); !b.isEmpty()) {
        refuse(b);
        return false;
    }
    const bool dying = job_ && tasks_.cancelled(taskId_);  // cancelled, its process not gone yet: it serves nothing
    // an auto request that changes nothing leaves a click waiting behind the run (design §3.3 step 4: nothing to do)
    const bool keepClick = autoStart_ && pending_ && !pendingAuto_;
    if (job_ && !dying && job_->key == job.key) {  // this very preview is being computed: it is the last request again
        if (!keepClick) pending_.reset();
        jobSuperseded_ = false;
        return true;
    }
    if (autoStart_ && shownJob_ && shownEntry_ && !shownEntry_->cacheable && model_.nrPreview() && shownJob_->key == job.key) {
        if (job_ && !dying && PrepareNrPreview(model_.previewPlan(), job_->mode, job_->first, jobRequest_.seconds, env()).key == job_->key)
            return true;  // an edit that changed neither the preview shown nor the run in flight: both stay
        if (!keepClick) pending_.reset();  // the preview on screen is of this key, only not in the cache: an edit outside it changes nothing
        if (job_) {
            jobSuperseded_ = true;
            if (jobRequest_.kind == Kind::Segment && !dying) tasks_.cancel(taskId_);
            setStatus(shownStatus_, "ok", shownTip_);
        }
        return true;
    }
    if (const auto hit = FindNrPreview(job)) {  // computed before: shown at once, no task
        const bool onScreen = autoStart_ && model_.nrPreview() && model_.nrPreview()->dir == hit->dir;
        if (onScreen && job_ && !dying &&
            PrepareNrPreview(model_.previewPlan(), job_->mode, job_->first, jobRequest_.seconds, env()).key == job_->key)
            return true;  // an edit that changed neither the preview shown nor the run in flight: both stay
        if (job_) {
            jobSuperseded_ = true;  // the run in flight is an older request: it fills the cache, it is not shown (the last one wins)
            if (jobRequest_.kind == Kind::Segment && !dying) tasks_.cancel(taskId_);
        }
        if (!(onScreen && keepClick)) pending_.reset();
        shownRequest_ = r;
        if (onScreen) {  // an edit that left the key: nothing to show anew
            if (job_) setStatus(shownStatus_, "ok", shownTip_);                       // the superseded run's progress had the status
        } else {
            show(*hit, job, true, true);
        }
        return true;
    }
    if (!job.gate.ok()) {  // over the disk budget and gone from the cache meanwhile
        refuse(blockText(job, nullptr));
        return false;
    }
    if (job_) {
        pending_ = r;  // the last request wins
        pendingAuto_ = autoStart_;
        if (jobRequest_.kind == Kind::Segment && !dying) tasks_.cancel(taskId_);  // a segment is long: it goes; a frame finishes (NGX's create)
        return true;
    }
    if (lastStart_.isValid() && lastStart_.elapsed() < kMinStartGapMs) {
        pending_ = r;
        pendingAuto_ = autoStart_;
        throttle_.start(static_cast<int>(kMinStartGapMs - lastStart_.elapsed()));
        return true;
    }
    try {
        WriteNrPreviewParams(job);
    } catch (const std::exception& e) {
        setStatus(Tr("Превью не удалось — подробности в логе"), "warn", QString::fromUtf8(e.what()));
        Log()->error("nr preview: {}", e.what());
        return false;
    }
    const auto [program, args] = command_ ? command_(job) : std::pair<QString, QStringList>{model_.cliPath(), [&] {
        QStringList l;
        for (const auto& a : job.Args()) l << QString::fromStdString(a);
        return l;
    }()};
    QString title = r.kind == Kind::Segment ? Tr("Превью улучшения · кадры %1–%2").arg(job.first).arg(job.first + job.frames - 1)
                                            : Tr("Превью улучшения · кадр %1 · %2").arg(job.first).arg(r.kind == Kind::Frame ? Tr("без истории") : Tr("с историей"));
    const int id = tasks_.enqueue(title, program, args, {{"", Tr("Превью улучшения"), job.estimatedSeconds, false}}, TaskQueue::Lane::Preview);
    if (id < 0) {
        removeWork(job.workDir);
        refuse(Tr("Идёт обработка — превью будет доступно после неё"));
        return false;
    }
    job_ = job;
    jobRequest_ = r;
    taskId_ = id;
    jobGeneration_ = model_.generation();
    jobCursor_ = model_.time();
    jobClock_.start();
    lastStart_.start();
    preempted_ = watchdogFired_ = jobSuperseded_ = false;
    lastLine_.clear();
    // a hang, not a slow disk: the demux before a seek gets the time of an external HDD on top
    watchdog_.start(watchdogMsForTests_ > 0 ? watchdogMsForTests_
                                            : static_cast<int>((std::max(30.0, 3.0 * job.estimatedSeconds + 30.0) + static_cast<double>(job.scanBytes) / kNrPreviewScanRateSlow) * 1000.0));
    tick_.start();
    Log()->info("nr preview: {} {} at frame {} ({} frames, warm-up {}), key {}", ToString(job.mode), id, job.first, job.frames, job.warmup, job.key);
    onProgress(id);
    return true;
}

void PreviewRunner::onStarted(int id) {
    if (id == taskId_) onProgress(id);
}

void PreviewRunner::onProgress(int id) {
    if (id < 0 || id != taskId_ || !job_ || jobSuperseded_ || jobGeneration_ != model_.generation()) return;  // superseded: the shown one's status
    const TaskQueue::Progress p = tasks_.progress(id);
    const double elapsed = jobClock_.isValid() ? static_cast<double>(jobClock_.elapsed()) / 1000.0 : 0.0;
    const double left = std::max(0.0, job_->estimatedSeconds - elapsed);
    const QString remain = left > 0.5 ? PreviewDuration(left) : Tr("секунды");
    if (jobRequest_.kind == Kind::Segment) {
        const int percent = p.total > 0 ? static_cast<int>(p.done * 100 / p.total) : 0;
        setStatus(Tr("Превью отрезка %1–%2 · %3 % · осталось %4").arg(job_->first).arg(job_->first + job_->frames - 1).arg(percent).arg(remain));
    } else if (p.done == 0) {
        if (!upstreamNote_.isEmpty()) setStatus(Tr("%1 — считаю превью кадра %2 · осталось %3").arg(upstreamNote_).arg(job_->first).arg(remain), "ok");
        else setStatus(Tr("Превью кадра %1 · запуск модели · осталось %2").arg(job_->first).arg(remain));
    } else {
        setStatus(Tr("Превью кадра %1 · %2 · осталось %3").arg(job_->first).arg(jobRequest_.kind == Kind::Frame ? Tr("без истории") : Tr("с историей"), remain));
    }
}

void PreviewRunner::onOutput(int id, const QString& line) {
    if (id == taskId_ && !line.trimmed().isEmpty()) lastLine_ = line.trimmed();
}

void PreviewRunner::removeWork(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);  // a held file: left to the clean-up (another process's folder older than 10 min, ours at the next GC)
    if (ec) Log()->warn("nr preview: cannot remove {}: {}", dir.string(), ec.message());
}

void PreviewRunner::onFinished(int id, bool ok) {
    if (id != taskId_ || !job_) return;
    const NrPreviewJob job = *job_;
    const Request request = jobRequest_;
    const bool superseded = jobSuperseded_;  // a newer request is on screen: the statuses are its own
    const double seconds = jobClock_.isValid() ? static_cast<double>(jobClock_.elapsed()) / 1000.0 : 0.0;
    jobSuperseded_ = false;
    upstreamNote_.clear();  // said while the run started; its end says what it gave
    job_.reset();
    taskId_ = -1;
    watchdog_.stop();
    tick_.stop();
    emit changed();  // not running: [Отменить] and «Очистить превью» follow, whatever the branch below says
    if (jobGeneration_ != model_.generation() || job.root != model_.project().passesRoot) {  // another project or root since: dropped
        removeWork(job.workDir);
        runPending();  // a request made since is of the project open now (the switch cleared the status and the older ones)
        return;
    }
    if (tasks_.cancelled(id)) {
        removeWork(job.workDir);
        if (preempted_) {  // the requests before the preemption went with it (previewPreempted); one made since runs below
            if (!superseded && !pending_) setStatus(Tr("Превью отменено: началась обработка"));
        } else if (watchdogFired_) {
            if (!superseded) setStatus(Tr("Превью не ответило за %1 — остановлено").arg(FormatDuration(seconds)), "warn");
            else Log()->warn("nr preview {}: stopped after {}", id, FormatDuration(seconds).toStdString());
        } else if (!pending_ && !superseded) {
            setStatus(Tr("Превью отменено"));
        }
        runPending();
        return;
    }
    if (!ok) {
        const nlohmann::json run = ReadJson(job.workDir / "nr.json");
        if (run.is_object() && run.value("code", std::string()) == "vfr") vfrSources_.insert(job.sourceHash);
        removeWork(job.workDir);
        QString text = Tr("Превью не удалось — подробности в логе");
        if (const QString vfr = appBlock(&job); !vfr.isEmpty() && vfrSources_.count(job.sourceHash)) text = vfr;  // the source cannot be sought
        if (!superseded) setStatus(text, "warn", lastLine_);
        Log()->warn("nr preview {} failed: {}", id, lastLine_.toStdString());
        runPending();
        return;
    }
    const NrPreviewVerdict v = VerifyNrPreviewRun(job);
    if (!v.ok()) {
        removeWork(job.workDir);
        QString why;
        switch (v.why) {
            case NrPreviewReject::InputChanged: why = Tr("входы изменились во время расчёта"); break;
            case NrPreviewReject::ControlMask: why = Tr("маска управления изменилась во время расчёта"); break;
            case NrPreviewReject::Frames: why = Tr("записано %1 из %2 кадров").arg(v.written).arg(job.frames); break;
            default: why = Tr("нет сводки прогона"); break;
        }
        Log()->warn("nr preview {} discarded: {}", id, v.detail);
        if (!superseded) setStatus(Tr("Превью отброшено: %1").arg(why), "warn", QString::fromStdString(v.detail));
        runPending();
        return;
    }
    NrPreviewEntry entry;
    try {
        entry = PublishNrPreview(job, seconds, v.cacheable);
    } catch (const std::exception& e) {
        removeWork(job.workDir);
        Log()->error("nr preview: {}", e.what());
        if (!superseded) setStatus(Tr("Превью не удалось — подробности в логе"), "warn", QString::fromUtf8(e.what()));
        runPending();
        return;
    }
    calibrate(job, seconds, entry);
    if (superseded) {
        Log()->info("nr preview: {} computed in {:.2f} s, not shown: a newer one is", entry.key, seconds);
    } else {
        shownRequest_ = request;
        uncachedTip_ = v.cacheable ? Tr("не сохранено в кэш: папку превью не удалось переименовать (занята) — подробности в логе")
                                   : Tr("не сохранено в кэш: у CLI другая nvngx_dlssnr.dll");
        const bool seek = std::abs(model_.time() - jobCursor_) < 1e-9 && !model_.playing();  // the cursor was not moved meanwhile
        show(entry, job, false, seek);
        updateStale();  // a setting changed while it ran: shown as a preview of the earlier settings
        Log()->info("nr preview: shown {} in {:.2f} s ({})", entry.key, seconds, entry.cacheable ? "cached" : "not cached");
    }
    // the clean-up keeps the entry just made and the one shown; a new run waits for it (housekeeping_)
    const fs::path root = job.root;
    std::vector<fs::path> pinned{entry.dir};
    if (shownEntry_) pinned.push_back(shownEntry_->dir);
    housekeep([root, pinned] { GcNrPreviews(root, pinned); }, [] {});
    runPending();
}

void PreviewRunner::runPending() {
    if (!pending_ || job_ || housekeeping_) return;
    if (lastStart_.isValid() && lastStart_.elapsed() < kMinStartGapMs && !throttle_.isActive()) {
        throttle_.start(static_cast<int>(kMinStartGapMs - lastStart_.elapsed()));
        return;
    }
    if (throttle_.isActive()) return;
    const Request r = *pending_;
    pending_.reset();
    autoStart_ = pendingAuto_;  // an auto request waited: it stays one
    start(r);
    autoStart_ = false;
}

void PreviewRunner::show(const NrPreviewEntry& entry, const NrPreviewJob& job, bool cached, bool seek) {
    AppModel::NrPreviewShown s;
    s.dir = entry.dir;
    s.root = job.root;
    s.key = entry.key;
    s.mode = job.mode;
    s.first = entry.first;
    s.last = entry.last;
    s.warmup = entry.warmup;
    s.cached = cached;
    s.dropped = job.gate.dropped;
    shownJob_ = job;
    shownEntry_ = entry;
    stale_.clear();
    model_.showNrPreview(s, seek);
    QString text;
    const QString took = cached ? Tr("из кэша") : Took(entry.seconds);
    if (job.mode == NrPreviewMode::Segment) {
        const double fps = model_.sourceInfo().fps > 0 ? model_.sourceInfo().fps : 24.0;
        const int64_t frames = entry.last - entry.first + 1;
        const QString length = frames < std::llround(shownRequest_.seconds * fps)  // the clip ended first
                                   ? Tr("%1 с, до конца клипа").arg(QLocale(QLocale::Russian).toString(static_cast<double>(frames) / fps, 'f', 1))
                                   : Tr("%1 с").arg(shownRequest_.seconds);
        text = Tr("Показано превью отрезка %1–%2 (%3) · %4").arg(entry.first).arg(entry.last).arg(length, took);
    } else {
        text = Tr("Показано превью кадра %1 · %2 · %3").arg(entry.first).arg(job.mode == NrPreviewMode::Frame ? Tr("без истории") : Tr("с историей"), took);
    }
    if (job.mode != NrPreviewMode::Frame && entry.warmup < kNrPreviewHistoryFrames)  // the history and a segment: the clip's start
        text += entry.warmup == 0 && job.mode == NrPreviewMode::History ? Tr(" · первый кадр клипа: без истории")
                                                                         : Tr(" · разгон %1 кадров (начало клипа)").arg(entry.warmup);
    text += NrPreviewDroppedText(job.gate.dropped);
    const QString tip = entry.cacheable ? QString() : uncachedTip_;
    if (!entry.cacheable && model_.engineerMode()) text += Tr(" · ") + tip;  // the engineer mode says it in the line (design §5)
    if (cached && !upstreamNote_.isEmpty()) text = upstreamNote_ + Tr(" — ") + text;  // the stages done, the preview from the cache
    upstreamNote_.clear();
    shownStatus_ = text;
    shownTip_ = tip;
    setStatus(text, "ok", tip);
    updateUsage();
}

void PreviewRunner::updateStale() {
    if (!shownJob_ || !model_.nrPreview()) return;
    std::error_code ec;
    if (shownEntry_ && !fs::exists(shownEntry_->ColorDir(), ec)) {  // removed by hand: nothing to show
        shownJob_.reset();
        shownEntry_.reset();
        model_.clearNrPreview();
        setStatus({});
        return;
    }
    const NrPreviewJob now = PrepareNrPreview(model_.previewPlan(), shownJob_->mode, shownJob_->first, shownRequest_.seconds, env());
    const QString reason = NrPreviewStaleReason(*shownJob_, now, !model_.planError().isEmpty() && !Decision(model_.previewPlan(), "nr"));
    if (reason == stale_) return;
    stale_ = reason;
    if (!stale_.isEmpty()) setStatus(Tr("Показано превью для прежних настроек — %1").arg(stale_), "warn");
    else setStatus(shownStatus_, "ok", shownTip_);  // the value came back: current again (the viewport as it is)
    emit model_.nrPreviewChanged();  // the cell label says «устарело»
}

void PreviewRunner::updateUsage() {
    usage_ = model_.hasProject() && !model_.project().passesRoot.empty() ? NrPreviewUsageOf(model_.project().passesRoot) : NrPreviewUsage{};
    emit changed();
}

void PreviewRunner::housekeep(std::function<void()> work, std::function<void()> done) {
    ++housekeeping_;
    emit changed();
    pool_.start([this, work = std::move(work), done = std::move(done)] {
        if (poolGate_) poolGate_();
        try {
            work();
        } catch (const std::exception& e) {
            Log()->warn("nr preview: {}", e.what());
        }
        QMetaObject::invokeMethod(this, [this, done] {
            --housekeeping_;
            done();
            updateUsage();
            if (!housekeeping_ && pending_ && job_) {  // parked while a run goes: as asked now (a hit shows, a segment goes)
                const Request r = *pending_;
                pending_.reset();
                autoStart_ = pendingAuto_;  // an auto request stays one: the preview on screen and the run in flight stay
                start(r);
                autoStart_ = false;
            } else {
                runPending();
            }
        }, Qt::QueuedConnection);
    });
}

std::pair<double, double> PreviewCalibration(double elapsed, int64_t modelFrames, const nlohmann::json& run, bool scanned) {
    auto number = [](const nlohmann::json& j, const char* key) {
        if (!j.is_object()) return 0.0;
        const auto it = j.find(key);
        return it != j.end() && it->is_number() ? it->get<double>() : 0.0;
    };
    const auto stats = run.is_object() ? run.find("stats") : run.end();
    const auto diag = run.is_object() ? run.find("diagnostics") : run.end();
    const double wallMs = stats != run.end() ? number(*stats, "ms_per_frame_wall") : 0.0;
    if (wallMs <= 0.0 || modelFrames <= 0) return {-1.0, -1.0};
    const double modelStart = diag != run.end() ? (number(*diag, "init_ms") + number(*diag, "create_ms")) / 1000.0 : 0.0;
    const double runSeconds = wallMs * static_cast<double>(modelFrames) / 1000.0;  // the stage's timer: the model's start included
    const double frameMs = std::max(0.0, (runSeconds - modelStart) * 1000.0 / static_cast<double>(modelFrames));
    const double start = std::max(0.0, elapsed - runSeconds) + std::min(modelStart, runSeconds);  // the process before the stage + the model's start
    return {scanned ? -1.0 : start, frameMs};
}

void PreviewRunner::calibrate(const NrPreviewJob& job, double seconds, const NrPreviewEntry& entry) {
    const auto [start, frameMs] = PreviewCalibration(seconds, job.warmup + job.frames, ReadJson(entry.dir / "nr.json"), job.scanBytes > 0);
    if (frameMs < 0.0) return;
    const StageDecision* nr = Decision(model_.previewPlan(), "nr");
    const bool ngx = nr && nr->tool.is_object() && nr->tool.value("backend", std::string()) == "ngx" && !NrBypassParams(nr->params);
    const NrPreviewEnv e = env();
    const QString base = QString("preview/nr/%1/%2x%3/").arg(ngx ? "ngx" : "other").arg(e.colorW).arg(e.colorH);
    QSettings s;
    const double oldMs = s.value(base + "frame_ms", frameMs).toDouble();
    if (start >= 0.0) {  // a moving average: one slow start does not skew the estimate
        const double oldStart = s.value(base + "start_s", start).toDouble();
        s.setValue(base + "start_s", 0.7 * oldStart + 0.3 * start);
    }
    s.setValue(base + "frame_ms", 0.7 * oldMs + 0.3 * frameMs);
}

void PreviewRunner::cancel() {
    pending_.reset();
    autoTimer_.stop();
    throttle_.stop();
    tick_.stop();  // no «осталось …» of a run going away
    if (taskId_ >= 0) tasks_.cancel(taskId_);
}

void PreviewRunner::cancelUpstream() {
    if (upstreamTaskId_ >= 0) tasks_.cancel(upstreamTaskId_);
}

int PreviewRunner::prepareInputs() {
    if (!model_.hasProject()) return -1;
    if (tasks_.mainBusy() || model_.controlMaskBusy()) {  // a stale button: said why
        refuse(tasks_.mainBusy() ? Tr("Идёт обработка — превью будет доступно после неё") : Tr("Загружается маска управления — подождите"));
        return -1;
    }
    std::vector<std::string> stages;
    for (const char* name : {"scenes", "depth", "flow", "upscale"})
        if (const StageEntry* e = model_.stageEntry(name); e && e->enabled) stages.push_back(name);
    if (stages.empty() || !model_.saveForRun()) return -1;
    std::vector<TaskQueue::StagePlan> plan;
    for (const auto& e : model_.outcome().stages)
        if (std::find(stages.begin(), stages.end(), e.stage) != stages.end()) {
            TaskQueue::StagePlan s;
            s.name = QString::fromStdString(e.stage);
            s.title = model_.stageTitle(e.stage);
            s.seconds = e.seconds;
            s.reused = e.action != "run";
            plan.push_back(s);
        }
    upstreamTitles_ = StageTitles(model_, stages);
    afterUpstream_ = lastFrameKind_;
    upstreamTaskId_ = tasks_.enqueue(Tr("Этапы до улучшения"), model_.cliPath(), model_.upstreamArgs(stages), plan);  // preempts a preview
    setStatus(Tr("Этапы до улучшения: %1").arg(upstreamTitles_));
    return upstreamTaskId_;
}

bool PreviewRunner::handleTaskFinished(int id, bool ok) {
    if (!isUpstreamTask(id)) return false;
    model_.reloadSources();
    upstreamFinished(id, ok);
    return true;
}

void PreviewRunner::upstreamFinished(int id, bool ok) {
    if (id != upstreamTaskId_) return;
    upstreamTaskId_ = -1;
    const double took = tasks_.progress(id).elapsed;
    if (!ok) {
        afterUpstream_.reset();
        setStatus(tasks_.cancelled(id) ? Tr("Этапы до улучшения отменены") : Tr("Этапы до улучшения не посчитаны — подробности в логе"), "warn");
        return;
    }
    const Request next = afterUpstream_.value_or(lastFrameKind_);
    afterUpstream_.reset();
    upstreamNote_ = Tr("Этапы до улучшения посчитаны за %1").arg(FormatDuration(took));  // onProgress / show say it while it starts
    const int64_t frame = model_.store().FrameAt("source", model_.time());
    setStatus(Tr("%1 — считаю превью кадра %2").arg(upstreamNote_).arg(frame), "ok");
    if (!start(next)) upstreamNote_.clear();  // the status says why
}

void PreviewRunner::hide() {
    shownJob_.reset();
    shownEntry_.reset();
    stale_.clear();
    model_.clearNrPreview();
    setStatus({});
}

void PreviewRunner::clearCache() {
    if (!model_.hasProject() || housekeeping_) return;
    const fs::path root = model_.project().passesRoot;
    if (root.empty()) return;
    shownJob_.reset();
    shownEntry_.reset();
    stale_.clear();
    model_.clearNrPreview();  // the viewport lets the files go first
    std::vector<fs::path> pinned;
    if (job_) pinned.push_back(job_->workDir);
    auto report = std::make_shared<NrPreviewGcReport>();
    const uint64_t generation = model_.generation();
    housekeep([root, pinned, report] { *report = RemoveNrPreviews(root, pinned); },
              [this, report, generation] {
                  const QString text = report->failed.empty() ? Tr("Превью очищено: %1").arg(HumanBytes(report->bytes))
                                                              : Tr("Превью очищено: %1; часть файлов занята — удалится позже").arg(HumanBytes(report->bytes));
                  if (generation == model_.generation()) setStatus(text);  // another project or root since: not its card's status
                  emit model_.message(text);  // the window's status bar (design §5.5): the card may be out of view
              });
}

void PreviewRunner::saveFrame(const QString& file) {
    if (!shownEntry_ || file.isEmpty()) return;
    const NrPreviewEntry entry = *shownEntry_;
    const int64_t frame = model_.store().FrameAt("source", model_.time());
    if (frame < entry.first || frame > entry.last) {
        setStatus(Tr("Курсор вне превью (кадры %1–%2)").arg(entry.first).arg(entry.last), "warn");
        return;
    }
    const fs::path out(file.toStdWString());
    auto error = std::make_shared<std::string>();
    housekeep(
        [entry, frame, out, error] {
            try {
                SaveNrPreviewFrame(entry, frame, out);
            } catch (const std::exception& e) {
                *error = e.what();
            }
        },
        [this, error, file, frame] {
            if (error->empty()) setStatus(Tr("Кадр %1 превью сохранён: %2").arg(frame).arg(file), "ok");
            else setStatus(Tr("Кадр превью не сохранён — подробности в логе"), "warn", QString::fromStdString(*error));
            if (!error->empty()) Log()->error("nr preview: save frame: {}", *error);
        });
}

void PreviewRunner::setBefore(const QString& which) { model_.setNrPreviewBefore(which.toStdString()); }

QString PreviewDuration(double seconds) { return seconds < 0.5 ? Tr("< 1 с") : FormatDuration(seconds); }

QString NrPreviewStaleReason(const NrPreviewJob& shown, const NrPreviewJob& now, bool planFailed) {
    if (!now.gate.ok() && now.gate.why != NrPreviewBlock::Disk) {  // the disk budget changes nothing of the preview shown
        if (planFailed) return Tr("в плане ошибка");                  // an empty plan has no nr decision: it reads as NR off
        switch (now.gate.why) {
            case NrPreviewBlock::Upstream:
            case NrPreviewBlock::Restore:
            case NrPreviewBlock::Pending: return Tr("этапы до улучшения нужно пересчитать");
            case NrPreviewBlock::NrOff: return Tr("улучшение выключено");
            case NrPreviewBlock::NoDll: return Tr("нет nvngx_dlssnr.dll");
            case NrPreviewBlock::ControlMask: return Tr("маска управления не читается");
            case NrPreviewBlock::NrError: return Tr("в плане ошибка");
            default: return Tr("входы пересчитаны");  // PastEnd: the clip is shorter now
        }
    }
    if (now.keyJson == shown.keyJson) return {};
    const nlohmann::json a = nlohmann::json::parse(shown.keyJson, nullptr, false), b = nlohmann::json::parse(now.keyJson, nullptr, false);
    auto differs = [&](const char* k) { return a.is_object() && b.is_object() && a.value(k, nlohmann::json()) != b.value(k, nlohmann::json()); };
    bool dropsMore = false;  // a guide the frame now goes without: its stage would run, nothing was recomputed
    for (const auto& d : now.gate.dropped) dropsMore = dropsMore || std::find(shown.gate.dropped.begin(), shown.gate.dropped.end(), d) == shown.gate.dropped.end();
    if (differs("params")) return Tr("настройки изменились");
    if (dropsMore) return Tr("этапы до улучшения нужно пересчитать");
    if (differs("inputs") || differs("source") || differs("source_format")) return Tr("входы пересчитаны");
    if (differs("masks")) return Tr("маски изменились");
    if (a.is_object() && b.is_object() && a.value("tool", nlohmann::json()).value("dll", std::string()) != b.value("tool", nlohmann::json()).value("dll", std::string()))
        return Tr("сменилась nvngx_dlssnr.dll");
    return Tr("программа обновлена");
}

QString NrPreviewDroppedText(const std::vector<std::string>& dropped) {
    const bool mv = std::find(dropped.begin(), dropped.end(), "mv_dlss") != dropped.end();
    const bool depth = std::find(dropped.begin(), dropped.end(), "depth_dlss") != dropped.end();
    if (mv && depth) return Tr(" · без векторов движения и глубины");
    if (mv) return Tr(" · без векторов движения");
    if (depth) return Tr(" · без глубины");
    return {};
}

QString NrPreviewDiskText(uint64_t estimatedBytes, uint64_t freeBytes) {
    if (estimatedBytes > kNrPreviewMaxBytes)  // the budget, not the volume, is the limit
        return Tr("≈ %1 на диске — больше %2, отведённых под превью").arg(HumanBytes(estimatedBytes), HumanBytes(kNrPreviewMaxBytes));
    return Tr("Не хватает места: превью ≈ %1, свободно %2, а %3 превью оставляют на диске свободными")
        .arg(HumanBytes(estimatedBytes), HumanBytes(freeBytes), HumanBytes(kNrPreviewDiskReserve));
}

QString NrPreviewCellSuffix(const AppModel::NrPreviewShown& shown, FrameState state, bool stale) {
    QString s;
    if (shown.mode == NrPreviewMode::Segment) s = Tr(" · отрезок %1–%2").arg(shown.first).arg(shown.last);
    else s = shown.mode == NrPreviewMode::Frame ? Tr(" · без истории") : Tr(" · с историей");
    s += NrPreviewDroppedText(shown.dropped);
    if (stale) s += Tr(" · устарело");
    switch (state) {
        case FrameState::Missing: s += Tr(" · нет кадра превью (кадры %1–%2)").arg(shown.first).arg(shown.last); break;
        case FrameState::Loading:
        case FrameState::Stale: s += Tr(" · загрузка…"); break;
        default: break;
    }
    return s;
}

}  // namespace dlssvid
