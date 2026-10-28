#include "PreviewRunner.h"

#include <QCoreApplication>
#include <QGuiApplication>
#include <QLocale>
#include <QSettings>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>

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
        tasks_.cancel(taskId_);
    });
    connect(&tick_, &QTimer::timeout, this, [this] { onProgress(taskId_); });
    connect(&throttle_, &QTimer::timeout, this, &PreviewRunner::runPending);
    connect(&autoTimer_, &QTimer::timeout, this, [this] {
        const bool held = pointerHeld_ ? pointerHeld_() : QGuiApplication::mouseButtons() != Qt::NoButton;
        if (held) {  // a slider being dragged: its release is the change
            autoTimer_.start();
            return;
        }
        if (autoUpdate_ && everStarted_ && model_.hasProject()) start(lastFrameKind_);
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
        setStatus({});
        QMetaObject::invokeMethod(this, [this] { updateUsage(); }, Qt::QueuedConnection);  // the new root is set after the signal
    });
    connect(&model_, &AppModel::planChanged, this, &PreviewRunner::updateStale);
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
    status_ = text;
    role_ = role;
    tip_ = tip;
    emit changed();
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
    e.freeBytes = model_.freeSpace();
    e.clipFrames = model_.sourceInfo().frames > 0 ? model_.sourceInfo().frames : -1;
    e.fps = model_.sourceInfo().fps > 0 ? model_.sourceInfo().fps : 24.0;
    e.colorW = model_.sourceInfo().width;
    e.colorH = model_.sourceInfo().height;
    for (const auto& s : model_.outcome().stages)
        if (s.stage == "nr") e.msPerFrame = s.msPerFrame;
    // the start: the process, the device, NGX and the feature (E8 ≈ 1.6-2 s), the stub's or the bypass's much less; then what the
    // previews of this size measured
    const StageDecision* nr = Decision(model_.previewPlan(), "nr");
    const bool ngx = nr && nr->tool.is_object() && nr->tool.value("backend", std::string()) == "ngx" && !NrBypassParams(nr->params);
    const QString size = QString("%1x%2").arg(e.colorW).arg(e.colorH);
    e.startSeconds = QSettings().value(QString("preview/nr/%1/%2/start_s").arg(ngx ? "ngx" : "other", size), ngx ? 2.0 : 0.5).toDouble();
    if (const double ms = QSettings().value(QString("preview/nr/%1/%2/frame_ms").arg(ngx ? "ngx" : "other", size), 0.0).toDouble(); ms > 0) e.msPerFrame = ms;
    return e;
}

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
        case NrPreviewBlock::ControlMask: return Tr("Маска управления не читается: %1").arg(QString::fromStdString(g.detail));
        case NrPreviewBlock::PastEnd: return Tr("Курсор за концом клипа");
        case NrPreviewBlock::Disk:
            return Tr("Не хватает места: превью ≈ %1, свободно %2").arg(HumanBytes(job.estimatedBytes), HumanBytes(model_.freeSpace()));
        case NrPreviewBlock::Upstream:
        case NrPreviewBlock::Restore:
        case NrPreviewBlock::Pending: {
            double seconds = 0;
            for (const auto& s : model_.previewPlan().stages)
                if (std::find(g.stages.begin(), g.stages.end(), s.name) != g.stages.end())
                    for (const auto& e : model_.outcome().stages)
                        if (e.stage == s.name && e.action == "run") seconds += e.seconds;
            if (a) {
                a->canPrepare = true;
                a->upstreamSeconds = seconds;
                for (const auto& s : g.stages) a->upstream << m.stageTitle(s);
            }
            if (g.why == NrPreviewBlock::Restore)
                return Tr("Нужная версия «%1» лежит в истории — «Посчитать этапы до улучшения» вернёт её за секунды").arg(StageTitles(m, g.stages));
            if (g.why == NrPreviewBlock::Pending) return Tr("Склейки ещё не найдены — сначала посчитайте этапы до улучшения");
            return Tr("Сначала посчитайте этапы до улучшения: %1 — по всему клипу, %2").arg(StageTitles(m, g.stages), PreviewDuration(seconds));
        }
    }
    return {};
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
    if (!job.gate.ok()) {
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
    if (housekeeping_) {            // a clean-up of the cache is moving folders: right after it
        pending_ = r;
        return true;
    }
    if (const QString b = appBlock(nullptr); !b.isEmpty()) {
        setStatus(b, "warn");
        return false;
    }
    const int64_t frame = model_.store().FrameAt("source", model_.time());
    NrPreviewJob job = PrepareNrPreview(model_.previewPlan(), ModeOf(r), frame, r.seconds, env());
    if (!job.gate.ok()) {
        setStatus(blockText(job, nullptr), "warn");
        return false;
    }
    if (const QString b = appBlock(&job); !b.isEmpty()) {
        setStatus(b, "warn");
        return false;
    }
    if (job_ && job_->key == job.key) return true;  // this very preview is being computed
    if (const auto hit = FindNrPreview(job)) {     // computed before: shown at once, no task
        if (job_ && jobRequest_.kind == Kind::Segment) tasks_.cancel(taskId_);
        pending_.reset();
        shownRequest_ = r;
        show(*hit, job, true, true);
        return true;
    }
    if (job_) {
        pending_ = r;  // the last request wins
        if (jobRequest_.kind == Kind::Segment) tasks_.cancel(taskId_);  // a segment is long: it goes; a frame finishes (NGX's create)
        return true;
    }
    if (lastStart_.isValid() && lastStart_.elapsed() < kMinStartGapMs) {
        pending_ = r;
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
        setStatus(Tr("Идёт обработка — превью будет доступно после неё"), "warn");
        return false;
    }
    job_ = job;
    jobRequest_ = r;
    taskId_ = id;
    jobGeneration_ = model_.generation();
    jobCursor_ = model_.time();
    jobClock_.start();
    lastStart_.start();
    preempted_ = watchdogFired_ = false;
    lastLine_.clear();
    watchdog_.start(static_cast<int>(std::max(30.0, 3.0 * job.estimatedSeconds + 30.0) * 1000.0));
    tick_.start();
    Log()->info("nr preview: {} {} at frame {} ({} frames, warm-up {}), key {}", ToString(job.mode), id, job.first, job.frames, job.warmup, job.key);
    onProgress(id);
    return true;
}

void PreviewRunner::onStarted(int id) {
    if (id == taskId_) onProgress(id);
}

void PreviewRunner::onProgress(int id) {
    if (id < 0 || id != taskId_ || !job_ || jobGeneration_ != model_.generation()) return;
    const TaskQueue::Progress p = tasks_.progress(id);
    const double elapsed = jobClock_.isValid() ? static_cast<double>(jobClock_.elapsed()) / 1000.0 : 0.0;
    const double left = std::max(0.0, job_->estimatedSeconds - elapsed);
    const QString remain = left > 0.5 ? PreviewDuration(left) : Tr("секунды");
    if (jobRequest_.kind == Kind::Segment) {
        const int percent = p.total > 0 ? static_cast<int>(p.done * 100 / p.total) : 0;
        setStatus(Tr("Превью отрезка %1–%2 · %3 % · осталось %4").arg(job_->first).arg(job_->first + job_->frames - 1).arg(percent).arg(remain));
    } else if (p.done == 0) {
        setStatus(Tr("Превью кадра %1 · запуск модели · осталось %2").arg(job_->first).arg(remain));
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
    const double seconds = jobClock_.isValid() ? static_cast<double>(jobClock_.elapsed()) / 1000.0 : 0.0;
    job_.reset();
    taskId_ = -1;
    watchdog_.stop();
    tick_.stop();
    if (jobGeneration_ != model_.generation() || job.root != model_.project().passesRoot) {  // another project or root since: dropped
        removeWork(job.workDir);
        pending_.reset();
        setStatus({});
        return;
    }
    if (tasks_.cancelled(id)) {
        removeWork(job.workDir);
        if (preempted_) {
            pending_.reset();
            setStatus(Tr("Превью отменено: началась обработка"));
        } else if (watchdogFired_) {
            pending_.reset();
            setStatus(Tr("Превью не ответило за %1 — остановлено").arg(FormatDuration(seconds)), "warn");
        } else if (!pending_) {
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
        setStatus(text, "warn", lastLine_);
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
        setStatus(Tr("Превью отброшено: %1").arg(why), "warn", QString::fromStdString(v.detail));
        runPending();
        return;
    }
    NrPreviewEntry entry;
    try {
        entry = PublishNrPreview(job, seconds, v.cacheable);
    } catch (const std::exception& e) {
        removeWork(job.workDir);
        Log()->error("nr preview: {}", e.what());
        setStatus(Tr("Превью не удалось — подробности в логе"), "warn", QString::fromUtf8(e.what()));
        runPending();
        return;
    }
    calibrate(job, seconds, entry);
    shownRequest_ = request;
    const bool seek = std::abs(model_.time() - jobCursor_) < 1e-9 && !model_.playing();  // the cursor was not moved meanwhile
    show(entry, job, false, seek);
    Log()->info("nr preview: shown {} in {:.2f} s ({})", entry.key, seconds, entry.cacheable ? "cached" : "not cached");
    // the clean-up keeps the entry shown and a run that started meanwhile
    const fs::path root = job.root;
    std::vector<fs::path> pinned{entry.dir};
    if (job_) pinned.push_back(job_->workDir);
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
    start(r);
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
        text = Tr("Показано превью отрезка %1–%2 (%3 с) · %4").arg(entry.first).arg(entry.last).arg(shownRequest_.seconds).arg(took);
    } else {
        text = Tr("Показано превью кадра %1 · %2 · %3").arg(entry.first).arg(job.mode == NrPreviewMode::Frame ? Tr("без истории") : Tr("с историей"), took);
        if (job.mode == NrPreviewMode::History && entry.warmup < kNrPreviewHistoryFrames)
            text += entry.warmup == 0 ? Tr(" · первый кадр клипа: без истории") : Tr(" · разгон %1 кадров (начало клипа)").arg(entry.warmup);
    }
    if (std::find(job.gate.dropped.begin(), job.gate.dropped.end(), "mv_dlss") != job.gate.dropped.end()) text += Tr(" · без векторов движения");
    const QString tip = entry.cacheable ? QString() : Tr("не сохранено в кэш: у CLI другая nvngx_dlssnr.dll");
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
    QString reason;
    if (!now.gate.ok()) {
        reason = Tr("входы пересчитаны");
    } else if (now.keyJson != shownJob_->keyJson) {
        const nlohmann::json a = nlohmann::json::parse(shownJob_->keyJson, nullptr, false), b = nlohmann::json::parse(now.keyJson, nullptr, false);
        auto differs = [&](const char* k) { return a.is_object() && b.is_object() && a.value(k, nlohmann::json()) != b.value(k, nlohmann::json()); };
        if (differs("params")) reason = Tr("настройки изменились");
        else if (differs("inputs") || differs("source") || differs("source_format")) reason = Tr("входы пересчитаны");
        else if (a.is_object() && b.is_object() && a.value("tool", nlohmann::json()).value("dll", std::string()) != b.value("tool", nlohmann::json()).value("dll", std::string()))
            reason = Tr("сменилась nvngx_dlssnr.dll");
        else reason = Tr("программа обновлена");
    }
    if (reason == stale_) return;
    stale_ = reason;
    if (!stale_.isEmpty()) setStatus(Tr("Показано превью для прежних настроек — %1").arg(stale_), "warn");
    else if (shownEntry_) show(*shownEntry_, *shownJob_, true, false);  // the value came back: current again
    emit model_.nrPreviewChanged();  // the cell label says «устарело»
}

void PreviewRunner::updateUsage() {
    usage_ = model_.hasProject() && !model_.project().passesRoot.empty() ? NrPreviewUsageOf(model_.project().passesRoot) : NrPreviewUsage{};
    emit changed();
}

void PreviewRunner::housekeep(std::function<void()> work, std::function<void()> done) {
    housekeeping_ = true;
    emit changed();
    pool_.start([this, work = std::move(work), done = std::move(done)] {
        try {
            work();
        } catch (const std::exception& e) {
            Log()->warn("nr preview: {}", e.what());
        }
        QMetaObject::invokeMethod(this, [this, done] {
            housekeeping_ = false;
            done();
            updateUsage();
            runPending();
        }, Qt::QueuedConnection);
    });
}

void PreviewRunner::calibrate(const NrPreviewJob& job, double seconds, const NrPreviewEntry& entry) {
    const double wallMs = entry.stats.is_object() && entry.stats.contains("ms_per_frame_wall") && entry.stats["ms_per_frame_wall"].is_number()
                              ? entry.stats["ms_per_frame_wall"].get<double>()
                              : 0.0;
    if (wallMs <= 0.0) return;
    const StageDecision* nr = Decision(model_.previewPlan(), "nr");
    const bool ngx = nr && nr->tool.is_object() && nr->tool.value("backend", std::string()) == "ngx" && !NrBypassParams(nr->params);
    const NrPreviewEnv e = env();
    const QString base = QString("preview/nr/%1/%2x%3/").arg(ngx ? "ngx" : "other").arg(e.colorW).arg(e.colorH);
    const double start = std::max(0.0, seconds - static_cast<double>(job.warmup + job.frames) * wallMs / 1000.0);
    QSettings s;
    const double oldStart = s.value(base + "start_s", start).toDouble(), oldMs = s.value(base + "frame_ms", wallMs).toDouble();
    s.setValue(base + "start_s", 0.7 * oldStart + 0.3 * start);  // a moving average: one slow start does not skew the estimate
    s.setValue(base + "frame_ms", 0.7 * oldMs + 0.3 * wallMs);
}

void PreviewRunner::cancel() {
    pending_.reset();
    autoTimer_.stop();
    throttle_.stop();
    tick_.stop();  // no «осталось …» of a run going away
    if (taskId_ >= 0) tasks_.cancel(taskId_);
}

int PreviewRunner::prepareInputs() {
    if (!model_.hasProject() || tasks_.mainBusy() || model_.controlMaskBusy()) return -1;
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
    const int64_t frame = model_.store().FrameAt("source", model_.time());
    setStatus(Tr("Этапы до улучшения посчитаны за %1 — считаю превью кадра %2").arg(FormatDuration(took)).arg(frame), "ok");
    if (!start(next)) return;  // the status says why
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
    housekeep([root, pinned, report] { *report = RemoveNrPreviews(root, pinned); },
              [this, report] {
                  setStatus(report->failed.empty() ? Tr("Превью очищено: %1").arg(HumanBytes(report->bytes))
                                                   : Tr("Превью очищено: %1; часть файлов занята — удалится позже").arg(HumanBytes(report->bytes)));
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

QString NrPreviewCellSuffix(const AppModel::NrPreviewShown& shown, FrameState state, bool stale) {
    QString s;
    if (shown.mode == NrPreviewMode::Segment) s = Tr(" · отрезок %1–%2").arg(shown.first).arg(shown.last);
    else s = shown.mode == NrPreviewMode::Frame ? Tr(" · без истории") : Tr(" · с историей");
    if (std::find(shown.dropped.begin(), shown.dropped.end(), "mv_dlss") != shown.dropped.end()) s += Tr(" · без векторов движения");
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
