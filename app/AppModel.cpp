#include "AppModel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <algorithm>
#include <cmath>
#include <optional>

#include "ControlMaskText.h"
#include "ParamText.h"
#include "PresetStore.h"
#include "SourceNames.h"
#include "passes/ControlMaskPass.h"
#include "passes/ScenesPass.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "stages/ParamSchema.h"
#include "stages/PresetLibrary.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/nr/INrBackend.h"
#include "stages/upscale/IUpscaler.h"
#include "util/Log.h"
#include "viewport/ViewportProxy.h"

namespace dlssvid {

AppModel::AppModel(bool warp, QObject* parent) : QObject(parent) {
    device_ = std::make_unique<D3D12Device>(D3D12Device::Options{warp, false});
    store_ = std::make_unique<FrameStore>(*device_);
    playTimer_.setTimerType(Qt::PreciseTimer);
    playTimer_.setSingleShot(true);
    connect(&playTimer_, &QTimer::timeout, this, &AppModel::onPlayTick);
    monotonic_.start();
    now_ = [this] { return monotonic_.nsecsElapsed(); };
    engineerMode_ = QSettings().value("view/engineerMode", false).toBool();
    loop_ = QSettings().value("view/loop", false).toBool();
    planTimer_.setSingleShot(true);
    planTimer_.setInterval(250);
    connect(&planTimer_, &QTimer::timeout, this, &AppModel::refreshPlan);
    // a frame of the current time is ready on a loader thread: upload it now instead of on the viewport's next poll
    // (up to 16 ms later; TASK-0040). Coalesced: one queued call at a time.
    store_->SetOnCurrentLoaded([this] {
        if (currentLoadedQueued_.exchange(true)) return;
        QMetaObject::invokeMethod(this, [this] {
            currentLoadedQueued_ = false;
            if (store_->Update()) notifyFramesUpdated();
        }, Qt::QueuedConnection);
    });
}

AppModel::~AppModel() {
    maskPool_.waitForDone();  // an import in flight finishes (or leaves .import/ to the next one); its queued end is dropped with us
    cancelProxies();
    store_->SetOnCurrentLoaded({});  // loaders may still finish a frame while the store shuts down
    store_.reset();
}

QStringList AppModel::sourceNames() const {
    QStringList names;
    for (const auto& s : store_->Sources()) names << QString::fromStdString(s.name);
    return names;
}

std::vector<std::string> AppModel::neededSources() const { return state().NeededSources(); }

QString AppModel::cliPath() const {
    const QString here = QCoreApplication::applicationDirPath();
    const QString cli = QDir(here).filePath(DLSSVID_CLI_NAME);
    if (QFileInfo::exists(cli)) return cli;
#ifdef DLSSVID_CLI_PATH
    if (QFileInfo::exists(DLSSVID_CLI_PATH)) return QString(DLSSVID_CLI_PATH);  // build tree
#endif
    return QString(DLSSVID_CLI_NAME);
}

void AppModel::openVideo(const QString& file) {
    try {
        project_ = Project::Create(std::filesystem::path(file.toStdWString()), {});
        // default project file next to the video
        project_.file = std::filesystem::path(file.toStdWString()).replace_extension(".dlssvid.json");
        const bool loaded = std::filesystem::exists(project_.file);
        if (loaded) project_ = Project::Load(project_.file);
        const QString seeded = loaded ? QString() : seedNewProject();  // before the first plan: it is made with those settings
        reloadSources();
        dirty_ = !loaded;  // a project made from the video is not on disk yet: it is saved next to the video on close
        emit dirtyChanged(dirty_);
        emit projectChanged();
        addRecent(file);
        emit message(tr("Открыт %1").arg(file) + seeded);
    } catch (const std::exception& e) {
        emit message(tr("Ошибка: %1").arg(e.what()));
    }
}

void AppModel::openProject(const QString& file) {
    try {
        project_ = Project::Load(std::filesystem::path(file.toStdWString()));
        reloadSources();
        dirty_ = false;
        emit dirtyChanged(false);
        emit projectChanged();
        addRecent(file);
        emit message(tr("Проект %1").arg(file));
    } catch (const std::exception& e) {
        emit message(tr("Ошибка: %1").arg(e.what()));
    }
}

bool AppModel::saveProject() {
    if (project_.file.empty()) return false;
    return saveProjectAs(QString::fromStdWString(project_.file.wstring()));
}

bool AppModel::saveProjectAs(const QString& file) {
    try {
        project_.Save(std::filesystem::path(file.toStdWString()));
        dirty_ = false;
        emit dirtyChanged(false);
        emit message(tr("Проект сохранён: %1").arg(file));
        return true;
    } catch (const std::exception& e) {
        emit message(tr("Ошибка сохранения: %1").arg(e.what()));
        return false;
    }
}

namespace {
// The control mask of a passes root as the card shows it.
AppModel::ControlMaskState ReadControlMaskState(const std::filesystem::path& root) {
    AppModel::ControlMaskState s;
    if (root.empty() || !std::filesystem::is_directory(root / kControlMaskPass)) return s;
    if (ControlMaskFolderBroken(root)) {
        s.status = AppModel::ControlMaskState::Status::Unreadable;
        s.problem = QCoreApplication::translate("ControlMask", "папка повреждена");
        s.broken = true;
        return s;
    }
    std::string why;
    const auto info = ReadControlMaskInfo(root / kControlMaskPass, &why);
    if (!info) {
        s.status = AppModel::ControlMaskState::Status::Unreadable;
        s.problem = ControlMaskWhyText(why);
        try {  // the name for the «Убрать» dialog when the manifest still reads
            const Manifest m = Manifest::Load(root / kControlMaskPass);
            if (m.stageParams.is_object()) s.file = QString::fromStdString(m.stageParams.value("source_name", std::string()));
        } catch (const std::exception&) {
        }
        return s;
    }
    s.status = AppModel::ControlMaskState::Status::Ready;
    s.file = QString::fromStdString(info->sourceName);
    s.fingerprint = QString::fromStdString(ShortFingerprint(info->fingerprint));
    s.width = info->width;
    s.height = info->height;
    s.grayscale = info->grayscale;
    s.alphaIgnored = info->alphaIgnored;
    s.clamped = info->clamped;
    return s;
}
}  // namespace

std::vector<ViewportSource> AppModel::viewportSources(bool dropControlMask) const {
    std::vector<ViewportSource> sources = project_.Sources();
    if (dropControlMask)  // the import renames mask_control: an open file of the viewport would fail it on Windows (Locked)
        sources.erase(std::remove_if(sources.begin(), sources.end(), [](const ViewportSource& s) { return s.pass == kControlMaskPass; }), sources.end());
    return sources;
}

void AppModel::importControlMask(const QString& file) {
    if (controlMaskBusy_) return;
    if (!hasProject() || project_.passesRoot.empty()) {
        emit controlMaskFinished(false, false, ControlMaskIssueText({ControlMaskIssue::Kind::NoRoot, {}, 0, 0}));
        return;
    }
    controlMaskBusy_ = true;
    rememberDir("mask", file);
    store_->SetSources(viewportSources(true));  // waits for loads in flight: then the folder can be renamed
    requestFrames();                             // the other sources again: SetSources cleared the cache
    controlMask_.status = ControlMaskState::Status::Importing;
    controlMask_.file = QFileInfo(file).fileName();
    emit controlMaskChanged();
    const std::filesystem::path image(file.toStdWString()), root = project_.passesRoot;
    maskPool_.start([this, image, root] {
        std::optional<ControlMaskImport> done;
        std::optional<ControlMaskIssue> issue;
        std::string other;
        try {
            done = ImportControlMask(image, root);  // no GPU, no Qt
        } catch (const ControlMaskError& e) {
            issue = e.issue;
        } catch (const std::exception& e) {
            other = e.what();
        }
        QMetaObject::invokeMethod(this, [this, done, issue, other, root] {
            controlMaskBusy_ = false;
            bool ok = false, changed = false;
            QString message;
            if (done) {
                ok = true;
                changed = done->outcome != ControlMaskImport::Outcome::Unchanged;
                message = changed ? tr("Маска управления загружена: %1").arg(QString::fromStdString(done->mask.sourceName)) : tr("эта маска уже загружена");
            } else if (issue) {
                message = ControlMaskIssueText(*issue);
            } else {
                message = tr("Маска не загружена: %1").arg(QString::fromStdString(other));
            }
            if (ok) Log()->info("mask_control: {}", message.toStdString());
            // the current project's mask and sources whichever project the import ran for (another one may have been opened meanwhile); after
            // a failure the previous mask comes back on the viewport
            (void)root;
            if (hasProject()) reloadSources();
            else controlMask_ = {};
            emit controlMaskFinished(ok, changed, message);
            emit controlMaskChanged();
        }, Qt::QueuedConnection);
    });
}

bool AppModel::removeControlMask(QString* error) {
    if (controlMaskBusy_ || !hasProject() || project_.passesRoot.empty()) return false;
    store_->SetSources(viewportSources(true));  // the viewport lets go of the folder before the rename
    bool ok = true;
    try {
        RemoveControlMask(project_.passesRoot);
    } catch (const ControlMaskError& e) {
        ok = false;
        if (error) *error = ControlMaskIssueText(e.issue);
    } catch (const std::exception& e) {
        ok = false;
        if (error) *error = QString::fromStdString(e.what());
    }
    reloadSources();
    emit controlMaskChanged();
    return ok;
}

void AppModel::reloadSources() {
    store_->SetSources(viewportSources(controlMaskBusy_));
    if (!controlMaskBusy_) controlMask_ = ReadControlMaskState(project_.passesRoot);  // while importing it stays «Importing»
    scrubStartNs_ = -1;
    resultAudio_ = false;
    resultCodec_.clear();
    resultFormat_ = FrameDesc{};
    if (!project_.resultVideo.empty() && std::filesystem::exists(project_.resultVideo)) {
        try {
            VideoDecoder probe(project_.resultVideo, VideoDecoder::Probe());
            resultAudio_ = probe.HasAudio();
            resultCodec_ = probe.Info().codecName;
            resultFormat_ = probe.Info().decoded;
        } catch (const std::exception&) {
        }
    }
    emit resultChanged();
    timelineFps_ = Rational{0, 1};
    state().ResolveLegacyFrame(store_->FpsOf(baseSource()));  // projects before stage 9: frame -> time at the base rate
    state().time = std::clamp(state().time, 0.0, lastTime());
    emit sourcesChanged();
    requestFrames();
    emit stateChanged();
    refreshPlan();
    startProxies();
}

void AppModel::requestFrames() { store_->SetCurrentTime(state().time, neededSources()); }

void AppModel::startProxies() {
    if (!hasProject() || project_.passesRoot.empty() || !std::filesystem::exists(project_.sourceVideo)) return;
    if (!QSettings().value("view/autoProxy", true).toBool()) return;
    QStringList names;
    for (const char* name : {"source", "result"}) {
        const std::filesystem::path video = std::string(name) == "source" ? project_.sourceVideo : project_.resultVideo;
        if (video.empty() || !std::filesystem::exists(video) || ValidProxy(video, project_.passesRoot, name)) continue;
        names << name;
    }
    if (names.isEmpty()) return;
    QStringList args{"proxy", "-i", QString::fromStdWString(project_.sourceVideo.wstring()), "--passes", QString::fromStdWString(project_.passesRoot.wstring()),
                     "--sources", names.join(','), "--background", "--max-gb", QString::number(QSettings().value("view/proxyMaxGiB", 8.0).toDouble())};
    if (!project_.resultVideo.empty()) args << "--result" << QString::fromStdWString(project_.resultVideo.wstring());
    if (proxyBuilding() && args == proxyArgs_) return;  // the same targets: let it go on
    cancelProxies();
    proxyArgs_ = args;
    proxyProcess_ = new QProcess(this);
    proxyProcess_->setProcessChannelMode(QProcess::MergedChannels);
    connect(proxyProcess_, &QProcess::finished, this, [this, p = proxyProcess_](int code, QProcess::ExitStatus status) {
        const QString out = QString::fromLocal8Bit(p->readAll());
        p->deleteLater();
        if (proxyProcess_ == p) proxyProcess_ = nullptr;
        if (status != QProcess::NormalExit || code != 0) {
            Log()->warn("viewport proxy: dlssvid proxy failed ({}): {}", code, out.right(400).toStdString());
            return;
        }
        QStringList ready;
        for (const char* name : {"source", "result"}) {
            const std::filesystem::path video = std::string(name) == "source" ? project_.sourceVideo : project_.resultVideo;
            if (const auto proxy = ValidProxy(video, project_.passesRoot, name)) {
                store_->SetProxy(name, *proxy);
                ready << HumanSourceName(name);
            }
        }
        if (!ready.isEmpty()) emit message(tr("Прокси для перемотки готов: %1").arg(ready.join(", ")));
        emit proxiesChanged();
    });
    Log()->info("viewport proxy: building {} in the background", names.join(", ").toStdString());
    proxyProcess_->start(cliPath(), args);
    emit proxiesChanged();
}

void AppModel::cancelProxies() {
    if (!proxyProcess_) return;
    QProcess* p = proxyProcess_;
    proxyProcess_ = nullptr;
    proxyArgs_.clear();
    p->disconnect(this);
    if (p->state() != QProcess::NotRunning) {
        p->kill();  // its partial file is removed by the next build (BuildProxy starts with that)
        p->waitForFinished(3000);
    }
    p->deleteLater();
    emit proxiesChanged();
}

QString AppModel::proxyState(const std::string& source) const {
    if (const ViewportSource* s = store_->FindSource(source); s && !s->proxy.empty()) return "ready";
    if (proxyBuilding() && proxyArgs_.join(' ').contains(QString::fromStdString(source))) return "building";
    return "none";
}

void AppModel::noteScrubReady() {
    if (scrubStartNs_ < 0) return;
    for (const auto& n : neededSources())
        if (store_->StatusAt(state().time, n) != FrameStore::Status::Ready && store_->StatusAt(state().time, n) != FrameStore::Status::Missing) return;
    scrubMs_.push_back(static_cast<double>(now_() - scrubStartNs_) / 1e6);
    scrubStartNs_ = -1;
    if (scrubMs_.size() >= 30) {
        std::vector<double> v = scrubMs_;
        std::sort(v.begin(), v.end());
        Log()->info("viewport scrub: {} jumps, p50 {:.1f} ms, p90 {:.1f} ms, max {:.1f} ms (time set -> exact frames on the GPU)", v.size(), v[v.size() / 2],
                    v[std::min(v.size() - 1, v.size() * 9 / 10)], v.back());
        scrubMs_.clear();
    }
}

// ---- time axis ----------------------------------------------------------------------------------

std::string AppModel::baseSource() const {
    const ViewportState& st = state();
    if (st.mode == ViewMode::Grid) return st.gridSources[0];
    return st.layers.empty() ? std::string("source") : st.layers[0].source;
}

int64_t AppModel::baseFrame() const { return store_->FrameAt(baseSource(), time()); }

Rational AppModel::timelineFps() const { return timelineFps_.num > 0 && timelineFps_.den > 0 ? timelineFps_ : store_->SourceFps(baseSource()); }

int64_t AppModel::timelineFrameCount() const { return static_cast<int64_t>(std::llround(store_->Duration() * timelineFps().ToDouble())); }

int64_t AppModel::timelineFrame() const { return static_cast<int64_t>(std::llround(time() * timelineFps().ToDouble())); }

double AppModel::lastTime() const {
    const int64_t n = timelineFrameCount();
    return n > 0 ? static_cast<double>(n - 1) / timelineFps().ToDouble() : 0.0;
}

std::vector<Rational> AppModel::fpsChoices() const {
    std::vector<Rational> out;
    for (const auto& s : store_->Sources()) {
        const Rational r = store_->SourceFps(s.name);
        bool dup = false;
        for (const auto& o : out) dup = dup || std::abs(o.ToDouble() - r.ToDouble()) < 1e-6;
        if (!dup) out.push_back(r);
    }
    std::sort(out.begin(), out.end(), [](const Rational& a, const Rational& b) { return a.ToDouble() < b.ToDouble(); });
    return out;
}

FrameState AppModel::frameStateOf(const std::string& source) const {
    switch (store_->StatusAt(time(), source)) {
        case FrameStore::Status::Ready: return FrameState::Ready;
        case FrameStore::Status::Missing: return FrameState::Missing;
        default: return store_->HeldFrameAt(time(), source) ? FrameState::Stale : FrameState::Loading;
    }
}

bool AppModel::framesReadyAt(double t) const {
    for (const auto& n : neededSources()) {
        const FrameStore::Status st = store_->StatusAt(t, n);
        if (st != FrameStore::Status::Ready && st != FrameStore::Status::Missing) return false;
    }
    return true;
}

FrameStates AppModel::frameStates() const {
    FrameStates out;
    for (const auto& n : neededSources()) out[n] = frameStateOf(n);
    return out;
}

void AppModel::setTime(double seconds) {
    seconds = std::clamp(seconds, 0.0, lastTime());
    if (std::abs(seconds - state().time) < 1e-9) return;
    state().time = seconds;
    if (playing_ && !inPlayTick_) {  // a jump while playing: go on from there
        clock_.Restart(now_(), seconds);
        clockTime_ = seconds;
    }
    if (!playing_) scrubStartNs_ = now_();  // a scrub: timed until its frames are on the GPU
    requestFrames();
    emit timeChanged(seconds);
    emit stateChanged();
}

void AppModel::setFrame(int64_t timelineFrame) { setTime(static_cast<double>(timelineFrame) / timelineFps().ToDouble()); }

void AppModel::stepFrame(int delta) {
    const std::string base = baseSource();
    setTime(store_->TimeOfFrame(base, store_->FrameAt(base, time()) + delta));
}

void AppModel::seekToSourceFrame(int64_t frame) {
    // the frame's own time, as stepFrame: FrameAt rounds time × fps, so a half frame more would show the next one
    const double fps = sourceInfo_.fps > 0 ? sourceInfo_.fps : 24.0;
    setTime(static_cast<double>(std::max<int64_t>(frame, 0)) / fps);
}

void AppModel::setTimelineFps(Rational fps) {
    timelineFps_ = fps;
    emit timelineFpsChanged();
}

void AppModel::setPlaying(bool playing) {
    if (playing == playing_) return;
    if (playing) {
        Rational rate = store_->MaxRate(neededSources());  // the fastest shown source: an FG result runs at 2x
        if (rate.num <= 0 || rate.den <= 0) rate = {1, 1};
        clock_.Start(now_(), time(), rate);
        clockTime_ = time();
        playing_ = true;
        schedulePlayTick();
    } else {
        playing_ = false;
        playTimer_.stop();
    }
    emit playingChanged(playing);
}

void AppModel::setPlaybackClock(std::function<int64_t()> nowNs) { now_ = nowNs ? std::move(nowNs) : std::function<int64_t()>([this] { return monotonic_.nsecsElapsed(); }); }

void AppModel::schedulePlayTick() {
    if (!playing_) return;
    const int64_t now = now_();
    const int64_t ms = std::clamp<int64_t>((clock_.WakeNs(now) - now + 999'999) / 1'000'000, 0, 1000);
    playTimer_.start(static_cast<int>(ms));
}

// Playback by the clock (audit play-fg-rate): frame k is due at k periods of the exact rational rate; a tick shows at
// most one frame. Every frame is shown: while a shown source has not decoded the next frame, the tick waits (finished
// loads are uploaded first, so the wait is as short as it can be) - a slow decode plays slower than real time instead
// of skipping frames or flashing plates. A frame that does not arrive within PlaybackClock::kStallNs is skipped, so
// playback never freezes.
void AppModel::onPlayTick() {
    if (!playing_) return;
    const int64_t now = now_();
    // the time changed without setTime (a project opened while playing restores its saved time): go on from there
    if (std::abs(time() - clockTime_) > 1e-9) {
        clock_.Start(now, time(), clock_.Rate());
        clockTime_ = time();
    }
    // the shown sources may change while playing (source 24 fps -> an FG result at 48): follow the fastest one
    if (const Rational rate = store_->MaxRate(neededSources()); rate.num > 0 && rate.den > 0 &&
                                                                (rate.num != clock_.Rate().num || rate.den != clock_.Rate().den))
        clock_.Start(now, time(), rate);
    if (now < clock_.DueNs()) {
        schedulePlayTick();
        return;
    }
    const double next = clock_.NextTime();
    if (next > lastTime() + 1e-9) {  // the end: wrap when looping, else stop
        if (loop_ && lastTime() > 0.0) {
            inPlayTick_ = true;
            setTime(0.0);
            inPlayTick_ = false;
            clock_.Restart(now, 0.0);
            clockTime_ = time();
            schedulePlayTick();
        } else {
            setPlaying(false);
        }
        return;
    }
    // play only over computed frames: stop where the base source has no frame
    if (store_->StatusAt(next, baseSource()) == FrameStore::Status::Missing) {
        setPlaying(false);
        return;
    }
    bool ready = framesReadyAt(next);
    if (!ready && store_->Update()) {
        notifyFramesUpdated();
        ready = framesReadyAt(next);
    }
    if (clock_.Decide(now, ready) == PlaybackClock::Action::Wait) {
        schedulePlayTick();
        return;
    }
    inPlayTick_ = true;
    setTime(next);
    inPlayTick_ = false;
    clockTime_ = time();
    clock_.Advance(now);
    schedulePlayTick();
}

void AppModel::setLoop(bool on) {
    if (loop_ == on) return;
    loop_ = on;
    QSettings().setValue("view/loop", on);
    emit loopChanged(on);
}

void AppModel::setMode(ViewMode mode) {
    if (state().mode == mode) return;
    state().mode = mode;
    markDirty();
    requestFrames();
    emit stateChanged();
}

void AppModel::setSingleSource(const QString& name) {
    state().SetSingleSource(name.toStdString());
    markDirty();
    requestFrames();
    emit stateChanged();
}

void AppModel::notifyStateChanged(bool structural) {
    if (structural) markDirty();
    requestFrames();
    emit stateChanged();
}

void AppModel::markDirty() {
    if (dirty_) return;
    dirty_ = true;
    emit dirtyChanged(true);
}

void AppModel::notifyFramesUpdated() {
    noteScrubReady();
    emit framesUpdated();
}

// ---- comparison in one action ----------------------------------------------------------------------

bool AppModel::hasSource(const std::string& name) const { return store_->FindSource(name) != nullptr; }

AppModel::CompareView AppModel::compareView() const {
    const ViewportState& st = state();
    if (st.mode == ViewMode::Grid) return CompareView::Grid;
    if (st.CompareConfigured()) return CompareView::BeforeAfter;
    return CompareView::AfterOnly;
}

std::string AppModel::beforeSource() const {
    for (const char* n : {"source", "color_sr", "color_nr", "color_fg", "result"})
        if (hasSource(n)) return n;
    return store_->Sources().empty() ? std::string("source") : store_->Sources().front().name;
}

std::string AppModel::afterSource() const {
    const std::string before = beforeSource();
    for (const char* n : {"result", "color_fg", "color_nr", "color_sr"})
        if (hasSource(n) && n != before) return n;
    return before;
}

std::vector<AppModel::Chip> AppModel::chips() const {
    std::vector<Chip> out;
    for (const auto& s : store_->Sources()) {
        if (!s.version.empty()) continue;
        Chip c;
        c.source = s.name;
        c.label = HumanSourceName(s.name);
        c.tooltip = SourceTooltip(s);
        if (!s.pass.empty())
            for (const auto& v : store_->Sources())
                if (!v.version.empty() && v.pass == s.pass) c.versions.push_back(v);
        out.push_back(std::move(c));
    }
    std::stable_sort(out.begin(), out.end(), [](const Chip& a, const Chip& b) {
        const int ra = SourceRank(a.source), rb = SourceRank(b.source);
        return ra != rb ? ra < rb : a.source < b.source;
    });
    for (size_t i = 0; i < out.size() && i < 9; ++i) out[i].hotkey = static_cast<int>(i) + 1;  // keys 1..9 follow the chip order
    return out;
}

void AppModel::setCompareView(CompareView view) {
    ViewportState& st = state();
    switch (view) {
        case CompareView::BeforeAfter: {
            const std::string before = beforeSource();
            std::string after = afterSource();
            // keep what the single view shows (when it exists) as the «after» side
            if (st.mode == ViewMode::Single && !st.layers.empty() && st.layers[0].source != before && hasSource(st.layers[0].source)) after = st.layers[0].source;
            st.SetCompare(before, after);
            break;
        }
        case CompareView::AfterOnly: {
            const LayerState* side = st.CompareSide(true);
            st.SetAfterOnly(side ? side->source : (st.layers.empty() ? afterSource() : st.layers[0].source));
            break;
        }
        case CompareView::Grid: st.mode = ViewMode::Grid; break;
    }
    notifyStateChanged();
}

bool AppModel::applyPreset(ComparePreset preset) {
    std::string a, b;
    switch (preset) {
        case ComparePreset::BeforeAfter:
            a = beforeSource();
            b = afterSource();
            break;
        case ComparePreset::SrVsNr:
            a = "color_sr";
            b = "color_nr";
            break;
        case ComparePreset::SourceVsDepth:
            a = "source";
            b = hasSource("depth_dlss") ? "depth_dlss" : "depth_raw";
            break;
    }
    if (!hasSource(a) || !hasSource(b) || a == b) {
        emit message(tr("Для сравнения нужны %1 и %2").arg(HumanSourceName(a), HumanSourceName(b)));
        return false;
    }
    state().SetCompare(a, b);
    notifyStateChanged();
    return true;
}

void AppModel::showSource(const std::string& source) {
    ViewportState& st = state();
    if (compareView() == CompareView::BeforeAfter) {
        LayerState& b = st.layers[static_cast<size_t>(st.wipe.layerB)];
        if (b.source != source) {
            b.source = source;
            b.display = LayerState::DefaultDisplayFor(source);
            b.autoRange = true;
        }
        st.selectedLayer = st.wipe.layerB;
    } else {
        st.SetAfterOnly(source);
    }
    notifyStateChanged();
}

void AppModel::toggleWipe() {
    ViewportState& st = state();
    if (st.mode == ViewMode::Overlay) {
        ViewportState probe = st;  // would the wipe compare two different sources?
        probe.wipe.enabled = true;
        if (probe.CompareConfigured()) {
            st.wipe.enabled = !st.wipe.enabled;
            notifyStateChanged();
            return;
        }
    }
    setCompareView(CompareView::BeforeAfter);
}

// ---- project screen ------------------------------------------------------------------------------

StageEntry* AppModel::stageEntry(const std::string& stage) {
    for (auto& s : project_.stages)
        if (s.name == stage) return &s;
    return nullptr;
}

void AppModel::refreshPlan() {
    planTimer_.stop();
    plan_ = ProcessPlan{};
    outcome_ = RunOutcome{};
    sourceInfo_ = SourceInfo{};
    planError_.clear();
    history_.clear();
    if (const ViewportSource* src = store_->FindSource("source")) {
        sourceInfo_.width = src->width;
        sourceInfo_.height = src->height;
        sourceInfo_.fps = store_->FpsOf("source");
        sourceInfo_.frames = src->lastFrame + 1;
    }
    if (!hasProject() || !std::filesystem::exists(project_.sourceVideo)) {
        planError_ = hasProject() ? tr("видео не найдено") : tr("нет проекта");
        emit planChanged();
        return;
    }
    try {
        VideoDecoder probe(project_.sourceVideo, VideoDecoder::Probe());
        sourceInfo_.audio = probe.HasAudio();
        sourceInfo_.codec = probe.Info().codecName;
        sourceInfo_.decoded = probe.Info().decoded;
        sourceInfo_.bitDepth = probe.Info().bitDepth;
        sourceInfo_.chroma = probe.Info().chroma;
        ProcessOptions o;
        o.input = project_.sourceVideo;
        o.output = project_.resultVideo.empty() ? project_.sourceVideo.parent_path() / (project_.sourceVideo.stem().string() + "_result" + ResultExtensionFor(project_.codec))
                                                : project_.resultVideo;
        o.passesRoot = project_.passesRoot;
        o.stages = StagesFromProject(project_);
        o.keepVersions = project_.passVersionsKeep;
        o.forceStages = force_;
        o.sourceHash = project_.SourceHash();
        o.vramBytes = device_ && !device_->IsWarp() ? device_->DedicatedVideoMemory() : 0;  // the VRAM advice of depth
        plan_ = PlanProcess(o);
        outcome_ = EstimateRun(plan_, o.stages, sourceInfo_);
        // the result's layout: the encoder card's choice against the codec and the source (an impossible pair is a plan error)
        const OutputFormat of = ResolveOutputFormat(project_.codec, project_.outputFormat, sourceInfo_.decoded);
        outcome_.codec = project_.codec;
        outcome_.format = of.desc;
        outcome_.format.width = outcome_.width;
        outcome_.format.height = outcome_.height;
        for (const auto& v : ListPassVersions(project_.passesRoot))
            if (!v.current) ++history_[v.pass];
    } catch (const std::exception& e) {
        planError_ = QString::fromUtf8(e.what());
    }
    emit planChanged();
}

void AppModel::schedulePlan() { planTimer_.start(); }

QString AppModel::planBlocker() const {
    for (const auto& d : plan_.stages)
        if (!d.error.empty()) return QString::fromStdString(d.error);
    return {};
}

void AppModel::setStageEnabled(const std::string& stage, bool on) {
    const bool had = stageEntry(stage) != nullptr;  // a v3 file without the entry: inserted (PR-13) - a change whatever its default
    if (StageEntry* e = project_.EnsureStage(stage); e && (!had || e->enabled != on)) {
        e->enabled = on;
        markDirty();
        emit projectChanged();
        schedulePlan();
    }
}

void AppModel::setStageParam(const std::string& stage, const std::string& key, const nlohmann::json& value) {
    const bool had = stageEntry(stage) != nullptr;
    StageEntry* e = project_.EnsureStage(stage);
    if (!e) return;
    if (!had) e->enabled = false;  // the card shows a stage the file lacks as off: an edit keeps it so (PR-13)
    if (had && e->params.contains(key) && e->params[key] == value) return;
    // another backend: a model from the old backend's list goes (VDA Large -> DA3 leaves no VDA model in the project)
    if (key == "backend" && value.is_string()) e->params = ParamsForBackendChange(stage, e->params, value.get<std::string>());
    e->params[key] = value;
    markDirty();
    emit projectChanged();
    schedulePlan();
}

void AppModel::setStageParams(const std::string& stage, const nlohmann::json& params) {
    if (!params.is_object()) return;
    const bool had = stageEntry(stage) != nullptr;
    StageEntry* e = project_.EnsureStage(stage);
    if (!e || (had && e->params == params)) return;
    if (!had) e->enabled = false;
    e->params = params;
    markDirty();
    emit projectChanged();
    schedulePlan();
}

void AppModel::setForce(const std::string& stage, bool on) {
    if (on) force_.insert(stage);
    else force_.erase(stage);
    schedulePlan();
}

void AppModel::clearForce() {
    if (force_.empty()) return;
    force_.clear();
    schedulePlan();
}

void AppModel::setEncode(const QString& codec, const std::map<std::string, std::string>& options) { setEncode(codec, options, project_.outputFormat); }

void AppModel::setEncode(const QString& codec, const std::map<std::string, std::string>& options, const OutputFormatRequest& format) {
    const std::string name = codec.toStdString();
    const bool same = project_.codec == name && project_.codecOptions == options && project_.outputFormat == format;
    project_.codec = name;
    project_.codecOptions = options;
    project_.outputFormat = format;
    const std::filesystem::path result = ResultPathFor(project_.resultVideo, name);  // ProRes lives in .mov, FFV1 in .mkv
    if (result != project_.resultVideo) {
        project_.resultVideo = result;
        reloadSources();  // a result under the old name is no longer the project's
        emit message(tr("Результат: %1").arg(QString::fromStdWString(result.filename().wstring())));
    }
    if (same) return;
    markDirty();
    emit projectChanged();
    schedulePlan();  // the layout and the container are part of the plan
}

void AppModel::setPassesRoot(const QString& dir) {
    const std::filesystem::path p(dir.toStdWString());
    if (p.empty() || p == project_.passesRoot) return;
    project_.passesRoot = p;
    markDirty();
    reloadSources();  // the passes found there and the plan against them
    emit projectChanged();
    emit message(tr("Папка пассов: %1").arg(dir));
}

QStringList AppModel::processArgs() const {
    QStringList args{"process", "--project", QString::fromStdWString(project_.file.wstring()), "--disable-unavailable"};
    if (!force_.empty()) {
        QString list;
        for (const auto& s : force_) list += (list.isEmpty() ? "" : ",") + QString::fromStdString(s);
        args << "--force" << list;
    }
    return args;
}

QString AppModel::stageTitle(const std::string& stage) const {
    if (stage == "encode") return tr("Кодирование");
    const StageSchema* schema = FindStageSchema(stage);
    QString title = schema ? QString::fromStdString(schema->title) : QString::fromStdString(stage);
    const StageEntry* entry = nullptr;
    for (const auto& s : project_.stages)
        if (s.name == stage) entry = &s;
    if (!entry) return title;
    const nlohmann::json p = EffectiveStageParams(stage, entry->params);
    if (stage == "upscale" && p.contains("scale") && p["scale"].is_number()) title += QString(" ×%1").arg(p["scale"].get<double>());
    if (stage == "fg" && p.contains("multiplier") && p["multiplier"].is_number()) title += QString(" ×%1").arg(p["multiplier"].get<int>());
    return title;
}

QString AppModel::sourceHashStatus() const {
    if (plan_.sourceHash.empty()) return {};
    int passes = 0, other = 0;
    for (const auto& s : store_->Sources()) {
        if (!s.manifest || !s.version.empty()) continue;
        ++passes;
        if (!s.manifest->sourceHash.empty() && s.manifest->sourceHash != plan_.sourceHash) ++other;
    }
    if (passes == 0) return tr("пассов пока нет");
    if (other == 0) return tr("совпадает с пассами");
    return tr("%1 от другого исходника").arg(Plural(other, tr("пасс"), tr("пасса"), tr("пассов")));
}

AppModel::StageCardState AppModel::cardState(const std::string& stage) const {
    StageCardState st;
    st.tone = "none";
    const StageEntry* entry = nullptr;
    for (const auto& s : project_.stages)
        if (s.name == stage) entry = &s;
    if (!entry || !entry->enabled) {
        st.state = tr("выключена");
        st.tone = "off";
        if (stage == kScenesPass) {  // PR-13: a list on disk is read all the same (the cache's convention); threshold 0 turns it off
            const auto& disk = plan_.scenesOnDisk;
            if (disk && !disk->off && disk->count > 0)
                st.sub = tr("действует список с диска: %1 · без сброса — порог 0").arg(Plural(static_cast<int>(disk->count), tr("склейка"), tr("склейки"), tr("склеек")));
            else st.sub = tr("стадии не сбрасывают историю на склейках");
        }
        return st;
    }
    const StageDecision* d = nullptr;
    for (const auto& s : plan_.stages)
        if (s.name == stage) d = &s;
    if (!d) {
        st.state = planError_.isEmpty() ? tr("—") : planError_;
        return st;
    }
    const std::string primary = d->outputs.empty() ? stage : d->outputs.back();
    const int history = history_.count(primary) ? history_.at(primary) : 0;
    const StageEstimate* est = nullptr;
    for (const auto& e : outcome_.stages)
        if (e.stage == stage) est = &e;
    const QString eta = est ? FormatDuration(est->seconds) : QString();
    std::string created;
    if (const ViewportSource* src = store_->FindSource(primary); src && src->manifest) created = src->manifest->created;
    if (created.empty()) created = d->currentCreated;  // a pass without frames (scenes) is no viewport source
    const StageSchema* schema = FindStageSchema(stage);
    if (d->action == "reuse") {
        st.state = d->kind == "legacy" ? tr("переиспользуется (без отпечатка)") : tr("переиспользуется");
        st.sub = created.empty() ? tr("0 мин") : tr("0 мин · посчитано %1").arg(HumanWhen(created));
        st.version = QString("v%1").arg(history + 1);
        st.tone = "ok";
    } else if (d->action == "restore") {
        st.state = tr("вернётся версия %1").arg(VersionLabel(d->version));
        st.sub = tr("0 мин · без пересчёта");
        st.version = QString("v%1").arg(history + 1);
        st.tone = "ok";
    } else {
        st.runs = true;
        st.tone = "run";
        st.version = QString("v%1").arg(d->current ? history + (d->retire ? 2 : 1) : 1);
        if (d->kind == "missing") {
            st.state = tr("будет посчитано");
        } else if (d->kind == "params_changed") {
            QStringList parts;
            if (d->diff.contains("params"))
                for (const auto& [key, pair] : d->diff["params"].items()) {
                    const ParamSpec* spec = schema ? schema->Find(key) : nullptr;
                    if (!spec && FindMovedParam(stage, key)) {  // PR-13: nr.scene_cut is the threshold of the stage scenes now
                        parts << tr("сброс на склейках перенесён в «Склейки»");
                        continue;
                    }
                    const QString label = spec ? QString::fromStdString(spec->label) : QString::fromStdString(key);
                    const nlohmann::json o = pair.is_array() && pair.size() > 0 ? pair[0] : nlohmann::json(), n = pair.is_array() && pair.size() > 1 ? pair[1] : nlohmann::json();
                    // labels of the backend each value belongs to (depth.model: «Small» of VDA, a raw id of DA3)
                    const std::string newBackend = StageBackend(stage, d->params);
                    const nlohmann::json& bd = d->diff["params"].contains("backend") ? d->diff["params"]["backend"] : nlohmann::json();
                    const std::string oldBackend = bd.is_array() && !bd.empty() && bd[0].is_string() ? bd[0].get<std::string>() : newBackend;
                    const QString now = spec ? QString::fromStdString(ParamValueLabel(*spec, n, newBackend)) : QString::fromStdString(n.dump());
                    const QString was = spec ? QString::fromStdString(ParamValueLabel(*spec, o, oldBackend)) : QString::fromStdString(o.dump());
                    // a null that is not «at the default» (omitDefault labels that): the key is absent on one side - the old pass predates it
                    // («null → 0.24» on every NR pass of before PR-6), or it is one backend's parameter across a switch
                    if (spec && !spec->omitDefault && o.is_null() != n.is_null()) {
                        const bool switched = !bd.is_null() || !spec->backends.empty();
                        if (o.is_null() && !switched) parts << tr("%1: новая настройка, %2").arg(label, now);
                        else parts << QString("%1: %2").arg(label, o.is_null() ? now : was);
                        continue;
                    }
                    parts << QString("%1 %2 → %3").arg(label, was, now);
                }
            st.state = tr("пересчёт: %1").arg(parts.isEmpty() ? QString::fromStdString(d->detail) : parts.join(", "));
        } else if (d->kind == "input_changed" && d->pending) {  // PR-13: the list is analysed first; the run may reuse the pass
            st.state = d->pendingRule == "no_cut"      ? tr("пересчёт, если найдутся склейки")
                       : d->pendingRule == "same_list" ? tr("пересчёт, если склейки изменятся")
                                                       : tr("пересчёт, если пересчитается стадия выше");
            st.sub = tr("список склеек посчитается первым");
        } else if (d->kind == "input_changed" && d->diff.contains("inputs") && d->diff["inputs"] == nlohmann::json::array({kScenesPass})) {
            st.state = tr("пересчёт: изменились склейки");
            const ViewportSource* src = store_->FindSource(primary);
            const int64_t was = src && src->manifest ? src->manifest->stageParams.value("scene_cut_count", int64_t{0}) : 0;
            QString now = "0";
            bool inPlan = false;
            for (const auto& s : plan_.stages)
                if (s.name == kScenesPass) {
                    inPlan = true;
                    now = s.sceneCuts.known ? QString::number(s.sceneCuts.count) : QString("?");  // not analysed yet: not known
                }
            if (!inPlan && plan_.scenesOnDisk) now = QString::number(plan_.scenesOnDisk->count);  // the stage off: the list on disk
            if (!d->inputs.count(kScenesPass)) now = "0";
            st.sub = QString("%1 → %2").arg(was).arg(now);
        } else if (d->kind == "input_changed") {
            st.state = tr("пересчёт: изменился вход");
            QStringList names;
            if (d->diff.contains("inputs"))
                for (const auto& n : d->diff["inputs"]) names << HumanSourceName(n.get<std::string>());
            if (!names.isEmpty()) st.sub = tr("из-за «%1»").arg(names.join(", "));
        } else if (d->kind == "source_changed") {
            st.state = tr("другой исходник");
        } else if (d->kind == "forced") {
            st.state = tr("пересчёт (принудительно)");
        } else if (d->kind == "incomplete") {
            st.state = tr("досчитать: %1").arg(QString::fromStdString(d->detail));
        } else if (d->kind == "ab_pass") {
            st.state = tr("пересчёт: пасс снят с переключателями A/B отдельного dlssvid nr");
        } else if (d->kind == "tool_changed") {
            st.state = tr("пересчёт: другая версия инструмента");
            const nlohmann::json& app = d->diff.contains("tool") ? d->diff["tool"].value("app", nlohmann::json()) : nlohmann::json();
            if (app.is_array() && app.size() == 2 && app[0].is_string() && app[1].is_string()) {
                const auto added = AddedToolRevisions(app[0].get<std::string>(), app[1].get<std::string>());
                auto has = [&](const char* r) { return std::find(added.begin(), added.end(), r) != added.end(); };
                if (has("nr.i0")) st.state = tr("пересчёт: при силе 0 модель не вызывается");
                else if (has("nr.msize")) st.state = tr("пересчёт: защита интерфейса при масках разного размера");
                else if (has("nr.gain")) st.state = tr("пересчёт: сила правки теперь после модели");
                else if (has("nr.ev")) st.state = tr("пересчёт: экспозиция в ступенях EV");
            }
        } else {
            st.state = tr("пересчёт");
        }
        QStringList sub;
        if (!eta.isEmpty()) sub << eta;
        if (!st.sub.isEmpty()) sub << st.sub;
        if (d->retire) sub << tr("v%1 останется на диске").arg(d->current ? history + 1 : 1);
        st.sub = sub.join(" · ");
    }
    if (stage == kScenesPass) {  // PR-13: the card of the list - how many cuts, and what a new threshold costs
        const SceneCutsPlan& sc = d->sceneCuts;
        const QString n = Plural(static_cast<int>(sc.count), tr("склейка"), tr("склейки"), tr("склеек"));
        if (sc.known && sc.off) {
            st.state = tr("выкл: без сброса");
        } else if (d->action != "run") {
            if (sc.known) st.sub += (st.sub.isEmpty() ? "" : " · ") + n;
        } else if (sc.rederive) {
            QStringList parts{tr("без декодирования"), sc.before >= 0 && sc.before != sc.count ? tr("станет %1 (было %2)").arg(n).arg(sc.before) : n};
            if (d->retire) parts << tr("v%1 останется на диске").arg(d->current ? history + 1 : 1);
            st.sub = parts.join(" · ");
        } else {
            st.sub += (st.sub.isEmpty() ? "" : " · ") + tr("один проход декодирования исходника");
        }
    } else if (StageReadsSceneCuts(stage, d->params)) {  // PR-13: what the cuts do to a stage that reads them
        const auto it = d->inputs.find(kScenesPass);
        int64_t count = 0;
        for (const auto& s : plan_.stages)
            if (s.name == kScenesPass && s.sceneCuts.known) count = s.sceneCuts.count;
        if (it != d->inputs.end() && count == 0 && plan_.scenesOnDisk) count = plan_.scenesOnDisk->count;  // the stage off: the list on disk
        if (d->pending || (it != d->inputs.end() && it->second == kScenesPendingId)) {
            st.scenes = tr("склейки найдутся перед этой стадией");
        } else if (it != d->inputs.end() && count > 0) {
            const std::string backend = StageBackend(stage, d->params);
            const QString what = stage == "flow"                         ? tr("векторы на них нулевые")
                                 : stage == "upscale" && backend == "worker" ? tr("окна RealBasicVSR через них не переходят")
                                 : stage == "upscale"                     ? tr("история DLSS начинается заново")
                                 : stage == "nr"                          ? tr("история модели и подавления мерцания сбрасывается")
                                                                          : tr("вместо промежуточных кадров повтор последнего кадра сцены");
            st.scenes = tr("склейки: %1 — %2").arg(count).arg(what);
        }
    }
    // upscale: only DLSS reads the guides (PR-13) - nis, bicubic, trt and worker without them is no loss
    if (schema && !schema->guides.empty() && (stage == "nr" || stage == "fg" || (stage == "upscale" && UpscaleReadsGuides(d->params)))) {
        bool any = false;
        for (const auto& g : schema->guides) any = any || d->inputs.count(g) > 0;
        const bool guidesOff = stage == "nr" && d->params.contains("guides") && d->params["guides"].is_boolean() && !d->params["guides"].get<bool>();
        if (stage == "nr" && NrBypassParams(d->params)) {
            // intensity 0: the model is not called, its guides are not needed (PR-7)
        } else if (guidesOff) st.warning = tr("глубина и векторы выключены: каждый кадр обрабатывается отдельно");
        else if (!any) st.warning = tr("без глубины и векторов: качество ниже");
    }
    // the advice of the plan (TASK-0039, depth): licence, VRAM; an error blocks «Обработать»
    QStringList warn;
    if (!st.warning.isEmpty()) warn << st.warning;
    if (!d->error.empty()) warn << QString::fromStdString(d->error);
    for (const auto& w : d->warnings) warn << QString::fromStdString(w);
    st.warning = warn.join("\n");
    // the advice of the stage; the list's card says it its own way (above)
    if (st.runs && !d->info.empty() && stage != kScenesPass) st.sub = st.sub.isEmpty() ? QString::fromStdString(d->info) : st.sub + " · " + QString::fromStdString(d->info);
    return st;
}

// ---- start screen, recents, unsaved changes -------------------------------------------------------

AppModel::ResultInfo AppModel::resultInfo() const {
    ResultInfo r;
    if (!hasProject() || project_.resultVideo.empty()) return r;
    const ViewportSource* src = store_->FindSource("result");
    if (!src) return r;
    r.exists = true;
    r.path = QString::fromStdWString(project_.resultVideo.wstring());
    r.width = src->width;
    r.height = src->height;
    r.fps = store_->FpsOf("result");
    r.frames = src->lastFrame >= 0 ? src->lastFrame + 1 : 0;
    r.audio = resultAudio_;
    r.codec = resultCodec_;
    r.format = resultFormat_;
    r.modified = HumanWhen(FileTimeIso8601(project_.resultVideo));
    r.runSeconds = lastRun_;
    return r;
}

QString AppModel::formatLabel(const std::string& codec, const FrameDesc& format) {
    QString name;
    if (codec == "hevc" || codec == "hevc_nvenc") name = "HEVC";
    else if (codec == "h264" || codec == "h264_nvenc") name = "H.264";
    else if (codec == "av1" || codec == "av1_nvenc") name = "AV1";
    else if (codec == "prores" || codec == "prores_ks") name = "ProRes";
    else if (codec == "ffv1") name = "FFV1";
    else if (!codec.empty()) name = QString::fromStdString(codec).toUpper();
    const std::string_view chroma = ChromaName(format.format);
    const QString layout = tr("%1 бит %2:%3:%4").arg(format.HighBitDepth() ? format.bitDepth : 8).arg(chroma[0]).arg(chroma[1]).arg(chroma[2]);
    return name.isEmpty() ? layout : name + " " + layout;
}

QString AppModel::resultSummary() const {
    const ResultInfo r = resultInfo();
    if (!r.exists) return {};
    QStringList parts;
    if (r.width > 0 && r.height > 0) parts << QString("%1×%2").arg(r.width).arg(r.height);
    if (r.fps > 0.0) parts << tr("%1 fps").arg(QString::number(r.fps, 'g', 4));
    if (r.frames > 0) parts << Plural(static_cast<int>(r.frames), tr("кадр"), tr("кадра"), tr("кадров"));
    if (!r.codec.empty()) parts << formatLabel(r.codec, r.format);
    parts << (r.audio ? tr("со звуком") : tr("без звука"));
    if (!r.modified.isEmpty()) parts << tr("обновлён %1").arg(r.modified);
    if (r.runSeconds > 0.0) parts << tr("готов за %1").arg(FormatClock(r.runSeconds));
    return parts.join(" · ");
}

std::vector<std::string> AppModel::exportablePasses() const {
    std::vector<std::string> out;
    for (const auto& s : store_->Sources())
        if (!s.isVideo && s.version.empty() && s.manifest && !s.pass.empty()) out.push_back(s.pass);
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
        const int ra = SourceRank(a), rb = SourceRank(b);
        return ra != rb ? ra < rb : a < b;
    });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

unsigned long long AppModel::freeSpace() const {
    if (!hasProject()) return 0;
    std::filesystem::path p = project_.passesRoot.empty() ? project_.sourceVideo.parent_path() : project_.passesRoot;
    std::error_code ec;
    while (!p.empty() && !std::filesystem::exists(p, ec)) {  // the passes folder may not exist yet: its nearest parent
        const std::filesystem::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }
    const std::filesystem::space_info info = std::filesystem::space(p, ec);
    return ec ? 0ULL : static_cast<unsigned long long>(info.available);
}

bool AppModel::exportResult(const QString& to) {
    const ResultInfo r = resultInfo();
    if (!r.exists) {
        emit message(tr("Результата ещё нет"));
        return false;
    }
    try {
        const std::filesystem::path dst(to.toStdWString());
        std::error_code ec;
        if (std::filesystem::exists(dst, ec) && std::filesystem::equivalent(dst, project_.resultVideo, ec)) {
            emit message(tr("Это и есть файл результата"));
            return false;
        }
        if (!dst.parent_path().empty()) std::filesystem::create_directories(dst.parent_path());
        std::filesystem::copy_file(project_.resultVideo, dst, std::filesystem::copy_options::overwrite_existing);
        emit message(tr("Результат сохранён: %1").arg(to));
        return true;
    } catch (const std::exception& e) {
        emit message(tr("Не удалось сохранить результат: %1").arg(e.what()));
        return false;
    }
}

bool AppModel::closeProject() {
    if (!hasProject()) return false;
    setPlaying(false);
    project_ = Project{};
    force_.clear();
    lastRun_ = 0.0;
    resultAudio_ = false;
    cancelProxies();
    store_->SetSources({});
    if (!controlMaskBusy_) controlMask_ = {};
    emit controlMaskChanged();
    dirty_ = false;
    emit dirtyChanged(false);
    emit sourcesChanged();
    emit stateChanged();
    refreshPlan();  // clears the plan
    emit resultChanged();
    emit projectChanged();
    emit message(tr("Проект закрыт"));
    return true;
}

void AppModel::setLastRun(double seconds) {
    lastRun_ = seconds;
    emit resultChanged();
}

AppModel::CloseAction AppModel::closeAction() const {
    if (!dirty_ || !hasProject()) return CloseAction::Nothing;
    return project_.file.empty() ? CloseAction::Ask : CloseAction::AutoSave;
}

// ---- presets (PR-9) --------------------------------------------------------------------------------

namespace {

nlohmann::json CanonEffective(const std::string& stage, const nlohmann::json& params) { return CanonicalizeJson(EffectiveStageParams(stage, params)); }

// the notes of a read preset for a message: what was left out, and a slot's values with errors apart
void SplitNotes(const std::vector<PresetIssue>& issues, QStringList* skipped, QStringList* slotErrors, bool withStage) {
    for (const auto& i : issues) {
        if (i.fatal() || i.kind == PresetIssueKind::MissingStage || i.kind == PresetIssueKind::NoEncode) continue;
        const QString t = PresetStore::IssueText(i, withStage);
        if (i.kind == PresetIssueKind::SlotValue) slotErrors->append(t);
        else skipped->append(t);
    }
    skipped->removeDuplicates();
    slotErrors->removeDuplicates();
}

std::string Utf8(const QString& s) { return s.toStdString(); }

}  // namespace

nlohmann::json AppModel::stageParamsOf(const std::string& stage) const {
    for (const auto& s : project_.stages)
        if (s.name == stage) return s.params;
    return nlohmann::json::object();
}

AppModel::PresetResult AppModel::applyStagePreset(const std::string& stage, const QString& file, bool withoutBackup) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const PresetLibrary lib = PresetStore::Library();
    const auto path = PresetStore::Path(file);
    const PresetLoad l = LoadPreset(path, stage, lib.IsSlotFile(path) ? PresetRead::Slot : PresetRead::Strict);
    r.name = QString::fromStdString(l.preset.name);
    r.file = file;
    if (l.scopeKnown && l.preset.scope == PresetScope::Pipeline) {
        r.problems << tr("Это пресет обработки — он применяется из «Пресет обработки» в шапке «Стадии»");
        return r;
    }
    if (!l.ok()) {
        r.problems = PresetStore::IssueTexts(l.issues, true);
        return r;
    }
    const nlohmann::json cur = stageParamsOf(stage);
    std::vector<std::string> ignored;
    const nlohmann::json next = ApplyPreset(stage, cur, l.preset.stages.at(stage).params, PresetScope::Stage, &ignored);
    if (CanonEffective(stage, next) == CanonEffective(stage, cur)) {  // nothing to change: no slot, not dirty
        r.ok = true;
        r.message = tr("Настройки «%1» уже совпадают с «%2»").arg(StageNameGenitive(stage), r.name);
        emit message(r.message);
        return r;
    }
    if (!withoutBackup) {
        try {
            lib.SaveSlot(PresetSlot::BeforeApply, stage, MakePreset(PresetNameFor("Перед применением", project_.sourceVideo), stage, cur));
        } catch (const std::exception& e) {
            r.backupFailed = true;
            r.problems << tr("Не удалось сохранить «Перед применением»: %1").arg(QString::fromUtf8(e.what()));
            return r;
        }
    }
    const auto changes = DiffStageParams(stage, cur, next, PresetScope::Stage);
    setStageParams(stage, next);
    refreshPlan();  // one event: the plan now, not after the sliders' timer
    QStringList parts;
    for (size_t i = 0; i < changes.size(); ++i) {
        if (i == 3) {
            parts << tr("ещё %1").arg(changes.size() - 3);
            break;
        }
        parts << ChangeText(stage, changes[i]);
    }
    QString msg = tr("Пресет «%1» применён").arg(r.name);
    if (!parts.isEmpty()) msg += ": " + parts.join(", ");
    if (!withoutBackup) msg += tr(" · прежние — в «Перед применением»");
    QStringList skipped, slotErrors;
    SplitNotes(l.issues, &skipped, &slotErrors, false);
    for (const auto& k : ignored) skipped << tr("%1 не относится к этому методу").arg(LabelOf(stage, k));
    if (!skipped.isEmpty()) msg += tr(" · пропущено: %1").arg(skipped.join("; "));
    if (!slotErrors.isEmpty()) msg += tr(" · в слоте были и ошибки: %1").arg(slotErrors.join("; "));
    if (const StageEntry* e = stageEntry(stage); e && !e->enabled) msg += tr(" · стадия выключена");
    r.ok = true;
    r.notes = skipped + slotErrors;
    r.message = msg;
    emit message(msg);
    return r;
}

AppModel::PresetResult AppModel::saveStagePreset(const std::string& stage, const QString& name, const QString& replacing, bool forNewProjects) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const std::string n = TrimPresetName(Utf8(name));
    r.name = QString::fromStdString(n);
    if (!PresetNameProblem(n).empty()) {
        r.problems << tr("Имя пресета — от 1 до 120 символов, без управляющих.");
        return r;
    }
    if (ReservedPresetName(n)) {
        r.problems << tr("Так называются автосохранения — выберите другое имя");
        return r;
    }
    if (replacing.isEmpty())
        if (const auto taken = PresetStore::FindByName(r.name, PresetScope::Stage, stage)) {
            r.exists = true;
            r.file = *taken;
            return r;
        }
    const nlohmann::json cur = stageParamsOf(stage);
    try {
        const auto path = PresetStore::Library().Save(MakePreset(n, stage, cur), replacing.isEmpty() ? std::filesystem::path() : PresetStore::Path(replacing));
        r.file = PresetStore::Text(path);
    } catch (const PresetError& e) {
        r.problems << tr("Текущие настройки содержат ошибку — исправьте её перед сохранением:");
        r.problems << PresetStore::IssueTexts(e.issues, true);
        return r;
    } catch (const std::exception& e) {
        r.problems << tr("Не удалось сохранить пресет: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    if (forNewProjects) PresetStore::SetNewProjectPreset(r.file);
    if (const QString left = presetLeftOut(stage); !left.isEmpty()) r.notes << left;
    r.ok = true;
    r.message = tr("Пресет «%1» сохранён").arg(r.name) + (forNewProjects ? tr(" · для новых проектов") : QString());
    emit message(r.message);
    return r;
}

AppModel::PresetResult AppModel::resetStageCard(const std::string& stage, bool withoutBackup) {
    PresetResult r;
    const StageSchema* schema = FindStageSchema(stage);
    if (!stageEntry(stage) || !schema) return r;
    const nlohmann::json cur = stageParamsOf(stage);
    // the form's keys go; the backend and the keys of the JSON editor (format, ...) stay - setStageParams replaces the whole object, so
    // ResetStageParams carries them over (TASK-0046)
    const nlohmann::json next = ResetStageParams(stage, cur);
    r.ok = true;
    if (CanonEffective(stage, next) == CanonEffective(stage, cur)) return r;  // nothing to reset: no slot
    if (schema->cardPresets && !withoutBackup) {
        try {
            PresetStore::Library().SaveSlot(PresetSlot::BeforeReset, stage, MakePreset(PresetNameFor("Перед сбросом", project_.sourceVideo), stage, cur));
        } catch (const std::exception& e) {
            r.ok = false;
            r.backupFailed = true;
            r.problems << tr("Не удалось сохранить «Перед сбросом»: %1").arg(QString::fromUtf8(e.what()));
            return r;
        }
    }
    setStageParams(stage, next);
    refreshPlan();
    if (schema->cardPresets) {
        r.message = tr("Настройки «%1» сброшены").arg(StageNameGenitive(stage)) + (withoutBackup ? QString() : tr(" · прежние — в «Перед сбросом»"));
        emit message(r.message);
    }
    return r;
}

AppModel::PresetResult AppModel::exportCurrentStagePreset(const std::string& stage, const QString& to) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const Preset p = MakePreset(PresetNameFor(Utf8(StageName(stage)), project_.sourceVideo), stage, stageParamsOf(stage));
    r.name = QString::fromStdString(p.name);
    try {
        SavePreset(PresetStore::Path(to), p);
    } catch (const PresetError& e) {
        r.problems << tr("Текущие настройки содержат ошибку — исправьте её перед сохранением:");
        r.problems << PresetStore::IssueTexts(e.issues, true);
        return r;
    } catch (const std::exception& e) {
        r.problems << tr("Не удалось записать файл: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    if (const QString left = presetLeftOut(stage); !left.isEmpty()) r.notes << left;
    r.ok = true;
    r.file = to;
    r.message = tr("Пресет сохранён в %1").arg(QFileInfo(to).fileName());
    if (!r.notes.isEmpty()) r.message += " · " + r.notes.join(" · ");  // what the file lacks, not in silence (principle 11)
    emit message(r.message);
    return r;
}

AppModel::PresetResult AppModel::applyPipelinePreset(const QString& file, bool withoutBackup) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const PresetLibrary lib = PresetStore::Library();
    const auto path = PresetStore::Path(file);
    const PresetLoad l = LoadPreset(path, {}, lib.IsSlotFile(path) ? PresetRead::Slot : PresetRead::Strict);
    r.name = QString::fromStdString(l.preset.name);
    r.file = file;
    if (l.scopeKnown && l.preset.scope == PresetScope::Stage && !l.preset.stage().empty()) {  // a card preset without its stage: the reason below
        r.problems << PresetStore::CardPresetRefusal(l.preset.stage());
        return r;
    }
    if (!l.ok()) {
        r.problems = PresetStore::IssueTexts(l.issues, true, true);
        return r;
    }
    Project next = project_;  // one copy, every stage and the encoder at once: one projectChanged, one plan
    std::vector<PresetIssue> notes = l.issues;
    ApplyPipelinePreset(next, l.preset, &notes);
    const PipelineDiff diff = DiffPipeline(project_, next);
    if (diff.empty()) {
        r.ok = true;
        r.message = tr("Настройки обработки уже совпадают с «%1»").arg(r.name);
        emit message(r.message);
        return r;
    }
    if (!withoutBackup) {
        try {
            lib.SaveSlot(PresetSlot::BeforeApply, kPipelineTarget, MakePipelinePreset(PresetNameFor("Перед применением", project_.sourceVideo), project_));
        } catch (const std::exception& e) {
            r.backupFailed = true;
            r.problems << tr("Не удалось сохранить «Перед применением»: %1").arg(QString::fromUtf8(e.what()));
            return r;
        }
    }
    const bool renamed = next.resultVideo != project_.resultVideo;
    project_ = std::move(next);
    markDirty();
    emit projectChanged();
    if (renamed) reloadSources();  // a result under the old name is no longer the project's; it plans too
    else refreshPlan();
    // the summary: up to four items, the full difference in the log
    QStringList items;
    for (const StageDiff& sd : diff.stages) {
        QString t = StageName(sd.stage);
        if (sd.enabledWas != sd.enabledNow) t += sd.enabledNow ? tr(" включена") : tr(" выключена");
        if (sd.enabledWas == sd.enabledNow && sd.params.size() == 1) t += ": " + ChangeText(sd.stage, sd.params[0]);
        else if (!sd.params.empty()) t += QString(" (%1)").arg(sd.params.size());
        items << t;
        Log()->info("{}", tr("Пресет обработки «%1»: %2: %3").arg(r.name, StageName(sd.stage), StageDiffText(sd, 1000)).toStdString());
    }
    if (!diff.encode.empty()) {
        items << (diff.encode.size() == 1 ? tr("Кодирование: %1").arg(EncodeChangeText(diff.encode[0])) : tr("Кодирование (%1)").arg(diff.encode.size()));
        QStringList enc;
        for (const auto& c : diff.encode) enc << EncodeChangeText(c);
        Log()->info("{}", tr("Пресет обработки «%1»: Кодирование: %2").arg(r.name, enc.join("; ")).toStdString());
    }
    if (items.size() > 4) {
        const qsizetype more = items.size() - 4;
        items = items.mid(0, 4);
        items << tr("ещё %1").arg(more);
    }
    QString msg = tr("Пресет обработки «%1» применён: %2").arg(r.name, items.join(", "));
    if (renamed) msg += tr(" · результат: %1").arg(QString::fromStdWString(project_.resultVideo.filename().wstring()));
    msg += tr(" · подробности — в «Лог»");
    if (!withoutBackup) msg += tr(" · прежние — в «Перед применением»");
    QStringList skipped, slotErrors;
    SplitNotes(notes, &skipped, &slotErrors, true);
    if (!skipped.isEmpty()) msg += tr(" · пропущено: %1").arg(skipped.join("; "));
    if (!slotErrors.isEmpty()) msg += tr(" · в слоте были и ошибки: %1").arg(slotErrors.join("; "));
    r.ok = true;
    r.notes = skipped + slotErrors;
    r.message = msg;
    emit message(msg);
    return r;
}

AppModel::PresetResult AppModel::savePipelinePreset(const QString& name, const QString& replacing, bool forNewProjects) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const std::string n = TrimPresetName(Utf8(name));
    r.name = QString::fromStdString(n);
    if (!PresetNameProblem(n).empty()) {
        r.problems << tr("Имя пресета — от 1 до 120 символов, без управляющих.");
        return r;
    }
    if (ReservedPresetName(n)) {
        r.problems << tr("Так называются автосохранения — выберите другое имя");
        return r;
    }
    if (replacing.isEmpty())
        if (const auto taken = PresetStore::FindByName(r.name, PresetScope::Pipeline)) {
            r.exists = true;
            r.file = *taken;
            return r;
        }
    try {
        const auto path = PresetStore::Library().Save(MakePipelinePreset(n, project_), replacing.isEmpty() ? std::filesystem::path() : PresetStore::Path(replacing));
        r.file = PresetStore::Text(path);
    } catch (const PresetError& e) {
        // every stage is checked, the switched-off ones too, and the encoder: the plan checks only what runs
        r.problems << tr("Текущие настройки содержат ошибку — исправьте её перед сохранением:");
        r.problems << PresetStore::IssueTexts(e.issues, true, true);
        return r;
    } catch (const std::exception& e) {
        r.problems << tr("Не удалось сохранить пресет: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    if (forNewProjects) PresetStore::SetNewProjectPreset(r.file);
    if (const QString left = presetLeftOut(kPipelineTarget); !left.isEmpty()) r.notes << left;
    r.ok = true;
    r.message = tr("Пресет обработки «%1» сохранён").arg(r.name) + (forNewProjects ? tr(" · для новых проектов") : QString());
    emit message(r.message);
    return r;
}

AppModel::PresetResult AppModel::exportCurrentPipelinePreset(const QString& to) {
    PresetResult r;
    if (!hasProject()) {
        r.problems << tr("Нет проекта");
        return r;
    }
    const Preset p = MakePipelinePreset(PresetNameFor("Обработка", project_.sourceVideo), project_);
    r.name = QString::fromStdString(p.name);
    try {
        SavePreset(PresetStore::Path(to), p);
    } catch (const PresetError& e) {
        r.problems << tr("Текущие настройки содержат ошибку — исправьте её перед сохранением:");
        r.problems << PresetStore::IssueTexts(e.issues, true, true);
        return r;
    } catch (const std::exception& e) {
        r.problems << tr("Не удалось записать файл: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    if (const QString left = presetLeftOut(kPipelineTarget); !left.isEmpty()) r.notes << left;
    r.ok = true;
    r.file = to;
    r.message = tr("Пресет обработки сохранён в %1").arg(QFileInfo(to).fileName());
    if (!r.notes.isEmpty()) r.message += " · " + r.notes.join(" · ");
    emit message(r.message);
    return r;
}

QString AppModel::presetLeftOut(const std::string& target) const {
    if (target == kPipelineTarget) return PresetStore::LeftOutText(PipelinePresetLeftOut(project_), true);
    return PresetStore::LeftOutText(StagePresetLeftOut(target, stageParamsOf(target), PresetScope::Stage), false);
}

std::vector<ParamChange> AppModel::previewStagePreset(const std::string& stage, const Preset& preset) const {
    const auto it = preset.stages.find(stage);
    if (it == preset.stages.end()) return {};
    const nlohmann::json cur = stageParamsOf(stage);
    return DiffStageParams(stage, cur, ApplyPreset(stage, cur, it->second.params, PresetScope::Stage), PresetScope::Stage);
}

PipelineDiff AppModel::previewPipelinePreset(const Preset& preset) const {
    Project next = project_;
    ApplyPipelinePreset(next, preset);
    return DiffPipeline(project_, next);
}

QString AppModel::seedNewProject() {
    const QString mark = PresetStore::NewProjectPreset();
    if (mark.isEmpty()) return {};
    const auto path = PresetStore::Path(PresetStore::Dir()) / PresetStore::Path(mark);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {  // removed by hand or not roamed here: the mark goes
        PresetStore::SetNewProjectPreset({});
        return tr(" · пресет для новых проектов «%1» не найден — отметка снята, взяты встроенные умолчания").arg(mark);
    }
    const PresetLoad l = LoadPreset(path);
    const QString name = l.preset.name.empty() ? mark : QString::fromStdString(l.preset.name);
    if (!l.ok())  // newer, broken, a value out of range: the built-in defaults, the mark stays (a version may be rolled back)
        return tr(" · пресет для новых проектов «%1» не применён: %2; взяты встроенные умолчания").arg(name, PresetStore::IssueTexts(l.issues, true, true).join("; "));
    std::vector<PresetIssue> notes = l.issues;
    ApplyPipelinePreset(project_, l.preset, &notes);  // a card preset goes to its stage; no slot is written for a new project
    for (const auto& n : notes) Log()->info("new project: preset '{}': {}", l.preset.name, n.text);
    if (l.preset.scope == PresetScope::Pipeline) return tr(" · настройки обработки — из «%1» (для новых проектов)").arg(name);
    return tr(" · «%1» — из пресета «%2» (для новых проектов)").arg(StageName(l.preset.stage()), name);
}

QString AppModel::lastDir(const QString& key) const { return QSettings().value("dialogs/" + key).toString(); }

void AppModel::rememberDir(const QString& key, const QString& file) {
    if (!file.isEmpty()) QSettings().setValue("dialogs/" + key, QFileInfo(file).absolutePath());
}

namespace {

// The stage a pass folder belongs to and how the recents name it: «глубина посчитана», «векторы посчитаны».
struct StageWord {
    const char* stage;
    const char* name;
    const char* done;
};
constexpr StageWord kStageWords[] = {{"scenes", "склейки", "посчитаны"},
                                     {"depth", "глубина", "посчитана"},
                                     {"flow", "векторы", "посчитаны"},
                                     {"upscale", "апскейл", "посчитан"},
                                     {"nr", "NR", "посчитан"},
                                     {"fg", "генерация кадров", "посчитана"}};

std::string StageOfPass(const std::string& pass) {
    if (pass.rfind("depth_", 0) == 0) return "depth";
    if (pass.rfind("mv_", 0) == 0) return "flow";
    if (pass == "color_sr") return "upscale";
    if (pass == "color_nr") return "nr";
    if (pass == "color_fg") return "fg";
    return {};
}

bool DlssSrDllPresent() {
    for (const auto& dir : NvidiaDllSearchPaths())
        if (std::filesystem::exists(dir / "nvngx_dlss.dll")) return true;
    return false;
}

}  // namespace

std::vector<AppModel::Recent> AppModel::recents() const {
    std::vector<Recent> out;
    for (const QString& path : QSettings().value("recent/files").toStringList()) {
        Recent r;
        r.path = path;
        const QFileInfo fi(path);
        r.exists = fi.exists();
        r.isProject = path.endsWith(".dlssvid.json", Qt::CaseInsensitive) || path.endsWith(".json", Qt::CaseInsensitive);
        r.title = fi.fileName();
        if (r.isProject && r.title.endsWith(".dlssvid.json", Qt::CaseInsensitive)) r.title.chop(13);  // ".dlssvid.json"
        else if (r.title.contains('.')) r.title = r.title.left(r.title.lastIndexOf('.'));
        if (!r.exists) {
            r.info = tr("файл не найден");
        } else if (r.isProject) {
            try {
                const Project p = Project::Load(std::filesystem::path(path.toStdWString()));
                const auto passes = FrameStore::DiscoverPasses(p.passesRoot);
                if (!p.sourceVideo.empty() && std::filesystem::exists(p.sourceVideo)) r.video = QString::fromStdWString(p.sourceVideo.wstring());
                const bool ready = !p.resultVideo.empty() && std::filesystem::exists(p.resultVideo);
                if (ready) r.result = QString::fromStdWString(p.resultVideo.wstring());
                // stages with a pass on disk, in the project's order: «глубина и NR посчитаны», «2 из 5 стадий»
                std::vector<std::string> computed;
                if (Manifest::Exists(p.passesRoot / kScenesPass)) computed.push_back("scenes");  // a data pass: DiscoverPasses leaves it out
                for (const auto& s : passes) {
                    const std::string stage = StageOfPass(s.name);
                    if (!stage.empty() && std::find(computed.begin(), computed.end(), stage) == computed.end()) computed.push_back(stage);
                }
                int enabled = 0, done = 0;
                QStringList names;
                QString participle;
                for (const StageEntry& st : p.stages) {
                    if (st.enabled) ++enabled;
                    if (std::find(computed.begin(), computed.end(), st.name) == computed.end()) continue;
                    if (st.enabled) ++done;
                    for (const StageWord& w : kStageWords)
                        if (st.name == w.stage) {
                            names << tr(w.name);
                            participle = tr(w.done);
                        }
                }
                if (ready) r.info = tr("результат готов");
                else if (!passes.empty()) r.info = Plural(static_cast<int>(passes.size()), tr("пасс"), tr("пасса"), tr("пассов"));
                else r.info = tr("не обработан");
                if (ready) {
                    r.state = tr("результат готов");
                } else if (!names.isEmpty()) {
                    const QString list = names.size() == 1 ? names[0] : names.mid(0, names.size() - 1).join(", ") + tr(" и ") + names.last();
                    r.state = list + " " + (names.size() == 1 ? participle : tr("посчитаны"));
                    r.progress = tr("%1 из %2 стадий").arg(done).arg(enabled);
                } else {
                    r.progress = tr("не обработан");
                }
            } catch (const std::exception&) {
                r.info = tr("проект не читается");
                r.state = r.info;
            }
        } else {
            const bool hasProject = QFileInfo::exists(QString::fromStdWString(std::filesystem::path(path.toStdWString()).replace_extension(".dlssvid.json").wstring()));
            r.info = hasProject ? tr("видео · есть проект") : tr("видео · не обработан");
            r.video = path;
            r.progress = hasProject ? tr("есть проект") : tr("не обработан");
        }
        if (r.exists) {
            r.when = HumanWhen(FileTimeIso8601(std::filesystem::path(path.toStdWString())));
            r.info += " · " + r.when;
        }
        out.push_back(std::move(r));
    }
    return out;
}

void AppModel::addRecent(const QString& path) {
    QSettings settings;
    QStringList list = settings.value("recent/files").toStringList();
    list.removeAll(path);
    list.prepend(path);
    while (list.size() > 10) list.removeLast();
    settings.setValue("recent/files", list);
    emit recentsChanged();
}

void AppModel::clearRecents() {
    QSettings().remove("recent/files");
    emit recentsChanged();
}

QString AppModel::readiness() const {
    QString adapter = QString::fromStdString(device_->AdapterName());
    if (device_->IsWarp()) return tr("%1 (WARP, программный рендер) — без NVIDIA GPU доступны NIS, бикубик и смешивание кадров").arg(adapter);
    if (adapter.startsWith("NVIDIA GeForce ")) adapter.remove(0, 15);  // mockup: «RTX 4070 Ti SUPER · драйвер 616.92 · …»
    QStringList parts{adapter};
    const std::string driver = NvidiaDriverFromUmd(device_->UmdDriverVersion()).ToString();
    if (!driver.empty()) parts << tr("драйвер %1").arg(QString::fromStdString(driver));
    const bool sr = DlssSrDllPresent(), nr = !FindNrDll().empty(), fg = !FindDlssgDll().empty();
    if (sr && nr && fg) {
        parts << tr("DLSS SR, Neural Rendering и генерация кадров доступны");
    } else {  // what is missing, by its DLL
        parts << (sr ? tr("DLSS SR ✓") : tr("DLSS SR ✗ (нет nvngx_dlss.dll)"));
        parts << (nr ? tr("Neural Rendering ✓") : tr("Neural Rendering ✗ (нет nvngx_dlssnr.dll)"));
        parts << (fg ? tr("генерация кадров ✓") : tr("генерация кадров ✗ (нет nvngx_dlssg.dll)"));
    }
    return parts.join(" · ");
}

AppModel::Readiness AppModel::readinessLevel() const {
    if (device_->IsWarp()) return Readiness::Software;
    return DlssSrDllPresent() && !FindNrDll().empty() && !FindDlssgDll().empty() ? Readiness::Ready : Readiness::Partial;
}

void AppModel::setEngineerMode(bool on) {
    if (on == engineerMode_) return;
    engineerMode_ = on;
    QSettings().setValue("view/engineerMode", on);
    emit engineerModeChanged(on);
}

}  // namespace dlssvid
