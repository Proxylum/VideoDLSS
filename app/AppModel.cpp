#include "AppModel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <algorithm>
#include <cmath>

#include "SourceNames.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "stages/ParamSchema.h"
#include "stages/fg/IFrameGenerator.h"
#include "stages/nr/INrBackend.h"
#include "stages/upscale/IUpscaler.h"
#include "util/Log.h"

namespace dlssvid {

AppModel::AppModel(bool warp, QObject* parent) : QObject(parent) {
    device_ = std::make_unique<D3D12Device>(D3D12Device::Options{warp, false});
    store_ = std::make_unique<FrameStore>(*device_);
    playTimer_.setTimerType(Qt::PreciseTimer);
    connect(&playTimer_, &QTimer::timeout, this, &AppModel::onPlayTick);
    engineerMode_ = QSettings().value("view/engineerMode", false).toBool();
    planTimer_.setSingleShot(true);
    planTimer_.setInterval(250);
    connect(&planTimer_, &QTimer::timeout, this, &AppModel::refreshPlan);
}

AppModel::~AppModel() = default;

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
        reloadSources();
        dirty_ = !loaded;  // a project made from the video is not on disk yet: it is saved next to the video on close
        emit dirtyChanged(dirty_);
        emit projectChanged();
        addRecent(file);
        emit message(tr("Открыт %1").arg(file));
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

void AppModel::reloadSources() {
    store_->SetSources(project_.Sources());
    timelineFps_ = Rational{0, 1};
    state().ResolveLegacyFrame(store_->FpsOf(baseSource()));  // projects before stage 9: frame -> time at the base rate
    state().time = std::clamp(state().time, 0.0, lastTime());
    emit sourcesChanged();
    requestFrames();
    emit stateChanged();
    refreshPlan();
}

void AppModel::requestFrames() { store_->SetCurrentTime(state().time, neededSources()); }

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
        default: return FrameState::Loading;
    }
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
    requestFrames();
    emit timeChanged(seconds);
    emit stateChanged();
}

void AppModel::setFrame(int64_t timelineFrame) { setTime(static_cast<double>(timelineFrame) / timelineFps().ToDouble()); }

void AppModel::stepFrame(int delta) {
    const std::string base = baseSource();
    setTime(store_->TimeOfFrame(base, store_->FrameAt(base, time()) + delta));
}

void AppModel::setTimelineFps(Rational fps) {
    timelineFps_ = fps;
    emit timelineFpsChanged();
}

void AppModel::setPlaying(bool playing) {
    if (playing == playTimer_.isActive()) return;
    if (playing) {
        const double rate = std::max(1.0, store_->MaxFps(neededSources()));  // the fastest shown source: an FG result runs at 2x
        playStep_ = 1.0 / rate;
        playTimer_.start(static_cast<int>(std::max(1.0, 1000.0 / rate)));
    } else {
        playTimer_.stop();
    }
    emit playingChanged(playing);
}

void AppModel::onPlayTick() {
    const double next = time() + playStep_;
    // play only over computed frames: stop at the end and where the base source has no frame
    if (next > lastTime() + 1e-9 || store_->StatusAt(next, baseSource()) == FrameStore::Status::Missing) {
        setPlaying(false);
        return;
    }
    setTime(next);
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

void AppModel::notifyFramesUpdated() { emit framesUpdated(); }

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
        VideoDecoder probe(project_.sourceVideo);
        sourceInfo_.audio = probe.HasAudio();
        ProcessOptions o;
        o.input = project_.sourceVideo;
        o.output = project_.resultVideo.empty() ? project_.sourceVideo.parent_path() / (project_.sourceVideo.stem().string() + "_result.mp4") : project_.resultVideo;
        o.passesRoot = project_.passesRoot;
        o.stages = StagesFromProject(project_);
        o.keepVersions = project_.passVersionsKeep;
        o.forceStages = force_;
        o.sourceHash = project_.SourceHash();
        plan_ = PlanProcess(o);
        outcome_ = EstimateRun(plan_, o.stages, sourceInfo_);
        for (const auto& v : ListPassVersions(project_.passesRoot))
            if (!v.current) ++history_[v.pass];
    } catch (const std::exception& e) {
        planError_ = QString::fromUtf8(e.what());
    }
    emit planChanged();
}

void AppModel::schedulePlan() { planTimer_.start(); }

void AppModel::setStageEnabled(const std::string& stage, bool on) {
    if (StageEntry* e = stageEntry(stage); e && e->enabled != on) {
        e->enabled = on;
        markDirty();
        emit projectChanged();
        schedulePlan();
    }
}

void AppModel::setStageParam(const std::string& stage, const std::string& key, const nlohmann::json& value) {
    StageEntry* e = stageEntry(stage);
    if (!e) return;
    if (e->params.contains(key) && e->params[key] == value) return;
    e->params[key] = value;
    markDirty();
    emit projectChanged();
    schedulePlan();
}

void AppModel::setStageParams(const std::string& stage, const nlohmann::json& params) {
    StageEntry* e = stageEntry(stage);
    if (!e || !params.is_object() || e->params == params) return;
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

void AppModel::setEncode(const QString& codec, const std::map<std::string, std::string>& options) {
    project_.codec = codec.toStdString();
    project_.codecOptions = options;
    markDirty();
    emit projectChanged();
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
                    const QString label = spec ? QString::fromStdString(spec->label) : QString::fromStdString(key);
                    const nlohmann::json o = pair.is_array() && pair.size() > 0 ? pair[0] : nlohmann::json(), n = pair.is_array() && pair.size() > 1 ? pair[1] : nlohmann::json();
                    parts << QString("%1 %2 → %3").arg(label, spec ? QString::fromStdString(ParamValueLabel(*spec, o)) : QString::fromStdString(o.dump()),
                                                          spec ? QString::fromStdString(ParamValueLabel(*spec, n)) : QString::fromStdString(n.dump()));
                }
            st.state = tr("пересчёт: %1").arg(parts.isEmpty() ? QString::fromStdString(d->detail) : parts.join(", "));
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
        } else if (d->kind == "tool_changed") {
            st.state = tr("пересчёт: другая версия инструмента");
        } else {
            st.state = tr("пересчёт");
        }
        QStringList sub;
        if (!eta.isEmpty()) sub << eta;
        if (!st.sub.isEmpty()) sub << st.sub;
        if (d->retire) sub << tr("v%1 останется на диске").arg(d->current ? history + 1 : 1);
        st.sub = sub.join(" · ");
    }
    if (schema && !schema->guides.empty() && (stage == "nr" || stage == "fg" || stage == "upscale")) {
        bool any = false;
        for (const auto& g : schema->guides) any = any || d->inputs.count(g) > 0;
        if (!any) st.warning = tr("без глубины и векторов: качество ниже");
    }
    return st;
}

// ---- start screen, recents, unsaved changes -------------------------------------------------------

AppModel::CloseAction AppModel::closeAction() const {
    if (!dirty_ || !hasProject()) return CloseAction::Nothing;
    return project_.file.empty() ? CloseAction::Ask : CloseAction::AutoSave;
}

QString AppModel::lastDir(const QString& key) const { return QSettings().value("dialogs/" + key).toString(); }

void AppModel::rememberDir(const QString& key, const QString& file) {
    if (!file.isEmpty()) QSettings().setValue("dialogs/" + key, QFileInfo(file).absolutePath());
}

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
                const int passes = static_cast<int>(FrameStore::DiscoverPasses(p.passesRoot).size());
                if (!p.resultVideo.empty() && std::filesystem::exists(p.resultVideo)) r.info = tr("результат готов");
                else if (passes > 0) r.info = Plural(passes, tr("пасс"), tr("пасса"), tr("пассов"));
                else r.info = tr("не обработан");
            } catch (const std::exception&) {
                r.info = tr("проект не читается");
            }
        } else {
            const bool hasProject = QFileInfo::exists(QString::fromStdWString(std::filesystem::path(path.toStdWString()).replace_extension(".dlssvid.json").wstring()));
            r.info = hasProject ? tr("видео · есть проект") : tr("видео · не обработан");
        }
        if (r.exists) r.info += " · " + HumanWhen(FileTimeIso8601(std::filesystem::path(path.toStdWString())));
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
    const QString adapter = QString::fromStdString(device_->AdapterName());
    if (device_->IsWarp()) return tr("%1 (WARP, программный рендер) — без NVIDIA GPU доступны NIS, бикубик и смешивание кадров").arg(adapter);
    QStringList parts{adapter};
    const std::string driver = NvidiaDriverFromUmd(device_->UmdDriverVersion()).ToString();
    if (!driver.empty()) parts << tr("драйвер %1").arg(QString::fromStdString(driver));
    bool sr = false;
    for (const auto& dir : NvidiaDllSearchPaths()) sr = sr || std::filesystem::exists(dir / "nvngx_dlss.dll");
    parts << (sr ? tr("DLSS SR ✓") : tr("DLSS SR ✗ (нет nvngx_dlss.dll)"));
    parts << (!FindNrDll().empty() ? tr("Neural Rendering ✓") : tr("Neural Rendering ✗ (нет nvngx_dlssnr.dll)"));
    parts << (!FindDlssgDll().empty() ? tr("генерация кадров ✓") : tr("генерация кадров ✗ (нет nvngx_dlssg.dll)"));
    return parts.join(" · ");
}

void AppModel::setEngineerMode(bool on) {
    if (on == engineerMode_) return;
    engineerMode_ = on;
    QSettings().setValue("view/engineerMode", on);
    emit engineerModeChanged(on);
}

}  // namespace dlssvid
