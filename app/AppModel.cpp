#include "AppModel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <algorithm>
#include <cmath>

#include "util/Log.h"

namespace dlssvid {

AppModel::AppModel(bool warp, QObject* parent) : QObject(parent) {
    device_ = std::make_unique<D3D12Device>(D3D12Device::Options{warp, false});
    store_ = std::make_unique<FrameStore>(*device_);
    playTimer_.setTimerType(Qt::PreciseTimer);
    connect(&playTimer_, &QTimer::timeout, this, &AppModel::onPlayTick);
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
        if (std::filesystem::exists(project_.file)) {
            project_ = Project::Load(project_.file);
        }
        reloadSources();
        emit projectChanged();
        emit message(tr("Открыт %1").arg(file));
    } catch (const std::exception& e) {
        emit message(tr("Ошибка: %1").arg(e.what()));
    }
}

void AppModel::openProject(const QString& file) {
    try {
        project_ = Project::Load(std::filesystem::path(file.toStdWString()));
        reloadSources();
        emit projectChanged();
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
    requestFrames();
    emit stateChanged();
}

void AppModel::setSingleSource(const QString& name) {
    state().SetSingleSource(name.toStdString());
    requestFrames();
    emit stateChanged();
}

void AppModel::notifyStateChanged() {
    requestFrames();
    emit stateChanged();
}

void AppModel::notifyFramesUpdated() { emit framesUpdated(); }

}  // namespace dlssvid
