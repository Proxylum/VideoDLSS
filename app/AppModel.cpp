#include "AppModel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <algorithm>

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
    if (state().frame >= store_->FrameCount() && store_->FrameCount() > 0) state().frame = store_->FrameCount() - 1;
    emit sourcesChanged();
    requestFrames();
    emit stateChanged();
}

void AppModel::requestFrames() { store_->SetCurrentFrame(state().frame, neededSources()); }

void AppModel::setFrame(int64_t frame) {
    const int64_t count = store_->FrameCount();
    if (count > 0) frame = std::clamp<int64_t>(frame, 0, count - 1);
    if (frame < 0) frame = 0;
    if (frame == state().frame) return;
    state().frame = frame;
    requestFrames();
    emit frameChanged(frame);
    emit stateChanged();
}

void AppModel::stepFrame(int delta) { setFrame(state().frame + delta); }

void AppModel::setPlaying(bool playing) {
    if (playing == playTimer_.isActive()) return;
    if (playing) {
        Rational fps = store_->Fps();
        double rate = fps.num > 0 ? fps.ToDouble() : 24.0;
        // FG results carry twice the frames of the source (ТЗ §6)
        const std::string base = state().mode == ViewMode::Grid ? state().gridSources[0] : state().layers.empty() ? "source" : state().layers[0].source;
        if (base == "color_fg") rate *= 2.0;
        playTimer_.start(static_cast<int>(std::max(1.0, 1000.0 / rate)));
    } else {
        playTimer_.stop();
    }
    emit playingChanged(playing);
}

void AppModel::onPlayTick() {
    const int64_t next = state().frame + 1;
    const std::string base = state().mode == ViewMode::Grid ? state().gridSources[0] : state().layers.empty() ? "source" : state().layers[0].source;
    // play only over computed frames: stop when the next frame does not exist
    if (store_->FrameCount() > 0 && next >= store_->FrameCount()) {
        setPlaying(false);
        return;
    }
    if (store_->GetStatus(next, base) == FrameStore::Status::Missing) {
        setPlaying(false);
        return;
    }
    setFrame(next);
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

}  // namespace dlssvid
