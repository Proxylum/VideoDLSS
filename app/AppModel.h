#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu/D3D12Device.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

// Application state shared by the panels: project, viewport state, frame store, playback.
// All Qt widgets talk to the model; the model owns the D3D12 device and the frame cache.
class AppModel : public QObject {
    Q_OBJECT
public:
    explicit AppModel(bool warp, QObject* parent = nullptr);
    ~AppModel() override;

    D3D12Device& device() { return *device_; }
    FrameStore& store() { return *store_; }
    Project& project() { return project_; }
    ViewportState& state() { return project_.viewport; }
    const ViewportState& state() const { return project_.viewport; }
    bool hasProject() const { return !project_.sourceVideo.empty(); }

    // Names the viewport can show: "source", "result" and pass names.
    QStringList sourceNames() const;
    // Sources the current state needs (for prefetch).
    std::vector<std::string> neededSources() const;
    QString cliPath() const;

public slots:
    void openVideo(const QString& file);
    void openProject(const QString& file);
    bool saveProject();
    bool saveProjectAs(const QString& file);
    void reloadSources();  // after a stage finished: rescan pass folders
    void setFrame(int64_t frame);
    void stepFrame(int delta);
    void setPlaying(bool playing);
    bool playing() const { return playTimer_.isActive(); }
    void setMode(ViewMode mode);
    void setSingleSource(const QString& name);
    void notifyStateChanged();  // after direct edits of state()

signals:
    void projectChanged();
    void sourcesChanged();
    void stateChanged();
    void frameChanged(qint64 frame);
    void playingChanged(bool playing);
    void message(const QString& text);

private:
    void onPlayTick();
    void requestFrames();

    std::unique_ptr<D3D12Device> device_;
    std::unique_ptr<FrameStore> store_;
    Project project_;
    QTimer playTimer_;
};

}  // namespace dlssvid
