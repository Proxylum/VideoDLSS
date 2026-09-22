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
//
// The viewport position is a time (stage 9): every source shows its own frame at that time, the
// timeline counts frames at a chosen rate (the base layer's by default, switchable when the sources
// differ), stepping moves by one frame of the base layer, playback runs at the fastest shown rate.
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

    // ---- time axis ----
    double time() const { return state().time; }
    std::string baseSource() const;           // grid: cell 1; single / overlay: layer 0
    int64_t baseFrame() const;                // the base source's frame at the current time
    Rational timelineFps() const;             // rate the timeline counts frames at
    int64_t timelineFrameCount() const;       // frames of that rate over the duration
    int64_t timelineFrame() const;            // the current time in those frames
    double lastTime() const;                  // time of the last timeline frame
    std::vector<Rational> fpsChoices() const;  // distinct source rates, ascending
    // Frame state of the needed sources at the current time (plates and cell labels).
    FrameStates frameStates() const;
    FrameState frameStateOf(const std::string& source) const;

public slots:
    void openVideo(const QString& file);
    void openProject(const QString& file);
    bool saveProject();
    bool saveProjectAs(const QString& file);
    void reloadSources();  // after a stage finished: rescan pass folders
    void setTime(double seconds);
    void setFrame(int64_t timelineFrame);  // in frames of the timeline rate
    void stepFrame(int delta);             // by frames of the base source
    void setTimelineFps(Rational fps);     // {0,1}: follow the base source
    void setPlaying(bool playing);
    bool playing() const { return playTimer_.isActive(); }
    void setMode(ViewMode mode);
    void setSingleSource(const QString& name);
    void notifyStateChanged();   // after direct edits of state()
    void notifyFramesUpdated();  // the frame store uploaded something (the viewport's poll)

signals:
    void projectChanged();
    void sourcesChanged();
    void stateChanged();
    void timeChanged(double seconds);
    void timelineFpsChanged();
    void framesUpdated();
    void playingChanged(bool playing);
    void message(const QString& text);

private:
    void onPlayTick();
    void requestFrames();

    std::unique_ptr<D3D12Device> device_;
    std::unique_ptr<FrameStore> store_;
    Project project_;
    QTimer playTimer_;
    Rational timelineFps_{0, 1};  // {0,1}: the base source's rate
    double playStep_ = 1.0 / 24.0;
};

}  // namespace dlssvid
