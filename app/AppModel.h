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

    // ---- comparison in one action (stage 9, MR C) ----
    enum class CompareView { BeforeAfter, AfterOnly, Grid };
    enum class ComparePreset { BeforeAfter, SrVsNr, SourceVsDepth };
    struct Chip {
        std::string source;                    // the current version of a pass, "source" or "result"
        QString label, tooltip;                // human name; the technical side
        std::vector<ViewportSource> versions;  // previous versions of the pass (the chip's menu)
    };
    CompareView compareView() const;   // derived from the state: a configured wipe, a grid, or a single view
    std::string beforeSource() const;  // "source" when present, else the first colour pass
    std::string afterSource() const;   // "result", else the last colour pass, different from the before side when it can
    std::vector<Chip> chips() const;   // one per current source in pipeline order, versions attached
    bool engineerMode() const { return engineerMode_; }

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
    void setCompareView(CompareView view);
    bool applyPreset(ComparePreset preset);      // false (with a message) when a source it needs is absent
    void showSource(const std::string& source);  // a chip: the «after» side of a comparison, else the single view
    void toggleWipe();                            // W: sets up «before | after» when no comparison is configured
    void setEngineerMode(bool on);                // remembered in QSettings
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
    void engineerModeChanged(bool on);

private:
    void onPlayTick();
    void requestFrames();
    bool hasSource(const std::string& name) const;

    std::unique_ptr<D3D12Device> device_;
    std::unique_ptr<FrameStore> store_;
    Project project_;
    QTimer playTimer_;
    Rational timelineFps_{0, 1};  // {0,1}: the base source's rate
    double playStep_ = 1.0 / 24.0;
    bool engineerMode_ = false;
};

}  // namespace dlssvid
