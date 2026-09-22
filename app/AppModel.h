#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <map>
#include <set>

#include "gpu/D3D12Device.h"
#include "pipeline/Estimates.h"
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
        int hotkey = 0;                        // 1..9: the key that shows this chip (chip order); 0 beyond the ninth
    };
    CompareView compareView() const;   // derived from the state: a configured wipe, a grid, or a single view
    std::string beforeSource() const;  // "source" when present, else the first colour pass
    std::string afterSource() const;   // "result", else the last colour pass, different from the before side when it can
    std::vector<Chip> chips() const;   // one per current source in pipeline order, versions attached
    bool engineerMode() const { return engineerMode_; }

    // ---- project screen (stage 9, MR D) ----
    struct StageCardState {
        QString state;    // «переиспользуется», «пересчёт: Интенсивность 1 → 1.4», «будет посчитано», «выключена» …
        QString sub;      // «0 мин · посчитано вчера 14:02», «≈ 2 мин · v1 останется на диске»
        QString version;  // «v1», «v2»
        QString warning;  // «без глубины и векторов: качество ниже»; empty when fine
        QString tone;     // ok | run | off | none
        bool runs = false;
    };
    const ProcessPlan& plan() const { return plan_; }        // PlanProcess over the project (empty without a source)
    const RunOutcome& outcome() const { return outcome_; }  // what the run gives: size, rate, time, disk
    const SourceInfo& sourceInfo() const { return sourceInfo_; }
    QString planError() const { return planError_; }         // why there is no plan (bad parameters, no video)
    QString sourceHashStatus() const;                        // «совпадает с пассами» / «N пассов от другого исходника» / «пассов нет»
    StageCardState cardState(const std::string& stage) const;
    StageEntry* stageEntry(const std::string& stage);
    bool forced(const std::string& stage) const { return force_.count(stage) > 0; }
    QStringList processArgs() const;  // dlssvid process --project <file> --disable-unavailable [--force a,b]
    QString stageTitle(const std::string& stage) const;  // «Апскейл ×2», «Генерация кадров ×3», «Кодирование»

    // ---- start screen, recents, unsaved changes (stage 9, MR E) ----
    struct Recent {
        QString path;   // a project file or a video
        QString title;  // «face»
        QString info;   // «результат готов · вчера 14:02», «3 пасса · сегодня 10:12», «не обработан», «файл не найден»
        bool exists = true;
        bool isProject = false;
    };
    std::vector<Recent> recents() const;  // newest first (QSettings recent/files, at most 10)
    QString readiness() const;            // the GPU, the driver and which DLSS features have their DLL
    bool dirty() const { return dirty_; }
    enum class CloseAction { Nothing, AutoSave, Ask };
    CloseAction closeAction() const;      // nothing to save | save into the project's file | ask (no file yet)
    QString lastDir(const QString& key) const;               // last folder of a file dialog (QSettings dialogs/<key>)
    void rememberDir(const QString& key, const QString& file);

    // ---- the result and what to do with it (TASK-0013) ----
    struct ResultInfo {
        bool exists = false;
        QString path;
        uint32_t width = 0, height = 0;
        double fps = 0.0;
        int64_t frames = 0;       // 0 when unknown
        bool audio = false;
        QString modified;         // «сегодня 19:23»
        double runSeconds = 0.0;  // the run of this session that made it; 0 when none
    };
    ResultInfo resultInfo() const;
    QString resultSummary() const;                      // «3840×1600 · 48 fps · 479 кадров · со звуком · обновлён сегодня 19:23 · готов за 11:22»; empty without a result
    std::vector<std::string> exportablePasses() const;  // current pass folders with a manifest, pipeline order
    unsigned long long freeSpace() const;               // bytes free on the volume of the passes root (0 when unknown)

public slots:
    void openVideo(const QString& file);
    void openProject(const QString& file);
    bool saveProject();
    bool saveProjectAs(const QString& file);
    bool exportResult(const QString& to);  // copy the result video to `to` (overwrites); a message either way
    bool closeProject();                   // forget the project (unsaved changes are settled by the caller): the start page
    void setLastRun(double seconds);       // the run that has just produced the result
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
    void refreshPlan();                           // plan the run now (manifests + the cached source hash)
    void schedulePlan();                          // after an edit: coalesced into one refresh
    void setStageEnabled(const std::string& stage, bool on);
    void setStageParam(const std::string& stage, const std::string& key, const nlohmann::json& value);
    void setStageParams(const std::string& stage, const nlohmann::json& params);
    void setForce(const std::string& stage, bool on);
    void clearForce();
    void setEncode(const QString& codec, const std::map<std::string, std::string>& options);
    void notifyStateChanged(bool structural = true);  // after direct edits of state(); structural ones (layers, modes) mark the project unsaved
    void markDirty();
    void addRecent(const QString& path);
    void clearRecents();
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
    void planChanged();
    void dirtyChanged(bool dirty);
    void recentsChanged();
    void resultChanged();

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
    bool dirty_ = false;
    ProcessPlan plan_;
    RunOutcome outcome_;
    SourceInfo sourceInfo_;
    QString planError_;
    std::set<std::string> force_;
    std::map<std::string, int> history_;  // pass -> previous versions on disk
    QTimer planTimer_;
    bool resultAudio_ = false;  // the result video has a sound track (probed on reload)
    double lastRun_ = 0.0;      // seconds the last run of this session took
};

}  // namespace dlssvid
