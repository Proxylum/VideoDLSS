#pragma once

#include <QElapsedTimer>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QTimer>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <map>
#include <optional>
#include <set>

#include "PlaybackClock.h"
#include "gpu/D3D12Device.h"
#include "pipeline/Estimates.h"
#include "pipeline/NrPreview.h"
#include "stages/StagePreset.h"
#include "StageAvailability.h"
#include "viewport/FrameStore.h"
#include "viewport/PipelinePreset.h"
#include "viewport/Project.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

// The NR preview's viewport source (PR-10): <passes root>/.preview/nr/<key>/color_nr - a «version» of color_nr for the chips, never in
// the project file.
inline constexpr const char* kNrPreviewSource = "color_nr@preview";

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
    // Frame state of the needed sources at the current time (plates and cell labels); Stale while the store holds the
    // last frame for one that is still loading.
    FrameStates frameStates() const;
    FrameState frameStateOf(const std::string& source) const;
    // Every needed source has its frame at `t` resident (or none at all): playback steps only onto such frames.
    bool framesReadyAt(double t) const;

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
        QString scenes;   // a stage that reads the scene list: what the cuts do to it («склейки: 3 — векторы на них нулевые», PR-13)
        bool runs = false;
    };
    const ProcessPlan& plan() const { return plan_; }        // PlanProcess over the project (empty without a source)
    const RunOutcome& outcome() const { return outcome_; }  // what the run gives: size, rate, time, disk
    const SourceInfo& sourceInfo() const { return sourceInfo_; }
    // The colour NR reads in the run: the frame the stages make (the upscale's; an anamorphic source's grid, not the result stretched after
    // it - TASK-0073), else the source's; {0, 0} unknown. The preview's estimate and calibration key
    std::pair<uint32_t, uint32_t> nrColorSize() const;
    QString planError() const { return planError_; }         // why there is no plan (bad parameters, no video)
    // a stage the run would refuse although the plan exists (depth does not fit the GPU, TASK-0039): «Обработать» waits
    QString planBlocker() const;
    QString sourceHashStatus() const;                        // «совпадает с пассами» / «N пассов от другого исходника» / «пассов нет»
    StageCardState cardState(const std::string& stage) const;
    StageEntry* stageEntry(const std::string& stage);
    bool stageOn(const std::string& stage) const;  // the project's stage is ticked (a stage the file lacks is off)
    bool forced(const std::string& stage) const { return force_.count(stage) > 0; }
    QStringList processArgs() const;  // dlssvid process --project <file> --disable-unavailable [--force a,b]
    // TASK-0071: whether NR / FG can run here - the line of their card. «Проверить» runs `dlssvid nr|fg --check` with the project's
    // settings in a process of its own (one feature on the GPU, a couple of seconds); its result stands while the settings and the DLL
    // stay the same. availabilityChanged when a check starts or ends.
    StageAvailability availability(const std::string& stage) const;
    void checkStage(const std::string& stage);
    bool checkRunning() const { return checkProcess_ != nullptr; }
    // A running check is stopped (nothing stored) when a task or a preview starts on the GPU (ProjectTasks), and with the project.
    void cancelCheck();
    void setCheckProgramForTests(const QString& program) { checkProgram_ = program; }  // empty: cliPath()
    // what checkStage runs (tests): `nr --check --params-file <paramsFile> --json <jsonFile> [--check-size WxH]` or `fg --check ...`;
    // checkParams - what goes into the settings file: the stage's effective settings without the keys the schema lacks
    QStringList checkArgs(const std::string& stage, const QString& paramsFile, const QString& jsonFile) const;
    nlohmann::json checkParams(const std::string& stage) const;
    QString stageTitle(const std::string& stage) const;  // «Апскейл ×2», «Генерация кадров ×3», «Кодирование»

    // ---- start screen, recents, unsaved changes (stage 9, MR E) ----
    struct Recent {
        QString path;   // a project file or a video
        QString title;  // «face»
        QString info;   // «результат готов · вчера 14:02», «3 пасса · сегодня 10:12», «не обработан», «файл не найден»
        bool exists = true;
        bool isProject = false;
        // The start screen's card (mockup «1 · Стартовый экран»): «face — результат готов» over «1920×800 24 fps →
        // 3840×1600 48 fps · вчера 14:02»; the size lines and the frame come from RecentPreviews.
        QString state;     // «результат готов», «глубина и NR посчитаны»; empty when nothing was computed
        QString progress;  // «2 из 5 стадий», «не обработан», «есть проект»; empty when the result is ready
        QString when;      // «вчера 14:02»
        QString video;     // the source video (its preview and size line); empty when it does not exist
        QString result;    // the result video when it exists (its size line after «→», its frame as the preview)
    };
    std::vector<Recent> recents() const;  // newest first (QSettings recent/files, at most 10)
    QString readiness() const;            // the GPU, the driver and which DLSS features have their DLL
    enum class Readiness { Ready, Partial, Software };  // every DLL in place / some missing / WARP
    Readiness readinessLevel() const;
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
        std::string codec;        // "hevc" ... and the layout the file decodes into (TASK-0034)
        FrameDesc format;
        QString modified;         // «сегодня 19:23»
        double runSeconds = 0.0;  // the run of this session that made it; 0 when none
    };
    // ---- presets (PR-9): the NR card's «Пресет ▾» and the «Пресет обработки ▾» of the stages' header ----
    // Applying and resetting save the state before them into the slot «Перед применением» / «Перед сбросом» first; when that cannot be
    // written nothing changes and `backupFailed` asks for a repeat with withoutBackup. One projectChanged and one plan per action.
    struct PresetResult {
        bool ok = false;
        bool backupFailed = false;  // the slot was not written: nothing changed
        bool exists = false;        // save: the name is taken - again with `replacing` (the file in `file`)
        QString name, message, file;
        QStringList notes, problems;  // in Russian
    };
    // the card (stage: one with card presets, "nr")
    PresetResult applyStagePreset(const std::string& stage, const QString& file, bool withoutBackup = false);
    PresetResult saveStagePreset(const std::string& stage, const QString& name, const QString& replacing = {}, bool forNewProjects = false);
    PresetResult resetStageCard(const std::string& stage, bool withoutBackup = false);  // «Сбросить» of every card; a slot for the ones with presets
    PresetResult exportCurrentStagePreset(const std::string& stage, const QString& to);
    // the whole processing: every stage's switch and settings and the encoder at once
    PresetResult applyPipelinePreset(const QString& file, bool withoutBackup = false);
    PresetResult savePipelinePreset(const QString& name, const QString& replacing = {}, bool forNewProjects = false);
    PresetResult exportCurrentPipelinePreset(const QString& to);
    // «Не войдут в пресет: …» of the current settings for the save dialog; "" when nothing is left out (target: a stage or kPipelineTarget)
    QString presetLeftOut(const std::string& target) const;
    // the menus' preview: what applying would change; nothing is written
    std::vector<ParamChange> previewStagePreset(const std::string& stage, const Preset& preset) const;
    PipelineDiff previewPipelinePreset(const Preset& preset) const;

    ResultInfo resultInfo() const;
    QString resultSummary() const;                      // «3840×1600 · 48 fps · 479 кадров · HEVC 10 бит 4:2:0 · со звуком · обновлён сегодня 19:23 · готов за 11:22»; empty without a result
    // «HEVC 10 бит 4:2:0», «ProRes 10 бит 4:2:2»: a codec name and a frame layout in the words of the GUI (empty codec: the layout alone).
    static QString formatLabel(const std::string& codec, const FrameDesc& format);
    std::vector<std::string> exportablePasses() const;  // current pass folders with a manifest (folder names), pipeline order
    // «Экспорт пассов», one task per folder: dlssvid export --from-dir <passes root>/<pass> -o <dir>/<pass> --format exr
    QStringList exportPassArgs(const std::string& pass, const QString& dir) const;
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
    // TASK-0070: a run encodes a new result into `file`: the viewer lets every video of that file go - the result, or the source of a
    // project made from an earlier result - and builds no proxy of it (no side shows it); Windows does not rename the new file over one
    // that is open. A running proxy build that reads the file is stopped in any case (one of a project open before too); when no video
    // shown is that file nothing else happens (resultReleased stays false). restoreResult when the run has ended, before the sources
    // are read again.
    void releaseResult(const std::filesystem::path& file);
    void restoreResult() { releasedFile_.clear(); }
    bool resultReleased() const { return !releasedFile_.empty(); }
    bool resultBeingReplaced() const { return isReleased(project_.resultVideo); }  // the file let go is this project's result
    void setTime(double seconds);
    // The time of frame `frame` of the source (its rate, not the base layer's: color_fg may run at twice it), in the middle of the
    // frame - the chips of the scene list (PR-13)
    void seekToSourceFrame(int64_t frame);
    void setFrame(int64_t timelineFrame);  // in frames of the timeline rate
    void stepFrame(int delta);             // by frames of the base source
    void setTimelineFps(Rational fps);     // {0,1}: follow the base source
    void setPlaying(bool playing);
    bool playing() const { return playing_; }
    // tests: the monotonic clock playback is paced by, in nanoseconds (default: QElapsedTimer), and one tick of the
    // playback timer driven by hand against that clock
    void setPlaybackClock(std::function<int64_t()> nowNs);
    void playbackTick() { onPlayTick(); }
    bool loop() const { return loop_; }              // playback wraps to the start at the end (QSettings view/loop)
    void setLoop(bool on);
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
    void setEncode(const QString& codec, const std::map<std::string, std::string>& options);  // keeps the bit depth / chroma
    // The result's encoder, its options and its bit depth / chroma (TASK-0034); the result file takes the container the codec
    // needs (.mp4 / .mov / .mkv) and the plan is refreshed (the layout is part of it).
    void setEncode(const QString& codec, const std::map<std::string, std::string>& options, const OutputFormatRequest& format);
    void setPassesRoot(const QString& dir);  // where the stages write and read their passes (the project's «passes»)
    void notifyStateChanged(bool structural = true);  // after direct edits of state(); structural ones (layers, modes) mark the project unsaved
    void markDirty();
    void addRecent(const QString& path);
    void clearRecents();
    void notifyFramesUpdated();  // the frame store uploaded something (the viewport's poll)

    // Scrubbing proxies (TASK-0040): after the sources load, `dlssvid proxy --background` builds the missing ones of the
    // source and result videos (long keyframe intervals only, within view/proxyMaxGiB and a quarter of the free disk);
    // when it ends the store takes them. cancelProxies stops it (a run that rewrites the result, closing the project).
    void startProxies();
    void cancelProxies();
    QString proxyState(const std::string& source) const;  // "ready" | "building" | "none"
    QStringList proxyBuildSources() const;                // the sources a running build makes proxies for ("source", "result")
    bool proxyBuilding() const { return proxyProcess_ && proxyProcess_->state() != QProcess::NotRunning; }
    // Scrubbing latency: a time set by the user (not playback) until the exact frames of every needed source are on
    // the GPU; logged as p50 / p90 every 30 jumps («viewport scrub» in the log). The samples since the last log line.
    const std::vector<double>& scrubLatencies() const { return scrubMs_; }

public:  // no slots: a struct and plain methods
    // ---- the control mask of NR (PR-8): one image under the passes root, imported off the GUI thread ----
    struct ControlMaskState {
        enum class Status { None, Importing, Ready, Unreadable } status = Status::None;
        QString file;         // the source's name (Importing: the file being read)
        QString problem;      // Unreadable: why, in Russian
        QString fingerprint;  // 8 hex digits
        uint32_t width = 0, height = 0;
        bool grayscale = false, alphaIgnored = false;
        bool broken = false;  // a folder without a loadable manifest: «Убрать» deletes it
        int64_t clamped = 0;
    };
    const ControlMaskState& controlMask() const { return controlMask_; }  // re-read from disk by reloadSources (not while importing)
    bool controlMaskBusy() const { return controlMaskBusy_; }              // an import runs: «Обработать», the versions panel wait
    // The image -> <passes root>/mask_control (ImportControlMask) on the model's thread; the viewport lets go of the folder first. The
    // end: controlMaskFinished, then controlMaskChanged (sources, plan and state reloaded).
    void importControlMask(const QString& file);
    // «Убрать»: into the history (a folder without a loadable manifest is deleted); false with the Russian text in `error`.
    bool removeControlMask(QString* error = nullptr);

    // ---- the NR preview (PR-10): computed by PreviewRunner under <passes root>/.preview/nr, shown here as color_nr@preview ----
    struct NrPreviewShown {
        std::filesystem::path dir;   // the entry (its color_nr is the source)
        std::filesystem::path root;  // the passes root it was made under
        std::string key;
        NrPreviewMode mode = NrPreviewMode::Frame;
        int64_t first = 0, last = 0, warmup = 0;
        bool cached = false;               // shown from the cache, without a run
        std::vector<std::string> dropped;  // inputs a frame went without (depth_dlss, mv_dlss)
    };
    const std::optional<NrPreviewShown>& nrPreview() const { return nrPreview_; }
    // On the «after» side of «before | after» (the wipe kept where it was when a comparison is set up), «before» by nrPreviewBefore.
    // `seekToFirst`: the cursor to the preview's first frame. Not a change of the project: nothing is marked unsaved.
    void showNrPreview(const NrPreviewShown& shown, bool seekToFirst);
    void clearNrPreview();  // the store lets its files go first (SetSources waits for the loads), its side back to the pass
    // «До: Текущее улучшение | Апскейл» (QSettings card/nr/previewBefore: "nr" | "sr"); the source shown on the «before» side
    void setNrPreviewBefore(const std::string& which);
    std::string nrPreviewBeforeChoice() const;  // "nr" | "sr"
    std::string nrPreviewBefore() const;        // color_nr when it covers the preview's frames (choice "nr"), else color_sr, else source
    // The plan without the force marks of the stages above NR: the preview's gate and key (a mark would rerun them for nothing).
    const ProcessPlan& previewPlan() const { return planNoForce_ ? *planNoForce_ : plan_; }
    void refreshPlanIfPending();  // a plan scheduled after an edit is made now (the preview reads it)
    bool saveForRun();            // the default result path set and the project saved before a run; false when it cannot be saved
    // «Посчитать этапы до улучшения»: process --project <file> --disable-unavailable --stages <stages> --no-encode (no --force: the
    // marks stay for «Обработать»)
    QStringList upstreamArgs(const std::vector<std::string>& stages) const;
    uint64_t generation() const { return generation_; }  // +1 when the project or its passes root changes: a late preview is dropped

signals:
    void projectChanged();
    void sourcesChanged();
    void stateChanged();
    void timeChanged(double seconds);
    void timelineFpsChanged();
    void framesUpdated();
    void playingChanged(bool playing);
    void loopChanged(bool on);
    void message(const QString& text);
    void engineerModeChanged(bool on);
    void planChanged();
    void dirtyChanged(bool dirty);
    void recentsChanged();
    void resultChanged();
    void proxiesChanged();
    void controlMaskChanged();
    // ok: imported, replaced or the same pixels again (changed false; message «эта маска уже загружена»); else the refusal in Russian
    void controlMaskFinished(bool ok, bool changed, const QString& message);
    void nrPreviewChanged();       // shown, cleared or its «before» side changed (PR-10): the chips, the labels, the viewport
    void passesContextChanged();   // another project or passes root: a preview in flight is cancelled
    void availabilityChanged();    // a check started or ended (TASK-0071)

private:
    // the project's and the NR preview shown; dropped: without mask_control and its versions
    std::vector<ViewportSource> viewportSources(bool dropControlMask) const;
    ProcessOptions planOptions(bool withForce);  // the plan's options: with the force marks (plan_) or without (planNoForce_); hashes the source once
    void dropNrPreview();                              // another project or root: the preview state goes, generation_ + 1
    void adoptProject(Project next);                   // a project read to the end replaces the open one (a file that cannot be read changes nothing)
    void sanitizePreviewSide();                        // after a load: a layer naming the preview goes to color_nr, else afterSource()
    QString seedNewProject();  // openVideo of a video without a project: the preset marked for new projects; the message's tail
    nlohmann::json stageParamsOf(const std::string& stage) const;  // the project's, {} for a stage it lacks
    void noteScrubReady();  // a scrub jump's frames are all on the GPU: one latency sample
    void onPlayTick();
    void schedulePlayTick();
    void requestFrames();
    bool hasSource(const std::string& name) const;

    std::unique_ptr<D3D12Device> device_;
    std::unique_ptr<FrameStore> store_;
    Project project_;
    QTimer playTimer_;  // single shot, re-armed for the next frame's due time (PlaybackClock)
    PlaybackClock clock_;
    bool playing_ = false;
    bool inPlayTick_ = false;  // setTime from playback itself, not a jump
    double clockTime_ = 0.0;   // the time playback last stood at: a time set in another way restarts the clock
    QElapsedTimer monotonic_;
    std::function<int64_t()> now_;
    Rational timelineFps_{0, 1};  // {0,1}: the base source's rate
    bool engineerMode_ = false;
    bool loop_ = false;
    bool dirty_ = false;
    bool adoptedView_ = false;  // a project's saved view was just adopted: reloadSources puts its centre back on the image once (TASK-0072)
    ProcessPlan plan_;
    std::optional<ProcessPlan> planNoForce_;  // PR-10: when a stage above NR is marked «пересчитать»
    std::optional<ProcessPlan> planNrSkipped_;  // TASK-0071: NR planned to run and FG on - the run's plan if NR is skipped (FG's card)
    std::optional<NrPreviewShown> nrPreview_;
    uint64_t generation_ = 0;
    RunOutcome outcome_;
    SourceInfo sourceInfo_;
    QString planError_;
    std::set<std::string> force_;
    std::map<std::string, int> history_;  // pass -> previous versions on disk
    QTimer planTimer_;
    bool resultAudio_ = false;  // the result video has a sound track (probed on reload)
    std::string resultCodec_;   // the result video's codec and decoded layout (probed on reload)
    FrameDesc resultFormat_;
    double lastRun_ = 0.0;      // seconds the last run of this session took
    QProcess* proxyProcess_ = nullptr;
    QStringList proxyArgs_;     // what it builds: a reload with the same targets does not restart it
    std::atomic<bool> currentLoadedQueued_{false};
    int64_t scrubStartNs_ = -1;
    std::vector<double> scrubMs_;
    ControlMaskState controlMask_;
    bool controlMaskBusy_ = false;
    // TASK-0071: the checks of NR / FG: per stage the last one, with what it depends on; the one running
    struct CheckRecord {
        std::string key;
        StageCheck check;
    };
    std::map<std::string, CheckRecord> checks_;
    QProcess* checkProcess_ = nullptr;
    std::string checkingStage_;
    std::string checkKey(const std::string& stage) const;  // the stage's effective settings, the DLL file, the driver, NR's frame size
    QString nrCheckSize() const;                           // "WxH" of NR's model frame in the run, empty before the plan
    int checkSerial_ = 0;
    AvailabilityFacts availabilityFacts() const;
    QString checkProgram_;
    std::filesystem::path releasedFile_;  // releaseResult: its videos are no sources until restoreResult
    bool isReleased(const std::filesystem::path& video) const;
    QThreadPool maskPool_;  // one thread: the control mask's import (decoding and a PNG16 write take seconds on 4K)
};

}  // namespace dlssvid
