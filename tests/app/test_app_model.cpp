// Qt application model without a window (QT_QPA_PLATFORM=offscreen): project handling,
// frame navigation, playback over computed frames and the task queue.

#include <catch2/catch_test_macros.hpp>

#include <QApplication>
#include <QElapsedTimer>
#include <QSignalSpy>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <QFileInfo>
#include <QSettings>

#include <catch2/catch_approx.hpp>

#include "AppModel.h"
#include "SourceNames.h"
#include "ProcessingPanel.h"
#include "TaskQueue.h"
#include "TestClips.h"
#include "util/Subprocess.h"
#include "pipeline/PassVersions.h"
#include "passes/PassSequence.h"
#include "viewport/Project.h"

using namespace dlssvid;
using namespace dlssvid::test;
using Catch::Approx;

namespace {

QApplication& App() {
    static int argc = 1;
    static char name[] = "dlssvid_app_tests";
    static char* argv[] = {name, nullptr};
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // A missing platform plugin makes Qt on Windows show a modal message box (no console) and
    // abort — an automated run would hang on it. Fall back to the Qt SDK plugin folder the
    // tests were built against and fail with a message instead of a dialog.
    QCoreApplication::addLibraryPath(QStringLiteral(DLSSVID_QT_PLUGINS));
    QString platform = qEnvironmentVariable("QT_QPA_PLATFORM", QStringLiteral("windows"));
    platform = platform.section(QLatin1Char(':'), 0, 0);
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::filesystem::path exeDir = std::filesystem::path(exe).parent_path();
    const std::string dll = "q" + platform.toStdString() + ".dll";
    const std::filesystem::path candidates[] = {exeDir / "platforms" / dll, std::filesystem::path(DLSSVID_QT_PLUGINS) / "platforms" / dll};
    bool found = false;
    for (const auto& c : candidates) found = found || std::filesystem::exists(c);
    if (!found) {
        std::fprintf(stderr, "dlssvid_app_tests: Qt platform plugin %s not found (looked in %s and %s)\n", dll.c_str(),
                     candidates[0].parent_path().string().c_str(), candidates[1].parent_path().string().c_str());
        std::exit(1);
    }
    static QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("dlssvid-tests");  // QSettings of the tests, not the user's
    QCoreApplication::setApplicationName("dlssvid-tests");
    return app;
}

std::filesystem::path Dir(const char* name) {
    const auto d = std::filesystem::path(DLSSVID_TEST_TMP) / "app" / name;
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

void WriteDepth(const std::filesystem::path& dir, int frames) {
    PassWriter w(dir, Manifest::ForPass(PassKind::DepthRaw, 16, 8, FileFormat::Exr));
    for (int f = 0; f < frames; ++f) w.WriteFrame(f, MakePassImage(PassKind::DepthRaw, 16, 8));
    w.Finish();
}

void WritePassFps(const std::filesystem::path& dir, PassKind kind, int frames, int fps) {
    Manifest m = Manifest::ForPass(kind, 16, 8, FileFormat::Png);
    m.fps = Rational{fps, 1};
    PassWriter w(dir, m);
    for (int f = 0; f < frames; ++f) w.WriteFrame(f, MakePassImage(kind, 16, 8));
    w.Finish();
}

void Pump(const std::function<bool()>& done, int ms) {
    QElapsedTimer t;
    t.start();
    while (!done() && t.elapsed() < ms) QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
}

QString Q(const std::filesystem::path& p) { return QString::fromStdWString(p.wstring()); }

}  // namespace

TEST_CASE("AppModel opens a project, navigates frames and saves the viewport state", "[app][model][gpu]") {
    App();
    const auto d = Dir("model");
    WriteDepth(d / "passes" / "depth_raw", 5);
    Project p = Project::Create(d / "missing.mp4", d / "passes");
    p.Save(d / "proj.dlssvid.json");

    AppModel model(true);
    QSignalSpy opened(&model, &AppModel::projectChanged);
    QSignalSpy frames(&model, &AppModel::timeChanged);
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(opened.count() == 1);
    CHECK(model.sourceNames() == QStringList{"depth_raw"});
    CHECK(model.store().FrameCount() == 5);
    CHECK(model.hasProject());

    model.setSingleSource("depth_raw");
    CHECK(model.state().layers[0].display == DisplayMode::Turbo);
    model.setFrame(99);
    CHECK(model.baseFrame() == 4);
    model.stepFrame(-10);
    CHECK(model.baseFrame() == 0);
    model.stepFrame(2);
    CHECK(model.baseFrame() == 2);
    CHECK(frames.count() == 3);
    model.store().WaitForCurrent();
    CHECK(model.store().GetStatus(2, "depth_raw") == FrameStore::Status::Ready);

    model.setMode(ViewMode::Grid);
    CHECK(model.neededSources().size() == 4);
    model.setMode(ViewMode::Single);
    CHECK(model.neededSources() == std::vector<std::string>{"depth_raw"});

    // playback runs only over frames that exist and stops at the end
    model.setFrame(3);
    model.setPlaying(true);
    CHECK(model.playing());
    Pump([&] { return !model.playing(); }, 3000);
    CHECK(!model.playing());
    CHECK(model.baseFrame() == 4);

    model.state().wipe.enabled = true;
    model.state().time = 1.0;  // fps-less passes: the base rate is 1 fps, a second per frame
    REQUIRE(model.saveProject());
    const Project q = Project::Load(d / "proj.dlssvid.json");
    CHECK(q.viewport.wipe.enabled);
    CHECK(q.viewport.time == 1.0);
    CHECK(q.viewport.layers[0].source == "depth_raw");

    // a bad file reports a message instead of throwing
    QSignalSpy messages(&model, &AppModel::message);
    model.openProject(Q(d / "nope.json"));
    CHECK(messages.count() == 1);
}

TEST_CASE("TaskQueue runs commands one after another and parses progress", "[app][tasks]") {
    App();
    TaskQueue queue;
    QSignalSpy finished(&queue, &TaskQueue::taskFinished);
    QSignalSpy output(&queue, &TaskQueue::taskOutput);
    const int a = queue.enqueue("ok", "cmd", {"/c", "echo", "3/10", "frames", "&&", "echo", "hello"});
    const int b = queue.enqueue("fail", "cmd", {"/c", "exit", "3"});
    CHECK(queue.busy());
    Pump([&] { return finished.count() >= 2; }, 15000);
    REQUIRE(finished.count() == 2);
    CHECK(finished[0][0].toInt() == a);
    CHECK(finished[0][1].toBool());
    CHECK(finished[1][0].toInt() == b);
    CHECK(!finished[1][1].toBool());
    CHECK(!queue.busy());
    bool sawHello = false;
    for (int i = 0; i < output.count(); ++i) sawHello = sawHello || output[i][1].toString().contains("hello");
    CHECK(sawHello);
}

TEST_CASE("TaskQueue survives a crashing stage process and runs the next task", "[app][tasks]") {
    App();
    TaskQueue queue;
    QSignalSpy finished(&queue, &TaskQueue::taskFinished);
    // `dlssvid fg --crash-after 0` terminates itself with an access-violation status before touching any input:
    // the stage process dies the way a runtime fault would, the queue reports it and moves on (ТЗ §9).
    const int crash = queue.enqueue("crash", DLSSVID_CLI_PATH, {"fg", "--crash-after", "0"});
    const int next = queue.enqueue("next", "cmd", {"/c", "echo", "alive"});
    Pump([&] { return finished.count() >= 2; }, 30000);
    REQUIRE(finished.count() == 2);
    CHECK(finished[0][0].toInt() == crash);
    CHECK(!finished[0][1].toBool());
    CHECK(finished[1][0].toInt() == next);
    CHECK(finished[1][1].toBool());
    CHECK(!queue.busy());
}

TEST_CASE("AppModel time axis: 24 and 48 fps sources, steps by the base layer, timeline rate, legacy frame", "[app][model][gpu]") {
    App();
    const auto d = Dir("time");
    WritePassFps(d / "passes" / "color_sr", PassKind::ColorSr, 6, 24);
    WritePassFps(d / "passes" / "color_fg", PassKind::ColorFg, 11, 48);
    // a project written before stage 9 stores a frame index instead of a time
    nlohmann::json j = Project::Create(d / "missing.mp4", d / "passes").ToJson(d);
    j["viewport"].erase("time");
    j["viewport"]["frame"] = 4;
    j["viewport"]["layers"][0]["source"] = "color_sr";
    std::ofstream(d / "proj.dlssvid.json") << j.dump(2);

    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    CHECK(model.baseSource() == "color_sr");
    CHECK(std::abs(model.time() - 4.0 / 24.0) < 1e-9);  // resolved at the base rate
    CHECK(model.state().legacyFrame == -1);
    CHECK(model.baseFrame() == 4);
    CHECK(model.timelineFps().num == 24);
    CHECK(model.timelineFrameCount() == 6);
    CHECK(std::abs(model.lastTime() - 5.0 / 24.0) < 1e-9);
    REQUIRE(model.fpsChoices().size() == 2);
    CHECK(model.fpsChoices()[0].num == 24);
    CHECK(model.fpsChoices()[1].num == 48);

    QSignalSpy times(&model, &AppModel::timeChanged);
    model.stepFrame(+1);
    CHECK(model.baseFrame() == 5);
    model.stepFrame(+1);  // past the end: clamped, no signal
    CHECK(model.baseFrame() == 5);
    CHECK(times.count() == 1);
    model.setTimelineFps(model.fpsChoices()[1]);
    CHECK(model.timelineFps().num == 48);
    CHECK(model.timelineFrameCount() == 12);
    model.setFrame(3);  // 48 fps frames
    CHECK(std::abs(model.time() - 3.0 / 48.0) < 1e-9);
    CHECK(model.timelineFrame() == 3);
    CHECK(model.baseFrame() == 2);  // round(1.5)
    CHECK(model.store().FrameAt("color_fg", model.time()) == 3);
    model.stepFrame(-1);  // by the base layer: one 24 fps frame
    CHECK(model.baseFrame() == 1);
    CHECK(std::abs(model.time() - 1.0 / 24.0) < 1e-9);

    // frame states drive the plates and the labels
    model.store().WaitForCurrent();
    CHECK(model.frameStateOf("color_sr") == FrameState::Ready);
    CHECK(model.frameStateOf("nope") == FrameState::Missing);
    model.state().mode = ViewMode::Grid;
    model.state().gridSources = {"color_fg", "color_sr", "color_fg", "color_sr"};
    model.notifyStateChanged();
    CHECK(model.baseSource() == "color_fg");
    CHECK(model.frameStates().size() == 2);

    // playback runs at the fastest shown rate (48 fps) and stops where the base source ends
    model.setTime(0.0);
    model.setPlaying(true);
    Pump([&] { return !model.playing(); }, 5000);
    CHECK(!model.playing());
    CHECK(model.timelineFrame() == 10);  // the last color_fg frame; the timeline itself runs to 11 (0.25 s of color_sr)
    REQUIRE(model.saveProject());
    CHECK(std::abs(Project::Load(d / "proj.dlssvid.json").viewport.time - 10.0 / 48.0) < 1e-9);  // saved as a time
}

TEST_CASE("AppModel compares in one action: views, presets, chips with versions, engineer mode", "[app][model][gpu]") {
    App();
    QSettings().clear();
    const auto d = Dir("compare");
    WritePassFps(d / "passes" / "color_sr", PassKind::ColorSr, 4, 24);
    WritePassFps(d / "passes" / "color_nr", PassKind::ColorNr, 4, 24);
    const auto old = RetirePassVersion(d / "passes", "color_nr");
    REQUIRE(old);
    WritePassFps(d / "passes" / "color_nr", PassKind::ColorNr, 4, 24);
    WritePassFps(d / "passes" / "depth_dlss", PassKind::DepthDlss, 4, 24);
    Project p = Project::Create(d / "missing.mp4", d / "passes");
    p.Save(d / "proj.dlssvid.json");

    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    CHECK(model.compareView() == AppModel::CompareView::AfterOnly);
    CHECK(model.beforeSource() == "color_sr");  // no source video: the first colour pass
    CHECK(model.afterSource() == "color_nr");

    // W sets up «before | after», then toggles the wipe of the configured comparison
    model.toggleWipe();
    CHECK(model.compareView() == AppModel::CompareView::BeforeAfter);
    REQUIRE(model.state().layers.size() == 2);
    CHECK(model.state().layers[0].source == "color_sr");
    CHECK(model.state().layers[1].source == "color_nr");
    CHECK(model.state().wipe.position == 0.5f);
    CHECK(model.state().selectedLayer == 1);
    model.toggleWipe();
    CHECK(!model.state().wipe.enabled);
    CHECK(model.compareView() == AppModel::CompareView::AfterOnly);
    model.toggleWipe();
    CHECK(model.state().wipe.enabled);
    CHECK(model.compareView() == AppModel::CompareView::BeforeAfter);

    // a chip changes the «after» side; in the single view it is the view itself; from the grid it returns to a single view
    model.showSource("depth_dlss");
    CHECK(model.state().layers[1].source == "depth_dlss");
    CHECK(model.state().layers[1].display == DisplayMode::Turbo);
    CHECK(model.state().layers[0].source == "color_sr");
    model.setCompareView(AppModel::CompareView::AfterOnly);
    CHECK(model.state().mode == ViewMode::Single);
    REQUIRE(model.state().layers.size() == 1);
    CHECK(model.state().layers[0].source == "depth_dlss");  // what the «after» side showed
    model.showSource("color_nr");
    CHECK(model.state().layers[0].source == "color_nr");
    model.setCompareView(AppModel::CompareView::Grid);
    CHECK(model.compareView() == AppModel::CompareView::Grid);
    model.showSource("color_sr");
    CHECK(model.compareView() == AppModel::CompareView::AfterOnly);
    CHECK(model.state().layers[0].source == "color_sr");
    model.setCompareView(AppModel::CompareView::BeforeAfter);  // the shown source equals the before side: the after side is the default
    CHECK(model.state().layers[0].source == "color_sr");
    CHECK(model.state().layers[1].source == "color_nr");

    // presets
    QSignalSpy messages(&model, &AppModel::message);
    CHECK(model.applyPreset(AppModel::ComparePreset::SrVsNr));
    CHECK(model.state().layers[0].source == "color_sr");
    CHECK(model.state().layers[1].source == "color_nr");
    CHECK(!model.applyPreset(AppModel::ComparePreset::SourceVsDepth));  // no source video here
    CHECK(messages.count() == 1);
    CHECK(model.applyPreset(AppModel::ComparePreset::BeforeAfter));

    // chips: one per current source in pipeline order, the previous color_nr version in its menu
    const auto chips = model.chips();
    REQUIRE(chips.size() == 3);
    CHECK(chips[0].source == "depth_dlss");
    CHECK(chips[1].source == "color_sr");
    CHECK(chips[2].source == "color_nr");
    CHECK(chips[2].label == QString::fromUtf8("Улучшение"));
    CHECK(chips[1].label == QString::fromUtf8("Апскейл"));
    CHECK(chips[0].versions.empty());
    REQUIRE(chips[2].versions.size() == 1);
    const std::string versionName = chips[2].versions[0].name;
    CHECK(versionName == "color_nr@" + *old);
    CHECK(chips[2].tooltip.contains("color_nr"));
    CHECK(HumanSourceName(versionName).startsWith(QString::fromUtf8("Улучшение · ")));
    CHECK(VersionLabel("20260922-140200_3f2a9c1d") == QString::fromUtf8("22.09 14:02 · 3f2a9c1d"));
    CHECK(SourceRank("source") < SourceRank("depth_raw"));
    CHECK(SourceRank("color_fg") < SourceRank("result"));
    CHECK(SourceRank("mask_face") > SourceRank("result"));
    // a previous version on the «after» side, saved and loaded back
    model.showSource(versionName);
    CHECK(model.state().layers[1].source == versionName);
    CHECK(model.compareView() == AppModel::CompareView::BeforeAfter);
    model.store().WaitForCurrent();
    CHECK(model.frameStateOf(versionName) == FrameState::Ready);
    REQUIRE(model.saveProject());
    CHECK(Project::Load(d / "proj.dlssvid.json").viewport.layers[1].source == versionName);

    // the engineer mode is remembered
    QSignalSpy engineer(&model, &AppModel::engineerModeChanged);
    CHECK(!model.engineerMode());
    model.setEngineerMode(true);
    model.setEngineerMode(true);
    CHECK(engineer.count() == 1);
    {
        AppModel other(true);
        CHECK(other.engineerMode());
    }
    model.setEngineerMode(false);
    CHECK(!AppModel(true).engineerMode());
}

TEST_CASE("AppModel plans the project screen: card states, outcome, forced stages, source hash, validation", "[app][model][gpu]") {
    App();
    const auto d = Dir("plan");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    spec.audio = true;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    Project p = Project::Create(clip, d / "passes");
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    p.resultVideo = d / "result.mkv";
    p.codec = "ffv1";
    p.Save(d / "proj.dlssvid.json");

    AppModel model(true);
    QSignalSpy plans(&model, &AppModel::planChanged);
    model.openProject(Q(d / "proj.dlssvid.json"));
    CHECK(plans.count() >= 1);
    CHECK(model.planError().isEmpty());
    CHECK(model.sourceInfo().width == 48);
    CHECK(model.sourceInfo().frames == 4);
    CHECK(model.sourceInfo().audio);
    REQUIRE(model.plan().stages.size() == 5);
    CHECK(model.outcome().width == 96);
    CHECK(model.outcome().fps == Approx(48.0));
    CHECK(model.outcome().stagesToRun == 6);
    CHECK(model.outcome().seconds > 0.0);
    CHECK(model.cardState("depth").state == QString::fromUtf8("будет посчитано"));
    CHECK(model.cardState("depth").tone == "run");
    CHECK(model.cardState("depth").version == "v1");
    CHECK(model.cardState("nr").runs);
    CHECK(model.cardState("nr").warning.isEmpty());  // depth and flow run first: the guides will be there
    CHECK(model.sourceHashStatus() == QString::fromUtf8("пассов пока нет"));
    CHECK(!model.plan().sourceHash.empty());
    CHECK(model.project().sourceHash == model.plan().sourceHash);  // cached in the project

    // the pipeline as the GUI runs it (dlssvid process --project), then everything is reused
    REQUIRE(model.saveProject());
    QStringList args = model.processArgs();
    CHECK(args.size() == 4);
    args << "--warp";
    std::vector<std::string> cmd{DLSSVID_CLI_PATH};
    for (const QString& a : args) cmd.push_back(a.toStdString());
    std::string out;
    REQUIRE(Subprocess::Run(cmd, &out) == 0);
    model.reloadSources();
    CHECK(model.cardState("depth").state == QString::fromUtf8("переиспользуется"));
    CHECK(model.cardState("depth").tone == "ok");
    CHECK(model.cardState("depth").version == "v1");
    CHECK(model.cardState("depth").sub.startsWith(QString::fromUtf8("0 мин")));
    CHECK(model.outcome().stagesToRun == 1);  // only the encode
    CHECK(model.outcome().reused.size() == 5);
    CHECK(model.sourceHashStatus() == QString::fromUtf8("совпадает с пассами"));

    // a changed parameter: nr recomputes, fg follows, the rest stays
    model.setStageParam("nr", "intensity", 1.4);
    model.refreshPlan();
    CHECK(model.cardState("nr").state.startsWith(QString::fromUtf8("пересчёт: Интенсивность")));
    CHECK(model.cardState("nr").state.contains("1.4"));
    CHECK(model.cardState("nr").version == "v2");
    CHECK(model.cardState("nr").sub.contains(QString::fromUtf8("v1 останется")));
    CHECK(model.cardState("fg").state == QString::fromUtf8("пересчёт: изменился вход"));
    CHECK(model.cardState("fg").sub.contains(QString::fromUtf8("Улучшение")));
    CHECK(model.cardState("upscale").state == QString::fromUtf8("переиспользуется"));
    CHECK(model.outcome().stagesToRun == 3);
    CHECK(model.outcome().newBytes > 0);
    model.setStageParam("nr", "intensity", 1.0);
    model.setForce("fg", true);
    model.refreshPlan();
    CHECK(model.cardState("fg").state == QString::fromUtf8("пересчёт (принудительно)"));
    CHECK(model.processArgs().contains("--force"));
    CHECK(model.processArgs().contains("fg"));
    model.clearForce();
    model.refreshPlan();
    CHECK(model.outcome().stagesToRun == 1);

    model.setStageEnabled("depth", false);
    model.refreshPlan();
    CHECK(model.cardState("depth").state == QString::fromUtf8("выключена"));
    CHECK(model.cardState("depth").tone == "off");
    model.setStageEnabled("depth", true);

    // a parameter outside the schema: the plan says what is wrong instead of failing later
    model.setStageParams("nr", nlohmann::json{{"backend", "stub"}, {"intensity", 9}});
    model.refreshPlan();
    CHECK(model.planError().contains("nr.intensity"));
    CHECK(model.cardState("nr").state.contains("nr.intensity"));
    model.setStageParams("nr", nlohmann::json{{"backend", "stub"}});
    model.refreshPlan();
    CHECK(model.planError().isEmpty());
    model.setEncode("hevc_nvenc", {{"b", "50M"}});
    CHECK(model.project().codec == "hevc_nvenc");
    CHECK(model.project().codecOptions.at("b") == "50M");
}

TEST_CASE("AppModel remembers recents and dialog folders, tracks unsaved changes and reports the GPU readiness", "[app][model][gpu]") {
    App();
    QSettings().clear();
    const auto d = Dir("recent");
    AppModel model(true);
    CHECK(model.recents().empty());
    CHECK(!model.dirty());
    CHECK(model.closeAction() == AppModel::CloseAction::Nothing);
    CHECK(model.readiness().contains("WARP"));  // the tests run on the software adapter

    // a project from its file: clean, first in the recents with its state
    WriteDepth(d / "passes" / "depth_raw", 3);
    Project p = Project::Create(d / "missing.mp4", d / "passes");
    p.Save(d / "proj.dlssvid.json");
    QSignalSpy dirtySpy(&model, &AppModel::dirtyChanged);
    QSignalSpy recentsSpy(&model, &AppModel::recentsChanged);
    model.openProject(Q(d / "proj.dlssvid.json"));
    CHECK(!model.dirty());
    CHECK(model.closeAction() == AppModel::CloseAction::Nothing);
    REQUIRE(model.recents().size() == 1);
    CHECK(model.recents()[0].title == "proj");
    CHECK(model.recents()[0].isProject);
    CHECK(model.recents()[0].exists);
    CHECK(model.recents()[0].info.startsWith(QString::fromUtf8("1 пасс")));
    CHECK(recentsSpy.count() == 1);

    // edits mark the project unsaved, a view change does not, saving cleans it
    model.setStageParam("nr", "intensity", 1.2);
    CHECK(model.dirty());
    CHECK(model.closeAction() == AppModel::CloseAction::AutoSave);
    REQUIRE(model.saveProject());
    CHECK(!model.dirty());
    model.notifyStateChanged(false);
    CHECK(!model.dirty());
    model.setMode(ViewMode::Grid);
    CHECK(model.dirty());
    CHECK(dirtySpy.count() == 4);  // open: false; edit: true; save: false; mode: true
    REQUIRE(model.saveProject());

    // a video without a project file: a new project, unsaved, saved next to the video on close
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    model.openVideo(Q(clip));
    CHECK(model.hasProject());
    CHECK(model.dirty());
    CHECK(model.closeAction() == AppModel::CloseAction::AutoSave);
    CHECK(model.project().file == std::filesystem::path(clip).replace_extension(".dlssvid.json"));
    REQUIRE(model.recents().size() == 2);
    CHECK(model.recents()[0].path == Q(clip));
    CHECK(!model.recents()[0].isProject);
    CHECK(model.recents()[0].title == "clip");
    CHECK(model.recents()[0].info.startsWith(QString::fromUtf8("видео · не обработан")));
    REQUIRE(model.saveProject());
    CHECK(!model.dirty());
    model.openVideo(Q(clip));  // the project file next to the video is picked up: clean
    CHECK(!model.dirty());
    CHECK(model.recents()[0].info.startsWith(QString::fromUtf8("видео · есть проект")));

    // the list dedupes, keeps ten entries, marks missing files and survives another model
    for (int i = 0; i < 12; ++i) model.addRecent(Q(d / ("gone" + std::to_string(i) + ".mkv")));
    CHECK(model.recents().size() == 10);
    CHECK(!model.recents()[0].exists);
    CHECK(model.recents()[0].info == QString::fromUtf8("файл не найден"));
    model.addRecent(Q(clip));
    CHECK(model.recents().size() == 10);
    CHECK(model.recents()[0].path == Q(clip));
    CHECK(model.recents()[0].exists);
    {
        AppModel other(true);
        CHECK(other.recents().size() == 10);
    }
    model.clearRecents();
    CHECK(model.recents().empty());

    // dialog folders
    CHECK(model.lastDir("video").isEmpty());
    model.rememberDir("video", Q(clip));
    CHECK(model.lastDir("video") == QFileInfo(Q(clip)).absolutePath());
    model.rememberDir("video", QString());
    CHECK(model.lastDir("video") == QFileInfo(Q(clip)).absolutePath());
}

TEST_CASE("TaskQueue cancels the running task, drops a queued one and keeps the order of events", "[app][tasks]") {
    App();
    TaskQueue queue;
    QStringList events;
    QObject::connect(&queue, &TaskQueue::taskStarted, [&](int id) { events << QString("start %1").arg(id); });
    QObject::connect(&queue, &TaskQueue::taskFinished, [&](int id, bool ok) { events << QString("finish %1 %2").arg(id).arg(ok ? "ok" : "fail"); });
    int progressOfLong = 0;
    // ping waits ~60 s between its echoes: the run that gets cancelled, a follow-up, and one dropped before it starts
    const int longTask = queue.enqueue("long", "ping", {"-n", "60", "127.0.0.1"});
    const int next = queue.enqueue("next", "cmd", {"/c", "echo", "alive"});
    const int dropped = queue.enqueue("dropped", "cmd", {"/c", "echo", "never"});
    QObject::connect(&queue, &TaskQueue::taskProgress, [&](int id, const TaskQueue::Progress&) { progressOfLong += id == longTask; });
    REQUIRE(queue.busy());
    CHECK(queue.currentId() == longTask);
    QElapsedTimer clock;
    clock.start();
    Pump([] { return false; }, 300);
    CHECK(queue.elapsed(longTask) >= 0.25);
    queue.cancel(dropped);
    CHECK(queue.cancelled(dropped));
    CHECK(queue.finished(dropped));
    CHECK(!queue.succeeded(dropped));
    CHECK(queue.busy());  // the running task is untouched by dropping a queued one
    queue.cancel(longTask);
    Pump([&] { return queue.finished(next); }, 20000);
    REQUIRE(queue.finished(next));
    CHECK(clock.elapsed() < 15000);  // the ping did not run its minute
    CHECK(queue.cancelled(longTask));
    CHECK(!queue.succeeded(longTask));
    CHECK(queue.succeeded(next));
    CHECK(!queue.cancelled(next));
    CHECK(!queue.busy());
    // the long task started first; the dropped one finished at once; the cancelled one finished before the next started
    const QStringList expected{QString("start %1").arg(longTask), QString("finish %1 fail").arg(dropped), QString("finish %1 fail").arg(longTask),
                               QString("start %1").arg(next), QString("finish %1 ok").arg(next)};
    CHECK(events == expected);
    CHECK(progressOfLong >= 1);  // the final progress of the cancelled task was published
    const TaskQueue::Progress p = queue.progress(longTask);
    CHECK(p.stages.empty());
    CHECK(p.eta == -1.0);
    CHECK(p.elapsed >= 0.25);
    CHECK(queue.progress(next).fraction == 1.0);
    CHECK(queue.progress(dropped).elapsed == 0.0);
}

TEST_CASE("TaskQueue counts reused stages at once and takes the time left from the plan and the measured rate", "[app][tasks]") {
    App();
    TaskQueue queue;
    std::vector<TaskQueue::Progress> seen;
    QObject::connect(&queue, &TaskQueue::taskProgress, [&](int, const TaskQueue::Progress& p) { seen.push_back(p); });
    const std::vector<TaskQueue::StagePlan> plan{{"depth", "Глубина", 8.0, false}, {"flow", "Векторы движения", 0.0, true}, {"upscale", "Апскейл ×2", 20.0, false}};
    // the "stage" reports two frames about a second apart (ping -n 2 waits ~1 s), says it is done the way the runner
    // does, then the next stage starts
    const int id = queue.enqueue("Обработка", "cmd",
                                 {"/c", "echo", "depth:", "1/4", "frames", "&", "ping", "-n", "2", "127.0.0.1", ">nul", "&", "echo", "depth:", "3/4", "frames", "&",
                                  "echo", "process:", "depth", "done", "-", "3", "frames", "in", "1.0", "s", "&", "echo", "upscale:", "1/8", "frames"},
                                 plan);
    REQUIRE(!seen.empty());
    // before any output: the reused stage is done, the rest is the plan
    CHECK(seen[0].stagesTotal == 3);
    CHECK(seen[0].stagesDone == 1);
    CHECK(seen[0].fraction == Approx(1.0 / 3));
    CHECK(seen[0].eta == Approx(28.0));
    CHECK(seen[0].stages[1].state == "reused");
    Pump([&] { return queue.finished(id); }, 20000);
    REQUIRE(queue.succeeded(id));
    auto snapshot = [&](const QString& stage, int done) -> const TaskQueue::Progress* {
        for (const auto& p : seen)
            if (p.stage == stage && p.done == done) return &p;
        return nullptr;
    };
    // the first frame line: the running stage still uses the estimate (8 s × 3/4) plus the queued stage
    const TaskQueue::Progress* first = snapshot("depth", 1);
    REQUIRE(first);
    CHECK(first->total == 4);
    CHECK(first->fraction == Approx((1.0 + 0.25) / 3));
    CHECK(first->eta == Approx(26.0));
    CHECK(first->stages[0].state == "running");
    CHECK(first->stages[0].percent == 25);
    // two frames measured in ~1 s: the stage's own rate (≈ 0.5 s for the last frame) replaces the estimate (2 s)
    const TaskQueue::Progress* measured = snapshot("depth", 3);
    REQUIRE(measured);
    CHECK(measured->fraction == Approx((1.0 + 0.75) / 3));
    CHECK(measured->eta >= 20.0);
    CHECK(measured->eta < 21.6);
    // the runner's "done" line: the stage's time is what it printed, nothing runs, the queued estimate remains
    const TaskQueue::Progress* next = snapshot("upscale", 1);
    REQUIRE(next);
    CHECK(next->stages[0].state == "done");
    CHECK(next->stages[0].seconds == Approx(1.0));
    CHECK(next->stagesDone == 2);
    CHECK(next->stages[2].state == "running");
    CHECK(next->stages[2].percent == 12);
    CHECK(next->eta == Approx(20.0 * 7 / 8));
    CHECK(next->fraction == Approx((2.0 + 0.12) / 3));  // the percent is an integer: 1/8 -> 12
    // the process exited 0: everything is done
    const TaskQueue::Progress last = queue.progress(id);
    CHECK(last.stagesDone == 3);
    CHECK(last.fraction == 1.0);
    CHECK(last.eta == 0.0);
    CHECK(last.stages[2].state == "done");
    CHECK(last.elapsed >= 0.9);
}

TEST_CASE("ProcessingPanel shows the stages, the time, the log tail and offers the result or the way back", "[app][tasks]") {
    App();
    TaskQueue queue;
    ProcessingPanel panel(queue);
    QSignalSpy result(&panel, &ProcessingPanel::showResultRequested);
    QSignalSpy back(&panel, &ProcessingPanel::backRequested);
    const std::vector<TaskQueue::StagePlan> plan{{"depth", "Глубина", 8.0, false}, {"flow", "Векторы движения", 0.0, true}, {"encode", "Кодирование", 4.0, false}};
    const int id = queue.enqueue("Обработка", "cmd", {"/c", "echo", "depth:", "2/4", "frames", "&", "echo", "[2026-09-22", "10:00:00.000]", "[info]", "hello", "from", "the", "stage"}, plan);
    panel.watch(id, "face.mp4 → face_result.mp4");
    CHECK(panel.headline().startsWith("Обработка"));
    Pump([&] { return queue.finished(id); }, 20000);
    REQUIRE(queue.succeeded(id));
    CHECK(panel.headline().startsWith("Готово за"));
    const QStringList lines = panel.stageLines();
    REQUIRE(lines.size() == 3);
    CHECK(lines[0].startsWith("Глубина · готово"));
    CHECK(lines[1] == "Векторы движения · переиспользуется · —");
    CHECK(lines[2].startsWith("Кодирование · готово"));
    CHECK(panel.logTail().contains("[info] hello from the stage"));
    CHECK(!panel.logTail().contains("2026-09-22"));  // the timestamp is stripped from the tail
    auto* showResult = panel.findChild<QPushButton*>("showResult");
    auto* cancel = panel.findChild<QPushButton*>("cancel");
    auto* backButton = panel.findChild<QPushButton*>("back");
    REQUIRE((showResult && cancel && backButton));
    CHECK(!showResult->isHidden());
    CHECK(cancel->isHidden());
    CHECK(backButton->isHidden());
    showResult->click();
    CHECK(result.count() == 1);

    // a cancelled run: the page says so, the stages that did not run are marked, the way back is offered
    const int slow = queue.enqueue("Обработка", "ping", {"-n", "60", "127.0.0.1"}, plan);
    panel.watch(slow, "face.mp4 → face_result.mp4");
    Pump([] { return false; }, 200);
    CHECK(panel.headline().startsWith("Обработка · Прошло"));
    CHECK(panel.headline().contains("Осталось ≈"));  // the plan's estimate: 12 s
    CHECK(!cancel->isHidden());
    CHECK(showResult->isHidden());
    cancel->click();
    Pump([&] { return queue.finished(slow); }, 20000);
    REQUIRE(queue.cancelled(slow));
    CHECK(panel.headline().startsWith("Отменено через"));
    const QStringList after = panel.stageLines();
    REQUIRE(after.size() == 3);
    CHECK(after[0].toStdString() == "Глубина · отменена · —");
    CHECK(after[1] == "Векторы движения · переиспользуется · —");
    CHECK(!backButton->isHidden());
    CHECK(cancel->isHidden());
    backButton->click();
    CHECK(back.count() == 1);
}
