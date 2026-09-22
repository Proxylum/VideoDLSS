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
