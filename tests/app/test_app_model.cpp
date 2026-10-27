// Qt application model without a window (QT_QPA_PLATFORM=offscreen): project handling,
// frame navigation, playback over computed frames and the task queue.

#include <catch2/catch_test_macros.hpp>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLocale>
#include <QMouseEvent>
#include <QElapsedTimer>
#include <QFormLayout>
#include <QLabel>
#include <QComboBox>
#include <QDir>
#include <QPushButton>
#include <QLineEdit>
#include <QMenu>
#include <QScrollArea>
#include <QShortcut>
#include <QTextDocument>
#include <QTimer>
#include <QScrollBar>
#include <QSignalSpy>
#include <QSlider>
#include <QTest>
#include <QToolButton>
#include <QPalette>
#include <QStyle>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <QFileInfo>
#include <QSettings>

#include <catch2/catch_approx.hpp>

#include "AppModel.h"
#include "ControlMaskRow.h"
#include "ControlMaskText.h"
#include "SourceNames.h"
#include "PlaybackClock.h"
#include "ParamText.h"
#include "PresetMenu.h"
#include "PresetStore.h"
#include "ProcessingPanel.h"
#include "ProjectPanel.h"
#include "RecentPreviews.h"
#include "ResultBar.h"
#include "SceneCutsView.h"
#include "StageCard.h"
#include "StartPage.h"
#include "Theme.h"
#include "TaskQueue.h"
#include "TestClips.h"
#include "util/Subprocess.h"
#include "pipeline/PassVersions.h"
#include "pipeline/ControlMask.h"
#include "pipeline/PassFingerprint.h"
#include "passes/formats/PngIO.h"
#include "viewport/ViewportState.h"
#include "passes/PassSequence.h"
#include "viewport/Project.h"
#include "viewport/ViewportProxy.h"

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
    // QSettings of the tests, not the user's: an INI file per test process under the build tree, so that the test
    // cases ctest runs in parallel (recents, dialog folders, engineer mode) do not see each other's settings
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, QString::fromStdWString((std::filesystem::path(DLSSVID_TEST_TMP) / "app" / "settings").wstring()));
    QCoreApplication::setOrganizationName("dlssvid-tests");
    QCoreApplication::setApplicationName(QString("dlssvid-tests-%1").arg(QCoreApplication::applicationPid()));
    // the presets of the tests, not the user's AppData (PR-9): a folder per test process
    PresetStore::SetDir(QString::fromStdWString((std::filesystem::path(DLSSVID_TEST_TMP) / "app" / ("presets-" + std::to_string(QCoreApplication::applicationPid()))).wstring()));
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

// PR-9: the presets of this test process start empty, without a new-projects mark
void ClearPresets() {
    std::filesystem::remove_all(PresetStore::Path(PresetStore::Dir()));
    QSettings().remove("presets/newProject");
}

std::filesystem::path PresetSlotFile(PresetSlot slot, const std::string& target) { return PresetStore::Library().SlotPath(slot, target); }

QString WritePreset(const std::filesystem::path& file, const nlohmann::json& j) {
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file, std::ios::binary) << j.dump(2);
    return Q(file);
}

nlohmann::json CardPresetJson(const char* name, const nlohmann::json& params, const char* stage = "nr") {
    nlohmann::json stages = nlohmann::json::object();
    stages[stage] = {{"params", params}};
    return {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "stage"}, {"name", name}, {"stages", stages}};
}

// a project of the stub methods without its video (the plan says «видео не найдено» at once)
void StubProjectFile(const std::filesystem::path& file, const std::filesystem::path& video) {
    Project p = Project::Create(video, file.parent_path() / "passes");
    for (auto& st : p.stages) {
        if (st.name == "depth" || st.name == "flow" || st.name == "nr") st.params["backend"] = "stub";
        if (st.name == "upscale") st.params["backend"] = "nis";
        if (st.name == "fg") st.params["backend"] = "blend";
    }
    p.resultVideo = file.parent_path() / "clip_result.mp4";
    p.Save(file);
}

QString LastMessage(const QSignalSpy& spy) { return spy.isEmpty() ? QString() : spy.last().at(0).toString(); }

// the next widget Tab reaches from `w` inside `root`
QWidget* NextTab(QWidget* w, QWidget* root) {
    for (QWidget* n = w->nextInFocusChain(); n && n != w; n = n->nextInFocusChain())
        if ((n->focusPolicy() & Qt::TabFocus) && n->isVisibleTo(root) && n->isEnabled()) return n;
    return nullptr;
}

QAction* ActionStarting(QMenu* menu, const QString& text) {
    for (QAction* a : menu->actions())
        if (a->text().startsWith(text)) return a;
    return nullptr;
}

// dialogs that answer as told and count what was asked
struct StubDialogs {
    int confirms = 0, warnings = 0, infos = 0;
    bool answer = false;
    std::optional<PresetSaveDialog::Setup> lastSetup;
    std::optional<PresetSaveDialog::Answer> saveAnswer;
    QString open, save;
    PresetDialogs::Clash clash = PresetDialogs::Clash::Cancel;
    QString clashName, clashCopy;
    QStringList texts;
    PresetDialogs Make() {
        PresetDialogs d;
        d.saveName = [this](QWidget*, const PresetSaveDialog::Setup& s) {
            lastSetup = s;
            return saveAnswer;
        };
        d.openFile = [this](QWidget*, const QString&, const QString&, const QString&) { return open; };
        d.saveFile = [this](QWidget*, const QString&, const QString&, const QString&) { return save; };
        d.confirm = [this](QWidget*, const QString&, const QString& text, const QString&) {
            ++confirms;
            texts << text;
            return answer;
        };
        d.clash = [this](QWidget*, const QString& name, const QString& copy) {
            clashName = name;
            clashCopy = copy;
            return clash;
        };
        d.warn = [this](QWidget*, const QString&, const QString& text) {
            ++warnings;
            texts << text;
        };
        d.inform = [this](QWidget*, const QString&, const QString& text) {
            ++infos;
            texts << text;
        };
        return d;
    }
};

}  // namespace

namespace {
// a control mask image: grey or RGB U16 of w x h
QString MaskPng(const std::filesystem::path& file, uint32_t w, uint32_t h, bool rgb, uint16_t v = 40000) {
    PassImage img;
    img.Allocate(w, h, PixelType::U16, rgb ? std::vector<std::string>{"R", "G", "B"} : std::vector<std::string>{"A"});
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < img.channels.size(); ++c) img.Set(x, y, c, static_cast<float>(rgb ? (v + 3000 * c + x) % 65536 : v));
    WritePng(file, img);
    return Q(file);
}
// a project of the stub methods on a short clip, its NR card with the test's dialogs
struct MaskCard {
    std::filesystem::path dir;
    std::unique_ptr<AppModel> model;
    StubDialogs dialogs;
    std::unique_ptr<dlssvid::StageCard> card;
    MaskCard(const char* name) : dir(Dir(name)) {
        ClipSpec spec;
        spec.frames = 2;
        spec.width = 48;
        spec.height = 32;
        Project p = Project::Create(WriteClip(dir / "clip.mkv", spec), dir / "passes");
        for (auto& st : p.stages)
            if (st.name == "nr") st.params["backend"] = "stub";
        p.Save(dir / "proj.dlssvid.json");
        model = std::make_unique<AppModel>(true);
        model->openProject(Q(dir / "proj.dlssvid.json"));
        card = std::make_unique<dlssvid::StageCard>(*model, "nr");
        card->setDialogs(dialogs.Make());
        card->refresh();
    }
    ControlMaskRow* row() const { return card->controlMaskRow(); }
    bool load(const QString& file) {  // through the card's «Загрузить…»: the dialog, the thread, the end
        dialogs.open = file;
        QSignalSpy done(model.get(), &AppModel::controlMaskFinished);
        row()->loadButton()->click();
        if (done.isEmpty() && !done.wait(20000)) return false;
        return done.last().at(0).toBool();
    }
    QLabel* label(const char* text) const {
        for (QLabel* l : card->findChildren<QLabel*>())
            if (l->text() == QString::fromUtf8(text)) return l;
        return nullptr;
    }
};
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

    // playback runs at the fastest shown rate (48 fps), visits every color_fg frame and stops where the base source ends
    model.setTime(0.0);
    std::vector<int64_t> visited;
    const auto conn = QObject::connect(&model, &AppModel::timeChanged, [&] { visited.push_back(model.store().FrameAt("color_fg", model.time())); });
    QElapsedTimer played;
    played.start();
    model.setPlaying(true);
    Pump([&] { return !model.playing(); }, 5000);
    const double seconds = static_cast<double>(played.nsecsElapsed()) / 1e9;
    QObject::disconnect(conn);
    CHECK(!model.playing());
    CHECK(visited == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});  // steps of 1/48 s, not 1/24
    CHECK(seconds >= 11.0 / 48.0 - 0.005);  // by the clock; the old 20 ms timer took 11 x 20 = 0.22 s
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
    REQUIRE(model.plan().stages.size() == 6);
    CHECK(model.outcome().width == 96);
    CHECK(model.outcome().fps == Approx(48.0));
    CHECK(model.outcome().stagesToRun == 7);  // scenes .. fg + the encode
    CHECK(model.outcome().seconds > 0.0);
    CHECK(model.cardState("depth").state == QString::fromUtf8("будет посчитано"));
    CHECK(model.cardState("depth").tone == "run");
    CHECK(model.cardState("depth").version == "v1");
    CHECK(model.cardState("nr").runs);
    CHECK(model.cardState("nr").warning.isEmpty());  // depth and flow run first: the guides will be there
    // TASK-0045: with the guides off the card says nr works frame by frame
    model.setStageParam("nr", "guides", false);
    model.refreshPlan();
    CHECK(model.cardState("nr").warning == QString::fromUtf8("глубина и векторы выключены: каждый кадр обрабатывается отдельно"));
    model.setStageParam("nr", "guides", true);
    model.refreshPlan();
    CHECK(model.cardState("nr").warning.isEmpty());
    // PR-13: nis reads no depth and vectors - without depth and flow only nr warns
    model.setStageEnabled("depth", false);
    model.setStageEnabled("flow", false);
    model.refreshPlan();
    CHECK(model.cardState("upscale").warning.isEmpty());
    CHECK(model.cardState("nr").warning == QString::fromUtf8("без глубины и векторов: качество ниже"));
    model.setStageEnabled("depth", true);
    model.setStageEnabled("flow", true);
    model.refreshPlan();
    CHECK(model.cardState("upscale").warning.isEmpty());
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
    CHECK(model.cardState("scenes").state == QString::fromUtf8("переиспользуется"));  // PR-13: the list, a pass without frames
    CHECK(model.cardState("scenes").sub.startsWith(QString::fromUtf8("0 мин · посчитано")));
    CHECK(model.outcome().stagesToRun == 1);  // only the encode
    CHECK(model.outcome().reused.size() == 6);
    CHECK(model.sourceHashStatus() == QString::fromUtf8("совпадает с пассами"));

    // PR-9: a preset of the current settings changes nothing - the pass stays reused; one with another intensity shows the recompute
    // right after it is applied, without the sliders' timer
    {
        ClearPresets();
        const auto same = model.saveStagePreset("nr", QString::fromUtf8("Как есть"));
        REQUIRE(same.ok);
        CHECK(model.applyStagePreset("nr", same.file).message.contains(QString::fromUtf8("уже совпадают")));
        CHECK(model.cardState("nr").state == QString::fromUtf8("переиспользуется"));
        model.setStageParam("nr", "intensity", 0.7);
        const auto soft = model.saveStagePreset("nr", QString::fromUtf8("Мягче"));
        REQUIRE(soft.ok);
        model.setStageParam("nr", "intensity", 1.0);
        model.refreshPlan();
        CHECK(model.cardState("nr").state == QString::fromUtf8("переиспользуется"));
        REQUIRE(model.applyStagePreset("nr", soft.file).ok);
        CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: Интенсивность 1 → 0.7"));
        REQUIRE(model.applyStagePreset("nr", Q(PresetSlotFile(PresetSlot::BeforeApply, "nr"))).ok);  // back: reused again
        CHECK(model.cardState("nr").state == QString::fromUtf8("переиспользуется"));
    }

    // PR-7 (review): an nr pass of before PR-7 with intensity != 1 is recomputed and discarded - the card names the revision, not
    // the guides dropped at 0, and promises no version on disk
    {
        const auto manifestPath = d / "passes" / "color_nr" / "manifest.json";
        std::string saved;
        {
            std::ifstream f(manifestPath, std::ios::binary);
            saved.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        auto stampOld = [&](double intensity) {
            nlohmann::json j = nlohmann::json::parse(saved);
            j["stage_params"].erase("model_intensity");
            j["stage_params"]["intensity"] = intensity;
            j["params_canonical"]["intensity"] = intensity;
            std::ofstream(manifestPath, std::ios::binary) << j.dump(2);
            model.setStageParam("nr", "intensity", intensity);
            model.reloadSources();
        };
        stampOld(0.5);
        CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: сила правки теперь после модели"));
        CHECK_FALSE(model.cardState("nr").sub.contains(QString::fromUtf8("останется")));
        model.setForce("nr", true);  // forced: discarded all the same, no version promised
        model.refreshPlan();
        CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт (принудительно)"));
        CHECK_FALSE(model.cardState("nr").sub.contains(QString::fromUtf8("останется")));
        model.clearForce();
        stampOld(0.0);
        CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: при силе 0 модель не вызывается"));
        CHECK(model.cardState("nr").warning.isEmpty());  // no guides read at 0: no «без глубины и векторов»
        // PR-6 (review): a pass of before the key - «новая настройка», not «null → 0.1»
        {
            nlohmann::json j = nlohmann::json::parse(saved);
            j["params_canonical"].erase("temporal_threshold");
            j["fingerprint"] = "sha256:" + std::string(64, '1');
            std::ofstream(manifestPath, std::ios::binary) << j.dump(2);
            model.setStageParam("nr", "intensity", 1.0);
            model.reloadSources();
            CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: Порог подавления мерцания: новая настройка, 0.1"));
            CHECK_FALSE(model.cardState("nr").state.contains("null"));
        }
        // PR-13: an nr pass of PR-6 hashed its own threshold - the card says where it went, not «scene_cut 0.24 → null»
        {
            nlohmann::json j = nlohmann::json::parse(saved);
            j["params_canonical"]["scene_cut"] = 0.24;
            j["fingerprint"] = "sha256:" + std::string(64, '2');
            std::ofstream(manifestPath, std::ios::binary) << j.dump(2);
            model.setStageParam("nr", "intensity", 1.0);
            model.reloadSources();
            CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: сброс на склейках перенесён в «Склейки»"));
        }
        std::ofstream(manifestPath, std::ios::binary) << saved;
        model.setStageParam("nr", "intensity", 1.0);
        model.reloadSources();
        CHECK(model.cardState("nr").state == QString::fromUtf8("переиспользуется"));
    }

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
    // TASK-0034: the result's bit depth and chroma, the container of the codec, the layout in the outcome
    model.refreshPlan();
    CHECK(model.sourceInfo().codec == "ffv1");
    CHECK(model.sourceInfo().decoded == FrameDesc{48, 32, PixelFormat::Yuv420p, 8});
    CHECK(model.outcome().codec == "hevc_nvenc");
    CHECK(model.outcome().format.format == PixelFormat::Yuv420p);  // an 8-bit source: 8-bit 4:2:0
    CHECK(model.outcome().format.width == 96);
    CHECK(AppModel::formatLabel("hevc_nvenc", model.outcome().format) == QString::fromUtf8("HEVC 8 бит 4:2:0"));
    model.setEncode("prores_ks", {}, OutputFormatRequest{0, "444"});
    CHECK(model.project().outputFormat.chroma == "444");
    CHECK(model.project().resultVideo.extension() == ".mov");  // ProRes lives in .mov
    model.refreshPlan();
    CHECK(model.planError().isEmpty());
    CHECK(model.outcome().format == FrameDesc{96, 64, PixelFormat::Yuv444p16, 10});
    CHECK(AppModel::formatLabel("prores_ks", model.outcome().format) == QString::fromUtf8("ProRes 10 бит 4:4:4"));
    model.setEncode("h264_nvenc", {}, OutputFormatRequest{10, "auto"});  // what the codec cannot do is a plan error, not a failed run
    CHECK(model.project().resultVideo.extension() == ".mp4");
    model.refreshPlan();
    CHECK(model.planError().contains("8-bit only"));
    model.setEncode("ffv1", {}, OutputFormatRequest{});
    CHECK(model.project().resultVideo.extension() == ".mkv");
    model.refreshPlan();
    CHECK(model.planError().isEmpty());
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
    // the start screen's card: «proj — глубина посчитана» over «1 из 6 стадий · сегодня …»; no source video, no preview
    CHECK(model.recents()[0].state == QString::fromUtf8("глубина посчитана"));
    CHECK(model.recents()[0].progress == QString::fromUtf8("1 из 6 стадий"));
    CHECK(!model.recents()[0].when.isEmpty());
    CHECK(model.recents()[0].video.isEmpty());
    CHECK(model.recents()[0].result.isEmpty());
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
    CHECK(model.recents()[0].video == Q(clip));  // the card previews the video itself
    CHECK(model.recents()[0].state.isEmpty());
    CHECK(model.recents()[0].progress == QString::fromUtf8("не обработан"));
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

    // a failed run: the stage that was running is the one in error, the stages after it never ran
    const int failing = queue.enqueue("Обработка", "cmd", {"/c", "echo", "depth:", "1/4", "frames", "&", "exit", "3"}, plan);
    panel.watch(failing, "face.mp4 → face_result.mp4");
    Pump([&] { return queue.finished(failing); }, 20000);
    REQUIRE(queue.finished(failing));
    CHECK(!queue.succeeded(failing));
    CHECK(!queue.cancelled(failing));
    CHECK(panel.headline().startsWith(QString::fromUtf8("Ошибка через")));
    const QStringList failed = panel.stageLines();
    REQUIRE(failed.size() == 3);
    CHECK(failed[0].toStdString().rfind("Глубина · ошибка · ", 0) == 0);
    CHECK(failed[1].toStdString() == "Векторы движения · переиспользуется · —");
    CHECK(failed[2].toStdString() == "Кодирование · не запускалась · —");
    CHECK(!backButton->isHidden());
}

TEST_CASE("AppModel result: summary and bar, export copy, exportable passes, chip hotkeys, free space, close project", "[app][model][gpu]") {
    App();
    const auto d = Dir("result");
    ClipSpec spec;
    spec.width = 32;
    spec.height = 16;
    spec.frames = 8;
    spec.fpsNum = 48;
    spec.audio = true;
    const auto result = WriteClip(d / "res.mkv", spec);
    WritePassFps(d / "passes" / "color_sr", PassKind::ColorSr, 4, 24);
    WritePassFps(d / "passes" / "depth_dlss", PassKind::DepthDlss, 4, 24);
    Project p = Project::Create(d / "missing.mp4", d / "passes");
    p.resultVideo = result;
    p.Save(d / "proj.dlssvid.json");

    AppModel model(true);
    ResultBar bar(model);
    CHECK(model.resultSummary().isEmpty());
    CHECK(bar.isHidden());
    model.openProject(Q(d / "proj.dlssvid.json"));
    const AppModel::ResultInfo info = model.resultInfo();
    REQUIRE(info.exists);
    CHECK(info.width == 32);
    CHECK(info.height == 16);
    CHECK(info.fps == Approx(48.0));
    CHECK(info.frames == 8);
    CHECK(info.audio);
    CHECK(!info.modified.isEmpty());
    CHECK(info.runSeconds == 0.0);
    CHECK(info.codec == "ffv1");
    CHECK(info.format == FrameDesc{32, 16, PixelFormat::Yuv420p, 8});
    QString summary = model.resultSummary();
    CHECK(summary.startsWith(QString::fromUtf8("32×16 · 48 fps · 8 кадров · FFV1 8 бит 4:2:0 · со звуком · обновлён ")));
    CHECK(!summary.contains(QString::fromUtf8("готов за")));
    model.setLastRun(683.0);
    summary = model.resultSummary();
    CHECK(summary.endsWith(QString::fromUtf8("готов за 11:23")));
    CHECK(!bar.isHidden());
    CHECK(bar.text().contains(QString::fromUtf8("<b>Результат</b> · 32×16")));

    // export = a copy of the result file; the result itself is refused
    QSignalSpy messages(&model, &AppModel::message);
    REQUIRE(model.exportResult(Q(d / "out" / "copy.mp4")));
    CHECK(std::filesystem::file_size(d / "out" / "copy.mp4") == std::filesystem::file_size(result));
    CHECK(!model.exportResult(Q(result)));
    CHECK(messages.count() == 2);

    // the passes on disk, in pipeline order; keys 1..9 follow the chips
    const std::vector<std::string> passes = model.exportablePasses();
    REQUIRE(passes.size() == 2);
    CHECK(passes[0] == "depth_dlss");
    CHECK(passes[1] == "color_sr");
    const auto chips = model.chips();
    REQUIRE(chips.size() == 3);
    CHECK(chips[0].hotkey == 1);
    CHECK(chips[2].hotkey == 3);
    CHECK(chips[2].source == "result");
    CHECK(model.freeSpace() > 0);

    // another passes folder: the project points there, the passes come from there, the file remembers it
    WritePassFps(d / "passes2" / "color_sr", PassKind::ColorSr, 4, 24);
    QSignalSpy projectChanges(&model, &AppModel::projectChanged);
    model.setPassesRoot(Q(d / "passes2"));
    CHECK(model.dirty());
    CHECK(projectChanges.count() == 1);
    CHECK(model.project().passesRoot == d / "passes2");
    CHECK(model.exportablePasses() == std::vector<std::string>{"color_sr"});
    model.setPassesRoot(Q(d / "passes2"));  // the same folder again: nothing happens
    CHECK(projectChanges.count() == 1);
    REQUIRE(model.saveProject());
    CHECK(Project::Load(d / "proj.dlssvid.json").passesRoot == d / "passes2");
    CHECK(!model.dirty());

    // closing the project: nothing is shown, the recents keep it
    QSignalSpy closed(&model, &AppModel::projectChanged);
    REQUIRE(model.closeProject());
    CHECK(!model.hasProject());
    CHECK(closed.count() == 1);
    CHECK(model.resultSummary().isEmpty());
    CHECK(model.chips().empty());
    CHECK(model.exportablePasses().empty());
    CHECK(model.freeSpace() == 0);
    CHECK(!model.dirty());
    CHECK(bar.isHidden());
    REQUIRE(!model.recents().empty());
    CHECK(QFileInfo(model.recents()[0].path) == QFileInfo(Q(d / "proj.dlssvid.json")));
    CHECK(!model.closeProject());
}

// Regression (TASK-0017): a wheel over the project panel used to change the backend of a stage when the pointer happened
// to be over a combo box; unfocused value controls now pass the wheel on to the page.
TEST_CASE("Theme: the wheel over a combo box in a scrolling page scrolls the page; outside it changes the value", "[app][theme]") {
    QApplication& app = App();
    ApplyTheme(app);
    CHECK(app.styleSheet().contains("#4aa3df"));
    CHECK(app.palette().color(QPalette::Window) == QColor("#17181b"));

    QScrollArea area;
    auto* content = new QWidget(&area);
    auto* layout = new QVBoxLayout(content);
    auto* combo = new QComboBox(content);
    combo->addItems({"a", "b", "c"});
    layout->addWidget(combo);
    auto* filler = new QWidget(content);
    filler->setMinimumHeight(3000);
    layout->addWidget(filler);
    area.setWidget(content);
    area.resize(300, 200);
    area.show();
    Pump([] { return false; }, 100);
    REQUIRE(area.verticalScrollBar()->maximum() > 0);
    CHECK(combo->currentIndex() == 0);
    // the page (the combo's parent) must receive the wheel the combo gave up
    struct WheelSpy : QObject {
        int wheels = 0;
        bool eventFilter(QObject*, QEvent* e) override {
            if (e->type() == QEvent::Wheel) ++wheels;
            return false;
        }
    } spy;
    content->installEventFilter(&spy);

    auto wheel = [&](QWidget* target) {
        const QPointF pos(10, 10);
        QWheelEvent e(pos, target->mapToGlobal(pos.toPoint()), QPoint(0, -120), QPoint(0, -120), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QApplication::sendEvent(target, &e);
    };
    // inside the page: the combo keeps its value, the page scrolls — focused or not
    wheel(combo);
    Pump([] { return false; }, 50);
    CHECK(combo->currentIndex() == 0);
    CHECK(spy.wheels >= 1);
    combo->setFocus();
    wheel(combo);
    CHECK(combo->currentIndex() == 0);
    CHECK(spy.wheels >= 2);
    // outside a scrolling page the wheel still changes the value (the timeline's frame box)
    QComboBox loose;
    loose.addItems({"a", "b", "c"});
    loose.show();
    Pump([] { return false; }, 50);
    wheel(&loose);
    CHECK(loose.currentIndex() == 1);
}

TEST_CASE("AppModel loops playback back to the start when the loop toggle is on", "[app][model][gpu]") {
    App();
    const auto d = Dir("loop");
    WritePassFps(d / "passes" / "color_sr", PassKind::ColorSr, 6, 24);
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setSingleSource("color_sr");
    REQUIRE(model.lastTime() > 0.0);
    QSignalSpy loops(&model, &AppModel::loopChanged);
    model.setLoop(true);
    CHECK(model.loop());
    CHECK(loops.count() == 1);
    model.setLoop(true);
    CHECK(loops.count() == 1);  // no change, no signal
    // from the last frame: the next tick wraps to the start and keeps playing
    model.setTime(model.lastTime());
    model.setPlaying(true);
    Pump([&] { return model.time() < model.lastTime() / 2; }, 3000);
    CHECK(model.playing());
    CHECK(model.time() < model.lastTime() / 2);
    // without the loop the same run stops at the end
    model.setLoop(false);
    Pump([&] { return !model.playing(); }, 5000);
    CHECK(!model.playing());
    CHECK(model.time() == Approx(model.lastTime()));
    AppModel again(true);
    CHECK(!again.loop());  // the setting was left off
}

// Incident 2026-09-23 (user report): the «До | После» split screen flickered during playback — on the ticks where a
// source's frame had not decoded yet the cell fell to a plate or an empty background (a 3840×1600 result decodes on the
// CPU slower than its 48 fps). Now the store holds the previous frame (test_frame_store.cpp), the model reports it as
// Stale (drawn as is, the label says «загрузка…»), and playback steps only onto frames every shown source has resident,
// uploading finished loads itself: a slow decode plays slower than real time instead of skipping frames.
TEST_CASE("AppModel reports held frames as stale and plays only over resident frames (regression: split-screen flicker)",
          "[app][model][gpu][regression]") {
    App();
    const auto d = Dir("stale");
    WritePassFps(d / "passes" / "color_sr", PassKind::ColorSr, 40, 24);
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setSingleSource("color_sr");
    model.setTime(0.0);
    model.store().WaitForCurrent();
    REQUIRE(model.frameStateOf("color_sr") == FrameState::Ready);
    // a jump beyond the prefetch window: frame 30 is not resident yet (nothing uploads until WaitForCurrent), a resident
    // frame stands in for it — Stale for the renderer (no plate) and for the label
    model.setFrame(30);
    REQUIRE(model.store().GetStatus(30, "color_sr") != FrameStore::Status::Ready);
    CHECK(model.frameStateOf("color_sr") == FrameState::Stale);
    CHECK(model.frameStates().at("color_sr") == FrameState::Stale);
    const auto held = model.store().HeldFrameAt(model.time(), "color_sr");
    REQUIRE(held.has_value());
    CHECK(*held < 30);
    model.store().WaitForCurrent();
    CHECK(model.frameStateOf("color_sr") == FrameState::Ready);
    CHECK(!model.store().HeldFrameAt(model.time(), "color_sr").has_value());
    // playback from the start visits every frame in order, and each one is resident when the time lands on it
    model.setTime(0.0);
    std::vector<int64_t> visited;
    bool resident = true;
    QObject::connect(&model, &AppModel::timeChanged, &model, [&](double) {
        visited.push_back(model.baseFrame());
        resident = resident && model.store().StatusAt(model.time(), "color_sr") == FrameStore::Status::Ready;
    });
    model.setPlaying(true);
    Pump([&] { return !model.playing(); }, 15000);
    CHECK(!model.playing());
    std::vector<int64_t> expected;
    for (int64_t f = 1; f < 40; ++f) expected.push_back(f);
    CHECK(visited == expected);
    CHECK(resident);
}

TEST_CASE("RecentPreviews makes a 96x40 frame and the size line of a clip and caches them", "[app][previews]") {
    App();
    const auto d = Dir("previews");
    RecentPreviews::SetCacheDir(Q(d / "cache"));
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    ClipSpec big;
    big.frames = 2;
    big.width = 96;
    big.height = 64;
    big.fpsNum = 48;
    const auto result = WriteClip(d / "result.mkv", big);
    const RecentPreviews::Preview p = RecentPreviews::Make(Q(clip));
    CHECK(p.meta == QString::fromUtf8("48×32 24 fps"));
    REQUIRE(!p.image.isNull());
    CHECK(p.image.width() == RecentPreviews::kWidth * RecentPreviews::kScale);
    CHECK(p.image.height() == RecentPreviews::kHeight * RecentPreviews::kScale);
    CHECK(std::filesystem::exists(std::filesystem::path(RecentPreviews::CacheFile(Q(clip), {}).toStdWString())));
    // the cached copy carries the same line
    const RecentPreviews::Preview again = RecentPreviews::Make(Q(clip));
    CHECK(again.meta == p.meta);
    CHECK(again.image.size() == p.image.size());
    // with a result: «→ …» and the result's frame
    const RecentPreviews::Preview both = RecentPreviews::Make(Q(clip), Q(result));
    CHECK(both.meta == QString::fromUtf8("48×32 24 fps → 96×64 48 fps"));
    CHECK(!both.image.isNull());
    // an unreadable file: no line, no image, nothing cached
    const RecentPreviews::Preview none = RecentPreviews::Make(Q(d / "nope.mp4"));
    CHECK(none.meta.isEmpty());
    CHECK(none.image.isNull());
    CHECK(!std::filesystem::exists(std::filesystem::path(RecentPreviews::CacheFile(Q(d / "nope.mp4"), {}).toStdWString())));
    // asynchronously: nothing at first, then previewReady on the caller's thread
    RecentPreviews previews;
    QSignalSpy ready(&previews, &RecentPreviews::previewReady);
    CHECK(!previews.get(Q(clip)).has_value());
    Pump([&] { return ready.count() > 0; }, 10000);
    REQUIRE(ready.count() == 1);
    REQUIRE(previews.get(Q(clip)).has_value());
    CHECK(previews.get(Q(clip))->meta == p.meta);
    CHECK(ready.count() == 1);  // a second get() is served from memory
}

// The start screen after the mockup «1 · Стартовый экран»: the recents on the right as cards — a 96×40 preview, «face —
// результат готов» and «1920×800 24 fps → 3840×1600 48 fps · вчера»; the size line and the frame arrive from a worker.
TEST_CASE("StartPage lays the recents out as cards with a preview, a state and a size line", "[app][start]") {
    App();
    QSettings().clear();
    const auto d = Dir("startpage");
    RecentPreviews::SetCacheDir(Q(d / "cache"));
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    WriteDepth(d / "passes" / "depth_raw", 2);
    Project p = Project::Create(clip, d / "passes");
    p.Save(d / "clip.dlssvid.json");  // next to the video: the video's card says «есть проект»
    AppModel model(true);
    dlssvid::StartPage page(model);  // qualified: <windows.h> has a GDI StartPage()
    CHECK(page.cardTitles().isEmpty());
    CHECK(page.readinessText().contains("WARP"));
    model.addRecent(Q(clip));
    model.addRecent(Q(d / "clip.dlssvid.json"));  // newest first
    REQUIRE(page.cardTitles().size() == 2);
    CHECK(page.cardTitles()[0] == QString::fromUtf8("clip — глубина посчитана"));
    CHECK(page.cardTitles()[1] == "clip");
    CHECK(page.cardMetas()[0].contains(QString::fromUtf8("1 из 6 стадий")));
    CHECK(page.cardMetas()[1].contains(QString::fromUtf8("есть проект")));
    // the third line is the file's path (native separators); the tooltip carries the title, the path and the state
    CHECK(page.cardPaths()[0] == QDir::toNativeSeparators(Q(d / "clip.dlssvid.json")));
    CHECK(page.cardPaths()[1] == QDir::toNativeSeparators(Q(clip)));
    // the previews arrive from the worker: the size line in front of the meta and the frame
    Pump([&] { return page.cardHasPreview(0) && page.cardHasPreview(1); }, 15000);
    CHECK(page.cardHasPreview(0));
    CHECK(page.cardHasPreview(1));
    CHECK(page.cardMetas()[0].startsWith(QString::fromUtf8("48×32 24 fps · 1 из 6 стадий · ")));
    CHECK(page.cardMetas()[1].startsWith(QString::fromUtf8("48×32 24 fps · есть проект · ")));
    // a click on a card asks the window to open that file; «очистить» empties the list
    QSignalSpy opened(&page, &dlssvid::StartPage::openPathRequested);
    std::vector<QFrame*> cards;
    for (QFrame* f : page.findChildren<QFrame*>())
        if (f->property("role").toString() == "recent") cards.push_back(f);
    REQUIRE(cards.size() == 2);
    QTest::mouseClick(cards[1], Qt::LeftButton);
    REQUIRE(opened.count() == 1);
    CHECK(opened[0][0].toString() == Q(clip));
    model.clearRecents();
    CHECK(page.cardTitles().isEmpty());
}

// TASK-0022: parameters scoped to a backend (`ParamSpec::backends`) are rows of the stage card only while that backend
// is chosen — the model of the TensorRT upscaler hides under DLSS and appears with «Нейросеть (TensorRT)».
TEST_CASE("StageCard shows backend-scoped parameters only for their backend", "[app][card]") {
    App();
    const auto d = Dir("card");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard card(model, "upscale");
    auto* form = card.findChild<QFormLayout*>();
    REQUIRE(form);
    int modelRow = -1, backendRow = -1;
    for (int r = 0; r < form->rowCount(); ++r) {
        QLayoutItem* item = form->itemAt(r, QFormLayout::LabelRole);
        auto* label = item ? qobject_cast<QLabel*>(item->widget()) : nullptr;
        if (!label) continue;
        if (label->text() == QString::fromUtf8("Модель")) modelRow = r;
        if (label->text() == QString::fromUtf8("Метод")) backendRow = r;
    }
    REQUIRE(modelRow >= 0);
    REQUIRE(backendRow >= 0);
    card.refresh();
    CHECK(form->isRowVisible(backendRow));
    CHECK(!form->isRowVisible(modelRow));  // the project's backend is dlss
    model.setStageParam("upscale", "backend", "trt");
    card.refresh();
    CHECK(form->isRowVisible(modelRow));
    model.setStageParam("upscale", "backend", "nis");
    card.refresh();
    CHECK(!form->isRowVisible(modelRow));
}

// TASK-0045: the slider stored v x step - 3 x 0.1 = 0.30000000000000004, 14 x 0.1 = 1.4000000000000001: one visible value
// from the slider and from the JSON gave two fingerprints, and the plan read «Интенсивность 1.4 → 1.4».
TEST_CASE("StageCard stores a slider value rounded to its step", "[app][card][regression]") {
    App();
    const auto d = Dir("card_quantize");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard card(model, "nr");
    card.refresh();
    QSlider* slider = nullptr;
    for (QSlider* s : card.findChildren<QSlider*>())
        if (s->accessibleName() == QString::fromUtf8("Интенсивность")) slider = s;
    REQUIRE(slider);
    REQUIRE(slider->maximum() == 40);  // step 0.05 (TASK-0046)
    for (const int ticks : {6, 14, 28}) {  // 6 x 0.05 = 0.30000000000000004
        slider->setValue(ticks);
        const dlssvid::StageEntry* e = model.stageEntry("nr");
        REQUIRE(e);
        const double v = e->params["intensity"].get<double>();
        const double want = ticks / 20.0;  // the nearest double to the decimal the form shows
        INFO(ticks << " ticks: " << nlohmann::json(v).dump());
        CHECK(v == want);
        CHECK(nlohmann::json(v).dump().size() <= 3);  // "0.3", "0.7", "1.4"
    }
}

// TASK-0046: the NR card shows every setting - the basic rows in groups, the rest in «Ещё настройки» (collapsed, saying
// what it hides), a row that cannot act disabled with a note, reset per row and for the card, changed values in accent.
TEST_CASE("StageCard NR: groups, the collapsed more block, conditions, reset and changed values", "[app][card][nr]") {
    App();
    const auto d = Dir("card_nr");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    QSettings().remove("card/nr/more");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard card(model, "nr");
    card.refresh();
    auto labelOf = [&](const char* text) -> QLabel* {
        for (QLabel* l : card.findChildren<QLabel*>())
            if (l->text() == QString::fromUtf8(text)) return l;
        return nullptr;
    };
    // groups and rows
    for (const char* group : {"Модель", "Локальные", "Временная стабильность", "Маска", "Вход", "Разрешение модели"}) {
        INFO(group);
        REQUIRE(labelOf(group));
        CHECK(labelOf(group)->property("role").toString() == "group");
    }
    for (const char* row : {"Стиль", "Интенсивность", "Проходы", "Локальный тон", "Локальная структура", "Подавление мерцания", "Автомаска кожи", "Структура кожи"}) {
        INFO(row);
        REQUIRE(labelOf(row));
        CHECK(labelOf(row)->isVisibleTo(&card));
    }
    CHECK_FALSE(labelOf("Сила на коже (маски)")->isVisibleTo(&card));  // no face / skin mask among the inputs
    // «Ещё настройки»: collapsed, opens on a click, says what it hides
    QToolButton* more = nullptr;
    for (QToolButton* b : card.findChildren<QToolButton*>())
        if (b->property("role").toString() == "more") more = b;
    REQUIRE(more);
    CHECK_FALSE(labelOf("Экспозиция")->isVisibleTo(&card));
    model.setStageParam("nr", "exposure", 1.0);
    card.refresh();
    QLabel* summary = card.findChild<QLabel*>("moreSummary");
    REQUIRE(summary);
    // the summary sits beside the toggle, wrapped: a long one no longer widens the card (final review of TASK-0046)
    CHECK(summary->text().contains(QString::fromUtf8("изменено: экспозиция ") + QLocale().toString(1.0, 'f', 2) + " EV"));
    CHECK(more->toolTip().contains(QString::fromUtf8("экспозиция")));
    more->click();
    CHECK(labelOf("Экспозиция")->isVisibleTo(&card));
    CHECK(labelOf("Экспозиция")->property("role").toString() == "changed");
    CHECK_FALSE(labelOf("Предел отношения")->isVisibleTo(&card));  // only at another model scale
    // skin structure: disabled with a note until the automatic mask is on
    QLabel* note = labelOf("действует только с автомаской");
    REQUIRE(note);
    CHECK(note->isVisibleTo(&card));
    model.setStageParam("nr", "auto_mask", true);
    card.refresh();
    CHECK_FALSE(note->isVisibleTo(&card));
    // reset of one row: the key leaves the project
    model.setStageParam("nr", "local_tone", 1.5);
    card.refresh();
    CHECK(labelOf("Локальный тон")->property("role").toString() == "changed");
    QToolButton* reset = nullptr;
    for (QToolButton* b : card.findChildren<QToolButton*>())
        if (b->accessibleName() == QString::fromUtf8("Сбросить «Локальный тон»")) reset = b;
    REQUIRE(reset);
    CHECK(reset->isVisibleTo(&card));
    CHECK(reset->focusPolicy() & Qt::TabFocus);
    reset->click();
    CHECK_FALSE(model.stageEntry("nr")->params.contains("local_tone"));
    // the card: the form's keys go, the backend and the engineer keys stay
    model.setStageParam("nr", "format", "png");
    model.setStageParam("nr", "intensity", 1.4);
    card.refresh();
    QPushButton* resetCard = nullptr;
    for (QPushButton* b : card.findChildren<QPushButton*>())
        if (b->text() == QString::fromUtf8("Сбросить")) resetCard = b;
    REQUIRE(resetCard);
    resetCard->click();
    const nlohmann::json after = model.stageEntry("nr")->params;
    CHECK(after["backend"] == "ngx");
    CHECK(after["format"] == "png");
    CHECK_FALSE(after.contains("intensity"));
    CHECK_FALSE(after.contains("exposure"));
    CHECK_FALSE(after.contains("auto_mask"));
    // PR-9 (spec :761): the settings before the reset are in «Перед сбросом» - the card's keys, not the backend or the pass format
    const auto slot = PresetStore::Library().Slot(PresetSlot::BeforeReset, "nr");
    REQUIRE(slot);
    CHECK(slot->load.ok());
    CHECK(slot->load.preset.stages.at("nr").params["intensity"] == 1.4);
    CHECK_FALSE(slot->load.preset.stages.at("nr").params.contains("backend"));
    CHECK_FALSE(slot->load.preset.stages.at("nr").params.contains("format"));
    CHECK(slot->load.preset.name == "Перед сбросом · missing");
    QSettings().remove("card/nr/more");
}

// TASK-0046, review: a changed value whose row its condition hides is still in the fingerprint - named on «Ещё
// настройки», dropped by «Сбросить»; the aliases of the JSON (skin-structure, no_guides) are shown and reset as the
// keys; skin structure «как структура» shows the value it follows; the label's menu resets the row; the controls are
// named for screen readers.
TEST_CASE("StageCard NR: hidden changed values, folded aliases, the followed value, the label's menu", "[app][card][nr][regression]") {
    App();
    const auto d = Dir("card_nr_review");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    QSettings().remove("card/nr/more");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard card(model, "nr");
    card.refresh();
    auto named = [&](auto* type, const char* name) {
        using T = std::remove_pointer_t<decltype(type)>;
        for (T* w : card.findChildren<T*>())
            if (w->accessibleName() == QString::fromUtf8(name)) return w;
        return static_cast<T*>(nullptr);
    };
    auto labelOf = [&](const char* text) -> QLabel* {
        for (QLabel* l : card.findChildren<QLabel*>())
            if (l->text() == QString::fromUtf8(text)) return l;
        return nullptr;
    };
    QToolButton* more = nullptr;
    for (QToolButton* b : card.findChildren<QToolButton*>())
        if (b->property("role").toString() == "more") more = b;
    QPushButton* resetCard = nullptr;
    for (QPushButton* b : card.findChildren<QPushButton*>())
        if (b->text() == QString::fromUtf8("Сбросить")) resetCard = b;
    REQUIRE(more);
    REQUIRE(resetCard);
    REQUIRE(named(static_cast<QComboBox*>(nullptr), "Стиль"));

    // the threshold changed, then the filter off: the row hides, the value stays in the run
    model.setStageParams("nr", {{"backend", "ngx"}, {"temporal", 0.3}, {"temporal_threshold", 0.2}});
    model.setStageParams("nr", {{"backend", "ngx"}, {"temporal_threshold", 0.2}});
    card.refresh();
    QLabel* summary = card.findChild<QLabel*>("moreSummary");
    REQUIRE(summary);
    CHECK(summary->text().contains(QString::fromUtf8("не действует: порог подавления мерцания ") + QLocale().toString(0.2, 'f', 2)));
    CHECK(resetCard->isVisibleTo(&card));  // before: hidden, the card had no way back
    resetCard->click();
    CHECK(model.stageEntry("nr")->params == nlohmann::json{{"backend", "ngx"}});

    // aliases from the engineer JSON
    model.setStageParams("nr", {{"backend", "ngx"}, {"auto-mask", true}, {"skin-structure", 0.5}, {"no_guides", true}});
    card.refresh();
    REQUIRE(named(static_cast<QCheckBox*>(nullptr), "Автомаска кожи"));
    CHECK(named(static_cast<QCheckBox*>(nullptr), "Автомаска кожи")->isChecked());
    CHECK_FALSE(named(static_cast<QCheckBox*>(nullptr), "Глубина и векторы")->isChecked());
    CHECK(labelOf("Автомаска кожи")->property("role").toString() == "changed");
    REQUIRE(named(static_cast<QCheckBox*>(nullptr), "Структура кожи: отдельно"));
    CHECK(named(static_cast<QCheckBox*>(nullptr), "Структура кожи: отдельно")->isChecked());
    CHECK(named(static_cast<QDoubleSpinBox*>(nullptr), "Структура кожи")->value() == Approx(0.5));
    resetCard->click();
    CHECK(model.stageEntry("nr")->params == nlohmann::json{{"backend", "ngx"}});  // before: the aliases stayed

    // «как структура»: the disabled number is the local structure it follows, not 0
    model.setStageParams("nr", {{"backend", "ngx"}, {"auto_mask", true}, {"local_structure", 0.7}});
    card.refresh();
    QDoubleSpinBox* skin = named(static_cast<QDoubleSpinBox*>(nullptr), "Структура кожи");
    REQUIRE(skin);
    CHECK_FALSE(skin->isEnabled());
    CHECK(skin->value() == Approx(0.7));
    named(static_cast<QCheckBox*>(nullptr), "Структура кожи: отдельно")->click();
    CHECK(model.stageEntry("nr")->params["skin_structure"] == 0.7);  // starts from what it followed

    // exposure keeps the 0.01 EV of the v1 migration
    model.setStageParam("nr", "exposure", 0.58);
    card.refresh();
    CHECK(named(static_cast<QDoubleSpinBox*>(nullptr), "Экспозиция")->value() == Approx(0.58));

    // a long summary wraps instead of widening the card
    const int narrow = card.minimumSizeHint().width();
    model.setStageParams("nr", {{"backend", "ngx"}, {"model_scale", 0.5}, {"tonemap", "aces"}, {"input_transfer", "pq"}, {"exposure", 0.5},
                                {"temporal_threshold", 0.2}, {"transfer", 0.8}, {"max_ratio", 6}});
    card.refresh();
    CHECK(card.minimumSizeHint().width() <= narrow);
    model.setStageParams("nr", {{"backend", "ngx"}});
    card.refresh();

    // a double click on a slider resets its row
    model.setStageParam("nr", "intensity", 1.4);
    card.refresh();
    QSlider* intensity = nullptr;
    for (QSlider* s : card.findChildren<QSlider*>())
        if (s->accessibleName() == QString::fromUtf8("Интенсивность")) intensity = s;
    REQUIRE(intensity);
    QMouseEvent dbl(QEvent::MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(intensity, &dbl);
    CHECK_FALSE(model.stageEntry("nr")->params.contains("intensity"));

    // the label's context menu resets the row
    model.setStageParam("nr", "local_tone", 1.5);
    card.refresh();
    QLabel* tone = labelOf("Локальный тон");
    REQUIRE(tone);
    REQUIRE(tone->actions().size() == 1);
    CHECK(tone->contextMenuPolicy() == Qt::ActionsContextMenu);
    CHECK(tone->actions()[0]->isEnabled());
    tone->actions()[0]->trigger();
    CHECK_FALSE(model.stageEntry("nr")->params.contains("local_tone"));
    card.refresh();
    CHECK_FALSE(tone->actions()[0]->isEnabled());
    QSettings().remove("card/nr/more");
}

// TASK-0046, review: ↺ of the backend erased the key only - the VDA model stayed under DA3 and the plan failed.
TEST_CASE("StageCard: the reset of the backend drops the old backend's model, as a switch in the list", "[app][card][regression]") {
    App();
    const auto d = Dir("card_backend_reset");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageParams("depth", {{"backend", "vda"}, {"model", "metric-vda-large"}});
    dlssvid::StageCard card(model, "depth");
    card.refresh();
    QToolButton* reset = nullptr;
    for (QToolButton* b : card.findChildren<QToolButton*>())
        if (b->accessibleName() == QString::fromUtf8("Сбросить «Метод»")) reset = b;
    REQUIRE(reset);
    reset->click();
    const nlohmann::json p = model.stageEntry("depth")->params;
    CHECK_FALSE(p.contains("backend"));
    CHECK_FALSE(p.contains("model"));  // before: metric-vda-large under DA3

    // «Сбросить» shows for what it removes: not for the backend alone (it keeps it), and a DA3 model from the JSON stays
    QPushButton* resetCard = nullptr;
    for (QPushButton* b : card.findChildren<QPushButton*>())
        if (b->text() == QString::fromUtf8("Сбросить")) resetCard = b;
    REQUIRE(resetCard);
    model.setStageParams("depth", {{"backend", "vda"}});
    card.refresh();
    CHECK_FALSE(resetCard->isVisibleTo(&card));  // before: shown, and a click did nothing
    model.setStageParams("depth", {{"backend", "da3"}, {"model", "da3mono-large"}, {"stabilize", "none"}});
    card.refresh();
    CHECK(resetCard->isVisibleTo(&card));
    resetCard->click();
    CHECK(model.stageEntry("depth")->params == nlohmann::json{{"backend", "da3"}, {"model", "da3mono-large"}});
}

// TASK-0046 (D7): the versions panel named only the intensity, so two NR versions with another style looked the same.
TEST_CASE("Versions panel names the values other than the default, the basic ones first", "[app][versions][regression]") {
    const nlohmann::json p = {{"backend", "ngx"},      {"style", "cinematic"},       {"intensity", 1.0}, {"local_tone", 1.5},
                              {"auto_mask", true},     {"skin_structure", 0.5},      {"model_scale", 0.5}, {"temporal_threshold", 0.2},
                              {"guides", false},       {"format", "png"}};
    QString tip;
    const QString s = dlssvid::HumanStageParams("nr", p, &tip);
    INFO(s.toStdString());
    CHECK(s.startsWith(QString::fromUtf8("Стиль Кинематографичный (2) · Интенсивность 1 · Локальный тон 1.5")));
    CHECK(s.indexOf(QString::fromUtf8("Структура кожи")) < s.indexOf(QString::fromUtf8("Порог подавления мерцания")));  // basic before «Ещё»
    CHECK_FALSE(s.contains(QString::fromUtf8("Разрешение модели")));  // the 8th of nine values: cut off
    CHECK(s.endsWith(QString::fromUtf8("…")));  // nine values, six shown
    CHECK_FALSE(s.contains(QString::fromUtf8("Формат пасса")));
    CHECK(tip.contains("style = "));
    CHECK(tip.contains("format = "));
    CHECK(dlssvid::HumanStageParams("nr", {{"backend", "ngx"}, {"format", "png"}}).contains(QString::fromUtf8("Формат пасса")));  // engineer keys too
    CHECK(dlssvid::HumanStageParams("nr", {{"backend", "ngx"}}).isEmpty());
    // a DA3 model set in the JSON sets two depth versions apart too (final review)
    QString depthTip;
    CHECK(dlssvid::HumanStageParams("depth", {{"backend", "da3"}, {"model", "da3mono-large"}}, &depthTip).contains("da3mono-large"));
    CHECK(depthTip.contains("model = da3mono-large"));
}

// TASK-0039: the VDA models in the depth card - Small / Base / Large for vda and worker:vda, hidden for DA3; a switch to
// DA3 drops the VDA model from the project; an id the list does not name is shown as it is.
TEST_CASE("StageCard lists the VDA models for VDA only and a switch to DA3 drops the VDA model", "[app][card][models]") {
    App();
    const auto d = Dir("card_vda");
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard card(model, "depth");
    auto* form = card.findChild<QFormLayout*>();
    REQUIRE(form);
    int modelRow = -1, windowRow = -1;
    for (int r = 0; r < form->rowCount(); ++r) {
        QLayoutItem* item = form->itemAt(r, QFormLayout::LabelRole);
        auto* label = item ? qobject_cast<QLabel*>(item->widget()) : nullptr;
        if (!label) continue;
        if (label->text() == QString::fromUtf8("Модель")) modelRow = r;
        if (label->text() == QString::fromUtf8("Окно кадров VDA")) windowRow = r;
    }
    REQUIRE(modelRow >= 0);
    REQUIRE(windowRow >= 0);
    card.refresh();
    CHECK(!form->isRowVisible(modelRow));  // DA3 by default: no list
    CHECK(!form->isRowVisible(windowRow));
    model.setStageParam("depth", "backend", "vda");
    card.refresh();
    CHECK(form->isRowVisible(modelRow));
    CHECK(form->isRowVisible(windowRow));
    // the field of a row is a cell: the control, the note of a disabled row and ↺ (TASK-0046)
    auto* combo = form->itemAt(modelRow, QFormLayout::FieldRole)->widget()->findChild<QComboBox*>();
    REQUIRE(combo);
    REQUIRE(combo->count() == 3);
    CHECK(combo->itemData(0).toString().isEmpty());
    CHECK(combo->itemData(1).toString() == "metric-vda-base");
    CHECK(combo->itemData(2).toString() == "metric-vda-large");
    CHECK(combo->itemText(2).contains("CC-BY-NC-4.0"));
    CHECK(combo->currentIndex() == 0);  // Small, the default
    combo->setCurrentIndex(2);
    CHECK(model.stageEntry("depth")->params["model"] == "metric-vda-large");
    model.setStageParam("depth", "backend", "worker:vda");  // the same list: the model stays
    CHECK(model.stageEntry("depth")->params["model"] == "metric-vda-large");
    model.setStageParam("depth", "backend", "da3");
    card.refresh();
    CHECK(!model.stageEntry("depth")->params.contains("model"));  // before: DA3 with a VDA model, refused by the plan
    CHECK(!form->isRowVisible(modelRow));
    model.setStageParam("depth", "backend", "vda");
    model.setStageParam("depth", "model", "vda-small");  // a registry id from the JSON
    card.refresh();
    REQUIRE(combo->count() == 4);
    CHECK(combo->currentText().contains("vda-small"));
}

// TASK-0039: the plan's advice on the depth card - a non-commercial model is a warning, Small is none.
TEST_CASE("AppModel: the depth card warns about the licence of VDA Large", "[app][model][models]") {
    App();
    const auto d = Dir("card_vda_licence");
    ClipSpec spec;
    spec.frames = 3;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    Project p = Project::Create(clip, d / "passes");
    const std::string models = (std::filesystem::path(DLSSVID_SOURCE_DIR) / "models").string();
    for (auto& st : p.stages) {
        st.enabled = st.name == "depth";
        if (st.name == "depth") st.params = {{"backend", "vda"}, {"model", "metric-vda-large"}, {"models_dir", models}};
    }
    p.Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(model.planError().isEmpty());
    CHECK(model.cardState("depth").warning.contains("CC-BY-NC-4.0"));
    CHECK(model.planBlocker().isEmpty());  // WARP: the memory is unknown, no VRAM error
    model.setStageParam("depth", "model", "");
    model.refreshPlan();
    CHECK(!model.cardState("depth").warning.contains("CC-BY-NC-4.0"));
}

// Audit fg-2 (TASK-0038): `process: fg failed - ...` from the runner (FG in its own process failed, the result was encoded
// without it) keeps the stage failed while the run finishes with a result - also when a progress update glued onto it.
TEST_CASE("ProcessingPanel: a stage the runner reports failed stays failed while the run finishes with a result", "[app][tasks][isolation]") {
    App();
    const auto d = Dir("stage_failed");
    TaskQueue queue;
    ProcessingPanel panel(queue);
    const std::vector<TaskQueue::StagePlan> plan{{"fg", "Генерация кадров ×2", 4.0, false}, {"encode", "Кодирование", 2.0, false}};
    // the runner's output as cmd scripts: whole lines, and a progress update without its newline glued to the log line
    auto script = [&](const char* name, const char* firstLine) {
        const auto path = d / name;
        std::ofstream(path) << "@echo off\r\n"
                            << firstLine << "\r\n"
                            << "echo [2026-10-12 10:00:00.000] [warning] process: fg failed - the fg process crashed\r\n"
                            << "echo encode: 4/4 frames\r\n";
        return QString::fromStdWString(path.wstring());
    };
    for (const QString& file : {script("lines.cmd", "echo fg: 1/4 frames"), script("glued.cmd", "<nul set /p \"=fg: 2/4 frames\"")}) {
        INFO(file.toStdString());
        const int id = queue.enqueue("Обработка", "cmd", {"/c", file}, plan);
        panel.watch(id, "clip.mp4 → clip_result.mkv");
        Pump([&] { return queue.finished(id); }, 20000);
        REQUIRE(queue.succeeded(id));
        const TaskQueue::Progress p = queue.progress(id);
        CHECK(p.failedStageTitles() == QStringList{"Генерация кадров ×2"});
        CHECK(p.stagesDone == 1);
        CHECK(p.fraction == 1.0);
        CHECK(panel.headline().startsWith("Готово за"));
        CHECK(panel.headline().contains("без стадии «Генерация кадров ×2»"));
        const QStringList lines = panel.stageLines();
        REQUIRE(lines.size() == 2);
        CHECK(lines[0].startsWith("Генерация кадров ×2 · ошибка"));  // before: done as soon as encode started
        CHECK(lines[1].startsWith("Кодирование · готово"));
        auto* showResult = panel.findChild<QPushButton*>("showResult");
        REQUIRE(showResult);
        CHECK(!showResult->isHidden());
    }
}

TEST_CASE("TaskQueue takes a failed stage only from the runner's line, not from any 'process: X failed' text", "[app][tasks][isolation][regression]") {
    App();
    const auto d = Dir("failed_line");
    TaskQueue queue;
    const std::vector<TaskQueue::StagePlan> plan{{"fg", "FG", 1.0, false}, {"encode", "Кодирование", 1.0, false}};
    // the warning Subprocess writes when the job object cannot be used contains "process: AssignProcessToJobObject failed"
    const auto path = d / "warn.cmd";
    std::ofstream(path) << "@echo off\r\n"
                        << "echo [2026-10-12 10:00:00.000] [warning] Subprocess: AssignProcessToJobObject failed (error 5); the child is not tied\r\n"
                        << "echo fg: 4/4 frames\r\n"
                        << "echo encode: 4/4 frames\r\n";
    const int id = queue.enqueue("Обработка", "cmd", {"/c", QString::fromStdWString(path.wstring())}, plan);
    Pump([&] { return queue.finished(id); }, 20000);
    REQUIRE(queue.succeeded(id));
    CHECK(queue.progress(id).failedStageTitles().isEmpty());  // before: a stage «AssignProcessToJobObject» failed
}

TEST_CASE("TaskQueue runs dlssvid process whose fg process crashes: fg failed, encode done, the task succeeded", "[app][tasks][process][isolation]") {
    App();
    const auto d = Dir("process_fg_crash");
    ClipSpec spec;
    spec.frames = 4;
    spec.width = 48;
    spec.height = 32;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    const auto out = d / "out.mkv";
    TaskQueue queue;
    QStringList output;
    QObject::connect(&queue, &TaskQueue::taskOutput, [&](int, const QString& line) { output << line; });
    const std::vector<TaskQueue::StagePlan> plan{{"depth", "Глубина", 1.0, false}, {"flow", "Векторы", 1.0, false}, {"upscale", "Апскейл", 1.0, false},
                                                  {"nr", "NR", 1.0, false},          {"fg", "FG", 1.0, false},      {"encode", "Кодирование", 1.0, false}};
    const QStringList args{"process", "-i", QString::fromStdWString(clip.wstring()), "-o", QString::fromStdWString(out.wstring()), "--passes",
                           QString::fromStdWString((d / "p").wstring()), "--warp", "--codec", "ffv1", "--disable-unavailable",
                           "--param", "depth.backend=stub", "--param", "flow.backend=stub", "--param", "upscale.backend=nis",
                           "--param", "nr.backend=stub", "--param", "fg.backend=blend", "--fg-crash-after", "1"};
    const int id = queue.enqueue("Обработка", DLSSVID_CLI_PATH, args, plan);
    Pump([&] { return queue.finished(id); }, 120000);
    REQUIRE(queue.finished(id));
    CHECK(queue.succeeded(id));  // before: the whole run died with 0xC0000005, no result
    const TaskQueue::Progress p = queue.progress(id);
    CHECK(p.failedStageTitles() == QStringList{"FG"});
    for (const auto& s : p.stages)
        if (s.name == "encode") CHECK(s.state == "done");
    CHECK(std::filesystem::exists(out));
    CHECK(DecodeAll(out).size() == 4);  // color_nr: 4 frames
    CHECK(std::any_of(output.begin(), output.end(), [](const QString& l) { return l.contains("process: fg failed"); }));
}

// Audit play-fg-rate (TASK-0038): playback is paced by a clock at the exact rational rate. Before, a QTimer of
// floor(1000 / fps) ms advanced a frame per tick: 48 fps played at 50 Hz, 24 fps at 24.39 Hz.
TEST_CASE("PlaybackClock paces frames at the exact rational rate: 48 fps is 48 Hz, not 50; 24 fps steps 1/24 s (regression: timer interval truncated to whole ms)",
          "[app][playback][regression]") {
    auto shownInOneSecond = [](Rational rate, const std::vector<int64_t>& stepsMs) {
        PlaybackClock clock;
        clock.Start(0, 0.0, rate);
        int shown = 0;
        size_t i = 0;
        // up to 1 s plus the longest step: the frame due at exactly 1 s is shown by the first tick at or after it
        for (int64_t now = 0; now <= 1'007'000'000;) {
            if (clock.Decide(now, true) == PlaybackClock::Action::Show) {
                ++shown;
                clock.Advance(now);
            }
            now += stepsMs[i++ % stepsMs.size()] * 1'000'000;
        }
        return shown;
    };
    for (const std::vector<int64_t>& steps : {std::vector<int64_t>{1}, std::vector<int64_t>{1, 3, 7}}) {
        CHECK(shownInOneSecond({48, 1}, steps) == 48);
        CHECK(shownInOneSecond({24, 1}, steps) == 24);
        CHECK(shownInOneSecond({50, 1}, steps) == 50);  // the test tells 48 from 50
    }
    PlaybackClock clock;
    clock.Start(0, 0.0, {48, 1});
    for (int k = 0; k < 5; ++k) {
        CHECK(clock.NextTime() == Catch::Approx((k + 1) / 48.0));
        clock.Advance(clock.DueNs());
    }
    clock.Start(0, 1.0 / 24.0 + 0.004, {48, 1});  // between two frames of the grid: snapped, the next is the following one
    CHECK(clock.NextTime() == Catch::Approx(3.0 / 48.0));
    PlaybackClock ntsc;
    ntsc.Start(0, 0.0, {24000, 1001});
    for (int k = 0; k < 1001; ++k) ntsc.Advance(ntsc.DueNs());
    CHECK(std::llabs(ntsc.DueNs() - std::llround(1002.0 * 1e9 * 1001 / 24000)) <= 1);  // no drift over 1001 frames
}

TEST_CASE("PlaybackClock waits for a slow decode, skips after the stall limit and restarts the count instead of racing", "[app][playback]") {
    PlaybackClock c;
    c.Start(0, 0.0, {48, 1});
    const int64_t period = c.PeriodNs();
    const int64_t due = c.DueNs();
    CHECK(due == period);
    CHECK(c.Decide(due - 1, true) == PlaybackClock::Action::Wait);
    CHECK(c.Decide(due, false) == PlaybackClock::Action::Wait);
    CHECK(c.Decide(due + PlaybackClock::kStallNs - 1, false) == PlaybackClock::Action::Wait);
    CHECK(c.Decide(due + PlaybackClock::kStallNs, false) == PlaybackClock::Action::Skip);
    CHECK(c.Decide(due, true) == PlaybackClock::Action::Show);
    c.Advance(due + 15'000'000);  // late by less than a period: the grid stays
    CHECK(c.DueNs() == std::llround(2e9 / 48));
    const int64_t now = c.DueNs() + 50'000'000;  // late by more than a period: count from now
    c.Advance(now);
    CHECK(c.DueNs() == now + period);
    c.Restart(now + 1, 1.5);  // a jump while playing
    CHECK(c.NextTime() == Catch::Approx(1.5 + 1.0 / 48.0));
    CHECK(c.WakeNs(0) == c.DueNs());
    CHECK(c.WakeNs(c.DueNs() + 5) == c.DueNs() + 5 + PlaybackClock::kPollNs);

    // a loop back to the start on time keeps the grid; one a period late counts from now
    PlaybackClock loop;
    loop.Start(0, 0.0, {48, 1});
    for (int k = 0; k < 4; ++k) loop.Advance(loop.DueNs());
    const int64_t loopDue = loop.DueNs();
    loop.Restart(loopDue + 5'000'000, 0.0);
    CHECK(loop.DueNs() == loopDue + loop.PeriodNs());  // the next frame on the old grid, not 5 ms late
    CHECK(loop.NextTime() == Catch::Approx(1.0 / 48.0));
    const int64_t late = loop.DueNs() + 2 * loop.PeriodNs();
    loop.Restart(late, 0.0);
    CHECK(loop.DueNs() == late + loop.PeriodNs());
}

TEST_CASE("AppModel plays at the real rate of the fastest shown source and shows every frame (regression: 20 ms timer made 48 fps play at 50 Hz)",
          "[app][model][gpu][regression]") {
    App();
    for (const int fps : {48, 24}) {
        INFO(fps << " fps");
        const auto d = Dir(fps == 48 ? "rate48" : "rate24");
        WritePassFps(d / "passes" / (fps == 48 ? "color_fg" : "color_sr"), fps == 48 ? PassKind::ColorFg : PassKind::ColorSr, 49, fps);
        Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
        AppModel model(true);
        model.openProject(Q(d / "proj.dlssvid.json"));
        model.setSingleSource(fps == 48 ? "color_fg" : "color_sr");
        model.setTime(0.0);
        model.store().WaitForCurrent();
        std::vector<int64_t> visited;
        QObject::connect(&model, &AppModel::timeChanged, [&] { visited.push_back(model.baseFrame()); });
        QElapsedTimer played;
        played.start();
        model.setPlaying(true);
        Pump([&] { return !model.playing(); }, 6000);
        const double seconds = static_cast<double>(played.nsecsElapsed()) / 1e9;
        std::vector<int64_t> all(48);
        for (int i = 0; i < 48; ++i) all[static_cast<size_t>(i)] = i + 1;
        CHECK(visited == all);                 // every frame shown, none skipped
        CHECK(seconds >= 49.0 / fps - 0.005);  // never faster than real time (the old timer: 49 x 20 ms = 0.98 s at 48 fps);
                                               // no upper bound: a loaded CI machine may only make it slower
    }
}

// Deterministic counterpart (audit play-fg-rate): the playback timer driven by hand against a fake clock - one second of
// the clock shows exactly as many frames as the source's rate, at the source's steps.
TEST_CASE("AppModel paces playback by its clock: 48 frames of a 48 fps source in one second, 50 at 50 fps (regression: whole-ms timer)",
          "[app][model][gpu][regression]") {
    App();
    for (const int fps : {48, 50}) {
        INFO(fps << " fps");
        const auto d = Dir(fps == 48 ? "clock48" : "clock50");
        WritePassFps(d / "passes" / "color_fg", PassKind::ColorFg, 80, fps);
        Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
        AppModel model(true);
        model.openProject(Q(d / "proj.dlssvid.json"));
        model.setSingleSource("color_fg");
        for (int k = 0; k <= 60; ++k) {  // every frame of the second resident: the clock alone decides
            model.setTime(static_cast<double>(k) / fps);
            model.store().WaitForCurrent();
        }
        model.setTime(0.0);
        model.store().WaitForCurrent();
        int64_t fake = 0;
        model.setPlaybackClock([&] { return fake; });
        std::vector<double> times;
        const auto conn = QObject::connect(&model, &AppModel::timeChanged, [&](double t) { times.push_back(t); });
        model.setPlaying(true);
        for (fake = 0; fake <= 1'000'000'000; fake += 1'000'000) model.playbackTick();
        model.setPlaying(false);
        QObject::disconnect(conn);
        REQUIRE(times.size() == static_cast<size_t>(fps));  // the old timer: 1000 / floor(1000 / 48) = 50 frames a second
        for (size_t k = 0; k < times.size(); ++k) CHECK(times[k] == Catch::Approx(static_cast<double>(k + 1) / fps));
    }
}

// Review of TASK-0038 (play-2): a time set without setTime while playing (a project opened during playback restores its
// saved time) restarts the clock from there instead of jumping back to where the clock was.
TEST_CASE("AppModel playback goes on from a time changed behind its back", "[app][model][gpu][regression]") {
    App();
    const auto d = Dir("clock_bypass");
    WritePassFps(d / "passes" / "color_fg", PassKind::ColorFg, 80, 48);
    Project::Create(d / "missing.mp4", d / "passes").Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setSingleSource("color_fg");
    for (int k = 0; k <= 70; ++k) {
        model.setTime(static_cast<double>(k) / 48);
        model.store().WaitForCurrent();
    }
    model.setTime(0.0);
    int64_t fake = 0;
    model.setPlaybackClock([&] { return fake; });
    model.setPlaying(true);
    for (fake = 0; fake <= 100'000'000; fake += 1'000'000) model.playbackTick();  // ~4 frames
    REQUIRE(model.time() < 0.2);
    model.state().time = 1.0;  // as reloadSources does with a project's saved time
    for (int i = 0; i < 40; ++i, fake += 1'000'000) model.playbackTick();
    model.setPlaying(false);
    CHECK(model.time() > 1.0);  // before: back to ~0.1 s, where the clock was
    CHECK(model.time() < 1.1);
}

// TASK-0040: the scrubbing proxy of a long-GOP source is built in the background by `dlssvid proxy` after the project
// opens and taken by the store when it is done; closing the project stops a build; a scrub jump is timed until its
// frames are on the GPU.
TEST_CASE("AppModel builds the scrubbing proxy in the background and the store takes it", "[app][model][proxy]") {
    App();
    const auto d = Dir("proxy_bg");
    ClipSpec spec;
    spec.frames = 40;
    spec.width = 96;
    spec.height = 64;
    spec.gop = 40;  // one keyframe: a proxy is needed
    const auto clip = WriteClip(d / "clip.mkv", spec);
    Project::Create(clip, d / "passes").Save(d / "proj.dlssvid.json");
    QSettings().setValue("view/autoProxy", true);
    AppModel model(true);
    QSignalSpy proxies(&model, &AppModel::proxiesChanged);
    model.openProject(Q(d / "proj.dlssvid.json"));
    CHECK((model.proxyBuilding() || model.proxyState("source") == "ready"));
    Pump([&] { return !model.proxyBuilding(); }, 60000);
    CHECK(model.proxyState("source") == "ready");
    CHECK(ValidProxy(clip, d / "passes", "source"));
    CHECK(proxies.count() >= 1);
    // the jumps are served by it and timed
    for (const double t : {1.2, 0.3, 1.5}) {
        model.setTime(t);
        Pump([&] { return model.framesReadyAt(model.time()) && model.store().Update() == false; }, 10000);
        model.notifyFramesUpdated();
    }
    CHECK(model.store().Stats("source").proxyLoads > 0);
    CHECK(!model.scrubLatencies().empty());
    // a reload with a valid proxy starts nothing; closing stops a build in progress
    RemoveProxy(d / "passes", "source");
    model.reloadSources();
    CHECK(model.proxyBuilding());
    model.closeProject();
    CHECK_FALSE(model.proxyBuilding());
    QSettings().setValue("view/autoProxy", false);
    AppModel off(true);
    off.openProject(Q(d / "proj.dlssvid.json"));
    CHECK_FALSE(off.proxyBuilding());  // turned off in «Вид»
    QSettings().remove("view/autoProxy");
}

// Screenshots of the NR card for an MR (docs/ux-guidelines.md: «до/после» at 100 % and 150 %). Hidden, run by hand:
//   set DLSSVID_SHOTS_DIR=<folder> & set QT_SCALE_FACTOR=1.5 & dlssvid_app_tests "[.shots]"
// The card is laid out off screen (WA_DontShowOnScreen) and grabbed; the desktop is not touched.
TEST_CASE("StageCard NR screenshots for a merge request", "[.shots]") {
    App();
    const QString out = qEnvironmentVariable("DLSSVID_SHOTS_DIR");
    if (out.isEmpty()) SKIP("DLSSVID_SHOTS_DIR is not set");
    dlssvid::ApplyTheme(App());
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto d = Dir("card_shots");
    Project p = Project::Create(WriteClip(d / "clip.mkv", spec), d / "passes");
    p.Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    const QString scale = QString::number(qEnvironmentVariable("QT_SCALE_FACTOR", "1").toDouble() * 100);
    auto shot = [&](const char* name, bool more) {
        QSettings().setValue("card/nr/more", more);
        dlssvid::StageCard card(model, "nr");
        card.setAttribute(Qt::WA_DontShowOnScreen);
        card.setFixedWidth(880);  // the project panel is at least 900 px wide (ProjectPanel), minus its margins
        card.show();
        card.refresh();
        card.adjustSize();
        QApplication::processEvents();
        const QString file = out + "/" + name + "_" + scale + ".png";
        REQUIRE(card.grab().save(file));
        std::printf("saved %s\n", file.toStdString().c_str());
    };
    shot("nr_default", false);
    {  // PR-9: the open preset menu
        model.setStageParam("nr", "intensity", 0.8);
        model.saveStagePreset("nr", QString::fromUtf8("Мягкий портрет"), {}, true);
        model.setStageParam("nr", "intensity", 1.0);
        dlssvid::StageCard card(model, "nr");
        card.refresh();
        PresetMenu* menu = card.presetMenu();
        menu->setAttribute(Qt::WA_DontShowOnScreen);
        emit menu->aboutToShow();
        menu->adjustSize();
        const QString file = out + "/nr_preset_menu_" + scale + ".png";
        REQUIRE(menu->grab().save(file));
        std::printf("saved %s\n", file.toStdString().c_str());
    }
    model.setStageParam("nr", "intensity", 1.4);
    model.setStageParam("nr", "local_tone", 1.5);
    model.setStageParam("nr", "exposure", 1.0);
    model.refreshPlan();
    shot("nr_changed", false);
    shot("nr_more", true);
    {  // PR-8: the row «Маска управления» with a mask loaded: the automatic mask off with its note
        QSignalSpy done(&model, &AppModel::controlMaskFinished);
        model.setStageParam("nr", "auto_mask", true);
        model.importControlMask(MaskPng(d / "portrait_mask.png", 48, 32, true));
        REQUIRE(done.wait(20000));
        shot("nr_control_mask", false);
    }
    QSettings().remove("card/nr/more");
}


// PR-8: the row «Маска управления» of the NR card loads the mask through the dialog on the model's thread; with it the automatic mask
// and «Структура кожи» are off with their notes; «Показать» puts it on the viewport; «Убрать» asks, moves it into the history.
TEST_CASE("StageCard NR control mask row: load through the dialog, status, auto mask and skin structure disabled with notes, Show selects the source, Remove asks, retires and clears; a cancelled Remove keeps it", "[app][card][nr][mask]") {
    App();
    MaskCard m("card_mask");
    REQUIRE(m.row());
    CHECK(m.label("Маска управления"));
    CHECK(m.row()->status()->text() == QString::fromUtf8("Не загружена"));
    CHECK(m.row()->loadButton()->isEnabled());
    CHECK(m.row()->loadButton()->text() == QString::fromUtf8("Загрузить…"));
    CHECK_FALSE(m.row()->showButton()->isVisibleTo(m.card.get()));
    CHECK_FALSE(m.label("выключена: её заменяет маска управления"));
    m.dialogs.open.clear();  // a cancelled dialog: nothing happens
    m.row()->loadButton()->click();
    CHECK_FALSE(m.model->controlMaskBusy());

    REQUIRE(m.load(MaskPng(m.dir / "mask.png", 16, 8, true)));
    CHECK(m.model->controlMask().status == AppModel::ControlMaskState::Status::Ready);
    CHECK(m.row()->status()->text() == QString::fromUtf8("Активна · RGB · 16×8\nmask.png"));
    CHECK(m.row()->status()->toolTip().contains("sha256:"));
    CHECK(m.row()->loadButton()->text() == QString::fromUtf8("Заменить…"));
    CHECK(m.row()->showButton()->isVisibleTo(m.card.get()));
    CHECK(m.row()->removeButton()->isVisibleTo(m.card.get()));
    QLabel* autoNote = m.label("выключена: её заменяет маска управления");  // the automatic mask: off, said why
    REQUIRE(autoNote);
    CHECK(autoNote->isVisibleTo(m.card.get()));
    REQUIRE(m.label("не действует при маске управления"));
    CHECK(m.label("не действует при маске управления")->isVisibleTo(m.card.get()));
    CHECK(m.model->sourceNames().contains("mask_control"));

    m.row()->showButton()->click();
    const auto needed = m.model->neededSources();
    CHECK(std::find(needed.begin(), needed.end(), "mask_control") != needed.end());

    m.dialogs.answer = false;  // «Убрать», cancelled: the mask stays
    m.row()->removeButton()->click();
    CHECK(m.dialogs.confirms == 1);
    CHECK(m.dialogs.texts.last().contains(QString::fromUtf8("уйдёт в историю версий")));
    CHECK(ReadControlMaskInfo(m.dir / "passes" / "mask_control"));
    m.dialogs.answer = true;
    m.row()->removeButton()->click();
    CHECK(!std::filesystem::exists(m.dir / "passes" / "mask_control"));
    CHECK(ListPassVersions(m.dir / "passes", "mask_control").size() == 1);  // in the history: «Вернуть» brings it back
    CHECK(m.row()->status()->text() == QString::fromUtf8("Не загружена"));
    CHECK_FALSE(autoNote->isVisibleTo(m.card.get()));
}

// PR-8: nothing can swap the mask under a run or take its folder in the middle of an import - the row, «Обработать», the folder of
// passes and «Вернуть» / «Удалить» / gc of the versions panel wait; a reload meanwhile keeps the mask off the viewport.
TEST_CASE("StageCard NR control mask row: buttons off while a task runs or an import is in progress; Process waits for the import; Use, Remove and gc in the versions panel are off meanwhile and a reload keeps the mask out of the viewport and Importing; the note at intensity 0; an unreadable folder offers Remove, a folder without a loadable manifest is deleted", "[app][card][mask]") {
    App();
    MaskCard m("card_mask_busy");
    bool taskRuns = false;
    m.card->setBusyCheck([&] { return taskRuns; });
    REQUIRE(m.load(MaskPng(m.dir / "a.png", 16, 8, false)));
    taskRuns = true;  // a task of the queue
    m.card->refresh();
    CHECK_FALSE(m.row()->loadButton()->isEnabled());
    CHECK_FALSE(m.row()->removeButton()->isEnabled());
    CHECK_FALSE(m.row()->showButton()->isEnabled());
    CHECK(m.row()->loadButton()->toolTip() == QString::fromUtf8("Недоступно во время обработки"));
    taskRuns = false;
    m.card->refresh();
    REQUIRE(m.row()->loadButton()->isEnabled());

    // an import held at its last step (after the previous mask went to the history)
    TaskQueue queue;
    ProjectPanel panel(*m.model, queue);
    std::atomic<bool> release{false}, reached{false};
    ControlMaskHooksForTests().beforePlace = [&](const std::filesystem::path&) {
        reached = true;
        while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };
    QSignalSpy done(m.model.get(), &AppModel::controlMaskFinished);
    m.dialogs.open = MaskPng(m.dir / "b.png", 16, 8, true);
    m.row()->loadButton()->click();
    Pump([&] { return reached.load(); }, 20000);
    REQUIRE(reached);
    CHECK(m.model->controlMaskBusy());
    m.card->refresh();
    CHECK(m.row()->status()->text() == QString::fromUtf8("Загружается…\nb.png"));
    CHECK_FALSE(m.row()->loadButton()->isEnabled());
    panel.refresh();
    QPushButton* process = nullptr;
    QPushButton* gc = nullptr;
    for (QPushButton* b : panel.findChildren<QPushButton*>()) {
        if (b->text().startsWith(QString::fromUtf8("Обработать"))) process = b;
        if (b->toolTip() == QString::fromUtf8("Недоступно во время обработки или загрузки маски") && b->text() != QString::fromUtf8("Вернуть") &&
            b->text() != QString::fromUtf8("Удалить"))
            gc = b;
    }
    REQUIRE(process);
    CHECK_FALSE(process->isEnabled());
    CHECK(gc);  // the gc button says why it waits (the history holds a.png)
    if (gc) CHECK_FALSE(gc->isEnabled());
    int versionButtons = 0;
    for (QPushButton* b : panel.findChildren<QPushButton*>())
        if (b->text() == QString::fromUtf8("Вернуть") || b->text() == QString::fromUtf8("Удалить")) {
            ++versionButtons;
            CHECK_FALSE(b->isEnabled());
        }
    CHECK(versionButtons >= 2);
    // review of PR-8: Ctrl+Return (processAll) waits for the import as the button does
    for (QShortcut* s : panel.findChildren<QShortcut*>())
        if (s->key() == QKeySequence("Ctrl+Return")) emit s->activated();
    CHECK_FALSE(queue.busy());
    m.model->reloadSources();  // another caller's reload meanwhile
    CHECK_FALSE(m.model->sourceNames().contains("mask_control"));
    for (const QString& s : m.model->sourceNames()) CHECK_FALSE(s.startsWith("mask_control@"));
    CHECK(m.model->controlMask().status == AppModel::ControlMaskState::Status::Importing);
    release = true;
    REQUIRE(done.wait(20000));
    ControlMaskHooksForTests().beforePlace = nullptr;
    CHECK(done.last().at(0).toBool());
    CHECK(m.model->controlMask().status == AppModel::ControlMaskState::Status::Ready);
    CHECK(m.model->sourceNames().contains("mask_control"));

    // intensity 0: loaded, not acting; the automatic mask is free again (no input)
    m.model->setStageParam("nr", "intensity", 0.0);
    m.card->refresh();
    CHECK(m.row()->status()->text() == QString::fromUtf8("Загружена · RGB · 16×8\nb.png"));
    CHECK(m.row()->note()->text() == QString::fromUtf8("не действует при интенсивности 0: модель не вызывается"));
    CHECK_FALSE(m.label("выключена: её заменяет маска управления"));  // the row is enabled again: no note
    m.model->setStageParam("nr", "intensity", 1.0);

    // unreadable (not static): «Убрать» offered, «Показать» not
    const auto cur = m.dir / "passes" / "mask_control";
    Manifest man = Manifest::Load(cur);
    man.stageParams["static"] = false;
    man.Save(cur);
    m.model->reloadSources();
    m.card->refresh();
    CHECK(m.row()->status()->text() == QString::fromUtf8("Не читается\nне статичная маска"));
    CHECK(m.row()->status()->property("role").toString() == "warn");
    CHECK(m.row()->removeButton()->isVisibleTo(m.card.get()));
    CHECK_FALSE(m.row()->showButton()->isVisibleTo(m.card.get()));
    m.dialogs.answer = false;  // the dialog names the mask from its manifest (review of PR-8: «Маска «» …»)
    m.row()->removeButton()->click();
    CHECK(m.dialogs.texts.last().contains(QString::fromUtf8("Маска «b.png» уйдёт в историю версий")));
    // a folder without a loadable manifest: deleted, not kept in the history as _broken
    { std::ofstream(cur / Manifest::kFileName) << "{ broken"; }
    m.model->reloadSources();
    m.card->refresh();
    CHECK(m.row()->status()->text() == QString::fromUtf8("Не читается\nпапка повреждена"));
    m.dialogs.answer = true;
    const int confirms = m.dialogs.confirms;
    const size_t history = ListPassVersions(m.dir / "passes", "mask_control").size();
    m.row()->removeButton()->click();
    CHECK(m.dialogs.confirms == confirms + 1);
    CHECK(m.dialogs.texts.last().contains(QString::fromUtf8("повреждена и будет удалена")));
    CHECK(!std::filesystem::exists(cur));
    CHECK(ListPassVersions(m.dir / "passes", "mask_control").size() == history);
}

// Review of PR-8: an import that ends after another project was opened re-reads the mask of the project open now - not «Загружается…» for
// good, and that project's mask back on the viewport.
TEST_CASE("AppModel: a mask import that ends after another project was opened shows the mask of the open project", "[app][mask][regression]") {
    App();
    MaskCard a("card_mask_switch_a");
    MaskCard b("card_mask_switch_b");
    REQUIRE(b.load(MaskPng(b.dir / "b_mask.png", 16, 8, true)));  // project B has a mask of its own
    std::atomic<bool> release{false}, reached{false};
    ControlMaskHooksForTests().beforePlace = [&](const std::filesystem::path&) {
        reached = true;
        while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };
    QSignalSpy done(a.model.get(), &AppModel::controlMaskFinished);
    a.dialogs.open = MaskPng(a.dir / "a_mask.png", 16, 8, false);
    a.row()->loadButton()->click();
    Pump([&] { return reached.load(); }, 20000);
    REQUIRE(reached);
    a.model->openProject(Q(b.dir / "proj.dlssvid.json"));  // another project while the import runs
    release = true;
    REQUIRE(done.wait(20000));
    ControlMaskHooksForTests().beforePlace = nullptr;
    CHECK(a.model->controlMask().status == AppModel::ControlMaskState::Status::Ready);
    CHECK(a.model->controlMask().file == QString::fromUtf8("b_mask.png"));
    CHECK(a.model->sourceNames().contains("mask_control"));
    CHECK(ReadControlMaskInfo(a.dir / "passes" / "mask_control"));  // the import itself finished into project A
}

// PR-8: one function behind the card and the CLI - one fingerprint for one file
TEST_CASE("The control mask from the card has the fingerprint ImportControlMask gives for the same file", "[app][mask]") {
    App();
    MaskCard m("card_mask_fp");
    const QString file = MaskPng(m.dir / "mask.png", 20, 10, true);
    REQUIRE(m.load(file));
    const std::string fp = Manifest::Load(m.dir / "passes" / "mask_control").fingerprint;
    CHECK(ImportControlMask(std::filesystem::path(file.toStdWString()), m.dir / "other").mask.fingerprint == fp);
    CHECK(m.model->controlMask().fingerprint == QString::fromStdString(ShortFingerprint(fp)));
    // the same file again: nothing written, said in the note
    REQUIRE(m.load(file));
    CHECK(m.row()->note()->text().contains(QString::fromUtf8("эта маска уже загружена")));
    CHECK(Manifest::Load(m.dir / "passes" / "mask_control").fingerprint == fp);
}

TEST_CASE("A failed control mask import shows the Russian text of its kind and keeps the previous mask", "[app][card][mask]") {
    App();
    MaskCard m("card_mask_fail");
    REQUIRE(m.load(MaskPng(m.dir / "mask.png", 16, 8, false)));
    const std::string fp = Manifest::Load(m.dir / "passes" / "mask_control").fingerprint;
    { std::ofstream(m.dir / "text.png") << "not an image"; }
    CHECK_FALSE(m.load(Q(m.dir / "text.png")));
    CHECK(m.dialogs.warnings == 1);
    CHECK(m.dialogs.texts.last() == QString::fromUtf8("Формат не поддерживается. Подходят PNG, TIFF и EXR."));
    CHECK(Manifest::Load(m.dir / "passes" / "mask_control").fingerprint == fp);  // the previous mask in place
    CHECK(m.model->controlMask().status == AppModel::ControlMaskState::Status::Ready);
    CHECK(m.model->sourceNames().contains("mask_control"));  // back on the viewport
    // the other kinds' texts
    using K = ControlMaskIssue::Kind;
    CHECK(ControlMaskIssueText({K::TooLarge, "", 20000, 100}) == QString::fromUtf8("Изображение 20000×100 слишком большое: сторона — не больше 16384 точек, всего — не больше 67 Мп."));
    CHECK(ControlMaskIssueText({K::Busy, "", 0, 0}).contains(QString::fromUtf8("другой импорт")));
    CHECK(ControlMaskIssueText({K::Locked, "", 0, 0}).contains(QString::fromUtf8("Прежняя маска осталась на месте")));
    CHECK(ControlMaskIssueText({K::Channels, "the EXR has no R, G, B, Y or a single channel (u, v)", 0, 0}).contains("(u, v)"));
    CHECK(ControlMaskIssueText({K::NoRoot, "", 0, 0}).startsWith(QString::fromUtf8("Сначала откройте видео")));
}

TEST_CASE("Source names, the static label and the versions panel name the control mask by file, size and kind", "[app][versions][mask]") {
    App();
    MaskCard m("card_mask_names");
    REQUIRE(m.load(MaskPng(m.dir / "first.png", 16, 8, true)));
    REQUIRE(m.load(MaskPng(m.dir / "second.png", 16, 8, false)));  // the first goes to the history
    CHECK(HumanSourceName("mask_control") == QString::fromUtf8("Маска управления"));
    CHECK(HumanSourceName("mask_control@20261027-120000_ab12cd34").startsWith(QString::fromUtf8("Маска управления · ")));
    const ViewportSource* src = m.model->store().FindSource("mask_control");
    REQUIRE(src);
    CHECK(src->isStatic);
    CHECK(SourceTooltip(*src).contains(QString::fromUtf8("статичная маска: один кадр на всё время")));
    TaskQueue queue;
    ProjectPanel panel(*m.model, queue);
    panel.refresh();
    QStringList texts;
    for (QLabel* l : panel.findChildren<QLabel*>()) texts << l->text();
    CHECK(texts.contains(QString::fromUtf8("second.png · 16×8 · серая")));
    CHECK(texts.contains(QString::fromUtf8("first.png · 16×8 · RGB")));
    CHECK(texts.contains(QString::fromUtf8("Маска управления")));
}

// PR-8: the mask is a pass, not a key - the card's reset, its presets and the pipeline presets neither drop nor hold it
TEST_CASE("Card reset, card presets and pipeline presets keep the control mask; a saved preset holds no mask", "[app][presets][mask]") {
    App();
    ClearPresets();
    MaskCard m("card_mask_presets");
    REQUIRE(m.load(MaskPng(m.dir / "mask.png", 16, 8, true)));
    const std::string fp = Manifest::Load(m.dir / "passes" / "mask_control").fingerprint;
    m.model->setStageParam("nr", "intensity", 1.4);
    const auto saved = m.model->saveStagePreset("nr", QString::fromUtf8("С маской"));
    REQUIRE(saved.ok);
    const std::string text = [&] {  // closed at once: ClearPresets removes the folder
        std::ifstream in(std::filesystem::path(saved.file.toStdWString()));
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }();
    CHECK(text.find("mask_control") == std::string::npos);
    CHECK(text.find("mask.png") == std::string::npos);
    REQUIRE(m.model->resetStageCard("nr").ok);
    CHECK(ReadControlMaskInfo(m.dir / "passes" / "mask_control")->fingerprint == fp);
    REQUIRE(m.model->applyStagePreset("nr", saved.file).ok);
    CHECK(ReadControlMaskInfo(m.dir / "passes" / "mask_control")->fingerprint == fp);
    const auto pipeline = m.model->savePipelinePreset(QString::fromUtf8("Всё с маской"));
    REQUIRE(pipeline.ok);
    m.model->setStageParam("nr", "intensity", 0.6);
    REQUIRE(m.model->applyPipelinePreset(pipeline.file).ok);
    CHECK(ReadControlMaskInfo(m.dir / "passes" / "mask_control")->fingerprint == fp);
    CHECK(m.model->controlMask().status == AppModel::ControlMaskState::Status::Ready);
    ClearPresets();
}

// PR-13: the card «Склейки» says what the list is and what a threshold costs (no decoding over the stored scores, «станет N (было M)»),
// the stages that read it say what the cuts do to them; its chips show a cut on the source's time; off, the list on disk still acts.
TEST_CASE("AppModel: the card of the scene list, its chips and the lines of the stages that read the list", "[app][model][scenes][gpu]") {
    App();
    const auto d = Dir("scenes");
    const uint32_t w = 48, h = 32;
    ClipSpec spec;
    spec.frames = 6;
    spec.width = w;
    spec.height = h;
    const auto clip = WriteClipWith(d / "clip.mkv", spec, [&](int64_t i) {  // a hard cut at frame 3
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < 3 ? 40 + c + 4 * i : 220 - c);
        return f;
    });
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
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(model.planError().isEmpty());
    REQUIRE(model.plan().stages.front().name == "scenes");
    // before the first analysis
    CHECK(model.cardState("scenes").state == QString::fromUtf8("будет посчитано"));
    CHECK(model.cardState("scenes").sub.contains(QString::fromUtf8("один проход декодирования исходника")));
    CHECK(model.cardState("flow").scenes == QString::fromUtf8("склейки найдутся перед этой стадией"));
    CHECK(model.cardState("upscale").scenes.isEmpty());  // nis keeps no history
    QWidget host;
    dlssvid::SceneCutsView view(model, &host);
    CHECK(view.isHidden());  // the list is not known yet

    REQUIRE(model.saveProject());
    QStringList args = model.processArgs();
    args << "--warp";
    std::vector<std::string> cmd{DLSSVID_CLI_PATH};
    for (const QString& a : args) cmd.push_back(a.toStdString());
    std::string out;
    REQUIRE(Subprocess::Run(cmd, &out) == 0);
    model.reloadSources();
    CHECK(model.cardState("scenes").state == QString::fromUtf8("переиспользуется"));
    CHECK(model.cardState("scenes").sub.contains(QString::fromUtf8("посчитано")));
    CHECK(model.cardState("scenes").sub.endsWith(QString::fromUtf8("1 склейка")));
    CHECK(model.cardState("flow").scenes == QString::fromUtf8("склейки: 1 — векторы на них нулевые"));
    CHECK(model.cardState("nr").scenes == QString::fromUtf8("склейки: 1 — история модели и подавления мерцания сбрасывается"));
    CHECK(model.cardState("fg").scenes == QString::fromUtf8("склейки: 1 — вместо промежуточных кадров повтор последнего кадра сцены"));
    CHECK(model.cardState("depth").scenes.isEmpty());
    // the chips: the cut on the source's time (the result runs at 48 fps, the source at 24)
    view.refresh();
    REQUIRE_FALSE(view.isHidden());
    const auto chips = view.findChildren<QToolButton*>();
    REQUIRE(chips.size() == 1);
    CHECK(chips[0]->text() == QString::fromStdString(FormatTimecode(3.0 / 24.0)));
    chips[0]->click();
    CHECK(model.time() == Approx(3.0 / 24.0));
    CHECK(model.store().FrameAt("source", model.time()) == 3);  // the cut frame itself, not the next one (review of PR-13)
    CHECK(model.store().FrameAt("result", model.time()) == 6);  // 48 fps: real frame 3 of the result
    model.seekToSourceFrame(2);
    CHECK(model.store().FrameAt("source", model.time()) == 2);
    const auto labels = view.findChildren<QLabel*>();
    CHECK(std::any_of(labels.begin(), labels.end(), [](QLabel* l) { return l->text() == QString::fromUtf8("Найдено 1 склейка"); }));

    // a re-analysis: the stages made with the list are reused if it comes out the same (review of PR-13)
    model.setForce("scenes", true);
    model.refreshPlan();
    CHECK(model.cardState("flow").state == QString::fromUtf8("пересчёт, если склейки изменятся"));
    CHECK(model.cardState("flow").sub.contains(QString::fromUtf8("список склеек посчитается первым")));
    model.clearForce();
    model.refreshPlan();
    // another threshold, the same list: over the stored scores, nothing below
    model.setStageParam("scenes", "threshold", 0.3);
    model.refreshPlan();
    CHECK(model.cardState("scenes").state.startsWith(QString::fromUtf8("пересчёт: Порог склейки")));
    CHECK(model.cardState("scenes").sub.startsWith(QString::fromUtf8("без декодирования · 1 склейка")));
    CHECK(model.cardState("flow").state == QString::fromUtf8("переиспользуется"));
    view.refresh();
    CHECK(std::any_of(labels.begin(), labels.end(), [](QLabel* l) { return l->text() == QString::fromUtf8("Список не изменится — стадии ниже не пересчитываются"); }));
    // no cut at 0.6: «станет 0 (было 1)», the readers say the list changed
    model.setStageParam("scenes", "threshold", 0.6);
    model.refreshPlan();
    CHECK(model.cardState("scenes").sub.contains(QString::fromUtf8("станет 0 склеек (было 1)")));
    CHECK(model.cardState("flow").state == QString::fromUtf8("пересчёт: изменились склейки"));
    CHECK(model.cardState("flow").sub.contains(QString::fromUtf8("1 → 0")));
    CHECK(model.cardState("flow").scenes.isEmpty());  // no cut: nothing to say
    view.refresh();
    const auto after = view.findChildren<QLabel*>();
    CHECK(std::any_of(after.begin(), after.end(), [](QLabel* l) { return l->text().startsWith(QString::fromUtf8("При этих настройках: 0 склеек (сейчас 1) — пересчитаются векторы")); }));
    // threshold 0
    model.setStageParam("scenes", "threshold", 0.0);
    model.refreshPlan();
    CHECK(model.cardState("scenes").state == QString::fromUtf8("выкл: без сброса"));
    // the stage off: the list on disk is read all the same, the card says so
    model.setStageParam("scenes", "threshold", 0.24);
    model.setStageEnabled("scenes", false);
    model.refreshPlan();
    CHECK(model.cardState("scenes").state == QString::fromUtf8("выключена"));
    CHECK(model.cardState("scenes").sub == QString::fromUtf8("действует список с диска: 1 склейка · без сброса — порог 0"));
    CHECK(model.cardState("flow").scenes == QString::fromUtf8("склейки: 1 — векторы на них нулевые"));
    view.refresh();
    CHECK(view.isHidden());
    // a v3 project without the entry: the switch creates it in its place (Project::EnsureStage) and counts as a change
    auto& stages = model.project().stages;
    stages.erase(std::remove_if(stages.begin(), stages.end(), [](const StageEntry& e) { return e.name == "scenes"; }), stages.end());
    const size_t before = stages.size();
    QSignalSpy changed(&model, &AppModel::projectChanged);
    model.setStageEnabled("scenes", true);
    REQUIRE(model.project().stages.size() == before + 1);
    CHECK(model.project().stages.front().name == "scenes");
    CHECK(model.project().stages.front().enabled);
    CHECK(changed.count() == 1);
    CHECK(model.dirty());
    // a parameter of a stage the file lacks: inserted as the card shows it - off
    stages.erase(stages.begin());
    model.setStageParam("scenes", "threshold", 0.3);
    CHECK(model.project().stages.front().name == "scenes");
    CHECK_FALSE(model.project().stages.front().enabled);
    CHECK(model.project().stages.front().params["threshold"] == 0.3);
}

// Review of PR-13, round 2: «Список не изменится» names the stages made with another list, which run again all the same; a stage that
// reads the list on disk (the stage off) says how many cuts it will meet
TEST_CASE("AppModel: the same list names the stages made with another one; the list on disk counts for a stage off", "[app][model][scenes][gpu][regression]") {
    App();
    const auto d = Dir("scenes_again");
    const uint32_t w = 48, h = 32;
    ClipSpec spec;
    spec.frames = 6;
    spec.width = w;
    spec.height = h;
    const auto clip = WriteClipWith(d / "clip.mkv", spec, [&](int64_t i) {  // a hard cut at frame 3
        CpuFrame f = SyntheticFrame(i, w, h);
        uint8_t* y = f.Plane(0);
        for (uint32_t r = 0; r < h; ++r)
            for (uint32_t c = 0; c < w; ++c) y[r * w + c] = static_cast<uint8_t>(i < 3 ? 40 + c + 4 * i : 220 - c);
        return f;
    });
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
    model.openProject(Q(d / "proj.dlssvid.json"));
    auto run = [&] {
        REQUIRE(model.saveProject());
        QStringList args = model.processArgs();
        args << "--warp";
        std::vector<std::string> cmd{DLSSVID_CLI_PATH};
        for (const QString& a : args) cmd.push_back(a.toStdString());
        std::string out;
        REQUIRE(Subprocess::Run(cmd, &out) == 0);
        model.reloadSources();
    };
    run();  // the list {3}: flow, nr, fg made with it
    model.setStageEnabled("nr", false);
    model.setStageParam("scenes", "threshold", 0.6);  // no cut: flow and fg run without a list, nr keeps {3}
    run();
    model.setStageEnabled("nr", true);
    model.setStageParam("scenes", "threshold", 0.7);  // still no cut: the same (empty) list - but nr was made with {3}
    model.refreshPlan();
    QWidget host;
    dlssvid::SceneCutsView view(model, &host);
    view.refresh();
    const auto labels = view.findChildren<QLabel*>();
    CHECK(std::any_of(labels.begin(), labels.end(),
                      [](QLabel* l) { return l->text() == QString::fromUtf8("Список не изменится — пересчитаются улучшение: сделаны с другим списком"); }));
    CHECK(model.cardState("nr").runs);  // its vectors and its list changed: «изменился вход»
    // the stage off: the stages read the list on disk, and the count of the cuts comes from it
    run();  // threshold 0.7: no list anywhere
    model.setStageEnabled("scenes", false);
    model.setStageEnabled("flow", false);  // its vectors stay: nr's only new input is the list
    std::string out;  // a list with a cut on disk, as `dlssvid scenes` (or «Вернуть» of a version) leaves it
    REQUIRE(Subprocess::Run({DLSSVID_CLI_PATH, "scenes", "-i", clip.string(), "-o", (d / "passes").string(), "--threshold", "0.24"}, &out) == 0);
    model.reloadSources();
    model.refreshPlan();
    CHECK(model.cardState("nr").state == QString::fromUtf8("пересчёт: изменились склейки"));
    CHECK(model.cardState("nr").sub.contains(QString::fromUtf8("0 → 1")));  // before: «0 → 0», the stage off had no count
    CHECK(model.cardState("nr").scenes == QString::fromUtf8("склейки: 1 — история модели и подавления мерцания сбрасывается"));
}

// ---- presets (PR-9) -----------------------------------------------------------------------------------------------------------------

TEST_CASE("Presets: save, list, apply keeps format; before-apply saved, applying it swaps back; an equal preset changes nothing", "[app][presets]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_apply");
    StubProjectFile(d / "proj.dlssvid.json", d / "clip.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    // the project's engineer and unknown keys - the preset must not lose them (the project's defaults have no format)
    model.setStageParams("nr", {{"backend", "stub"}, {"format", "png"}, {"extra_key", 1}, {"intensity", 0.7}, {"local_structure", 0.6}});
    const auto saved = model.saveStagePreset("nr", QString::fromUtf8("Мягкий"));
    REQUIRE(saved.ok);
    CHECK(saved.notes.join(" ").contains(QString::fromUtf8("Формат пасса")));  // named: it stays the project's
    const auto listing = PresetStore::List("nr");
    REQUIRE(listing.items.size() == 1);
    CHECK(listing.items[0].name == QString::fromUtf8("Мягкий"));
    CHECK(listing.items[0].file == saved.file);
    CHECK(model.saveStagePreset("nr", QString::fromUtf8("мягкий")).exists);  // the case is ignored: «Заменить?»

    model.setStageParams("nr", {{"backend", "stub"}, {"format", "png"}, {"extra_key", 1}, {"intensity", 1.4}});
    QSignalSpy plans(&model, &AppModel::planChanged);
    const auto applied = model.applyStagePreset("nr", saved.file);
    REQUIRE(applied.ok);
    CHECK(plans.count() == 1);  // the plan at once, one
    const nlohmann::json p = model.stageEntry("nr")->params;
    CHECK(p["format"] == "png");
    CHECK(p["backend"] == "stub");
    CHECK(p["extra_key"] == 1);
    CHECK(p["intensity"] == 0.7);
    CHECK(p["local_structure"] == 0.6);
    CHECK_FALSE(p.contains("passes"));  // sparse: the preset's defaults are not written
    const QString change = QString::fromUtf8("Интенсивность %1 → %2").arg(QLocale().toString(1.4, 'f', 2), QLocale().toString(0.7, 'f', 2));
    INFO(applied.message.toStdString());
    CHECK(applied.message.contains(change));
    CHECK(applied.message.contains(QString::fromUtf8("прежние — в «Перед применением»")));
    // the slot holds what was there; applying it swaps the states
    const auto slot = PresetSlotFile(PresetSlot::BeforeApply, "nr");
    REQUIRE(std::filesystem::exists(slot));
    REQUIRE(model.applyStagePreset("nr", Q(slot)).ok);
    CHECK(model.stageEntry("nr")->params["intensity"] == 1.4);
    CHECK(model.stageEntry("nr")->params["format"] == "png");
    REQUIRE(model.applyStagePreset("nr", Q(slot)).ok);
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.7);
    // an equal preset: nothing - not dirty, no signal, the slot untouched
    REQUIRE(model.saveProjectAs(Q(d / "saved.dlssvid.json")));
    CHECK_FALSE(model.dirty());
    const auto size = std::filesystem::file_size(slot);
    const auto time = std::filesystem::last_write_time(slot);
    QSignalSpy dirty(&model, &AppModel::dirtyChanged), project(&model, &AppModel::projectChanged), plan(&model, &AppModel::planChanged);
    const auto same = model.applyStagePreset("nr", saved.file);
    CHECK(same.ok);
    CHECK(same.message.contains(QString::fromUtf8("уже совпадают с «Мягкий»")));
    CHECK_FALSE(model.dirty());
    CHECK(dirty.count() == 0);
    CHECK(project.count() == 0);
    CHECK(plan.count() == 0);
    CHECK(std::filesystem::file_size(slot) == size);
    CHECK(std::filesystem::last_write_time(slot) == time);
}

TEST_CASE("Presets: a refused preset changes nothing and says why in Russian; a failed backup changes nothing until asked", "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_refused");
    StubProjectFile(d / "proj.dlssvid.json", d / "clip.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    const nlohmann::json before = model.stageEntry("nr")->params;
    const auto bad = model.applyStagePreset("nr", WritePreset(d / "bad.json", CardPresetJson("Плохой", {{"intensity", 5}, {"passes", 9}})));
    CHECK_FALSE(bad.ok);
    CHECK_FALSE(bad.backupFailed);
    INFO(bad.problems.join(" | ").toStdString());
    CHECK(bad.problems.contains(QString::fromUtf8("Интенсивность: 5 — допустимо %1–%2").arg(QLocale().toString(0.0, 'g', 9), QLocale().toString(2.0, 'g', 9))));
    CHECK(bad.problems.size() == 2);  // every value at once
    CHECK(model.stageEntry("nr")->params == before);
    CHECK(model.applyStagePreset("nr", WritePreset(d / "depth.json", CardPresetJson("Глубина", {{"backend", "vda"}}, "depth"))).problems.contains(
        QString::fromUtf8("В пресете нет настроек «Улучшения».")));
    nlohmann::json newer = CardPresetJson("Новый", {{"intensity", 1}});
    newer["schema"] = 2;
    CHECK(model.applyStagePreset("nr", WritePreset(d / "newer.json", newer)).problems.join(" ").contains(QString::fromUtf8("более новой версией")));
    // the slot cannot be written (the folder is inside a file): nothing changes until asked
    const QString dir = PresetStore::Dir();
    std::ofstream(d / "blocker", std::ios::binary) << "x";
    PresetStore::SetDir(Q(d / "blocker" / "presets"));
    const QString good = WritePreset(d / "good.json", CardPresetJson("Хороший", {{"intensity", 0.5}}));
    const auto failed = model.applyStagePreset("nr", good);
    CHECK_FALSE(failed.ok);
    CHECK(failed.backupFailed);
    CHECK(failed.problems.join(" ").contains(QString::fromUtf8("Не удалось сохранить «Перед применением»")));
    CHECK(model.stageEntry("nr")->params == before);
    const auto forced = model.applyStagePreset("nr", good, true);
    CHECK(forced.ok);
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.5);
    CHECK_FALSE(forced.message.contains(QString::fromUtf8("Перед применением")));
    // the reset too
    const auto reset = model.resetStageCard("nr");
    CHECK(reset.backupFailed);
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.5);
    CHECK(model.resetStageCard("nr", true).ok);
    CHECK_FALSE(model.stageEntry("nr")->params.contains("intensity"));
    PresetStore::SetDir(dir);
}

TEST_CASE("Presets: new projects take the marked preset; a project file next to the video wins; a missing mark is cleared and a broken one falls back, with a message",
          "[app][presets]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_new");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageParam("nr", "intensity", 0.6);
    const auto soft = model.saveStagePreset("nr", QString::fromUtf8("Мягкий"), {}, true);
    REQUIRE(soft.ok);
    CHECK(PresetStore::NewProjectPreset() == QFileInfo(soft.file).fileName());
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 32;
    spec.height = 16;
    const auto clip = WriteClip(d / "clip.mkv", spec);
    QSignalSpy messages(&model, &AppModel::message);
    model.openVideo(Q(clip));
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.6);
    CHECK(model.dirty());
    CHECK(LastMessage(messages).contains(QString::fromUtf8("«Улучшение» — из пресета «Мягкий» (для новых проектов)")));
    CHECK_FALSE(std::filesystem::exists(PresetSlotFile(PresetSlot::BeforeApply, "nr")));  // seeding writes no slot
    // a project file next to the video wins
    REQUIRE(model.saveProject());
    model.setStageParam("nr", "intensity", 1.3);
    REQUIRE(model.saveProject());
    model.openVideo(Q(clip));
    CHECK(model.stageEntry("nr")->params["intensity"] == 1.3);
    // a mark on a file that is gone: cleared, the built-in defaults
    PresetStore::SetNewProjectPreset("nothing.json");
    const auto other = WriteClip(d / "other.mkv", spec);
    model.openVideo(Q(other));
    CHECK(LastMessage(messages).contains(QString::fromUtf8("не найден — отметка снята")));
    CHECK(PresetStore::NewProjectPreset().isEmpty());
    CHECK(model.stageEntry("nr")->params == Project::DefaultStages()[4].params);
    // a broken one: the built-in defaults, the mark stays
    nlohmann::json newer = CardPresetJson("Новый", {{"intensity", 1}});
    newer["schema"] = 2;
    WritePreset(PresetStore::Path(PresetStore::Dir()) / "newer.json", newer);
    PresetStore::SetNewProjectPreset("newer.json");
    const auto third = WriteClip(d / "third.mkv", spec);
    model.openVideo(Q(third));
    CHECK(LastMessage(messages).contains(QString::fromUtf8("не применён")));
    CHECK(PresetStore::NewProjectPreset() == "newer.json");
    // a value out of range: the problem names its stage
    WritePreset(PresetStore::Path(PresetStore::Dir()) / "strong.json", CardPresetJson("Сильный", {{"intensity", 5}}));
    PresetStore::SetNewProjectPreset("strong.json");
    model.openVideo(Q(WriteClip(d / "fourth.mkv", spec)));
    CHECK(LastMessage(messages).contains(QString::fromUtf8("не применён: Улучшение: Интенсивность: 5 — допустимо")));
    CHECK(model.stageEntry("nr")->params == Project::DefaultStages()[4].params);
    QSettings().remove("presets/newProject");
}

TEST_CASE("Presets for new projects: one mark - a pipeline preset seeds every stage, enabled and the encoder, an NR preset only NR; marking in one menu moves the mark",
          "[app][presets]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_new_pipeline");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageEnabled("fg", false);
    model.setStageParam("scenes", "threshold", 0.3);
    model.setStageParam("nr", "intensity", 0.6);
    model.setEncode("hevc_nvenc", {{"b", "50M"}}, {10, "auto"});
    const auto anime = model.savePipelinePreset(QString::fromUtf8("Аниме"), {}, true);
    REQUIRE(anime.ok);
    CHECK(PresetStore::NewProjectPreset() == QFileInfo(anime.file).fileName());
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 32;
    spec.height = 16;
    QSignalSpy messages(&model, &AppModel::message);
    model.openVideo(Q(WriteClip(d / "clip.mkv", spec)));
    CHECK(LastMessage(messages).contains(QString::fromUtf8("настройки обработки — из «Аниме» (для новых проектов)")));
    {  // seeded before reloadSources: the first plan is made with those settings
        const auto& stages = model.plan().stages;
        CHECK(std::none_of(stages.begin(), stages.end(), [](const StageDecision& s) { return s.name == "fg"; }));  // off in the preset
        const auto nr = std::find_if(stages.begin(), stages.end(), [](const StageDecision& s) { return s.name == "nr"; });
        REQUIRE(nr != stages.end());
        CHECK(nr->params["intensity"] == 0.6);
    }
    CHECK_FALSE(model.stageEntry("fg")->enabled);
    CHECK(model.stageEntry("scenes")->params["threshold"] == 0.3);
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.6);
    CHECK(model.stageEntry("depth")->params["backend"] == "stub");
    CHECK(model.project().codec == "hevc_nvenc");
    CHECK(model.project().outputFormat.bitDepth == 10);
    CHECK(model.project().codecOptions.at("b") == "50M");
    CHECK(model.project().resultVideo.empty());  // a new project's result: next to the video by the plan
    CHECK(model.dirty());
    // an NR preset marked in its menu: the one mark moves to it, and a new project takes NR only
    model.setStageParam("nr", "intensity", 0.4);
    const auto soft = model.saveStagePreset("nr", QString::fromUtf8("Мягкий"), {}, true);
    REQUIRE(soft.ok);
    CHECK(PresetStore::NewProjectPreset() == QFileInfo(soft.file).fileName());
    model.openVideo(Q(WriteClip(d / "other.mkv", spec)));
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.4);
    CHECK(model.stageEntry("fg")->enabled);  // NR only
    CHECK(model.project().codec == "h264_nvenc");
    QSettings().remove("presets/newProject");
}

TEST_CASE("PresetStore: the folder is the test's; import refuses a preset without nr and renames on a clash; delete clears the new-projects mark", "[app][presets]") {
    App();
    ClearPresets();
    CHECK(PresetStore::Path(PresetStore::Dir()).u8string().find(u8"test-tmp") != std::u8string::npos);  // never the user's AppData
    const auto d = Dir("presets_store");
    const QString depth = WritePreset(d / "depth.json", CardPresetJson("Глубина", {{"backend", "vda"}}, "depth"));
    CHECK(PresetStore::Import("nr", depth).problems.contains(QString::fromUtf8("В пресете нет настроек «Улучшения».")));
    const QString soft = WritePreset(d / "soft.json", CardPresetJson("Мягкий", {{"intensity", 0.7}, {"format", "png"}}));
    const auto first = PresetStore::Import("nr", soft);
    REQUIRE(first.ok);
    CHECK(first.notes.join(" ").contains(QString::fromUtf8("Формат пасса")));  // the engineer key dropped, said
    CHECK(PresetStore::List("nr").items.size() == 1);
    const auto clash = PresetStore::Import("nr", soft);
    CHECK(clash.exists);
    CHECK(clash.file == first.file);
    const auto copy = PresetStore::Import("nr", soft, {}, true);
    REQUIRE(copy.ok);
    CHECK(copy.name == QString::fromUtf8("Мягкий (2)"));
    CHECK(PresetStore::Import("nr", soft, first.file).ok);  // replace
    CHECK(PresetStore::List("nr").items.size() == 2);
    // review of PR-9: the clash names the copy it will save, «(2)» being taken
    const auto clash3 = PresetStore::Import("nr", soft);
    REQUIRE(clash3.exists);
    CHECK(clash3.copyName == QString::fromUtf8("Мягкий (3)"));
    // a name of 120 code points (a long video's «Улучшение · …») gets its suffix: the name is cut before it, not with it
    const std::string longName = PresetNameFor("Улучшение", PresetStore::Path(Q(d / (std::string(200, 'x') + ".mp4"))));
    const QString longFile = WritePreset(d / "long.json", CardPresetJson(longName.c_str(), {{"intensity", 0.9}}));
    REQUIRE(PresetStore::Import("nr", longFile).ok);
    const auto copy2 = PresetStore::Import("nr", longFile, {}, true);
    const auto copy3 = PresetStore::Import("nr", longFile, {}, true);
    REQUIRE(copy2.ok);
    REQUIRE(copy3.ok);
    CHECK(copy2.name.endsWith(" (2)"));
    CHECK(copy3.name.endsWith(" (3)"));
    CHECK(copy2.name.size() <= 120);
    // a card preset's switch in its entry: the card's text, not the pipeline's
    nlohmann::json withSwitch = CardPresetJson("Со включением", {{"intensity", 0.6}});
    withSwitch["stages"]["nr"]["enabled"] = false;
    const auto switched = PresetStore::Import("nr", WritePreset(d / "switch.json", withSwitch));
    REQUIRE(switched.ok);
    CHECK(switched.notes.contains(QString::fromUtf8("включение и кодирование в пресете карточки не хранятся — пропущено")));
    // a card preset with two stages: the real reason, not «Это пресет карточки «»»
    nlohmann::json two = CardPresetJson("Две", {{"intensity", 0.6}});
    two["stages"]["depth"] = {{"params", {{"backend", "vda"}}}};
    const auto twoStages = PresetStore::Import(kPipelineTarget, WritePreset(d / "two.json", two));
    CHECK(twoStages.problems.contains(QString::fromUtf8("В пресете карточки должна быть ровно одна стадия.")));
    CHECK(PresetStore::List("nr").items.size() == 6);
    // a card preset is no pipeline preset; a pipeline preset gives its nr to the card
    CHECK(PresetStore::Import(kPipelineTarget, soft).problems.join(" ").contains(QString::fromUtf8("Это пресет карточки «Улучшение»")));
    nlohmann::json pipe = {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "pipeline"}, {"name", "Аниме"}, {"stages", nlohmann::json::object()}};
    pipe["stages"]["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.6}, {"backend", "stub"}, {"foo", 1}, {"scene_cut", 0.3}}}};
    pipe["stages"]["depth"] = {{"enabled", true}, {"params", {{"backend", "vda"}}}};
    const QString pipeFile = WritePreset(d / "pipe.json", pipe);
    const auto fromPipe = PresetStore::Import("nr", pipeFile);
    REQUIRE(fromPipe.ok);
    CHECK(fromPipe.message == QString::fromUtf8("Из пресета обработки «Аниме» взяты настройки «Улучшения»."));
    INFO(fromPipe.notes.join(" | ").toStdString());
    CHECK(fromPipe.notes.contains(QString::fromUtf8("нет в этой версии: «foo»")));  // the entry's own keys keep their notes
    CHECK(fromPipe.notes.contains(QString::fromUtf8("Сброс на склейках теперь в карточке «Склейки»")));
    const PresetLoad card = LoadPreset(PresetStore::Path(fromPipe.file));
    REQUIRE(card.ok());
    CHECK(card.preset.scope == PresetScope::Stage);
    CHECK(card.preset.stages.at("nr").params == nlohmann::json{{"intensity", 0.6}});  // the backend is the card's engineer key
    const auto whole = PresetStore::Import(kPipelineTarget, pipeFile);
    REQUIRE(whole.ok);
    CHECK_FALSE(whole.notes.join(" ").contains(QString::fromUtf8("оставлен")));  // the stages it lacks are no skip of the import
    CHECK(PresetStore::List(kPipelineTarget).items.size() == 1);
    CHECK(PresetStore::List("nr").items.size() == 7);  // the pipeline preset is not in the card's menu
    // a slot's name changes in front
    const auto slotNamed = PresetStore::Import("nr", WritePreset(d / "slot.json", CardPresetJson("Перед применением · clip", {{"intensity", 0.9}})));
    REQUIRE(slotNamed.ok);
    CHECK(slotNamed.name == QString::fromUtf8("Импорт — Перед применением · clip"));
    // export: checked and written in the canonical form
    const auto exported = PresetStore::Export(first.file, Q(d / "out" / "soft.json"));
    REQUIRE(exported.ok);
    CHECK(LoadPreset(d / "out" / "soft.json").ok());
    // delete: permanent; the mark on it goes
    PresetStore::SetNewProjectPreset(first.file);
    const auto removed = PresetStore::Remove(first.file);
    REQUIRE(removed.ok);
    CHECK(removed.message.contains(QString::fromUtf8("встроенными умолчаниями")));
    CHECK(PresetStore::NewProjectPreset().isEmpty());
    CHECK_FALSE(std::filesystem::exists(PresetStore::Path(first.file)));
    // an unreadable file is listed with its reason in Russian, in both menus
    WritePreset(PresetStore::Path(PresetStore::Dir()) / "broken.json", nlohmann::json::array());
    const auto listing = PresetStore::List("nr");
    REQUIRE(listing.refused.size() == 1);
    CHECK(listing.refused[0].error == QString::fromUtf8("Файл повреждён: это не JSON-объект."));
    CHECK(PresetStore::List(kPipelineTarget).refused.size() == 1);
}

TEST_CASE("Pipeline presets: save, apply every card and the encoder at once with one plan; before-apply saved and applying it swaps back; an equal preset changes nothing",
          "[app][presets]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_pipeline");
    StubProjectFile(d / "proj.dlssvid.json", d / "clip.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageEnabled("fg", false);
    model.setStageParam("scenes", "threshold", 0.3);
    model.setStageParam("nr", "intensity", 0.6);
    model.setEncode("prores_ks", {}, {10, "422"});
    const auto anime = model.savePipelinePreset(QString::fromUtf8("Аниме"));
    REQUIRE(anime.ok);
    REQUIRE(PresetStore::List(kPipelineTarget).items.size() == 1);
    CHECK(PresetStore::List("nr").items.empty());
    // back to other settings
    model.setStageEnabled("fg", true);
    model.setStageParam("scenes", "threshold", 0.24);
    model.setStageParam("nr", "intensity", 1.4);
    model.setEncode("h264_nvenc", {}, {});
    CHECK(model.project().resultVideo.filename() == "clip_result.mp4");
    QSignalSpy project(&model, &AppModel::projectChanged), plans(&model, &AppModel::planChanged), messages(&model, &AppModel::message);
    const auto applied = model.applyPipelinePreset(anime.file);
    REQUIRE(applied.ok);
    CHECK(project.count() == 1);
    CHECK(plans.count() == 1);
    QTest::qWait(300);
    CHECK(plans.count() == 1);  // no timer left behind
    CHECK(messages.count() == 1);  // the summary only: no «Результат: …» of its own
    INFO(applied.message.toStdString());
    CHECK(applied.message.startsWith(QString::fromUtf8("Пресет обработки «Аниме» применён: ")));
    CHECK(applied.message.contains(QString::fromUtf8(" · результат: clip_result.mov")));
    CHECK(applied.message.contains(QString::fromUtf8("Генерация кадров выключена")));
    CHECK(applied.message.contains(QString::fromUtf8("прежние — в «Перед применением»")));
    CHECK_FALSE(model.stageEntry("fg")->enabled);
    CHECK(model.stageEntry("scenes")->params["threshold"] == 0.3);
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.6);
    CHECK(model.project().codec == "prores_ks");
    CHECK(model.project().outputFormat == OutputFormatRequest{10, "422"});
    CHECK(model.project().resultVideo.filename() == "clip_result.mov");
    // the slot, then back and forth
    const auto slot = PresetSlotFile(PresetSlot::BeforeApply, kPipelineTarget);
    REQUIRE(std::filesystem::exists(slot));
    REQUIRE(model.applyPipelinePreset(Q(slot)).ok);
    CHECK(model.stageEntry("fg")->enabled);
    CHECK(model.project().codec == "h264_nvenc");
    CHECK(model.stageEntry("nr")->params["intensity"] == 1.4);
    REQUIRE(model.applyPipelinePreset(Q(slot)).ok);
    CHECK(model.project().codec == "prores_ks");
    // equal: nothing
    REQUIRE(model.saveProjectAs(Q(d / "saved.dlssvid.json")));
    const auto size = std::filesystem::file_size(slot);
    const auto time = std::filesystem::last_write_time(slot);
    QSignalSpy dirty(&model, &AppModel::dirtyChanged), project2(&model, &AppModel::projectChanged), plans2(&model, &AppModel::planChanged);
    const auto same = model.applyPipelinePreset(anime.file);
    CHECK(same.ok);
    CHECK(same.message.contains(QString::fromUtf8("Настройки обработки уже совпадают с «Аниме»")));
    CHECK_FALSE(model.dirty());
    CHECK(dirty.count() == 0);
    CHECK(project2.count() == 0);
    CHECK(plans2.count() == 0);
    CHECK(std::filesystem::file_size(slot) == size);
    CHECK(std::filesystem::last_write_time(slot) == time);
    // the preview of the menu: what a preset would change, nothing written
    const PresetLoad back = LoadPreset(slot, {}, PresetRead::Slot);
    const PipelineDiff preview = model.previewPipelinePreset(back.preset);
    CHECK_FALSE(preview.empty());
    CHECK(model.project().codec == "prores_ks");
}

TEST_CASE("Pipeline presets: a refused preset or a card preset changes nothing and says why in Russian; a failed backup changes nothing until asked; a project with an error is not saved",
          "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_pipeline_refused");
    StubProjectFile(d / "proj.dlssvid.json", d / "clip.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    const Project before = model.project();
    const auto good = model.savePipelinePreset(QString::fromUtf8("Хороший"));
    REQUIRE(good.ok);
    nlohmann::json h264 = nlohmann::json::parse(std::ifstream(PresetStore::Path(good.file), std::ios::binary));
    h264["encode"]["bit_depth"] = 10;
    const auto refused = model.applyPipelinePreset(WritePreset(d / "h264.json", h264));
    CHECK_FALSE(refused.ok);
    INFO(refused.problems.join(" | ").toStdString());
    CHECK(refused.problems.contains(QString::fromUtf8("Разрядность: 10 бит — H.264 NVENC кодирует только 8 бит")));
    CHECK(model.project().ToJson({}) == before.ToJson({}));
    CHECK(model.applyPipelinePreset(WritePreset(d / "card.json", CardPresetJson("Карточка", {{"intensity", 0.5}}))).problems.join(" ").contains(
        QString::fromUtf8("Это пресет карточки «Улучшение»")));
    nlohmann::json two = CardPresetJson("Две", {{"intensity", 0.5}});
    two["stages"]["depth"] = {{"params", {{"backend", "vda"}}}};
    const auto twoStages = model.applyPipelinePreset(WritePreset(d / "two.json", two));
    CHECK(twoStages.problems.contains(QString::fromUtf8("В пресете карточки должна быть ровно одна стадия.")));
    // a failed backup: nothing until asked
    model.setStageParam("nr", "intensity", 1.4);
    const QString dir = PresetStore::Dir();
    std::ofstream(d / "blocker", std::ios::binary) << "x";
    PresetStore::SetDir(Q(d / "blocker" / "presets"));
    const auto failed = model.applyPipelinePreset(good.file);
    CHECK(failed.backupFailed);
    CHECK(model.stageEntry("nr")->params["intensity"] == 1.4);
    CHECK(model.applyPipelinePreset(good.file, true).ok);
    CHECK_FALSE(model.stageEntry("nr")->params.contains("intensity"));  // the preset's 1.0 is the default: sparse
    PresetStore::SetDir(dir);
    // the project's errors: a switched-off stage with a value out of range, an encoder the card cannot hold - not saved as a preset
    model.setStageEnabled("nr", false);
    model.setStageParams("nr", {{"backend", "stub"}, {"intensity", 5}});
    const auto badSave = model.savePipelinePreset(QString::fromUtf8("С ошибкой"));
    CHECK_FALSE(badSave.ok);
    CHECK(badSave.problems.join(" | ").contains(QString::fromUtf8("Улучшение: Интенсивность: 5 — допустимо")));
    model.setEncode("libx264", {});
    model.setEncode("h264_nvenc", {}, {10, "auto"});
    // ... but they do not keep the slot from being written: applying a preset goes without a question
    model.setEncode("libx264", {});
    const auto overErrors = model.applyPipelinePreset(good.file);
    CHECK(overErrors.ok);
    CHECK_FALSE(overErrors.backupFailed);
    CHECK(model.project().codec == "h264_nvenc");
    const auto back = model.applyPipelinePreset(Q(PresetSlotFile(PresetSlot::BeforeApply, kPipelineTarget)));
    REQUIRE(back.ok);
    CHECK(model.project().codec == "libx264");
    CHECK(model.stageEntry("nr")->params["intensity"] == 5);
    CHECK(back.message.contains(QString::fromUtf8("в слоте были и ошибки")));
    // a bitrate the card cannot hold
    model.setEncode("hevc_nvenc", {{"b", "12.5M"}}, {});
    model.setStageParams("nr", {{"backend", "stub"}});
    const auto bitrate = model.savePipelinePreset(QString::fromUtf8("Битрейт"));
    CHECK_FALSE(bitrate.ok);
    CHECK(bitrate.problems.join(" ").contains(QString::fromUtf8("Битрейт: %1 Мбит/с — карточка принимает целые 1–400 Мбит/с").arg(QLocale().toString(12.5, 'g', 6))));
    model.setEncode("hevc_nvenc", {{"b", "50M"}}, {});
    const auto ok = model.savePipelinePreset(QString::fromUtf8("Битрейт"));
    CHECK(ok.ok);
}

TEST_CASE("StageCard: the preset menu sits in the NR card header only, after the stage switch in the Tab order; items show the diff, the match and the new-projects mark; a click applies",
          "[app][card][presets]") {
    App();
    ClearPresets();
    const auto d = Dir("card_presets");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageParam("nr", "intensity", 0.7);
    REQUIRE(model.saveStagePreset("nr", "A & B", {}, true).ok);
    model.setStageParam("nr", "intensity", 1.0);
    dlssvid::StageCard nr(model, "nr");
    nr.refresh();
    dlssvid::StageCard depth(model, "depth");
    depth.refresh();
    CHECK_FALSE(depth.presetMenu());
    CHECK(depth.findChildren<PresetButton*>().isEmpty());
    REQUIRE(nr.presetMenu());
    PresetButton* button = nr.findChild<PresetButton*>();
    REQUIRE(button);
    CHECK(button->accessibleName() == QString::fromUtf8("Пресеты «Улучшения»"));
    CHECK(button->isEnabled());
    CHECK(button->focusPolicy() & Qt::TabFocus);
    QCheckBox* stageSwitch = nullptr;
    for (QCheckBox* c : nr.findChildren<QCheckBox*>())
        if (c->toolTip() == QString::fromUtf8("Стадия участвует в обработке")) stageSwitch = c;
    REQUIRE(stageSwitch);
    CHECK(NextTab(stageSwitch, &nr) == button);  // the switch, then «Пресет», then «Сбросить», then the form
    model.setStageParam("nr", "intensity", 1.2);
    nr.refresh();
    QPushButton* reset = nullptr;
    for (QPushButton* b : nr.findChildren<QPushButton*>())
        if (b->text() == QString::fromUtf8("Сбросить")) reset = b;
    REQUIRE(reset);
    CHECK(NextTab(button, &nr) == reset);
    CHECK(reset->accessibleName() == QString::fromUtf8("Сбросить «Улучшение»"));
    // the menu: built on aboutToShow, the name escaped, the mark in words, the difference in the tooltip
    PresetMenu* menu = nr.presetMenu();
    emit menu->aboutToShow();
    QAction* item = ActionStarting(menu, QString::fromUtf8("A && B · для новых проектов"));
    REQUIRE(item);
    CHECK(item->isCheckable());
    CHECK_FALSE(item->isChecked());
    const QString change = QString::fromUtf8("Интенсивность %1 → %2").arg(QLocale().toString(1.2, 'f', 2), QLocale().toString(0.7, 'f', 2));
    INFO(item->toolTip().toStdString());
    CHECK(item->toolTip().startsWith(QString::fromUtf8("Изменит: ") + change));
    CHECK(item->toolTip().endsWith(QString::fromUtf8("Файл: A & B.json")));  // two presets of one name tell apart
    CHECK(item->statusTip().startsWith(QString::fromUtf8("Изменит: ")));
    REQUIRE(ActionStarting(menu, QString::fromUtf8("&Сохранить текущие…")));  // the mnemonics of the menu are its own
    CHECK(ActionStarting(menu, QString::fromUtf8("&Импорт…")));
    item->trigger();
    CHECK(model.stageEntry("nr")->params["intensity"] == 0.7);
    emit menu->aboutToShow();
    item = ActionStarting(menu, QString::fromUtf8("A && B"));
    REQUIRE(item);
    CHECK(item->isChecked());  // matches now
    CHECK(item->toolTip().startsWith(QString::fromUtf8("Совпадает с текущими настройками")));
    // the slot of that application, after a separator
    CHECK(ActionStarting(menu, QString::fromUtf8("Перед применением · missing")));
    // «Для новых проектов ▸»: the built-in defaults and the presets, one checked
    QMenu* mark = nullptr;
    for (QAction* a : menu->actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) mark = a->menu();
    REQUIRE(mark);
    REQUIRE(mark->actions().size() == 2);
    CHECK_FALSE(mark->actions()[0]->isChecked());
    CHECK(mark->actions()[1]->isChecked());
    mark->actions()[0]->trigger();
    CHECK(PresetStore::NewProjectPreset().isEmpty());
    // review of PR-9: every opening rebuilds the menu - the old submenus go (clear() kept them)
    emit menu->aboutToShow();
    emit menu->aboutToShow();
    CHECK(menu->findChildren<QMenu*>(QString(), Qt::FindDirectChildrenOnly).size() == 3);
    // a mark on a file that is gone: cleared, «Встроенные умолчания» checked
    // a mark on a file that is gone: said in the submenu, kept for the next new project to clear with its message
    PresetStore::SetNewProjectPreset("gone.json");
    emit menu->aboutToShow();
    for (QAction* a : menu->actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) mark = a->menu();
    CHECK_FALSE(mark->actions()[0]->isChecked());
    CHECK(mark->actions().last()->text().startsWith(QString::fromUtf8("gone.json — не найден")));
    CHECK(mark->actions().last()->isChecked());
    CHECK(PresetStore::NewProjectPreset() == "gone.json");
    // a refused preset of the other kind is no mark of it: the pipeline menu says the file cannot be read (review of PR-9)
    nlohmann::json bad = CardPresetJson("Плохой", {{"intensity", 5}});
    WritePreset(PresetStore::Path(PresetStore::Dir()) / "bad.json", bad);
    PresetStore::SetNewProjectPreset("bad.json");
    PresetMenu pipelineMenu(model, kPipelineTarget);
    emit pipelineMenu.aboutToShow();
    QMenu* pipelineMark = nullptr;
    for (QAction* a : pipelineMenu.actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) pipelineMark = a->menu();
    REQUIRE(pipelineMark);
    CHECK(pipelineMark->actions().last()->text().startsWith(QString::fromUtf8("bad.json — не читается")));
    StubDialogs saveStub;
    pipelineMenu.setDialogs(saveStub.Make());
    pipelineMenu.saveCurrent();  // no answer: nothing saved
    REQUIRE(saveStub.lastSetup);
    CHECK(saveStub.lastSetup->markNote.isEmpty());
    // a mark on a file no menu reads: a line says so, nothing else checked
    nlohmann::json newer = CardPresetJson("Новый", {{"intensity", 1}});
    newer["schema"] = 2;
    WritePreset(PresetStore::Path(PresetStore::Dir()) / "newer.json", newer);
    PresetStore::SetNewProjectPreset("newer.json");
    emit menu->aboutToShow();
    for (QAction* a : menu->actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) mark = a->menu();
    CHECK_FALSE(mark->actions()[0]->isChecked());
    CHECK(mark->actions().last()->text().startsWith(QString::fromUtf8("newer.json — не читается")));
    CHECK(mark->actions().last()->isChecked());
    QSettings().remove("presets/newProject");
    // a stage switched off still takes a preset
    model.setStageEnabled("nr", false);
    nr.refresh();
    CHECK(button->isEnabled());
}

TEST_CASE("StageCard: card reset saves the before-reset preset first and asks when it cannot; cards without presets reset as before", "[app][card][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("card_reset_presets");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    dlssvid::StageCard nr(model, "nr");
    StubDialogs dialogs;
    nr.setDialogs(dialogs.Make());
    auto resetButton = [](QWidget& card) {
        QPushButton* reset = nullptr;
        for (QPushButton* b : card.findChildren<QPushButton*>())
            if (b->text() == QString::fromUtf8("Сбросить")) reset = b;
        return reset;
    };
    // the slot cannot be written: asked once per click; «Отмена» keeps the settings, «Сбросить» resets
    const QString dir = PresetStore::Dir();
    std::ofstream(d / "blocker", std::ios::binary) << "x";
    PresetStore::SetDir(Q(d / "blocker" / "presets"));
    model.setStageParam("nr", "intensity", 1.4);
    nr.refresh();
    REQUIRE(resetButton(nr));
    dialogs.answer = false;
    resetButton(nr)->click();
    CHECK(dialogs.confirms == 1);
    CHECK(dialogs.texts.last().contains(QString::fromUtf8("Сбросить без автосохранения?")));
    CHECK(model.stageEntry("nr")->params["intensity"] == 1.4);
    dialogs.answer = true;
    resetButton(nr)->click();
    CHECK(dialogs.confirms == 2);
    CHECK_FALSE(model.stageEntry("nr")->params.contains("intensity"));
    PresetStore::SetDir(dir);
    // an error in the current settings (from the JSON) does not keep the slot from being written: it is a snapshot
    model.setStageParams("nr", {{"backend", "ngx"}, {"intensity", 5}});
    nr.refresh();
    resetButton(nr)->click();
    CHECK(dialogs.confirms == 2);
    const auto slot = PresetStore::Library().Slot(PresetSlot::BeforeReset, "nr");
    REQUIRE(slot);
    CHECK(slot->load.ok());
    CHECK(slot->load.preset.stages.at("nr").params["intensity"] == 5);
    CHECK_FALSE(model.stageEntry("nr")->params.contains("intensity"));
    // a card without presets: no slot, no question
    dlssvid::StageCard depth(model, "depth");
    depth.setDialogs(dialogs.Make());
    model.setStageParams("depth", {{"backend", "da3"}, {"stabilize", "none"}});
    depth.refresh();
    REQUIRE(resetButton(depth));
    resetButton(depth)->click();
    CHECK(model.stageEntry("depth")->params == nlohmann::json{{"backend", "da3"}});
    CHECK_FALSE(std::filesystem::exists(PresetSlotFile(PresetSlot::BeforeReset, "depth")));
    CHECK(dialogs.confirms == 2);
}

// Space is the window's playback (MainWindow: a QShortcut on the window) and QAbstractButton does not claim ShortcutOverride: without
// PresetButton the key would start playback instead of opening the menu. The control check proves the shortcut is live in this window.
TEST_CASE("StageCard: Space on the NR preset button opens the menu, not playback", "[app][card][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("card_presets_space");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    QWidget window;
    auto* v = new QVBoxLayout(&window);
    auto* card = new dlssvid::StageCard(model, "nr", &window);
    auto* plain = new QPushButton("plain", &window);
    v->addWidget(card);
    v->addWidget(plain);
    auto* space = new QShortcut(QKeySequence(Qt::Key_Space), &window);
    QObject::connect(space, &QShortcut::activated, &model, [&] { model.setPlaying(!model.playing()); });
    int processRequests = 0;
    auto* process = new QShortcut(QKeySequence("Ctrl+Return"), &window);
    QObject::connect(process, &QShortcut::activated, &window, [&] { ++processRequests; });
    card->refresh();
    window.show();
    window.activateWindow();
    REQUIRE(QTest::qWaitForWindowActive(&window));
    // the control: on a plain button Space reaches the window's shortcut
    plain->setFocus();
    QTest::keyClick(plain, Qt::Key_Space);
    REQUIRE(model.playing());
    model.setPlaying(false);
    PresetButton* button = card->findChild<PresetButton*>();
    REQUIRE(button);
    QMenu* menu = button->menu();
    REQUIRE(menu);
    int shown = 0;
    QObject::connect(menu, &QMenu::aboutToShow, menu, [&] {
        ++shown;
        QTimer::singleShot(0, menu, &QMenu::close);
    });
    button->setFocus();
    QTest::keyClick(button, Qt::Key_Return, Qt::ControlModifier);  // «Обработать» stays the window's (before a menu takes the activation)
    CHECK(processRequests == 1);
    CHECK(shown == 0);
    QTest::keyClick(button, Qt::Key_Space);
    CHECK(shown == 1);
    CHECK_FALSE(model.playing());
    QTest::keyClick(button, Qt::Key_Return);
    CHECK(shown == 2);
    QTest::keyClick(button, Qt::Key_Down, Qt::AltModifier);
    CHECK(shown == 3);
    QTest::keyClick(button, Qt::Key_Down);
    CHECK(shown == 3);
    model.setPlaying(false);
}

TEST_CASE("PresetSaveDialog: a bad name disables Save, an existing name turns it into Replace; it names the mark it replaces and what the preset leaves out",
          "[app][presets]") {
    App();
    ClearPresets();
    PresetSaveDialog::Setup setup;
    setup.title = QString::fromUtf8("Сохранить настройки «Улучшения»");
    setup.markNote = QString::fromUtf8("Сейчас для новых проектов: все настройки обработки из «Y» — будет заменено");
    setup.leftOut = QString::fromUtf8("Не войдут в пресет: Формат пасса PNG 16 бит (инженерная настройка) — их переносит только файл проекта");
    setup.problem = [](const QString& n) { return n.isEmpty() ? QString::fromUtf8("пусто") : QString(); };
    setup.exists = [](const QString& n) { return n == QString::fromUtf8("Мягкий"); };
    PresetSaveDialog dialog(setup);
    CHECK_FALSE(dialog.saveButton()->isEnabled());
    CHECK(dialog.nameNote()->text() == QString::fromUtf8("пусто"));
    dialog.nameEdit()->setText(QString::fromUtf8("Новый"));
    CHECK(dialog.saveButton()->isEnabled());
    CHECK(dialog.saveButton()->text() == QString::fromUtf8("Сохранить"));
    dialog.nameEdit()->setText(QString::fromUtf8("Мягкий"));
    CHECK(dialog.saveButton()->text() == QString::fromUtf8("Заменить"));
    CHECK(dialog.nameNote()->text() == QString::fromUtf8("Пресет «Мягкий» уже есть — будет заменён"));
    CHECK(dialog.nameEdit()->maxLength() == 120);
    CHECK_FALSE(dialog.markNote()->isVisibleTo(&dialog));  // only with «Применять к новым проектам»
    dialog.newProjects()->setChecked(true);
    CHECK(dialog.markNote()->isVisibleTo(&dialog));
    CHECK(dialog.leftOut()->isVisibleTo(&dialog));
    CHECK(dialog.answer().forNewProjects);
    CHECK(dialog.answer().name == QString::fromUtf8("Мягкий"));
    setup.leftOut.clear();
    PresetSaveDialog bare(setup);
    CHECK_FALSE(bare.leftOut()->isVisibleTo(&bare));
    // the NR menu fills it: the pipeline preset's mark it would move, the pass format the preset leaves out
    const auto d = Dir("save_dialog");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(model.savePipelinePreset("Y", {}, true).ok);
    model.setStageParam("nr", "format", "png");
    PresetMenu menu(model, "nr");
    StubDialogs dialogs;
    menu.setDialogs(dialogs.Make());
    dialogs.saveAnswer = PresetSaveDialog::Answer{QString::fromUtf8("Мягкий"), true};
    menu.saveCurrent();
    REQUIRE(dialogs.lastSetup);
    CHECK(dialogs.lastSetup->title == QString::fromUtf8("Сохранить настройки «Улучшения»"));
    CHECK(dialogs.lastSetup->markNote == QString::fromUtf8("Сейчас для новых проектов: все настройки обработки из «Y» — будет заменено"));
    CHECK(dialogs.lastSetup->leftOut.contains(QString::fromUtf8("Формат пасса")));
    CHECK(dialogs.lastSetup->problem(QString::fromUtf8("Перед сбросом · clip")).contains(QString::fromUtf8("автосохранения")));
    CHECK(dialogs.lastSetup->exists(QString::fromUtf8("мягкий")));  // saved by this very call
    CHECK(PresetStore::List("nr").items.size() == 1);
    CHECK(PresetStore::NewProjectPreset() == QFileInfo(PresetStore::List("nr").items[0].file).fileName());  // the one mark moved
    model.setStageParam("nr", "format", "exr");
    dialogs.lastSetup.reset();
    menu.saveCurrent();
    CHECK_FALSE(dialogs.lastSetup->leftOut.contains(QString::fromUtf8("Формат пасса")));  // at its default: nothing to name
    CHECK(dialogs.lastSetup->leftOut.contains(QString::fromUtf8("Бэкенд stub")));       // the stub method of this project is
    CHECK(dialogs.lastSetup->markNote.isEmpty());  // the mark is on this menu's kind now
    CHECK(PresetStore::List("nr").items.size() == 1);  // the same name: replaced
}

namespace {
PresetButton* PipelineButton(QWidget* panel) {
    for (PresetButton* b : panel->findChildren<PresetButton*>())
        if (b->accessibleName() == QString::fromUtf8("Пресеты обработки")) return b;
    return nullptr;
}
}  // namespace

TEST_CASE("ProjectPanel: the pipeline preset button sits in the Stages header after Process in the Tab order; Space opens its menu instead of playback; items preview the diff per stage",
          "[app][panel][presets][regression]") {
    App();
    ClearPresets();
    {  // no project: the button waits
        AppModel empty(true);
        TaskQueue queue;
        ProjectPanel panel(empty, queue);
        panel.refresh();
        REQUIRE(PipelineButton(&panel));
        CHECK_FALSE(PipelineButton(&panel)->isEnabled());
    }
    const auto d = Dir("panel_presets");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageParam("scenes", "threshold", 0.3);
    model.setStageEnabled("fg", false);
    model.setEncode("hevc_nvenc", {{"b", "50M"}}, {10, "auto"});
    REQUIRE(model.savePipelinePreset(QString::fromUtf8("Аниме")).ok);
    model.setStageParam("scenes", "threshold", 0.24);
    model.setStageEnabled("fg", true);
    model.setEncode("h264_nvenc", {}, {});
    TaskQueue queue;
    QWidget window;
    auto* v = new QVBoxLayout(&window);
    auto* panel = new ProjectPanel(model, queue, &window);
    auto* plain = new QPushButton("plain", &window);
    v->addWidget(panel);
    v->addWidget(plain);
    auto* space = new QShortcut(QKeySequence(Qt::Key_Space), &window);
    QObject::connect(space, &QShortcut::activated, &model, [&] { model.setPlaying(!model.playing()); });
    panel->refresh();
    PresetButton* button = PipelineButton(panel);
    REQUIRE(button);
    CHECK(button->isEnabled());
    CHECK(button->text() == QString::fromUtf8("Пресет &обработки"));  // Alt+О
    QPushButton* process = nullptr;
    for (QPushButton* b : panel->findChildren<QPushButton*>())
        if (b->text().startsWith(QString::fromUtf8("Обработать"))) process = b;
    REQUIRE(process);
    process->setEnabled(true);  // the plan has no video here: the order is what counts
    CHECK(NextTab(process, panel) == button);
    QWidget* after = NextTab(button, panel);
    REQUIRE(after);
    CHECK(after->toolTip() == QString::fromUtf8("Стадия участвует в обработке"));  // the switch of the first card
    window.show();
    window.activateWindow();
    REQUIRE(QTest::qWaitForWindowActive(&window));
    plain->setFocus();
    QTest::keyClick(plain, Qt::Key_Space);
    REQUIRE(model.playing());  // the control: the window's Space is live
    model.setPlaying(false);
    QMenu* menu = button->menu();
    REQUIRE(menu);
    int shown = 0;
    QObject::connect(menu, &QMenu::aboutToShow, menu, [&] {
        ++shown;
        QTimer::singleShot(0, menu, &QMenu::close);
    });
    button->setFocus();
    QTest::keyClick(button, Qt::Key_Space);
    CHECK(shown == 1);
    CHECK_FALSE(model.playing());
    // the preview per stage
    emit menu->aboutToShow();
    QAction* item = ActionStarting(menu, QString::fromUtf8("Аниме"));
    REQUIRE(item);
    CHECK_FALSE(item->isChecked());
    const QString tip = item->toolTip();
    INFO(tip.toStdString());
    CHECK(tip.startsWith(QString::fromUtf8("Изменит:\n")));
    CHECK(tip.contains(QString::fromUtf8("  Склейки: Порог склейки %1 → %2").arg(QLocale().toString(0.24, 'f', 2), QLocale().toString(0.3, 'f', 2))));
    CHECK(tip.contains(QString::fromUtf8("  Генерация кадров: выключится")));
    CHECK(tip.contains(QString::fromUtf8("  Кодирование: Кодек H.264 NVENC → HEVC NVENC; Разрядность Авто → 10 бит; Битрейт Авто → 50 Мбит/с")));
    CHECK(tip.contains(QString::fromUtf8("Без изменений: Глубина, Векторы движения, Апскейл, Улучшение")));
    CHECK(item->statusTip() == QString::fromUtf8("Изменит: Склейки, Генерация кадров, Кодирование"));
    StubDialogs quiet;
    panel->pipelinePresetMenu()->setDialogs(quiet.Make());
    item->trigger();
    CHECK(quiet.warnings == 0);  // the summary goes to the status bar and the log: no dialog
    CHECK(quiet.infos == 0);
    CHECK(quiet.confirms == 0);
    CHECK_FALSE(model.stageEntry("fg")->enabled);
    CHECK(model.project().codec == "hevc_nvenc");
    emit menu->aboutToShow();
    item = ActionStarting(menu, QString::fromUtf8("Аниме"));
    REQUIRE(item);
    CHECK(item->isChecked());
    CHECK(ActionStarting(menu, QString::fromUtf8("Перед применением · missing")));
    // the save dialog of this menu names the NR mark it would move
    model.setStageParam("nr", "intensity", 0.8);
    REQUIRE(model.saveStagePreset("nr", QString::fromUtf8("Мягкий"), {}, true).ok);
    StubDialogs dialogs;
    panel->pipelinePresetMenu()->setDialogs(dialogs.Make());
    panel->pipelinePresetMenu()->saveCurrent();
    REQUIRE(dialogs.lastSetup);
    CHECK(dialogs.lastSetup->title == QString::fromUtf8("Сохранить настройки обработки"));
    CHECK(dialogs.lastSetup->markNote == QString::fromUtf8("Сейчас для новых проектов: «Улучшение» из «Мягкий» — будет заменено"));
    emit panel->pipelinePresetMenu()->aboutToShow();
    QMenu* mark = nullptr;
    for (QAction* a : panel->pipelinePresetMenu()->actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) mark = a->menu();
    REQUIRE(mark);
    QAction* other = mark->actions().last();
    CHECK(other->text() == QString::fromUtf8("«Улучшение» из «Мягкий»"));
    CHECK(other->isChecked());
    CHECK_FALSE(other->isEnabled());
    CHECK_FALSE(mark->actions()[0]->isChecked());  // «Встроенные умолчания»: the mark is elsewhere
    QSettings().remove("presets/newProject");
    model.setPlaying(false);
}

TEST_CASE("ProjectPanel: a codec outside the list is its own item of the encode card; choosing one of the list changes the project's codec", "[app][panel][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("panel_codec");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setEncode("libx264", {});
    TaskQueue queue;
    ProjectPanel panel(model, queue);
    panel.refresh();
    QComboBox* codec = nullptr;
    for (QComboBox* c : panel.findChildren<QComboBox*>())
        if (c->count() > 0 && c->itemData(0).toString() == "h264_nvenc") codec = c;
    REQUIRE(codec);
    CHECK(codec->count() == 6);
    CHECK(codec->currentText() == QString::fromUtf8("libx264 (из файла проекта)"));
    const auto refused = model.savePipelinePreset(QString::fromUtf8("Не выйдет"));
    CHECK_FALSE(refused.ok);
    CHECK(refused.problems.join(" ").contains(QString::fromUtf8("Кодек: «libx264» — нет в списке")));
    codec->setCurrentIndex(0);  // H.264 NVENC: before, the index stayed 0 and nothing changed
    CHECK(model.project().codec == "h264_nvenc");
    panel.refresh();
    CHECK(codec->count() == 5);
    CHECK(codec->currentIndex() == 0);
    CHECK(model.savePipelinePreset(QString::fromUtf8("Теперь")).ok);
}

// The stages' header with «Пресет обработки» and its open menu (as the NR card's shots: set DLSSVID_SHOTS_DIR, [.shots])
TEST_CASE("ProjectPanel pipeline preset screenshots for a merge request", "[.shots]") {
    App();
    const QString out = qEnvironmentVariable("DLSSVID_SHOTS_DIR");
    if (out.isEmpty()) SKIP("DLSSVID_SHOTS_DIR is not set");
    dlssvid::ApplyTheme(App());
    ClearPresets();
    ClipSpec spec;
    spec.frames = 2;
    spec.width = 48;
    spec.height = 32;
    const auto d = Dir("panel_shots");
    Project p = Project::Create(WriteClip(d / "clip.mkv", spec), d / "passes");
    p.Save(d / "proj.dlssvid.json");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    model.setStageParam("nr", "intensity", 0.8);
    model.setStageEnabled("fg", false);
    model.savePipelinePreset(QString::fromUtf8("Аниме 4K"), {}, true);
    model.setStageParam("nr", "intensity", 1.0);
    model.setStageEnabled("fg", true);
    TaskQueue queue;
    const QString scale = QString::number(qEnvironmentVariable("QT_SCALE_FACTOR", "1").toDouble() * 100);
    ProjectPanel panel(model, queue);
    panel.setAttribute(Qt::WA_DontShowOnScreen);
    panel.resize(1100, 900);
    panel.show();
    panel.refresh();
    QApplication::processEvents();
    PresetButton* button = PipelineButton(&panel);
    REQUIRE(button);
    const QPoint at = button->mapTo(&panel, QPoint(0, 0));
    const QString head = out + "/stages_header_" + scale + ".png";
    REQUIRE(panel.grab(QRect(0, at.y() - 12, panel.width(), button->height() + 24)).save(head));
    std::printf("saved %s\n", head.toStdString().c_str());
    PresetMenu* menu = panel.pipelinePresetMenu();
    menu->setAttribute(Qt::WA_DontShowOnScreen);
    emit menu->aboutToShow();
    menu->adjustSize();
    const QString file = out + "/pipeline_preset_menu_" + scale + ".png";
    REQUIRE(menu->grab().save(file));
    std::printf("saved %s\n", file.toStdString().c_str());
    QSettings().remove("presets/newProject");
}

// Review of PR-9: an empty text value (flow.target "" = the default) read as nothing before the arrow
TEST_CASE("ParamText: a change of a text key from or to its empty default reads as the default", "[app][presets][regression]") {
    App();
    const QString t = ChangeText("flow", ParamChange{"target", "", "1920x1080", "ofa", "ofa"});
    INFO(t.toStdString());
    CHECK(t == QString::fromUtf8("Целевое разрешение mv_dlss (WxH) по умолчанию → 1920x1080"));
    CHECK(ChangeText("flow", ParamChange{"target", "1920x1080", "", "ofa", "ofa"}).endsWith(QString::fromUtf8("→ по умолчанию")));
}

// Review of PR-9: the menus' import, export and delete through their dialogs - the copy the clash names, replace, cancel, a pipeline
// preset into the NR card, the current settings exported with what they leave out, a permanent delete
TEST_CASE("PresetMenu: import, export and delete through the dialogs", "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("menu_actions");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    QSignalSpy messages(&model, &AppModel::message);
    PresetMenu nr(model, "nr");
    StubDialogs dialogs;
    nr.setDialogs(dialogs.Make());
    const QString soft = WritePreset(d / "soft.json", CardPresetJson("Мягкий", {{"intensity", 0.7}}));
    dialogs.open = soft;
    nr.importFile();
    REQUIRE(PresetStore::List("nr").items.size() == 1);
    dialogs.clash = PresetDialogs::Clash::Copy;
    nr.importFile();
    CHECK(dialogs.clashName == QString::fromUtf8("Мягкий"));
    CHECK(dialogs.clashCopy == QString::fromUtf8("Мягкий (2)"));
    nr.importFile();
    CHECK(dialogs.clashCopy == QString::fromUtf8("Мягкий (3)"));  // the button names what is saved
    auto names = [] {
        QStringList n;
        for (const auto& i : PresetStore::List("nr").items) n << i.name;
        return n;
    };
    CHECK(names() == QStringList{QString::fromUtf8("Мягкий"), QString::fromUtf8("Мягкий (2)"), QString::fromUtf8("Мягкий (3)")});
    auto stored = [] {
        for (const auto& i : PresetStore::List("nr").items)
            if (i.name == QString::fromUtf8("Мягкий")) return LoadPreset(PresetStore::Path(i.file)).preset.stages.at("nr").params.at("intensity").get<double>();
        return -1.0;
    };
    WritePreset(d / "soft.json", CardPresetJson("Мягкий", {{"intensity", 0.4}}));
    dialogs.clash = PresetDialogs::Clash::Replace;
    nr.importFile();
    CHECK(names().size() == 3);
    CHECK(stored() == Approx(0.4));  // replaced
    CHECK(LastMessage(messages) == QString::fromUtf8("Пресет «Мягкий» импортирован"));
    WritePreset(d / "soft.json", CardPresetJson("Мягкий", {{"intensity", 0.3}}));
    dialogs.clash = PresetDialogs::Clash::Cancel;
    nr.importFile();
    CHECK(names().size() == 3);
    CHECK(stored() == Approx(0.4));  // cancelled: untouched
    CHECK(dialogs.warnings == 0);
    // a pipeline preset into the card: its nr settings, said in a dialog
    nlohmann::json pipe = {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "pipeline"}, {"name", "Аниме"}, {"stages", nlohmann::json::object()}};
    pipe["stages"]["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.6}}}};
    dialogs.open = WritePreset(d / "pipe.json", pipe);
    nr.importFile();
    CHECK(dialogs.infos == 1);
    CHECK(dialogs.texts.last().startsWith(QString::fromUtf8("Из пресета обработки «Аниме» взяты настройки «Улучшения».")));
    // the current settings exported: a valid preset of their kind; what it lacks is said
    model.setStageParam("nr", "format", "png");
    dialogs.save = Q(d / "out" / "nr.json");
    nr.exportFile({});
    const PresetLoad card = LoadPreset(d / "out" / "nr.json");
    REQUIRE(card.ok());
    CHECK(card.preset.scope == PresetScope::Stage);
    CHECK(LastMessage(messages).contains(QString::fromUtf8("Не войдут в пресет")));
    PresetMenu all(model, kPipelineTarget);
    all.setDialogs(dialogs.Make());
    model.setEncode("hevc_nvenc", {{"b", "50M"}}, {});
    dialogs.save = Q(d / "out" / "all.json");
    all.exportFile({});
    const PresetLoad whole = LoadPreset(d / "out" / "all.json");
    REQUIRE(whole.ok());
    CHECK(whole.preset.scope == PresetScope::Pipeline);
    CHECK(whole.preset.stages.size() == 6);
    REQUIRE(whole.preset.encode);
    CHECK(whole.preset.encode->codec == "hevc_nvenc");
    CHECK(dialogs.warnings == 0);
    // delete: asked, permanent, the mark on it goes
    const auto listed = PresetStore::List("nr").items;
    const QString victim = listed[0].file;
    PresetStore::SetNewProjectPreset(victim);
    dialogs.answer = false;
    nr.removeFile(victim, listed[0].name);
    CHECK(std::filesystem::exists(PresetStore::Path(victim)));
    dialogs.answer = true;
    nr.removeFile(victim, listed[0].name);
    CHECK(dialogs.texts.last().contains(QFileInfo(victim).fileName()));  // the file named in the question
    CHECK_FALSE(std::filesystem::exists(PresetStore::Path(victim)));
    CHECK(PresetStore::NewProjectPreset().isEmpty());
}

// Review of PR-9: the Russian texts of a few file problems say what the file holds and what it should
TEST_CASE("PresetStore texts: a whole number, a chroma given as a number or in the card's spelling, a list key without a list", "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_texts");
    const auto fractional = PresetStore::Import("nr", WritePreset(d / "passes.json", CardPresetJson("Проходы", {{"passes", 2.5}})));
    INFO(fractional.problems.join(" | ").toStdString());
    CHECK(fractional.problems.join(" ").contains(QString::fromUtf8("Проходы: нужно целое число")));
    nlohmann::json pipe = {{"format", "dlssvid-preset"}, {"schema", 1}, {"scope", "pipeline"}, {"name", "Цвет"}, {"stages", nlohmann::json::object()}};
    pipe["stages"]["nr"] = {{"enabled", true}, {"params", {{"intensity", 0.6}}}};
    for (const nlohmann::json& chroma : {nlohmann::json(420), nlohmann::json("4:2:0")}) {
        INFO(chroma.dump());
        pipe["encode"] = {{"codec", "h264_nvenc"}, {"chroma", chroma}};
        const auto r = PresetStore::Import(kPipelineTarget, WritePreset(d / "chroma.json", pipe));
        INFO(r.problems.join(" | ").toStdString());
        CHECK(r.problems.join(" ").contains(QString::fromUtf8("в файле допустимо строкой \"auto\", \"420\", \"422\" или \"444\"")));
        CHECK_FALSE(r.problems.join(" ").contains(QString::fromUtf8("кодирует")));
    }
    PresetIssue listless;
    listless.kind = PresetIssueKind::ForeignKey;
    listless.stage = "depth";
    listless.key = "model";
    listless.to = "da3";
    listless.kept = true;
    CHECK(PresetStore::IssueText(listless).contains(QString::fromUtf8("нет списка — пресет её не задаёт, оставлено значение проекта")));
    listless.kept = false;  // the preset's switch of the method dropped the project's value
    CHECK(PresetStore::IssueText(listless).contains(QString::fromUtf8("работает значение метода по умолчанию")));
    listless.key = "window";
    CHECK(PresetStore::IssueText(listless).contains(QString::fromUtf8("не относится к методу")));
}

// Review of PR-9: in the GUI too a card preset's name is taken per stage - an NR «Мягкий» next to a depth one is saved and imported
// as a new file, the depth preset untouched (before: «Заменить» overwrote it)
TEST_CASE("Presets: the NR menu never replaces a card preset of another stage with the same name", "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_other_stage");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    const auto depthFile = PresetStore::Path(PresetStore::Dir()) / "depth-soft.json";
    WritePreset(depthFile, CardPresetJson("Мягкий", {{"backend", "vda"}}, "depth"));
    auto bytes = [](const std::filesystem::path& f) {
        std::ifstream in(f, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    };
    const std::string depthBytes = bytes(depthFile);
    CHECK_FALSE(PresetStore::FindByName(QString::fromUtf8("Мягкий"), PresetScope::Stage, "nr"));
    CHECK(PresetStore::FindByName(QString::fromUtf8("Мягкий"), PresetScope::Stage, "depth"));
    PresetMenu menu(model, "nr");
    StubDialogs dialogs;
    menu.setDialogs(dialogs.Make());
    menu.saveCurrent();  // cancelled: the dialog's setup only
    REQUIRE(dialogs.lastSetup);
    CHECK_FALSE(dialogs.lastSetup->exists(QString::fromUtf8("Мягкий")));  // no «Заменить» over the depth preset
    dialogs.saveAnswer = PresetSaveDialog::Answer{QString::fromUtf8("Мягкий"), false};
    menu.saveCurrent();
    const auto listed = PresetStore::List("nr").items;
    REQUIRE(listed.size() == 1);
    CHECK(PresetStore::Path(listed[0].file) != depthFile);
    CHECK(bytes(depthFile) == depthBytes);
    const auto again = model.saveStagePreset("nr", QString::fromUtf8("Мягкий"));
    CHECK(again.exists);
    CHECK(again.file == listed[0].file);  // the clash is with the NR preset
    // an import of an NR «Мягкий» clashes with the NR one only
    const auto imported = PresetStore::Import("nr", WritePreset(d / "soft.json", CardPresetJson("Мягкий", {{"intensity", 0.5}})));
    CHECK(imported.exists);
    CHECK(imported.file == listed[0].file);
    CHECK(bytes(depthFile) == depthBytes);
    // a card preset of a card without a menu, where a pipeline preset is expected: the command line, not a menu that does not exist
    const auto refused = model.applyPipelinePreset(Q(depthFile));
    CHECK(refused.problems.join(" ").contains(QString::fromUtf8("--param-file depth=")));
    CHECK_FALSE(refused.problems.join(" ").contains(QString::fromUtf8("Пресет ▾")));
    // a name with a no-break space at its end: the clash is found before the core would refuse the save
    const auto nbsp = PresetStore::Import("nr", WritePreset(d / "nbsp.json", CardPresetJson("Ночь\xC2\xA0", {{"intensity", 0.4}})));
    REQUIRE(nbsp.ok);
    const auto nbspAgain = PresetStore::Import("nr", WritePreset(d / "nbsp.json", CardPresetJson("Ночь\xC2\xA0", {{"intensity", 0.4}})));
    CHECK(nbspAgain.exists);
    CHECK(nbspAgain.problems.isEmpty());
}

// Review of PR-9: the tooltip of the other kind's mark is explicit rich text - «A & B» reads as written, not «A &amp; B»
TEST_CASE("PresetMenu: the other kind's mark names a preset with & as written", "[app][presets][regression]") {
    App();
    ClearPresets();
    const auto d = Dir("presets_mark_tip");
    StubProjectFile(d / "proj.dlssvid.json", d / "missing.mp4");
    AppModel model(true);
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(model.savePipelinePreset("A & B", {}, true).ok);
    PresetMenu menu(model, "nr");
    emit menu.aboutToShow();
    QMenu* mark = nullptr;
    for (QAction* a : menu.actions())
        if (a->menu() && a->text() == QString::fromUtf8("Для &новых проектов")) mark = a->menu();
    REQUIRE(mark);
    QAction* other = mark->actions().last();
    CHECK(other->text() == QString::fromUtf8("Пресет обработки «A && B»"));
    QTextDocument doc;
    doc.setHtml(other->toolTip());
    CHECK(doc.toPlainText().contains(QString::fromUtf8("из «A & B»")));
    QSettings().remove("presets/newProject");
}
