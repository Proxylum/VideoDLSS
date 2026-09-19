// Qt application model without a window (QT_QPA_PLATFORM=offscreen): project handling,
// frame navigation, playback over computed frames and the task queue.

#include <catch2/catch_test_macros.hpp>

#include <QApplication>
#include <QElapsedTimer>
#include <QSignalSpy>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "AppModel.h"
#include "TaskQueue.h"
#include "passes/PassSequence.h"
#include "viewport/Project.h"

using namespace dlssvid;

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
    QSignalSpy frames(&model, &AppModel::frameChanged);
    model.openProject(Q(d / "proj.dlssvid.json"));
    REQUIRE(opened.count() == 1);
    CHECK(model.sourceNames() == QStringList{"depth_raw"});
    CHECK(model.store().FrameCount() == 5);
    CHECK(model.hasProject());

    model.setSingleSource("depth_raw");
    CHECK(model.state().layers[0].display == DisplayMode::Turbo);
    model.setFrame(99);
    CHECK(model.state().frame == 4);
    model.stepFrame(-10);
    CHECK(model.state().frame == 0);
    model.stepFrame(2);
    CHECK(model.state().frame == 2);
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
    CHECK(model.state().frame == 4);

    model.state().wipe.enabled = true;
    model.state().frame = 1;
    REQUIRE(model.saveProject());
    const Project q = Project::Load(d / "proj.dlssvid.json");
    CHECK(q.viewport.wipe.enabled);
    CHECK(q.viewport.frame == 1);
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
