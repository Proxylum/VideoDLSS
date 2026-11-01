// dlssvid-gui — Qt 6 shell around the viewport (ТЗ §6). Everything it does is also reachable
// from the CLI (project init / render / depth / flow ...).

#include <QApplication>
#include <QCommandLineParser>

#include "MainWindow.h"
#include "Theme.h"
#include "util/Log.h"
#include "util/Paths.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("dlssvid");
    QApplication::setOrganizationName("dlssvid");
    dlssvid::ApplyTheme(app);
    QCommandLineParser parser;
    parser.setApplicationDescription("DLSS Video Pipeline — viewport");
    parser.addHelpOption();
    parser.addPositionalArgument("file", "project (*.dlssvid.json) or video file");
    QCommandLineOption warp("warp", "use the WARP software adapter");
    parser.addOption(warp);
    parser.process(app);

    dlssvid::MainWindow window(parser.isSet(warp));
    // after the window: its log panel attaches the sink that shows it (the console sink has no console in a GUI process)
    if (!dlssvid::Utf8CodePage()) dlssvid::Log()->warn("the process code page is not UTF-8: non-ASCII paths need Windows 10 version 1903 or newer");
    window.show();
    const QStringList args = parser.positionalArguments();
    if (!args.isEmpty()) window.openPath(args.first());
    return app.exec();
}
