// dlssvid-gui — Qt 6 shell around the viewport (ТЗ §6). Everything it does is also reachable
// from the CLI (project init / render / depth / flow ...).

#include <QApplication>
#include <QCommandLineParser>

#include "MainWindow.h"
#include "util/Log.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("dlssvid");
    QApplication::setOrganizationName("dlssvid");
    QCommandLineParser parser;
    parser.setApplicationDescription("DLSS Video Pipeline — viewport");
    parser.addHelpOption();
    parser.addPositionalArgument("file", "project (*.dlssvid.json) or video file");
    QCommandLineOption warp("warp", "use the WARP software adapter");
    parser.addOption(warp);
    parser.process(app);

    dlssvid::MainWindow window(parser.isSet(warp));
    window.show();
    const QStringList args = parser.positionalArguments();
    if (!args.isEmpty()) window.openPath(args.first());
    return app.exec();
}
