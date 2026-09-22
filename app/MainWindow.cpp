#include "MainWindow.h"

#include <QAction>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFileDialog>
#include <QGridLayout>
#include <QImage>
#include <QMenuBar>
#include <QPainter>
#include <QShortcut>
#include <QStatusBar>
#include <QVBoxLayout>
#include <cmath>

#include "InspectorPanel.h"
#include "LogPanel.h"
#include "ProjectPanel.h"
#include "TaskQueue.h"
#include "TimelineWidget.h"
#include "ViewportWindow.h"
#include "util/Log.h"

namespace dlssvid {

MainWindow::MainWindow(bool warp, QWidget* parent) : QMainWindow(parent), model_(warp) {
    setWindowTitle("dlssvid");
    resize(1500, 900);

    // centre: cell labels + viewport + timeline
    auto* central = new QWidget(this);
    auto* v = new QVBoxLayout(central);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(2);
    auto* labels = new QWidget(central);
    auto* lg = new QGridLayout(labels);
    lg->setContentsMargins(4, 0, 4, 0);
    for (size_t i = 0; i < 4; ++i) {
        cellLabels_[i] = new QLabel(labels);
        cellLabels_[i]->setAlignment(Qt::AlignCenter);
        lg->addWidget(cellLabels_[i], 0, static_cast<int>(i));
    }
    v->addWidget(labels);
    viewport_ = new ViewportWindow(model_);
    viewportContainer_ = QWidget::createWindowContainer(viewport_, central);
    viewportContainer_->setMinimumSize(320, 200);
    viewportContainer_->setFocusPolicy(Qt::StrongFocus);
    v->addWidget(viewportContainer_, 1);
    timeline_ = new TimelineWidget(model_, central);
    v->addWidget(timeline_);
    setCentralWidget(central);

    // docks
    tasks_ = new TaskQueue(this);
    projectPanel_ = new ProjectPanel(model_, *tasks_, this);
    auto* projectDock = new QDockWidget(tr("Проект"), this);
    projectDock->setWidget(projectPanel_);
    addDockWidget(Qt::LeftDockWidgetArea, projectDock);
    inspector_ = new InspectorPanel(model_, *viewport_, this);
    auto* inspectorDock = new QDockWidget(tr("Инспектор"), this);
    inspectorDock->setWidget(inspector_);
    addDockWidget(Qt::RightDockWidgetArea, inspectorDock);
    log_ = new LogPanel(this);
    auto* logDock = new QDockWidget(tr("Лог"), this);
    logDock->setWidget(log_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock);
    auto* tasksDock = new QDockWidget(tr("Задачи"), this);
    tasksDock->setWidget(tasks_);
    addDockWidget(Qt::BottomDockWidgetArea, tasksDock);
    tabifyDockWidget(logDock, tasksDock);

    zoomLabel_ = new QLabel(this);
    statusBar()->addPermanentWidget(zoomLabel_);
    connect(&model_, &AppModel::message, this, [this](const QString& m) { statusBar()->showMessage(m, 5000); });
    connect(viewport_, &ViewportWindow::zoomChanged, this, [this](float z) { zoomLabel_->setText(QString("%1 %").arg(static_cast<int>(std::lround(z * 100)))); });
    connect(&model_, &AppModel::stateChanged, this, &MainWindow::updateCellLabels);
    connect(&model_, &AppModel::timeChanged, this, [this](double) { updateCellLabels(); });
    connect(&model_, &AppModel::framesUpdated, this, &MainWindow::updateCellLabels);  // «загрузка…» -> the frame
    connect(&model_, &AppModel::sourcesChanged, this, &MainWindow::updateCellLabels);
    connect(tasks_, &TaskQueue::taskOutput, this, [](int id, const QString& line) { Log()->info("[task {}] {}", id, line.toStdString()); });

    buildMenus();
    updateCellLabels();
}

void MainWindow::buildMenus() {
    auto* file = menuBar()->addMenu(tr("&Файл"));
    file->addAction(tr("Открыть видео…"), QKeySequence::Open, this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, tr("Видео"), {}, tr("Видео (*.mp4 *.mov *.mkv *.avi);;Все файлы (*)"));
        if (!f.isEmpty()) model_.openVideo(f);
    });
    file->addAction(tr("Открыть проект…"), this, [this] {
        const QString f = QFileDialog::getOpenFileName(this, tr("Проект"), {}, tr("Проект dlssvid (*.dlssvid.json)"));
        if (!f.isEmpty()) model_.openProject(f);
    });
    file->addAction(tr("Сохранить проект"), QKeySequence::Save, this, [this] {
        if (!model_.saveProject()) {
            const QString f = QFileDialog::getSaveFileName(this, tr("Сохранить проект"), {}, tr("Проект dlssvid (*.dlssvid.json)"));
            if (!f.isEmpty()) model_.saveProjectAs(f);
        }
    });
    file->addAction(tr("Сохранить проект как…"), this, [this] {
        const QString f = QFileDialog::getSaveFileName(this, tr("Сохранить проект"), {}, tr("Проект dlssvid (*.dlssvid.json)"));
        if (!f.isEmpty()) model_.saveProjectAs(f);
    });
    file->addSeparator();
    file->addAction(tr("Скриншот вьюпорта в PNG…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S), this, &MainWindow::saveScreenshot);
    file->addSeparator();
    file->addAction(tr("Выход"), QKeySequence::Quit, this, &QWidget::close);

    auto* view = menuBar()->addMenu(tr("&Вид"));
    view->addAction(tr("Одиночный"), QKeySequence("Ctrl+1"), this, [this] { model_.setMode(ViewMode::Single); });
    view->addAction(tr("Наложение"), QKeySequence("Ctrl+2"), this, [this] { model_.setMode(ViewMode::Overlay); });
    view->addAction(tr("Сетка 2x2"), QKeySequence("Ctrl+3"), this, [this] { model_.setMode(ViewMode::Grid); });
    view->addSeparator();
    view->addAction(tr("Вписать в окно"), QKeySequence("F"), this, [this] { viewport_->fitView(); });
    view->addAction(tr("1:1"), QKeySequence("Ctrl+0"), this, [this] { viewport_->oneToOne(); });
    view->addAction(tr("Шторка вкл/выкл"), QKeySequence("W"), this, [this] {
        model_.state().wipe.enabled = !model_.state().wipe.enabled;
        model_.notifyStateChanged();
    });

    // hotkeys 1..9 switch the single-view source (ТЗ §6)
    for (int k = 1; k <= 9; ++k) {
        auto* sc = new QShortcut(QKeySequence(QString::number(k)), this);
        connect(sc, &QShortcut::activated, this, [this, k] { selectSource(k); });
    }
    new QShortcut(QKeySequence(Qt::Key_Space), this, [this] { model_.setPlaying(!model_.playing()); });
    new QShortcut(QKeySequence(Qt::Key_Period), this, [this] { model_.stepFrame(+1); });
    new QShortcut(QKeySequence(Qt::Key_Comma), this, [this] { model_.stepFrame(-1); });
    new QShortcut(QKeySequence(Qt::Key_Right), this, [this] { model_.stepFrame(+1); });
    new QShortcut(QKeySequence(Qt::Key_Left), this, [this] { model_.stepFrame(-1); });
}

void MainWindow::selectSource(int hotkey) {
    const QStringList names = model_.sourceNames();
    if (hotkey - 1 < names.size()) {
        if (model_.state().mode == ViewMode::Grid) model_.setMode(ViewMode::Single);
        model_.setSingleSource(names[hotkey - 1]);
    }
}

void MainWindow::updateCellLabels() {
    const ViewportState& st = model_.state();
    const bool grid = st.mode == ViewMode::Grid;
    for (size_t i = 0; i < 4; ++i) {
        cellLabels_[i]->setVisible(grid ? (st.expandedCell < 0 || st.expandedCell == static_cast<int>(i)) : i == 0);
        if (grid) cellLabels_[i]->setText(cellLabel(QString::fromStdString(st.gridSources[i]), st.gridSources[i]));
    }
    if (!grid) {
        QString t;
        for (const auto& l : st.layers) t += (t.isEmpty() ? "" : " + ") + QString::fromStdString(l.source);
        cellLabels_[0]->setText(cellLabel(t, model_.baseSource()));
    }
    setWindowTitle(QString("dlssvid — %1").arg(model_.hasProject() ? QString::fromStdWString(model_.project().sourceVideo.filename().wstring()) : tr("нет проекта")));
}

// «source  #123 · 00:05.12», plus the state while the frame is not on screen (stage 9: no silent black cells).
QString MainWindow::cellLabel(const QString& title, const std::string& source) {
    const double t = model_.time();
    QString s = QString("%1  #%2 · %3").arg(title).arg(model_.store().FrameAt(source, t)).arg(QString::fromStdString(FormatTimecode(t)));
    switch (model_.frameStateOf(source)) {
        case FrameState::Loading: s += tr(" · загрузка…"); break;
        case FrameState::Missing: s += tr(" · нет кадра"); break;
        default: break;
    }
    return s;
}

void MainWindow::saveScreenshot() {
    const QString f = QFileDialog::getSaveFileName(this, tr("Скриншот"), "viewport.png", "PNG (*.png)");
    if (f.isEmpty()) return;
    try {
        const PassImage img = viewport_->screenshot();
        QImage q(img.data.data(), static_cast<int>(img.width), static_cast<int>(img.height), static_cast<int>(img.RowBytes()), QImage::Format_RGBA8888);
        QImage out = q.copy();
        // cell labels into the picture (ТЗ §6: подписи ячеек с именем пасса и номером кадра)
        QPainter p(&out);
        p.setPen(Qt::white);
        const ViewportState& st = model_.state();
        const auto rects = ViewportRenderer::CellRects(st, static_cast<uint32_t>(out.width()), static_cast<uint32_t>(out.height()));
        for (size_t i = 0; i < rects.size(); ++i) {
            const int cell = st.mode == ViewMode::Grid ? (st.expandedCell >= 0 ? st.expandedCell : static_cast<int>(i)) : 0;
            const QString label = st.mode == ViewMode::Grid ? cellLabels_[static_cast<size_t>(cell)]->text() : cellLabels_[0]->text();
            p.fillRect(QRectF(rects[i].x + 4, rects[i].y + 4, 8 + p.fontMetrics().horizontalAdvance(label), 18), QColor(0, 0, 0, 160));
            p.drawText(QPointF(rects[i].x + 8, rects[i].y + 18), label);
        }
        p.end();
        out.save(f, "PNG");
        statusBar()->showMessage(tr("Скриншот сохранён: %1").arg(f), 5000);
    } catch (const std::exception& e) {
        statusBar()->showMessage(tr("Ошибка скриншота: %1").arg(e.what()), 5000);
    }
}

void MainWindow::openPath(const QString& path) {
    if (path.endsWith(".json", Qt::CaseInsensitive)) model_.openProject(path);
    else model_.openVideo(path);
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (model_.hasProject() && !model_.project().file.empty()) model_.saveProject();
    QMainWindow::closeEvent(e);
}

}  // namespace dlssvid
