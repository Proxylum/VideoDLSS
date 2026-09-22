#include "MainWindow.h"

#include <QAction>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QEvent>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QImage>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QSettings>
#include <QUrl>
#include <QShortcut>
#include <QStatusBar>
#include <QStyle>
#include <QVBoxLayout>
#include <cmath>

#include "CompareBar.h"
#include "InspectorPanel.h"
#include "LogPanel.h"
#include "ProcessingPanel.h"
#include "ProjectPanel.h"
#include "ResultBar.h"
#include "SourceNames.h"
#include "StartPage.h"
#include "TaskQueue.h"
#include "TimelineWidget.h"
#include "ViewportWindow.h"
#include "util/Log.h"

namespace dlssvid {

MainWindow::MainWindow(bool warp, QWidget* parent) : QMainWindow(parent), model_(warp) {
    setWindowTitle("dlssvid");
    setAcceptDrops(true);

    // pages: the start screen, then the work area (cell labels + viewport + timeline)
    pages_ = new QStackedWidget(this);
    startPage_ = new StartPage(model_, pages_);
    workArea_ = new QWidget(pages_);
    auto* v = new QVBoxLayout(workArea_);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(2);
    compareBar_ = new CompareBar(model_, workArea_);
    v->addWidget(compareBar_);
    resultBar_ = new ResultBar(model_, workArea_);
    v->addWidget(resultBar_);
    auto* labels = new QWidget(workArea_);
    auto* lg = new QGridLayout(labels);
    lg->setContentsMargins(4, 0, 4, 0);
    for (size_t i = 0; i < 4; ++i) {
        cellLabels_[i] = new QLabel(labels);
        cellLabels_[i]->setAlignment(Qt::AlignCenter);
        lg->addWidget(cellLabels_[i], 0, static_cast<int>(i));
    }
    v->addWidget(labels);
    viewport_ = new ViewportWindow(model_);
    viewportContainer_ = QWidget::createWindowContainer(viewport_, workArea_);
    viewportContainer_->setMinimumSize(320, 200);
    viewportContainer_->setFocusPolicy(Qt::StrongFocus);
    v->addWidget(viewportContainer_, 1);
    timeline_ = new TimelineWidget(model_, workArea_);
    v->addWidget(timeline_);
    pages_->addWidget(startPage_);
    pages_->addWidget(workArea_);
    setCentralWidget(pages_);

    // docks (object names: QMainWindow::saveState needs them)
    tasks_ = new TaskQueue(this);
    processing_ = new ProcessingPanel(*tasks_, pages_);
    pages_->addWidget(processing_);
    projectPanel_ = new ProjectPanel(model_, *tasks_, this);
    projectDock_ = new QDockWidget(tr("Проект"), this);
    projectDock_->setObjectName("projectDock");
    projectDock_->setWidget(projectPanel_);
    addDockWidget(Qt::LeftDockWidgetArea, projectDock_);
    inspector_ = new InspectorPanel(model_, *viewport_, this);
    inspectorDock_ = new QDockWidget(tr("Инспектор"), this);
    inspectorDock_->setObjectName("inspectorDock");
    inspectorDock_->setWidget(inspector_);
    addDockWidget(Qt::RightDockWidgetArea, inspectorDock_);
    log_ = new LogPanel(this);
    logDock_ = new QDockWidget(tr("Лог"), this);
    logDock_->setObjectName("logDock");
    logDock_->setWidget(log_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock_);
    tasksDock_ = new QDockWidget(tr("Задачи"), this);
    tasksDock_->setObjectName("tasksDock");
    tasksDock_->setWidget(tasks_);
    addDockWidget(Qt::BottomDockWidgetArea, tasksDock_);
    tabifyDockWidget(logDock_, tasksDock_);

    zoomLabel_ = new QLabel(this);
    statusBar()->addPermanentWidget(zoomLabel_);
    connect(&model_, &AppModel::message, this, [this](const QString& m) { statusBar()->showMessage(m, 5000); });
    connect(viewport_, &ViewportWindow::zoomChanged, this, [this](float z) { zoomLabel_->setText(QString("%1 %").arg(static_cast<int>(std::lround(z * 100)))); });
    connect(&model_, &AppModel::stateChanged, this, &MainWindow::updateCellLabels);
    connect(&model_, &AppModel::timeChanged, this, [this](double) { updateCellLabels(); });
    connect(&model_, &AppModel::framesUpdated, this, &MainWindow::updateCellLabels);  // «загрузка…» -> the frame
    connect(&model_, &AppModel::sourcesChanged, this, &MainWindow::updateCellLabels);
    connect(&model_, &AppModel::projectChanged, this, [this] {
        updatePage();
        updateTitle();
    });
    connect(&model_, &AppModel::dirtyChanged, this, [this](bool) { updateTitle(); });
    connect(&model_, &AppModel::recentsChanged, this, &MainWindow::rebuildRecentMenu);
    connect(startPage_, &StartPage::openVideoRequested, this, &MainWindow::chooseVideo);
    connect(startPage_, &StartPage::openProjectRequested, this, &MainWindow::chooseProject);
    connect(startPage_, &StartPage::openPathRequested, this, &MainWindow::openPath);
    connect(tasks_, &TaskQueue::taskOutput, this, [](int id, const QString& line) { Log()->info("[task {}] {}", id, line.toStdString()); });
    connect(tasks_, &TaskQueue::taskFinished, this, &MainWindow::onTaskFinished);
    connect(projectPanel_, &ProjectPanel::processQueued, this, &MainWindow::watchProcess);
    connect(processing_, &ProcessingPanel::showResultRequested, this, [this] {
        processingActive_ = false;
        updatePage();
    });
    connect(processing_, &ProcessingPanel::backRequested, this, [this] {
        processingActive_ = false;
        updatePage();
    });
    connect(processing_, &ProcessingPanel::minimizeRequested, this, &QWidget::showMinimized);
    connect(processing_, &ProcessingPanel::fullLogRequested, this, [this] {
        logDock_->show();
        logDock_->raise();
    });
    connect(resultBar_, &ResultBar::saveAsRequested, this, &MainWindow::saveResultAs);
    connect(resultBar_, &ResultBar::openFolderRequested, this, &MainWindow::openResultFolder);
    connect(resultBar_, &ResultBar::exportPassesRequested, this, &MainWindow::exportPasses);
    connect(resultBar_, &ResultBar::anotherVideoRequested, this, &MainWindow::closeProject);

    buildMenus();

    // the window as it was left; the first time: maximised
    QSettings settings;
    const QByteArray geometry = settings.value("window/geometry").toByteArray();
    if (!geometry.isEmpty()) restoreGeometry(geometry);
    else {
        resize(1500, 900);
        setWindowState(Qt::WindowMaximized);
    }
    workState_ = settings.value("window/state").toByteArray();
    docksSized_ = !workState_.isEmpty();
    updatePage();
    updateTitle();
    updateCellLabels();
}

void MainWindow::buildMenus() {
    auto* file = menuBar()->addMenu(tr("&Файл"));
    file->addAction(tr("Открыть видео…"), QKeySequence::Open, this, &MainWindow::chooseVideo);
    file->addAction(tr("Открыть проект…"), QKeySequence("Ctrl+Shift+O"), this, &MainWindow::chooseProject);
    recentMenu_ = file->addMenu(tr("Недавние"));
    rebuildRecentMenu();
    file->addAction(tr("Сохранить проект"), QKeySequence::Save, this, [this] { saveProjectDialog(false); });
    file->addAction(tr("Сохранить проект как…"), QKeySequence::SaveAs, this, [this] { saveProjectDialog(true); });
    file->addAction(tr("Закрыть проект"), QKeySequence("Ctrl+W"), this, &MainWindow::closeProject);
    file->addSeparator();
    file->addAction(tr("Сохранить результат как…"), this, &MainWindow::saveResultAs);
    file->addAction(tr("Открыть папку результата"), this, &MainWindow::openResultFolder);
    file->addAction(tr("Экспорт пассов…"), this, &MainWindow::exportPasses);
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
    view->addSeparator();
    view->addAction(tr("До | После"), QKeySequence("W"), this, [this] { model_.toggleWipe(); });
    view->addAction(tr("Только после"), this, [this] { model_.setCompareView(AppModel::CompareView::AfterOnly); });
    view->addAction(tr("Сравнить: Апскейл ↔ Улучшение"), this, [this] { model_.applyPreset(AppModel::ComparePreset::SrVsNr); });
    view->addAction(tr("Сравнить: Исходник ↔ Глубина"), this, [this] { model_.applyPreset(AppModel::ComparePreset::SourceVsDepth); });
    view->addSeparator();
    // panels: a closed dock comes back from here (audit finding 6)
    for (auto [dock, text] : {std::pair<QDockWidget*, QString>{projectDock_, tr("Панель «Проект»")},
                              {inspectorDock_, tr("Панель «Инспектор»")},
                              {logDock_, tr("Панель «Лог»")},
                              {tasksDock_, tr("Панель «Задачи»")}}) {
        QAction* a = dock->toggleViewAction();
        a->setText(text);
        view->addAction(a);
    }
    view->addSeparator();
    auto* engineer = view->addAction(tr("Инженерный режим"));
    engineer->setCheckable(true);
    engineer->setChecked(model_.engineerMode());
    engineer->setShortcut(QKeySequence("Ctrl+E"));
    connect(engineer, &QAction::toggled, this, [this](bool on) { model_.setEngineerMode(on); });
    connect(&model_, &AppModel::engineerModeChanged, engineer, &QAction::setChecked);

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

void MainWindow::rebuildRecentMenu() {
    if (!recentMenu_) return;
    recentMenu_->clear();
    const auto recents = model_.recents();
    for (const AppModel::Recent& r : recents) {
        QAction* a = recentMenu_->addAction(QString("%1 — %2").arg(r.title, r.info));
        a->setToolTip(r.path);
        a->setEnabled(r.exists);
        const QString path = r.path;
        connect(a, &QAction::triggered, this, [this, path] { openPath(path); });
    }
    if (recents.empty()) recentMenu_->addAction(tr("(пусто)"))->setEnabled(false);
    recentMenu_->addSeparator();
    recentMenu_->addAction(tr("Очистить список"), this, [this] { model_.clearRecents(); })->setEnabled(!recents.empty());
}

void MainWindow::chooseVideo() {
    const QString f = QFileDialog::getOpenFileName(this, tr("Видео"), model_.lastDir("video"), tr("Видео (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;Все файлы (*)"));
    if (f.isEmpty()) return;
    model_.rememberDir("video", f);
    model_.openVideo(f);
}

void MainWindow::chooseProject() {
    const QString f = QFileDialog::getOpenFileName(this, tr("Проект"), model_.lastDir("project"), tr("Проект dlssvid (*.dlssvid.json)"));
    if (f.isEmpty()) return;
    model_.rememberDir("project", f);
    model_.openProject(f);
}

void MainWindow::saveProjectDialog(bool forceDialog) {
    if (!model_.hasProject()) return;
    if (!forceDialog && model_.saveProject()) return;
    const Project& p = model_.project();
    QString start = model_.lastDir("project");
    if (!p.file.empty()) start = QString::fromStdWString(p.file.wstring());
    else if (!p.sourceVideo.empty()) start = QString::fromStdWString(std::filesystem::path(p.sourceVideo).replace_extension(".dlssvid.json").wstring());
    const QString f = QFileDialog::getSaveFileName(this, tr("Сохранить проект"), start, tr("Проект dlssvid (*.dlssvid.json)"));
    if (f.isEmpty()) return;
    model_.rememberDir("project", f);
    model_.saveProjectAs(f);
}

void MainWindow::updatePage() {
    const bool project = model_.hasProject();
    if (project) {
        pages_->setCurrentWidget(processingActive_ ? static_cast<QWidget*>(processing_) : workArea_);
        if (!workState_.isEmpty()) {
            restoreState(workState_);
            workState_.clear();
        } else {
            for (QDockWidget* d : {projectDock_, inspectorDock_, logDock_, tasksDock_}) d->show();
        }
        if (!docksSized_) {  // the first project: the dock wide enough for the stage cards
            resizeDocks({projectDock_}, {820}, Qt::Horizontal);
            docksSized_ = true;
        }
    } else {
        pages_->setCurrentWidget(startPage_);
        if (workState_.isEmpty()) workState_ = saveState();
        for (QDockWidget* d : {projectDock_, inspectorDock_, logDock_, tasksDock_}) d->hide();
        startPage_->refresh();
    }
}

void MainWindow::updateTitle() {
    QString title = "dlssvid";
    if (model_.hasProject()) {
        title += QString(" — %1").arg(QString::fromStdWString(model_.project().sourceVideo.filename().wstring()));
        if (model_.dirty()) title += "*";
    }
    setWindowTitle(title);
}

// 1..9 follow the chips of the compare bar (the order the user sees): the «after» side of a comparison, else the single view.
void MainWindow::selectSource(int hotkey) {
    const auto chips = model_.chips();
    if (hotkey < 1 || static_cast<size_t>(hotkey) > chips.size()) return;
    if (model_.state().mode == ViewMode::Grid) model_.setMode(ViewMode::Single);
    model_.showSource(chips[static_cast<size_t>(hotkey - 1)].source);
}

void MainWindow::updateCellLabels() {
    const ViewportState& st = model_.state();
    const bool grid = st.mode == ViewMode::Grid;
    for (size_t i = 0; i < 4; ++i) {
        cellLabels_[i]->setVisible(grid ? (st.expandedCell < 0 || st.expandedCell == static_cast<int>(i)) : i == 0);
        if (grid) {
            cellLabels_[i]->setText(cellLabel(HumanSourceName(st.gridSources[i]), st.gridSources[i]));
            cellLabels_[i]->setToolTip(QString::fromStdString(st.gridSources[i]));
        }
    }
    if (!grid) {
        if (const LayerState* before = st.CompareSide(false)) {
            const LayerState* after = st.CompareSide(true);
            cellLabels_[0]->setText(tr("ДО: %1   |   ПОСЛЕ: %2").arg(cellLabel(HumanSourceName(before->source), before->source, false),
                                                                       cellLabel(HumanSourceName(after->source), after->source)));
            cellLabels_[0]->setToolTip(QString::fromStdString(before->source + " | " + after->source));
        } else {
            QString t, tech;
            for (const auto& l : st.layers) {
                t += (t.isEmpty() ? "" : " + ") + HumanSourceName(l.source);
                tech += (tech.isEmpty() ? "" : " + ") + QString::fromStdString(l.source);
            }
            cellLabels_[0]->setText(cellLabel(t, model_.baseSource()));
            cellLabels_[0]->setToolTip(tech);
        }
    }
}

// «Исходник · 1920×800  #123 · 00:05.12», plus the state while the frame is not on screen (stage 9: no silent black cells).
QString MainWindow::cellLabel(const QString& title, const std::string& source, bool withTime) {
    const double t = model_.time();
    QString s = title;
    if (const ViewportSource* src = model_.store().FindSource(source); src && src->width > 0) s += QString(" · %1×%2").arg(src->width).arg(src->height);
    s += QString("  #%1").arg(model_.store().FrameAt(source, t));
    if (withTime) s += QString(" · %1").arg(QString::fromStdString(FormatTimecode(t)));
    switch (model_.frameStateOf(source)) {
        case FrameState::Loading: s += tr(" · загрузка…"); break;
        case FrameState::Missing: s += tr(" · нет кадра"); break;
        default: break;
    }
    return s;
}

void MainWindow::saveScreenshot() {
    const QString f = QFileDialog::getSaveFileName(this, tr("Скриншот"), QDir(model_.lastDir("screenshot")).filePath("viewport.png"), "PNG (*.png)");
    if (f.isEmpty()) return;
    model_.rememberDir("screenshot", f);
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

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    for (const QUrl& url : e->mimeData()->urls())
        if (url.isLocalFile() && IsOpenablePath(url.toLocalFile())) {
            e->acceptProposedAction();
            return;
        }
}

void MainWindow::dropEvent(QDropEvent* e) {
    for (const QUrl& url : e->mimeData()->urls())
        if (url.isLocalFile() && IsOpenablePath(url.toLocalFile())) {
            openPath(url.toLocalFile());
            e->acceptProposedAction();
            return;
        }
}

void MainWindow::watchProcess(int taskId, const QString& what) {
    processTaskId_ = taskId;
    processingActive_ = true;
    processing_->watch(taskId, what);
    updatePage();
}

void MainWindow::onTaskFinished(int taskId, bool ok) {
    if (taskId == processTaskId_ && ok) model_.setLastRun(tasks_->progress(taskId).elapsed);  // «готов за …» in the result line
    model_.clearForce();
    model_.reloadSources();  // the new passes / the result: sources, the plan, the cards
    if (taskId != processTaskId_) return;
    processTaskId_ = -1;
    const QString video = QString::fromStdWString(model_.project().sourceVideo.filename().wstring());
    const QString took = FormatClock(tasks_->progress(taskId).elapsed);
    if (ok) {
        model_.setCompareView(AppModel::CompareView::BeforeAfter);  // the result against the source
        processingActive_ = false;
        updatePage();
        statusBar()->showMessage(tr("Обработка завершена за %1 — «До | После»").arg(took), 10000);
        if (inBackground()) notify(tr("Обработка завершена"), tr("%1 — результат готов за %2").arg(video, took));
    } else {
        // the processing page stays: it says what happened and leads back to the project
        const bool cancelled = tasks_->cancelled(taskId);
        if (inBackground())
            notify(cancelled ? tr("Обработка отменена") : tr("Обработка не удалась"),
                   cancelled ? tr("%1 — готовые стадии сохранены").arg(video) : tr("%1 — подробности в логе").arg(video));
    }
}

bool MainWindow::inBackground() const { return isMinimized() || !isActiveWindow(); }

void MainWindow::notify(const QString& title, const QString& text) {
    if (!QSystemTrayIcon::isSystemTrayAvailable()) return;
    if (!tray_) {
        tray_ = new QSystemTrayIcon(this);
        QIcon icon = windowIcon();
        if (icon.isNull()) icon = style()->standardIcon(QStyle::SP_ComputerIcon);
        tray_->setIcon(icon);
        tray_->setToolTip("dlssvid");
        auto restore = [this] {
            setWindowState((windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
            raise();
            activateWindow();
        };
        connect(tray_, &QSystemTrayIcon::messageClicked, this, restore);
        connect(tray_, &QSystemTrayIcon::activated, this, [restore](QSystemTrayIcon::ActivationReason) { restore(); });
    }
    tray_->show();  // the icon must be visible for the message to appear
    tray_->showMessage(title, text, QSystemTrayIcon::Information, 10000);
    Log()->info("notification: {} / {}", title.toStdString(), text.toStdString());
}

bool MainWindow::event(QEvent* e) {
    if (e->type() == QEvent::WindowActivate && tray_ && !tasks_->busy()) tray_->hide();  // the user is back: no icon to leave behind
    return QMainWindow::event(e);
}

bool MainWindow::settleUnsaved() {
    switch (model_.closeAction()) {
        case AppModel::CloseAction::Nothing: return true;
        case AppModel::CloseAction::AutoSave:
            if (model_.saveProject()) return true;
            return QMessageBox::question(this, tr("Проект не сохранён"), tr("Не удалось сохранить проект. Продолжить без сохранения?")) == QMessageBox::Yes;
        case AppModel::CloseAction::Ask: {
            const auto r = QMessageBox::question(this, tr("Несохранённый проект"), tr("Сохранить проект?"),
                                                 QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
            if (r == QMessageBox::Cancel) return false;
            if (r == QMessageBox::Save) {
                saveProjectDialog(true);
                return !model_.dirty();
            }
            return true;
        }
    }
    return true;
}

void MainWindow::closeProject() {
    if (!model_.hasProject()) return;
    if (tasks_->busy()) {
        QMessageBox::information(this, tr("Идёт обработка"), tr("Дождитесь окончания обработки или отмените её, затем закройте проект."));
        return;
    }
    if (!settleUnsaved()) return;
    processingActive_ = false;
    model_.closeProject();  // projectChanged -> the start page
}

void MainWindow::saveResultAs() {
    const AppModel::ResultInfo r = model_.resultInfo();
    if (!r.exists) {
        statusBar()->showMessage(tr("Результата ещё нет — сначала «Обработать»"), 5000);
        return;
    }
    const QString last = model_.lastDir("result");
    const QString start = last.isEmpty() ? r.path : QDir(last).filePath(QFileInfo(r.path).fileName());
    const QString f = QFileDialog::getSaveFileName(this, tr("Сохранить результат как"), start, tr("Видео MP4 (*.mp4);;Все файлы (*)"));
    if (f.isEmpty()) return;
    model_.rememberDir("result", f);
    model_.exportResult(f);
}

void MainWindow::openResultFolder() {
    const AppModel::ResultInfo r = model_.resultInfo();
    const QString folder = r.exists ? QFileInfo(r.path).absolutePath() : (model_.hasProject() ? QString::fromStdWString(model_.project().sourceVideo.parent_path().wstring()) : QString());
    if (folder.isEmpty()) return;
    QDesktopServices::openUrl(QUrl::fromLocalFile(folder));
}

void MainWindow::exportPasses() {
    const std::vector<std::string> passes = model_.exportablePasses();
    if (passes.empty()) {
        statusBar()->showMessage(tr("Пассов для экспорта пока нет"), 5000);
        return;
    }
    const QString dir = QFileDialog::getExistingDirectory(this, tr("Папка для экспорта пассов (EXR-последовательности с manifest.json, по подпапке на пасс)"), model_.lastDir("export"));
    if (dir.isEmpty()) return;
    model_.rememberDir("export", QDir(dir).filePath("."));
    for (const std::string& pass : passes) {
        const QString from = QString::fromStdWString((model_.project().passesRoot / pass).wstring());
        tasks_->enqueue(tr("Экспорт: %1").arg(HumanSourceName(pass)), model_.cliPath(),
                        {"export", "--from-dir", from, "-o", QDir(dir).filePath(QString::fromStdString(pass)), "--format", "exr"});
    }
    tasksDock_->show();
    tasksDock_->raise();
    statusBar()->showMessage(tr("%1 в очереди → %2").arg(Plural(static_cast<int>(passes.size()), tr("задача экспорта"), tr("задачи экспорта"), tr("задач экспорта")), dir), 8000);
}

void MainWindow::saveSettings() {
    QSettings settings;
    settings.setValue("window/geometry", saveGeometry());
    settings.setValue("window/state", model_.hasProject() ? saveState() : workState_);
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (tasks_->busy()) {
        if (QMessageBox::question(this, tr("Идёт обработка"), tr("Прервать обработку и закрыть? Готовые стадии останутся на диске.")) != QMessageBox::Yes) {
            e->ignore();
            return;
        }
        tasks_->cancel(tasks_->currentId());
    }
    if (!settleUnsaved()) {
        e->ignore();
        return;
    }
    saveSettings();
    QMainWindow::closeEvent(e);
}

}  // namespace dlssvid
