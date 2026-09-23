#include "StartPage.h"

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMimeData>
#include <QMouseEvent>
#include <QVBoxLayout>

#include "AppModel.h"
#include "Theme.h"

namespace dlssvid {

bool IsOpenablePath(const QString& path) {
    const QString lower = path.toLower();
    for (const char* ext : {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".dlssvid.json", ".json"})
        if (lower.endsWith(ext)) return true;
    return false;
}

StartPage::StartPage(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    setAcceptDrops(true);
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* column = new QWidget(this);
    column->setMaximumWidth(760);
    auto* layout = new QVBoxLayout(column);
    layout->setContentsMargins(40, 48, 40, 24);
    layout->setSpacing(20);
    auto* title = new QLabel(tr("Улучшить видео с DLSS"), column);
    SetRole(title, "h1");
    layout->addWidget(title);
    auto* subtitle = new QLabel(tr("Апскейл ×2, Neural Rendering и генерация кадров для готового ролика. Откройте видео — стадии уже настроены по умолчанию."), column);
    subtitle->setWordWrap(true);
    SetRole(subtitle, "lead");
    layout->addWidget(subtitle);

    dropZone_ = new QFrame(column);
    dropZone_->setObjectName("dropZone");
    SetRole(dropZone_, "dropzone");
    dropZone_->setMinimumHeight(168);
    dropZone_->setCursor(Qt::PointingHandCursor);
    dropZone_->installEventFilter(this);
    auto* dz = new QVBoxLayout(dropZone_);
    dz->setContentsMargins(24, 30, 24, 30);
    dz->setSpacing(8);
    auto* dropTitle = new QLabel(tr("Перетащите видео сюда"), dropZone_);
    dropTitle->setAlignment(Qt::AlignCenter);
    dropTitle->setStyleSheet("font-size: 16px; font-weight: 500;");
    auto* dropHint = new QLabel(tr("MP4, MOV, MKV, AVI · или нажмите, чтобы выбрать файл"), dropZone_);
    dropHint->setAlignment(Qt::AlignCenter);
    SetRole(dropHint, "muted");
    dz->addWidget(dropTitle);
    dz->addWidget(dropHint);
    layout->addWidget(dropZone_);

    auto* buttons = new QHBoxLayout();
    auto* openVideo = new QPushButton(tr("Открыть видео…"), column);
    openVideo->setDefault(true);
    SetRole(openVideo, "start-primary");
    auto* openProject = new QPushButton(tr("Открыть проект…"), column);
    SetRole(openProject, "start");
    buttons->setSpacing(12);
    buttons->addWidget(openVideo);
    buttons->addWidget(openProject);
    buttons->addStretch(1);
    layout->addLayout(buttons);
    connect(openVideo, &QPushButton::clicked, this, &StartPage::openVideoRequested);
    connect(openProject, &QPushButton::clicked, this, &StartPage::openProjectRequested);

    readiness_ = new QLabel(column);
    readiness_->setWordWrap(true);
    SetRole(readiness_, "muted");
    layout->addWidget(readiness_);

    auto* recentsHead = new QHBoxLayout();
    auto* recentsTitle = new QLabel(tr("Недавние"), column);
    SetRole(recentsTitle, "title");
    clearRecents_ = new QPushButton(tr("очистить"), column);
    clearRecents_->setFlat(true);
    recentsHead->addWidget(recentsTitle);
    recentsHead->addStretch(1);
    recentsHead->addWidget(clearRecents_);
    layout->addLayout(recentsHead);
    recents_ = new QListWidget(column);
    recents_->setAlternatingRowColors(true);
    recents_->setToolTip(tr("Двойной щелчок открывает"));
    layout->addWidget(recents_, 1);
    connect(clearRecents_, &QPushButton::clicked, this, [this] { model_.clearRecents(); });
    connect(recents_, &QListWidget::itemActivated, this, [this](QListWidgetItem* item) {
        if (item) emit openPathRequested(item->data(Qt::UserRole).toString());
    });

    footer_ = new QLabel(tr("Готов · пассы по умолчанию: рядом с видео"), column);
    SetRole(footer_, "hint");
    layout->addWidget(footer_);

    outer->addWidget(column, 0, Qt::AlignHCenter | Qt::AlignTop);
    outer->addStretch(1);
    connect(&model_, &AppModel::recentsChanged, this, &StartPage::refresh);
    refresh();
}

void StartPage::refresh() {
    readiness_->setText(model_.readiness());
    recents_->clear();
    const auto recents = model_.recents();
    for (const AppModel::Recent& r : recents) {
        auto* item = new QListWidgetItem(QString("%1 — %2").arg(r.title, r.info), recents_);
        item->setData(Qt::UserRole, r.path);
        item->setToolTip(r.path);
        if (!r.exists) item->setForeground(QColor("#8a8f98"));
    }
    if (recents.empty()) {
        auto* item = new QListWidgetItem(tr("пока пусто"), recents_);
        item->setFlags(Qt::NoItemFlags);
    }
    clearRecents_->setVisible(!recents.empty());
}

void StartPage::dragEnterEvent(QDragEnterEvent* e) {
    for (const QUrl& url : e->mimeData()->urls())
        if (url.isLocalFile() && IsOpenablePath(url.toLocalFile())) {
            e->acceptProposedAction();
            return;
        }
}

void StartPage::dropEvent(QDropEvent* e) {
    for (const QUrl& url : e->mimeData()->urls())
        if (url.isLocalFile() && IsOpenablePath(url.toLocalFile())) {
            emit openPathRequested(url.toLocalFile());
            e->acceptProposedAction();
            return;
        }
}

bool StartPage::eventFilter(QObject* watched, QEvent* e) {
    if (watched == dropZone_ && e->type() == QEvent::MouseButtonRelease) {
        if (static_cast<QMouseEvent*>(e)->button() == Qt::LeftButton) emit openVideoRequested();
        return true;
    }
    return QWidget::eventFilter(watched, e);
}

}  // namespace dlssvid
