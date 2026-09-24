#include "StartPage.h"

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScrollArea>
#include <QVBoxLayout>

#include "AppModel.h"
#include "RecentPreviews.h"
#include "Theme.h"

namespace dlssvid {

bool IsOpenablePath(const QString& path) {
    const QString lower = path.toLower();
    for (const char* ext : {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".dlssvid.json", ".json"})
        if (lower.endsWith(ext)) return true;
    return false;
}

// A label that elides its text at its width instead of being clipped by the card's edge; keeps the full text.
class ElidedLabel : public QLabel {
public:
    explicit ElidedLabel(QWidget* parent) : QLabel(parent) { setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred); }
    void setFullText(const QString& text) {
        full_ = text;
        setToolTip(text);
        elide();
    }
    QString fullText() const { return full_; }

protected:
    void resizeEvent(QResizeEvent* e) override {
        QLabel::resizeEvent(e);
        elide();
    }

private:
    void elide() { setText(fontMetrics().elidedText(full_, Qt::ElideRight, std::max(16, width()))); }
    QString full_;
};

// «вчера 21:22» → «вчера», «22.09 19:12» → «22.09»: the card shows the day as the mockup does, the tooltip keeps the time.
QString DayOnly(const QString& when) {
    static const QRegularExpression time("\\s\\d{1,2}:\\d{2}$");
    QString out = when;
    out.remove(time);
    return out;
}

// One recent file (mockup: preview 96×40 · «face — результат готов» / «1920×800 24 fps → 3840×1600 48 fps · вчера»).
class RecentCard : public QFrame {
    Q_OBJECT
public:
    RecentCard(const AppModel::Recent& recent, QWidget* parent) : QFrame(parent), recent_(recent) {
        SetRole(this, "recent");
        setAttribute(Qt::WA_Hover);
        setToolTip(recent.path + "\n" + recent.info);
        if (recent.exists) setCursor(Qt::PointingHandCursor);
        auto* row = new QHBoxLayout(this);
        row->setContentsMargins(12, 10, 12, 10);
        row->setSpacing(12);
        preview_ = new QLabel(this);
        SetRole(preview_, "preview");
        preview_->setFixedSize(RecentPreviews::kWidth, RecentPreviews::kHeight);
        row->addWidget(preview_, 0, Qt::AlignVCenter);
        auto* text = new QVBoxLayout();
        text->setContentsMargins(0, 0, 0, 0);
        text->setSpacing(3);
        title_ = new ElidedLabel(this);
        title_->setFullText(recent.state.isEmpty() ? recent.title : QString("%1 — %2").arg(recent.title, recent.state));
        SetRole(title_, recent.exists ? "recent-title" : "recent-title-missing");
        meta_ = new ElidedLabel(this);
        SetRole(meta_, "recent-meta");
        text->addWidget(title_);
        text->addWidget(meta_);
        row->addLayout(text, 1);
        setMeta({});
    }
    const AppModel::Recent& recent() const { return recent_; }
    QString title() const { return title_->fullText(); }
    QString meta() const { return meta_->fullText(); }
    bool hasPreview() const { return hasPreview_; }

    // «<size line> · <progress> · <when>»: the size line comes with the preview, the rest is known at once.
    void setMeta(const QString& sizeLine) {
        QStringList parts;
        if (!sizeLine.isEmpty()) parts << sizeLine;
        if (!recent_.progress.isEmpty()) parts << recent_.progress;
        if (!recent_.when.isEmpty()) parts << DayOnly(recent_.when);
        if (!recent_.exists) parts << recent_.info;
        meta_->setFullText(parts.join(" · "));
    }
    void setPreview(const QImage& image) {
        if (image.isNull()) return;
        // the frame with 3 px corners, at the device pixel ratio of the screen
        const qreal dpr = devicePixelRatioF();
        QPixmap pm(QSize(RecentPreviews::kWidth, RecentPreviews::kHeight) * dpr);
        pm.setDevicePixelRatio(dpr);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        QPainterPath clip;
        clip.addRoundedRect(QRectF(0, 0, RecentPreviews::kWidth, RecentPreviews::kHeight), 3, 3);
        p.setClipPath(clip);
        p.drawImage(QRectF(0, 0, RecentPreviews::kWidth, RecentPreviews::kHeight), image);
        p.end();
        preview_->setPixmap(pm);
        hasPreview_ = true;
    }

signals:
    void activated(const QString& path);

protected:
    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton && rect().contains(e->pos()) && recent_.exists) emit activated(recent_.path);
        QFrame::mouseReleaseEvent(e);
    }

private:
    AppModel::Recent recent_;
    QLabel* preview_;
    ElidedLabel* title_;
    ElidedLabel* meta_;
    bool hasPreview_ = false;
};

StartPage::StartPage(AppModel& model, QWidget* parent) : QWidget(parent), model_(model), previews_(new RecentPreviews(this)) {
    setAcceptDrops(true);
    // mockup: the two columns (560 and 420 px, 64 px apart) centred in the page; the right one is 440 here, see below
    auto* outer = new QHBoxLayout(this);
    outer->setContentsMargins(40, 40, 40, 40);
    outer->setSpacing(64);
    outer->addStretch(1);

    auto* left = new QWidget(this);
    left->setFixedWidth(560);
    auto* column = new QVBoxLayout(left);
    column->setContentsMargins(0, 0, 0, 0);
    column->setSpacing(20);
    auto* heading = new QVBoxLayout();
    heading->setSpacing(6);
    auto* title = new QLabel(tr("Улучшить видео с DLSS"), left);
    SetRole(title, "h1");
    heading->addWidget(title);
    auto* subtitle = new QLabel(tr("Апскейл ×2, Neural Rendering и генерация кадров для готового ролика. Откройте видео — стадии уже настроены по умолчанию."), left);
    subtitle->setWordWrap(true);
    SetRole(subtitle, "lead");
    heading->addWidget(subtitle);
    column->addLayout(heading);

    dropZone_ = new QFrame(left);
    dropZone_->setObjectName("dropZone");
    SetRole(dropZone_, "dropzone");
    dropZone_->setFixedHeight(168);
    dropZone_->setCursor(Qt::PointingHandCursor);
    dropZone_->installEventFilter(this);
    auto* dz = new QVBoxLayout(dropZone_);
    dz->setContentsMargins(24, 20, 24, 20);
    dz->setSpacing(8);
    dz->addStretch(1);
    auto* dropIcon = new QLabel(dropZone_);
    dropIcon->setPixmap(ThemeIcon(Glyph::Upload, ThemeAccent()).pixmap(34, 34));
    dropIcon->setAlignment(Qt::AlignCenter);
    auto* dropTitle = new QLabel(tr("Перетащите видео сюда"), dropZone_);
    dropTitle->setAlignment(Qt::AlignCenter);
    SetRole(dropTitle, "dropzone-title");
    auto* dropHint = new QLabel(tr("MP4, MOV, MKV, AVI · или нажмите, чтобы выбрать файл"), dropZone_);
    dropHint->setAlignment(Qt::AlignCenter);
    SetRole(dropHint, "hint-13");
    dz->addWidget(dropIcon);
    dz->addWidget(dropTitle);
    dz->addWidget(dropHint);
    dz->addStretch(1);
    column->addWidget(dropZone_);

    auto* buttons = new QHBoxLayout();
    buttons->setSpacing(12);
    auto* openVideo = new QPushButton(tr("Открыть видео…"), left);
    openVideo->setDefault(true);
    SetRole(openVideo, "start-primary");
    auto* openProject = new QPushButton(tr("Открыть проект…"), left);
    SetRole(openProject, "start");
    buttons->addWidget(openVideo);
    buttons->addWidget(openProject);
    buttons->addStretch(1);
    column->addLayout(buttons);
    connect(openVideo, &QPushButton::clicked, this, &StartPage::openVideoRequested);
    connect(openProject, &QPushButton::clicked, this, &StartPage::openProjectRequested);

    auto* readinessCard = new QFrame(left);
    SetRole(readinessCard, "card");
    auto* rc = new QHBoxLayout(readinessCard);
    rc->setContentsMargins(14, 12, 14, 12);
    rc->setSpacing(10);
    readinessDot_ = new QLabel(readinessCard);
    SetRole(readinessDot_, "dot-ok");
    readiness_ = new QLabel(readinessCard);
    readiness_->setWordWrap(true);
    SetRole(readiness_, "text-13");
    rc->addWidget(readinessDot_, 0, Qt::AlignVCenter);
    rc->addWidget(readiness_, 1);
    column->addWidget(readinessCard);
    outer->addWidget(left, 0, Qt::AlignVCenter);

    auto* right = new QWidget(this);
    right->setFixedWidth(440);  // mockup: 420; +20 so «1920×800 24 fps → 3840×1600 48 fps · вчера» fits the Windows mono font
    auto* rcol = new QVBoxLayout(right);
    rcol->setContentsMargins(0, 0, 0, 0);
    rcol->setSpacing(10);
    auto* recentsHead = new QHBoxLayout();
    auto* recentsTitle = new QLabel(tr("Недавние"), right);
    SetRole(recentsTitle, "section");
    clearRecents_ = new QPushButton(tr("очистить"), right);
    clearRecents_->setFlat(true);
    clearRecents_->setCursor(Qt::PointingHandCursor);
    recentsHead->addWidget(recentsTitle, 0, Qt::AlignBaseline);
    recentsHead->addStretch(1);
    recentsHead->addWidget(clearRecents_, 0, Qt::AlignBaseline);
    rcol->addLayout(recentsHead);
    auto* scroll = new QScrollArea(right);
    scroll->setObjectName("recentsScroll");
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    cardsHost_ = new QWidget(scroll);
    auto* cards = new QVBoxLayout(cardsHost_);
    cards->setContentsMargins(0, 0, 0, 0);
    cards->setSpacing(10);
    empty_ = new QLabel(tr("пока пусто"), cardsHost_);
    SetRole(empty_, "muted");
    cards->addWidget(empty_);
    cards->addStretch(1);
    scroll->setWidget(cardsHost_);
    rcol->addWidget(scroll);
    outer->addWidget(right, 0, Qt::AlignVCenter);
    outer->addStretch(1);

    connect(clearRecents_, &QPushButton::clicked, this, [this] { model_.clearRecents(); });
    connect(&model_, &AppModel::recentsChanged, this, &StartPage::refresh);
    connect(previews_, &RecentPreviews::previewReady, this, [this](const QString& video) {
        for (RecentCard* card : cards_)
            if (card->recent().video == video) applyPreview(card);
    });
    refresh();
}

void StartPage::refresh() {
    readiness_->setText(model_.readiness());
    switch (model_.readinessLevel()) {
        case AppModel::Readiness::Ready: SetRole(readinessDot_, "dot-ok"); break;
        case AppModel::Readiness::Partial: SetRole(readinessDot_, "dot-warn"); break;
        case AppModel::Readiness::Software: SetRole(readinessDot_, "dot-danger"); break;
    }
    for (RecentCard* card : cards_) delete card;
    cards_.clear();
    auto* layout = static_cast<QVBoxLayout*>(cardsHost_->layout());
    const auto recents = model_.recents();
    int at = 0;
    for (const AppModel::Recent& r : recents) {
        auto* card = new RecentCard(r, cardsHost_);
        connect(card, &RecentCard::activated, this, &StartPage::openPathRequested);
        layout->insertWidget(at++, card);
        cards_.push_back(card);
        applyPreview(card);
    }
    empty_->setVisible(recents.empty());
    clearRecents_->setVisible(!recents.empty());
}

void StartPage::applyPreview(RecentCard* card) {
    const AppModel::Recent& r = card->recent();
    if (!r.exists || r.video.isEmpty()) return;
    if (const auto p = previews_->get(r.video, r.result)) {
        card->setMeta(p->meta);
        card->setPreview(p->image);
    }
}

QStringList StartPage::cardTitles() const {
    QStringList out;
    for (const RecentCard* c : cards_) out << c->title();
    return out;
}

QStringList StartPage::cardMetas() const {
    QStringList out;
    for (const RecentCard* c : cards_) out << c->meta();
    return out;
}

bool StartPage::cardHasPreview(int index) const { return index >= 0 && index < static_cast<int>(cards_.size()) && cards_[static_cast<size_t>(index)]->hasPreview(); }

QString StartPage::readinessText() const { return readiness_->text(); }

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

#include "StartPage.moc"
