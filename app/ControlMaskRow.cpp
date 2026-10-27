#include "ControlMaskRow.h"

#include <QHBoxLayout>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "AppModel.h"
#include "Theme.h"
#include "passes/ControlMaskPass.h"

namespace dlssvid {

namespace {
// A file name for the narrow field column: a long one keeps its start and its end.
QString ShortName(const QString& name) { return name.size() <= 34 ? name : name.left(16) + QString::fromUtf8("…") + name.right(17); }
}  // namespace

ControlMaskRow::ControlMaskRow(AppModel& model, QWidget* parent) : QWidget(parent), model_(model), dialogs_(PresetDialogs::Default()) {
    auto* lines = new QVBoxLayout(this);
    lines->setContentsMargins(0, 0, 0, 0);
    lines->setSpacing(2);
    // the state on a line of its own, the buttons under it at their full width (side by side the file name squeezed them)
    status_ = new QLabel(this);  // two lines of its own (the state; the file or why): no wrap, which a form row clipped
    lines->addWidget(status_);
    auto* h = new QHBoxLayout();
    h->setSpacing(6);
    lines->addLayout(h);
    auto keep = [](QWidget* w) {  // hidden buttons keep their place: the row does not jump
        QSizePolicy p = w->sizePolicy();
        p.setRetainSizeWhenHidden(true);
        w->setSizePolicy(p);
    };
    load_ = new QPushButton(tr("Загрузить…"), this);
    show_ = new QPushButton(tr("Показать"), this);
    remove_ = new QPushButton(tr("Убрать"), this);
    for (QPushButton* b : {load_, show_, remove_}) {
        SetRole(b, "small");
        keep(b);
        h->addWidget(b);
    }
    h->addStretch(1);
    load_->setAccessibleName(tr("Загрузить маску управления"));
    show_->setAccessibleName(tr("Показать маску управления"));
    remove_->setAccessibleName(tr("Убрать маску управления"));
    note_ = new QLabel(this);
    SetRole(note_, "hint");
    note_->setWordWrap(true);
    note_->hide();
    lines->addWidget(note_);
    connect(load_, &QPushButton::clicked, this, &ControlMaskRow::load);
    connect(show_, &QPushButton::clicked, this, [this] { model_.showSource(kControlMaskPass); });
    connect(remove_, &QPushButton::clicked, this, &ControlMaskRow::remove);
    connect(&model_, &AppModel::controlMaskFinished, this, &ControlMaskRow::finished);
}

void ControlMaskRow::refresh(bool busy, bool bypass) {
    busy_ = busy;
    bypass_ = bypass;
    const AppModel::ControlMaskState& s = model_.controlMask();
    const bool root = model_.hasProject() && !model_.project().passesRoot.empty();
    QString status, tip, note;
    const char* role = "muted";
    bool loadable = root, showable = false, removable = false;
    load_->setText(tr("Загрузить…"));
    switch (s.status) {
        case AppModel::ControlMaskState::Status::None:
            status = tr("Не загружена");
            if (!root) note = tr("Сначала откройте видео: маска хранится в папке пассов проекта");
            break;
        case AppModel::ControlMaskState::Status::Importing:
            status = tr("Загружается…") + "\n" + ShortName(s.file);
            tip = s.file;
            loadable = false;
            break;
        case AppModel::ControlMaskState::Status::Ready: {
            const QString what = tr("%1 · %2×%3").arg(s.grayscale ? tr("серая") : QStringLiteral("RGB")).arg(s.width).arg(s.height);
            status = (bypass ? tr("Загружена · %1") : tr("Активна · %1")).arg(what) + "\n" + ShortName(s.file);
            role = bypass ? "muted" : "";
            load_->setText(tr("Заменить…"));
            showable = removable = true;
            QStringList t{s.file, QStringLiteral("sha256:") + s.fingerprint};
            if (s.alphaIgnored) t << tr("альфа не учтена");
            if (s.clamped > 0) t << tr("обрезано значений: %1").arg(s.clamped);
            tip = t.join(QStringLiteral(" · "));
            if (bypass) note = tr("не действует при интенсивности 0: модель не вызывается");
            break;
        }
        case AppModel::ControlMaskState::Status::Unreadable:
            status = tr("Не читается") + "\n" + s.problem;
            role = "warn";
            removable = true;
            note = s.broken ? tr("«Убрать» удалит папку маски") : tr("Загрузите маску заново или уберите её");
            break;
    }
    if (!notice_.isEmpty()) note = note.isEmpty() ? notice_ : note + " · " + notice_;
    status_->setText(status);
    status_->setToolTip(tip);
    SetRole(status_, role);
    const bool importing = s.status == AppModel::ControlMaskState::Status::Importing;
    const QString busyTip = busy && !importing ? tr("Недоступно во время обработки") : QString();
    load_->setEnabled(loadable && !busy);
    load_->setToolTip(busyTip.isEmpty() ? tr("Выбрать изображение маски (PNG, TIFF, EXR). Прежняя маска уйдёт в историю версий.") : busyTip);
    show_->setVisible(showable);
    show_->setEnabled(showable && !busy);
    show_->setToolTip(busyTip.isEmpty() ? tr("Показать маску во вьюпорте") : busyTip);
    remove_->setVisible(removable);
    remove_->setEnabled(removable && !busy);
    remove_->setToolTip(busyTip.isEmpty() ? (s.broken ? tr("Папка маски повреждена: она будет удалена, исходный файл не меняется")
                                                       : tr("Маска уйдёт в историю версий, исходный файл не меняется"))
                                          : busyTip);
    note_->setText(note);
    note_->setVisible(!note.isEmpty());
}

void ControlMaskRow::load() {
    const QString file = dialogs_.openFile(window(), tr("Маска управления"), model_.lastDir("mask"), tr("Изображения (*.png *.tif *.tiff *.exr);;Все файлы (*)"));
    if (file.isEmpty()) return;
    notice_.clear();
    pending_ = true;
    model_.importControlMask(file);
}

void ControlMaskRow::remove() {
    const AppModel::ControlMaskState& s = model_.controlMask();
    const bool broken = s.broken;
    const QString tail = tr("вернуть её можно в панели версий. Исходный файл не меняется. «Улучшение» посчитается заново без маски.");
    const QString text = broken           ? tr("Папка маски повреждена и будет удалена. Исходный файл не меняется.")
                         : s.file.isEmpty() ? tr("Маска уйдёт в историю версий — %1").arg(tail)
                                            : tr("Маска «%1» уйдёт в историю версий — %2").arg(s.file, tail);
    if (!dialogs_.confirm(window(), tr("Убрать маску управления?"), text, broken ? tr("Удалить") : tr("Убрать"))) return;
    notice_.clear();
    QString error;
    if (!model_.removeControlMask(&error)) dialogs_.warn(window(), tr("Маска не убрана"), error);
}

void ControlMaskRow::finished(bool ok, bool changed, const QString& message) {
    if (!pending_) return;  // another row's import (a second card of the stage)
    pending_ = false;
    if (!ok) dialogs_.warn(window(), tr("Маска не загружена"), message);
    else if (!changed) notice_ = message;  // «эта маска уже загружена»: the card shows it on controlMaskChanged, emitted next
}

}  // namespace dlssvid
