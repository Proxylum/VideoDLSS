#include "PresetDialogs.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include "Theme.h"
#include "stages/StagePreset.h"

namespace dlssvid {

PresetSaveDialog::PresetSaveDialog(const Setup& setup, QWidget* parent) : QDialog(parent), setup_(setup) {
    setWindowTitle(setup.title);
    auto* v = new QVBoxLayout(this);
    auto* label = new QLabel(tr("&Название"), this);
    name_ = new QLineEdit(setup.name, this);
    name_->setMaxLength(static_cast<int>(kPresetNameMax));
    label->setBuddy(name_);
    nameNote_ = new QLabel(this);
    nameNote_->setTextFormat(Qt::PlainText);  // names come from files: never markup
    nameNote_->setWordWrap(true);
    SetRole(nameNote_, "hint");
    newProjects_ = new QCheckBox(tr("Применять к новым проектам"), this);
    newProjects_->setToolTip(tr("Новые проекты (видео без файла проекта рядом) получат эти настройки"));
    markNote_ = new QLabel(setup.markNote, this);
    markNote_->setTextFormat(Qt::PlainText);
    markNote_->setWordWrap(true);
    SetRole(markNote_, "hint");
    markNote_->setVisible(false);
    leftOut_ = new QLabel(setup.leftOut, this);
    leftOut_->setTextFormat(Qt::PlainText);
    leftOut_->setWordWrap(true);
    SetRole(leftOut_, "hint");
    leftOut_->setVisible(!setup.leftOut.isEmpty());
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
    save_ = buttons->button(QDialogButtonBox::Save);
    buttons->button(QDialogButtonBox::Cancel)->setText(tr("Отмена"));
    v->addWidget(label);
    v->addWidget(name_);
    v->addWidget(nameNote_);
    v->addWidget(newProjects_);
    v->addWidget(markNote_);
    v->addWidget(leftOut_);
    v->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(name_, &QLineEdit::textChanged, this, [this] { update(); });
    connect(newProjects_, &QCheckBox::toggled, this, [this] { update(); });
    setMinimumWidth(420);
    name_->setFocus();
    update();
}

PresetSaveDialog::Answer PresetSaveDialog::answer() const { return {name_->text().trimmed(), newProjects_->isChecked()}; }

void PresetSaveDialog::update() {
    const QString name = name_->text().trimmed();
    const QString problem = setup_.problem ? setup_.problem(name) : QString();
    const bool exists = problem.isEmpty() && setup_.exists && setup_.exists(name);
    nameNote_->setText(!problem.isEmpty() ? problem : exists ? tr("Пресет «%1» уже есть — будет заменён").arg(name) : QString());
    nameNote_->setVisible(!nameNote_->text().isEmpty());
    save_->setText(exists ? tr("Заменить") : tr("Сохранить"));
    save_->setEnabled(problem.isEmpty());
    markNote_->setVisible(newProjects_->isChecked() && !setup_.markNote.isEmpty());
}

PresetDialogs PresetDialogs::Default() {
    PresetDialogs d;
    d.saveName = [](QWidget* parent, const PresetSaveDialog::Setup& setup) -> std::optional<PresetSaveDialog::Answer> {
        PresetSaveDialog dialog(setup, parent);
        if (dialog.exec() != QDialog::Accepted) return std::nullopt;
        return dialog.answer();
    };
    d.openFile = [](QWidget* parent, const QString& title, const QString& dir, const QString& filter) {
        return QFileDialog::getOpenFileName(parent, title, dir, filter);
    };
    d.saveFile = [](QWidget* parent, const QString& title, const QString& path, const QString& filter) {
        return QFileDialog::getSaveFileName(parent, title, path, filter);
    };
    d.confirm = [](QWidget* parent, const QString& title, const QString& text, const QString& yes) {
        QMessageBox box(QMessageBox::Question, title, text, QMessageBox::NoButton, parent);
        QPushButton* ok = box.addButton(yes, QMessageBox::AcceptRole);
        box.addButton(QObject::tr("Отмена"), QMessageBox::RejectRole);
        box.setDefaultButton(ok);
        box.setTextFormat(Qt::PlainText);
        box.exec();
        return box.clickedButton() == ok;
    };
    d.clash = [](QWidget* parent, const QString& name, const QString& copyName) {
        QMessageBox box(QMessageBox::Question, QObject::tr("Пресет уже есть"), QObject::tr("Пресет «%1» уже есть.").arg(name), QMessageBox::NoButton, parent);
        QPushButton* replace = box.addButton(QObject::tr("Заменить"), QMessageBox::AcceptRole);
        QPushButton* copy = box.addButton(QObject::tr("Сохранить как «%1»").arg(QString(copyName).replace("&", "&&")), QMessageBox::AcceptRole);  // no mnemonic
        box.addButton(QObject::tr("Отмена"), QMessageBox::RejectRole);
        box.setTextFormat(Qt::PlainText);
        box.exec();
        return box.clickedButton() == replace ? Clash::Replace : box.clickedButton() == copy ? Clash::Copy : Clash::Cancel;
    };
    // the texts name presets from files: plain text, so «<b>Ночь</b>» reads as it is written (QMessageBox guesses markup otherwise)
    d.warn = [](QWidget* parent, const QString& title, const QString& text) {
        QMessageBox box(QMessageBox::Warning, title, text, QMessageBox::Ok, parent);
        box.setTextFormat(Qt::PlainText);
        box.exec();
    };
    d.inform = [](QWidget* parent, const QString& title, const QString& text) {
        QMessageBox box(QMessageBox::Information, title, text, QMessageBox::Ok, parent);
        box.setTextFormat(Qt::PlainText);
        box.exec();
    };
    return d;
}

}  // namespace dlssvid
