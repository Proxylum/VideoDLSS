#pragma once

#include <QDialog>
#include <QString>
#include <functional>
#include <optional>

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QWidget;

namespace dlssvid {

// «Сохранить текущие…» of a preset menu (PR-9): the name, «Применять к новым проектам», and before anything is written the name's
// problem or «будет заменён», the mark it would move and what the preset leaves out.
class PresetSaveDialog : public QDialog {
    Q_OBJECT
public:
    struct Setup {
        QString title;     // «Сохранить настройки «Улучшения»» / «Сохранить настройки обработки»
        QString name;      // the initial text of the field
        QString markNote;  // «Сейчас для новых проектов: … — будет заменено»; "" when the mark is not the other kind's
        QString leftOut;   // «Не войдут в пресет: …»; "" when nothing is left out
        std::function<QString(const QString&)> problem;  // the name's problem in Russian, "" when fine
        std::function<bool(const QString&)> exists;      // a preset of this kind has the name: the button says «Заменить»
    };
    struct Answer {
        QString name;
        bool forNewProjects = false;
    };
    PresetSaveDialog(const Setup& setup, QWidget* parent = nullptr);
    Answer answer() const;
    QLineEdit* nameEdit() const { return name_; }
    QCheckBox* newProjects() const { return newProjects_; }
    QPushButton* saveButton() const { return save_; }
    QLabel* nameNote() const { return nameNote_; }
    QLabel* markNote() const { return markNote_; }
    QLabel* leftOut() const { return leftOut_; }

private:
    void update();
    Setup setup_;
    QLineEdit* name_;
    QLabel* nameNote_;
    QCheckBox* newProjects_;
    QLabel* markNote_;
    QLabel* leftOut_;
    QPushButton* save_;
};

// The dialogs of the preset menus and of «Сбросить» as functions: the real ones by default, the tests put their own (a modal dialog
// offscreen blocks). The model and the store only get names and paths.
struct PresetDialogs {
    enum class Clash { Replace, Copy, Cancel };
    std::function<std::optional<PresetSaveDialog::Answer>(QWidget*, const PresetSaveDialog::Setup&)> saveName;
    std::function<QString(QWidget*, const QString& title, const QString& dir, const QString& filter)> openFile;
    std::function<QString(QWidget*, const QString& title, const QString& path, const QString& filter)> saveFile;
    std::function<bool(QWidget*, const QString& title, const QString& text, const QString& yes)> confirm;  // [yes] [Отмена]
    std::function<Clash(QWidget*, const QString& name, const QString& copyName)> clash;  // [Заменить] [Сохранить как «copy»] [Отмена]
    std::function<void(QWidget*, const QString& title, const QString& text)> warn;
    std::function<void(QWidget*, const QString& title, const QString& text)> inform;
    static PresetDialogs Default();
};

}  // namespace dlssvid
