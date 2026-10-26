#pragma once

#include <QMenu>
#include <QPushButton>
#include <string>

#include "PresetDialogs.h"
#include "PresetStore.h"

namespace dlssvid {

class AppModel;

// The button of a preset menu (PR-9): Space, Enter and Return without modifiers and Alt+↓ open the menu while it has the focus. The
// window's Space (playback, MainWindow) would take the key otherwise - QAbstractButton does not claim ShortcutOverride - and Enter
// does nothing on a QPushButton outside a dialog. Ctrl+Return («Обработать») and Shift+Space stay the window's.
class PresetButton : public QPushButton {
    Q_OBJECT
public:
    using QPushButton::QPushButton;
    static bool Opens(int key, Qt::KeyboardModifiers modifiers);

protected:
    bool event(QEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
};

// A menu of presets (PR-9, docs/ux-guidelines.md principle 11): the person's presets of the target with the difference each would make
// (tooltip and status tip) and a check at the one that matches, the automatic slots, «Сохранить текущие…», «Импорт…», «Экспорт ▸»,
// «Удалить ▸», «Для новых проектов ▸», the folder, «Не прочитаны». target: a stage with card presets ("nr") or kPipelineTarget. Built
// again on every aboutToShow: the folder is the source of truth (another copy of the app writes into it too).
class PresetMenu : public QMenu {
    Q_OBJECT
public:
    PresetMenu(AppModel& model, std::string target, QWidget* parent = nullptr);
    void setDialogs(PresetDialogs dialogs) { dialogs_ = std::move(dialogs); }
    const PresetDialogs& dialogs() const { return dialogs_; }
    const std::string& target() const { return target_; }
    void rebuild();

    // the actions as the menu calls them (also for the tests)
    void applyFile(const QString& file);
    void saveCurrent();
    void importFile();
    void exportFile(const QString& file);  // "" = the current settings
    void removeFile(const QString& file, const QString& name);
    void markForNewProjects(const QString& file);  // "" = the built-in defaults

private:
    bool pipeline() const;
    QWidget* dialogParent() const;
    QString preview(const Preset& preset, QString* statusLine) const;  // the tooltip; *statusLine: its one line
    QString markText() const;  // the mark's description when it is on a preset of the other kind; "" otherwise

    AppModel& model_;
    std::string target_;
    PresetDialogs dialogs_;
};

}  // namespace dlssvid
