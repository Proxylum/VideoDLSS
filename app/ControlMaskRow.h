#pragma once

#include <QLabel>
#include <QPushButton>
#include <QString>
#include <QWidget>
#include <filesystem>

#include "PresetDialogs.h"

namespace dlssvid {

class AppModel;

// The row «Маска управления» of the NR card (PR-8, docs/ux-guidelines.md): the mask's state, «Загрузить…» / «Заменить…»,
// «Показать» (the viewport) and «Убрать» (into the history), a note under them. The import runs on the model's thread; the
// buttons are off meanwhile and while a task runs. The dialogs go through PresetDialogs: the tests answer them.
class ControlMaskRow : public QWidget {
    Q_OBJECT
public:
    explicit ControlMaskRow(AppModel& model, QWidget* parent = nullptr);
    void setDialogs(PresetDialogs dialogs) { dialogs_ = std::move(dialogs); }
    // From the card's refresh: busy - a task runs or a mask is being imported; bypass - the stage's intensity is 0
    void refresh(bool busy, bool bypass);

    QLabel* status() const { return status_; }
    QPushButton* loadButton() const { return load_; }
    QPushButton* showButton() const { return show_; }
    QPushButton* removeButton() const { return remove_; }
    QLabel* note() const { return note_; }

private:
    void load();
    void remove();
    void finished(bool ok, bool changed, const QString& message);

    AppModel& model_;
    PresetDialogs dialogs_;
    QLabel* status_;
    QPushButton* load_;
    QPushButton* show_;
    QPushButton* remove_;
    QLabel* note_;
    QString notice_;        // «эта маска уже загружена»: while the same mask of the same project stays loaded
    QString noticeKey_;     // the passes root and the fingerprint it was said of
    bool pending_ = false;  // this row started the import in flight: it answers its end
    std::filesystem::path pendingRoot_;  // the passes root it was for: another project open at its end shows no note of it
    bool busy_ = false, bypass_ = false;
};

}  // namespace dlssvid
