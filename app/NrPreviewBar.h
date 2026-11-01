#pragma once

#include <QWidget>
#include <functional>

class QCheckBox;
class QComboBox;
class QLabel;
class QMenu;
class QPushButton;
class QTimer;

namespace dlssvid {

class AppModel;
class PresetButton;
class PreviewRunner;
class SplitButton;

// The preview row of the card «Улучшение» (PR-10): [Превью кадра │▾] (the menu: «Кадр с историей», «Очистить превью (…)…»),
// [Превью отрезка ▾] (1 / 3 / 5 / 10 с, beyond the disk budget disabled), ☐ «обновлять при изменении»; below them the status and,
// by the situation, [Отменить], [Посчитать этапы до улучшения · ≈ …], «До:» [Текущее улучшение | Апскейл], [Сохранить кадр…],
// [Убрать], [К превью]. Inside the card's form: off with the stage.
class NrPreviewBar : public QWidget {
    Q_OBJECT
public:
    NrPreviewBar(AppModel& model, PreviewRunner& runner, QWidget* parent = nullptr);
    void refresh();
    // tests and the shots
    SplitButton* frameButton() const { return frame_; }
    PresetButton* segmentButton() const { return segment_; }
    QCheckBox* autoBox() const { return auto_; }
    QLabel* status() const { return status_; }
    QPushButton* cancelButton() const { return cancel_; }
    QPushButton* prepareButton() const { return prepare_; }
    QPushButton* saveButton() const { return save_; }
    QPushButton* hideButton() const { return hide_; }
    QPushButton* toPreviewButton() const { return toPreview_; }
    QComboBox* beforeBox() const { return before_; }
    QMenu* frameMenu() const { return frameMenu_; }
    QMenu* segmentMenu() const { return segmentMenu_; }
    void setSaveFileForTests(QString file) { saveFile_ = std::move(file); }
    void setSaveDialogForTests(std::function<QString()> dialog) { saveDialog_ = std::move(dialog); }  // in place of the file dialog

private:
    AppModel& model_;
    PreviewRunner& runner_;
    SplitButton* frame_;
    QMenu* frameMenu_;
    PresetButton* segment_;
    QMenu* segmentMenu_;
    QCheckBox* auto_;
    QLabel* status_;
    QPushButton* cancel_;
    QPushButton* prepare_;
    QLabel* beforeLabel_;
    QComboBox* before_;
    QPushButton* save_;
    QPushButton* hide_;
    QPushButton* toPreview_;
    QTimer* timeTimer_;
    QString saveFile_;  // tests: the file of «Сохранить кадр…» without the dialog
    std::function<QString()> saveDialog_;
};

}  // namespace dlssvid
