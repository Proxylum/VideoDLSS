#pragma once

#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QWidget>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dlssvid {

class AppModel;
class PresetButton;
class PresetMenu;
class StageCard;
class TaskQueue;

// «Стиль Кинематографичный (2) · Интенсивность 1.4 · Локальный тон 1.5 · …»: what sets a pass version apart, in the
// schema's words (TASK-0046, D7) - the values other than the default, the basic rows first, then «Ещё настройки», then
// the engineer keys (format); at most six, then «…». `tip` gets every «key = value» of the version.
QString HumanStageParams(const std::string& stage, const nlohmann::json& params, QString* tip = nullptr);

// Left dock (ТЗ §6; stage 9, MR D — the project screen of the mockups): the source (file, frame, sound, passes,
// hash), «Что получится» (the result's size and rate, the time and disk of what the plan runs, one button
// «Обработать · N стадий»), one StageCard per stage in run order, the encode card, and the pass versions on disk
// with «Вернуть» / «Сравнить» / «Удалить» / «Очистить старые».
class ProjectPanel : public QWidget {
    Q_OBJECT
public:
    ProjectPanel(AppModel& model, TaskQueue& tasks, QWidget* parent = nullptr);
    // «Пресет обработки ▾» of the stages' header (PR-9): every stage card and the encoder at once
    PresetMenu* pipelinePresetMenu() const { return pipelineMenu_; }

public slots:
    void refresh();

signals:
    void processQueued(int taskId, const QString& what);  // «Обработать»: the run is queued; what = «face.mp4 → face_result.mp4»

private:
    void refreshVersions();
    void runStage(const std::string& stage);
    void processAll();  // save the project and run `dlssvid process --project` (the whole pipeline -> result)
    void patchDll();    // nr stage: dlssnr-patcher over the user's nvngx_dlssnr.dll (`dlssvid nr-patch`)
    void choosePassesRoot();  // «Пассы · изменить…»: another folder for the passes (tens of GB)

    AppModel& model_;
    TaskQueue& tasks_;
    QLabel* sourceFile_;
    QLabel* sourceFrame_;
    QLabel* sourceAudio_;
    QLabel* sourcePasses_;
    QPushButton* changePasses_;
    QLabel* sourceHash_;
    QLabel* outVideo_;
    QLabel* outTime_;
    QLabel* outReused_;
    QLabel* outDisk_;
    QLabel* outError_;
    QPushButton* process_;
    PresetButton* pipelinePresets_ = nullptr;
    PresetMenu* pipelineMenu_ = nullptr;
    int codecItems_ = 0;  // the encoders of the list; an item past them is a codec from the project file (libx264 ...)
    std::vector<StageCard*> cards_;
    QComboBox* codec_;
    QComboBox* bitDepth_;     // «Авто» | 8 | 10 | 12 бит (TASK-0034; items outside the codec's reach are disabled)
    QComboBox* chroma_;       // «Авто» | 4:2:0 | 4:2:2 | 4:4:4
    QComboBox* bitrate_;      // «Авто» | presets in Mbit/s | «Свой…»
    QSpinBox* bitrateCustom_;  // the custom value, Mbit/s
    QLabel* encodeAudio_;
    QLabel* encodeState_;
    QLabel* versionsInfo_;
    QPushButton* gc_;
    QWidget* versionsRows_;
    QGridLayout* versionsGrid_;
    QLabel* footer_;
    bool updating_ = false;
};

}  // namespace dlssvid
