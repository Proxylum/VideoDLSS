#pragma once

#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QWidget>
#include <string>
#include <vector>

namespace dlssvid {

class AppModel;
class StageCard;
class TaskQueue;

// Left dock (ТЗ §6; stage 9, MR D — the project screen of the mockups): the source (file, frame, sound, passes,
// hash), «Что получится» (the result's size and rate, the time and disk of what the plan runs, one button
// «Обработать · N стадий»), one StageCard per stage in run order, the encode card, and the pass versions on disk
// with «Вернуть» / «Сравнить» / «Удалить» / «Очистить старые».
class ProjectPanel : public QWidget {
    Q_OBJECT
public:
    ProjectPanel(AppModel& model, TaskQueue& tasks, QWidget* parent = nullptr);

public slots:
    void refresh();

signals:
    void processQueued(int taskId, const QString& what);  // «Обработать»: the run is queued; what = «face.mp4 → face_result.mp4»

private:
    void refreshVersions();
    void runStage(const std::string& stage);
    void processAll();  // save the project and run `dlssvid process --project` (the whole pipeline -> result)
    void patchDll();    // nr stage: dlssnr-patcher over the user's nvngx_dlssnr.dll (`dlssvid nr-patch`)

    AppModel& model_;
    TaskQueue& tasks_;
    QLabel* sourceFile_;
    QLabel* sourceFrame_;
    QLabel* sourceAudio_;
    QLabel* sourcePasses_;
    QLabel* sourceHash_;
    QLabel* outVideo_;
    QLabel* outTime_;
    QLabel* outReused_;
    QLabel* outDisk_;
    QLabel* outError_;
    QPushButton* process_;
    std::vector<StageCard*> cards_;
    QComboBox* codec_;
    QLineEdit* bitrate_;
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
