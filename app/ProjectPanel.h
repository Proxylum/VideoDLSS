#pragma once

#include <QTreeWidget>
#include <QWidget>

namespace dlssvid {

class AppModel;
class TaskQueue;

// Left dock (ТЗ §6): source, passes with status, stages with toggles/params and a run button.
class ProjectPanel : public QWidget {
    Q_OBJECT
public:
    ProjectPanel(AppModel& model, TaskQueue& tasks, QWidget* parent = nullptr);

public slots:
    void refresh();

private:
    void runStage(int index);
    void processAll();  // save the project and run `dlssvid process --project` (the whole pipeline -> result)
    void patchDll();  // nr stage: dlssnr-patcher over the user's nvngx_dlssnr.dll (`dlssvid nr-patch`)
    AppModel& model_;
    TaskQueue& tasks_;
    QTreeWidget* tree_;
    QTreeWidgetItem* sourceItem_ = nullptr;
    QTreeWidgetItem* passesItem_ = nullptr;
    QTreeWidgetItem* stagesItem_ = nullptr;
    QTreeWidgetItem* resultItem_ = nullptr;
};

}  // namespace dlssvid
