#pragma once

#include <QObject>
#include <QString>
#include <cstdint>
#include <filesystem>
#include <map>

namespace dlssvid {

class AppModel;
class TaskQueue;

// The project each Main task of the queue was queued for. Another project may be opened while a task runs or waits, so what the
// end of a task does to the model depends on whether that project is still the open one (TASK-0067).
class ProjectTasks : public QObject {
public:
    struct Finished {
        bool ownProject = true;  // the project it was queued for is still the open one
        QString video;           // the source file of that project, for the messages
    };

    ProjectTasks(AppModel& model, TaskQueue& tasks, QObject* parent = nullptr);

    // Task `id` has ended. Its own project: the «пересчитать» marks go and the sources are read again; `processDone` (the `process`
    // run of the processing page, ended ok) also records its time and shows the result against the source. Another project is open:
    // its marks and view stay, its sources are read again only if the task wrote into its passes folder.
    Finished finish(int id, bool processDone);
    void forget(int id) { origins_.erase(id); }  // a task whose end was handled elsewhere

private:
    struct Origin {
        uint64_t generation = 0;
        std::filesystem::path passesRoot;
        QString video;
    };

    AppModel& model_;
    TaskQueue& tasks_;
    std::map<int, Origin> origins_;
};

}  // namespace dlssvid
