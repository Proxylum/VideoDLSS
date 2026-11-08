#include "ProjectTasks.h"

#include <system_error>

#include "AppModel.h"
#include "TaskQueue.h"

namespace dlssvid {

namespace {
bool SameFolder(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code ec;
    return a == b || (std::filesystem::equivalent(a, b, ec) && !ec);
}
}  // namespace

ProjectTasks::ProjectTasks(AppModel& model, TaskQueue& tasks, QObject* parent) : QObject(parent), model_(model), tasks_(tasks) {
    // at the queueing, not at the start: a task waits in the queue while another project is opened
    connect(&tasks_, &TaskQueue::taskQueued, this, [this](int id) {
        const Project& p = model_.project();
        origins_[id] = {model_.generation(), p.passesRoot, QString::fromStdWString(p.sourceVideo.filename().wstring())};
    });
}

ProjectTasks::Finished ProjectTasks::finish(int id, bool processDone) {
    const auto it = origins_.find(id);
    if (it == origins_.end()) return {};
    const Origin origin = it->second;
    origins_.erase(it);
    const Finished finished{origin.generation == model_.generation(), origin.video};
    if (!finished.ownProject) {
        if (SameFolder(origin.passesRoot, model_.project().passesRoot)) model_.reloadSources();  // reopened, or a project over the same passes
        else model_.refreshPlan();  // its passes are not ours, but `nr-patch` changes what every project can run
        return finished;
    }
    if (processDone) model_.setLastRun(tasks_.progress(id).elapsed);  // «готов за …» in the result line
    model_.clearForce();
    model_.reloadSources();  // the new passes / the result: sources, the plan, the cards
    if (processDone) model_.setCompareView(AppModel::CompareView::BeforeAfter);  // the result against the source
    return finished;
}

}  // namespace dlssvid
