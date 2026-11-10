#include "ProjectTasks.h"

#include <system_error>

#include "AppModel.h"
#include "TaskQueue.h"
#include "io/ResultFile.h"

namespace dlssvid {

namespace {
bool SamePath(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code ec;
    return a == b || (std::filesystem::equivalent(a, b, ec) && !ec);
}
}  // namespace

ProjectTasks::ProjectTasks(AppModel& model, TaskQueue& tasks, QObject* parent) : QObject(parent), model_(model), tasks_(tasks) {
    // at the queueing, not at the start: a task waits in the queue while another project is opened
    connect(&tasks_, &TaskQueue::taskQueued, this, [this](int id) {
        const Project& p = model_.project();
        origins_[id] = {model_.generation(), p.passesRoot, QString::fromStdWString(p.sourceVideo.filename().wstring()), p.resultVideo};
    });
    // TASK-0070: the run encodes the new result next to the old one, to be renamed over it: the viewer lets the old file go. By the file,
    // not the project: a project opened again since, or another one with the same result file, shows that file too.
    connect(&tasks_, &TaskQueue::taskProgress, this, [this](int id, const TaskQueue::Progress& progress) {
        const auto it = origins_.find(id);
        if (progress.stage != "encode" || it == origins_.end() || it->second.result.empty()) return;
        it->second.encoding = true;
        if (!model_.resultReleased()) model_.releaseResult(it->second.result);  // the videos shown of that file, a proxy build reading it
    });
}

ProjectTasks::Finished ProjectTasks::finish(int id, bool processDone) {
    const auto it = origins_.find(id);
    if (it == origins_.end()) return {};
    const Origin origin = it->second;
    origins_.erase(it);
    const Finished finished{origin.generation == model_.generation(), origin.video};
    // a run cancelled (killed) or crashed while it encoded left the new result beside the old one, which it had not touched (TASK-0070):
    // that file goes. A run that failed on an error it handled removed its own, or kept a finished one it could not put in place - its
    // message says where
    if (origin.encoding && (tasks_.cancelled(id) || tasks_.endedAbnormally(id))) RemovePartialResult(PartialResultPath(origin.result));
    // the result again - the new one or the old one, as the run ended - when this task let it go: one Main task runs at a time, the end
    // of another (a waiting one dropped) leaves it to the run still encoding
    const bool released = origin.encoding && model_.resultReleased();
    if (released) model_.restoreResult();
    if (!finished.ownProject) {
        if (released || SamePath(origin.passesRoot, model_.project().passesRoot)) model_.reloadSources();  // reopened, or the same passes
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
