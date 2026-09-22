#pragma once

#include <QCheckBox>
#include <QFrame>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QWidget>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dlssvid {

class AppModel;
struct ParamSpec;
struct StageSchema;

// One stage of the project screen (stage 9, MR D; docs/ux-guidelines.md principles 3–5): the switch, the human
// title, a form built from the stage schema (backend list, scale toggle, intensity slider — the advanced keys stay in
// the JSON editor of the engineer mode), and on the right what the plan will do with it («переиспользуется»,
// «пересчёт: Интенсивность 1 → 1.4», «будет посчитано» …), the version tag, the time, and a warning when the
// stage would run without its guides.
class StageCard : public QFrame {
    Q_OBJECT
public:
    StageCard(AppModel& model, const std::string& stage, QWidget* parent = nullptr);
    const std::string& stage() const { return stage_; }

public slots:
    void refresh();  // parameters from the project, the state from the plan, engineer widgets shown or hidden

signals:
    void runRequested(const std::string& stage);  // engineer mode: this stage alone as `dlssvid <stage>`
    void patchDllRequested();                     // nr: dlssnr-patcher over the user's nvngx_dlssnr.dll

private:
    QWidget* MakeParamWidget(const ParamSpec& spec);
    void setParam(const std::string& key, const nlohmann::json& value);

    AppModel& model_;
    std::string stage_;
    const StageSchema* schema_ = nullptr;
    bool updating_ = false;
    QCheckBox* enabled_;
    QLabel* title_;
    QLabel* subtitle_;
    QWidget* form_;
    QLabel* state_;
    QLabel* sub_;
    QLabel* version_;
    QLabel* warning_;
    QCheckBox* force_;
    QWidget* engineer_;
    QPlainTextEdit* json_;
    QPushButton* applyJson_;
    QPushButton* run_;
    QPushButton* patch_ = nullptr;
    std::vector<std::function<void()>> pull_;  // per form widget: show the project's value
};

}  // namespace dlssvid
