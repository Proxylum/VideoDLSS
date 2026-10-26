#pragma once

#include <QAction>
#include <QCheckBox>
#include <QFormLayout>
#include <QFrame>
#include <QToolButton>
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
    // A widget of the stage's own under the form (PR-13: the list of the card «Склейки»), refreshed by itself
    void setExtra(QWidget* extra);

public slots:
    void refresh();  // parameters from the project, the state from the plan, engineer widgets shown or hidden

signals:
    void runRequested(const std::string& stage);  // engineer mode: this stage alone as `dlssvid <stage>`
    void patchDllRequested();                     // nr: dlssnr-patcher over the user's nvngx_dlssnr.dll

private:
    // One parameter row of the form (TASK-0046): where it is, its label (accent when changed), the note of a disabled
    // row and the reset button.
    struct Row {
        const ParamSpec* spec = nullptr;
        QFormLayout* form = nullptr;
        int index = 0;
        QLabel* label = nullptr;
        QWidget* field = nullptr;
        QLabel* note = nullptr;
        QToolButton* reset = nullptr;
        QAction* resetAction = nullptr;  // the same in the context menu of the label
    };
    QWidget* MakeParamWidget(const ParamSpec& spec);
    void setParam(const std::string& key, const nlohmann::json& value);
    void resetParam(const std::string& key);  // the row's ↺: the key leaves the project (the schema default runs)
    void resetCard();                          // «Сбросить»: the backend and the engineer keys stay
    bool changed(const ParamSpec& spec) const; // the project sets it to something else than the default
    // The stage's parameters with the aliases folded (skin-structure, no_guides, style 1 -> natural): what the stage runs
    // with, so the card shows, compares and resets the same values (TASK-0046).
    nlohmann::json stageParams() const;
    nlohmann::json valueOf(const ParamSpec& spec) const;   // the project's value, else the default
    double followedValue(const ParamSpec& spec) const;     // what a special value stands for (skin_structure -1: local_structure)
    std::string currentBackend() const;  // the stage's backend in the project (the schema default when unset)

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
    QLabel* scenes_;  // a stage that reads the scene list: what its cuts do (StageCardState::scenes, PR-13)
    QCheckBox* force_;
    QWidget* engineer_;
    QPlainTextEdit* json_;
    QPushButton* applyJson_;
    QPushButton* run_;
    QPushButton* patch_ = nullptr;
    QPushButton* resetCard_ = nullptr;
    QToolButton* moreToggle_ = nullptr;  // «Ещё настройки ▸/▾»
    QLabel* moreSummary_ = nullptr;      // «изменено: … · не действует: …», wrapped: it never sets the card's width
    QWidget* more_ = nullptr;
    std::vector<Row> rows_;
    std::vector<std::function<void()>> pull_;  // per form widget: show the project's value
};

}  // namespace dlssvid
