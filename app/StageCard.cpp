#include "StageCard.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <cmath>

#include "AppModel.h"
#include "Theme.h"
#include "stages/ParamSchema.h"

namespace dlssvid {

namespace {

nlohmann::json ChoiceValue(const ParamSpec& spec, const std::string& value) {
    if (spec.type == ParamType::Int) return std::stoi(value);
    if (spec.type == ParamType::Float) {
        const double d = std::stod(value);
        return d == std::floor(d) ? nlohmann::json(static_cast<int64_t>(d)) : nlohmann::json(d);
    }
    return value;
}

int ChoiceIndex(const ParamSpec& spec, const nlohmann::json& value) {
    for (size_t i = 0; i < spec.choices.size(); ++i) {
        const auto& c = spec.choices[i];
        if (value.is_string() && value.get<std::string>() == c.value) return static_cast<int>(i);
        if (value.is_number()) {
            try {
                if (std::fabs(value.get<double>() - std::stod(c.value)) < 1e-9) return static_cast<int>(i);
            } catch (...) {
            }
        }
    }
    return -1;
}

const char* kToneRole[] = {"ok", "warn", "muted", "muted"};  // ok, run, off, none

int ToneIndex(const QString& tone) {
    if (tone == "ok") return 0;
    if (tone == "run") return 1;
    if (tone == "off") return 2;
    return 3;
}

}  // namespace

StageCard::StageCard(AppModel& model, const std::string& stage, QWidget* parent) : QFrame(parent), model_(model), stage_(stage), schema_(FindStageSchema(stage)) {
    setFrameShape(QFrame::NoFrame);
    setObjectName("stageCard");
    SetRole(this, "card");
    auto* grid = new QGridLayout(this);
    grid->setContentsMargins(14, 9, 14, 9);
    grid->setHorizontalSpacing(14);
    grid->setVerticalSpacing(6);

    enabled_ = new QCheckBox(this);
    enabled_->setToolTip(tr("Стадия участвует в обработке"));
    title_ = new QLabel(schema_ ? QString::fromStdString(schema_->title) : QString::fromStdString(stage), this);
    SetRole(title_, "title");
    subtitle_ = new QLabel(schema_ ? QString::fromStdString(schema_->subtitle) : QString(), this);
    SetRole(subtitle_, "hint");
    auto* head = new QVBoxLayout();
    head->setSpacing(2);
    head->addWidget(title_);
    head->addWidget(subtitle_);
    grid->addWidget(enabled_, 0, 0, Qt::AlignTop);
    grid->addLayout(head, 0, 1, Qt::AlignTop);

    form_ = new QWidget(this);
    auto* form = new QFormLayout(form_);
    form->setContentsMargins(0, 0, 0, 0);
    form->setHorizontalSpacing(8);
    form->setVerticalSpacing(6);
    form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    if (schema_)
        for (const ParamSpec& spec : schema_->params) {
            if (spec.advanced) continue;
            QWidget* w = MakeParamWidget(spec);
            w->setToolTip(QString::fromStdString(spec.hint.empty() ? spec.key : spec.hint + " (" + spec.key + ")"));
            form->addRow(QString::fromStdString(spec.label), w);
        }
    grid->addWidget(form_, 0, 2, 2, 1);

    auto* side = new QVBoxLayout();
    side->setSpacing(3);
    state_ = new QLabel(this);
    state_->setWordWrap(true);
    sub_ = new QLabel(this);
    sub_->setWordWrap(true);
    SetRole(sub_, "hint");
    version_ = new QLabel(this);
    SetRole(version_, "version");
    auto* stateRow = new QHBoxLayout();
    stateRow->addWidget(version_, 0, Qt::AlignTop);
    stateRow->addWidget(state_, 1);
    side->addLayout(stateRow);
    side->addWidget(sub_);
    warning_ = new QLabel(this);
    warning_->setWordWrap(true);
    SetRole(warning_, "warn");
    side->addWidget(warning_);
    force_ = new QCheckBox(tr("пересчитать"), this);
    force_->setToolTip(tr("Посчитать заново, даже если пасс совпадает по отпечатку"));
    side->addWidget(force_);
    grid->addLayout(side, 0, 3, 2, 1, Qt::AlignTop);
    grid->setColumnStretch(2, 1);
    grid->setColumnMinimumWidth(3, 190);
    state_->setMinimumWidth(120);
    sub_->setMinimumWidth(120);
    warning_->setMinimumWidth(120);

    // engineer mode: the JSON of all parameters and the stage alone as a process
    engineer_ = new QWidget(this);
    auto* eng = new QHBoxLayout(engineer_);
    eng->setContentsMargins(0, 0, 0, 0);
    json_ = new QPlainTextEdit(engineer_);
    json_->setMaximumHeight(52);
    json_->setToolTip(tr("Все параметры стадии (JSON): ключи становятся опциями CLI"));
    applyJson_ = new QPushButton(tr("Применить JSON"), engineer_);
    SetRole(applyJson_, "small");
    run_ = new QPushButton(tr("Запустить отдельно"), engineer_);
    SetRole(run_, "small");
    run_->setToolTip(tr("Только эта стадия как процесс dlssvid %1").arg(QString::fromStdString(stage)));
    eng->addWidget(json_, 1);
    auto* buttons = new QVBoxLayout();
    buttons->addWidget(applyJson_);
    buttons->addWidget(run_);
    if (stage == "nr") {
        patch_ = new QPushButton(tr("Пропатчить DLL…"), engineer_);
        SetRole(patch_, "small");
        patch_->setToolTip(tr("Пропатчить вашу nvngx_dlssnr.dll для RTX 20/30/40 (dlssnr-patcher, CUDA Toolkit 13.3) и положить результат в bin/nvidia/"));
        buttons->addWidget(patch_);
        connect(patch_, &QPushButton::clicked, this, &StageCard::patchDllRequested);
    }
    eng->addLayout(buttons);
    grid->addWidget(engineer_, 2, 0, 1, 4);

    connect(enabled_, &QCheckBox::toggled, this, [this](bool on) {
        if (!updating_) model_.setStageEnabled(stage_, on);
    });
    connect(force_, &QCheckBox::toggled, this, [this](bool on) {
        if (!updating_) model_.setForce(stage_, on);
    });
    connect(applyJson_, &QPushButton::clicked, this, [this] {
        try {
            model_.setStageParams(stage_, nlohmann::json::parse(json_->toPlainText().toStdString()));
            json_->setStyleSheet("");
        } catch (...) {
            json_->setStyleSheet("background: #553333");
        }
    });
    connect(run_, &QPushButton::clicked, this, [this] { emit runRequested(stage_); });
    refresh();
}

QWidget* StageCard::MakeParamWidget(const ParamSpec& spec) {
    const std::string key = spec.key;
    switch (spec.widget) {
        case ParamWidget::Toggle: {
            auto* row = new QWidget(form_);
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(0, 0, 0, 0);
            h->setSpacing(0);
            auto* group_frame = new QFrame(row);  // the segmented control of the mockups: one bordered group
            SetRole(group_frame, "segments");
            auto* segments = new QHBoxLayout(group_frame);
            segments->setContentsMargins(0, 0, 0, 0);
            segments->setSpacing(0);
            auto* group = new QButtonGroup(row);
            group->setExclusive(true);
            std::vector<QToolButton*> buttons;
            for (const ParamChoice& c : spec.choices) {
                auto* b = new QToolButton(row);
                b->setText(QString::fromStdString(c.label));
                b->setToolTip(QString::fromStdString(c.hint));
                b->setCheckable(true);
                b->setAutoRaise(true);
                SetRole(b, "segment");
                group->addButton(b);
                segments->addWidget(b);
                buttons.push_back(b);
                const std::string value = c.value;
                connect(b, &QToolButton::clicked, this, [this, spec, value] { setParam(spec.key, ChoiceValue(spec, value)); });
            }
            h->addWidget(group_frame);
            h->addStretch(1);
            pull_.push_back([this, spec, buttons] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) ? e->params[spec.key] : spec.def;
                const int idx = ChoiceIndex(spec, v);
                for (size_t i = 0; i < buttons.size(); ++i) {
                    const QSignalBlocker b(buttons[i]);
                    buttons[i]->setChecked(static_cast<int>(i) == idx);
                }
            });
            return row;
        }
        case ParamWidget::Select: {
            auto* combo = new QComboBox(form_);
            combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);  // the card, not the longest item, sets the width
            combo->setMinimumContentsLength(10);
            for (const ParamChoice& c : spec.choices) combo->addItem(QString::fromStdString(c.label), QString::fromStdString(c.value));
            connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, spec, combo](int i) {
                if (!updating_ && i >= 0) setParam(spec.key, ChoiceValue(spec, combo->itemData(i).toString().toStdString()));
            });
            pull_.push_back([this, spec, combo] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) ? e->params[spec.key] : spec.def;
                const QSignalBlocker b(combo);
                combo->setCurrentIndex(std::max(0, ChoiceIndex(spec, v)));
            });
            return combo;
        }
        case ParamWidget::Slider: {
            auto* row = new QWidget(form_);
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(0, 0, 0, 0);
            auto* slider = new QSlider(Qt::Horizontal, row);
            slider->setMinimumWidth(80);
            const double step = spec.step > 0 ? spec.step : 0.1;
            slider->setRange(static_cast<int>(std::lround(spec.min / step)), static_cast<int>(std::lround(spec.max / step)));
            auto* spin = new QDoubleSpinBox(row);
            spin->setRange(spec.min, spec.max);
            spin->setSingleStep(step);
            spin->setDecimals(step < 0.01 ? 3 : step < 0.1 ? 2 : 1);
            h->addWidget(slider, 1);
            h->addWidget(spin);
            connect(slider, &QSlider::valueChanged, this, [this, spec, spin, step](int v) {
                if (updating_) return;
                const QSignalBlocker b(spin);
                spin->setValue(v * step);
                setParam(spec.key, spec.type == ParamType::Int ? nlohmann::json(static_cast<int>(std::lround(v * step))) : nlohmann::json(v * step));
            });
            connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, spec, slider, step](double v) {
                if (updating_) return;
                const QSignalBlocker b(slider);
                slider->setValue(static_cast<int>(std::lround(v / step)));
                setParam(spec.key, spec.type == ParamType::Int ? nlohmann::json(static_cast<int>(std::lround(v))) : nlohmann::json(v));
            });
            pull_.push_back([this, spec, slider, spin, step] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) && e->params[spec.key].is_number() ? e->params[spec.key] : spec.def;
                const double d = v.is_number() ? v.get<double>() : 0.0;
                const QSignalBlocker b1(slider), b2(spin);
                slider->setValue(static_cast<int>(std::lround(d / step)));
                spin->setValue(d);
            });
            return row;
        }
        case ParamWidget::Spin: {
            if (spec.type == ParamType::Int) {
                auto* spin = new QSpinBox(form_);
                spin->setRange(static_cast<int>(spec.min), static_cast<int>(spec.max > spec.min ? spec.max : 1e9));
                spin->setSingleStep(std::max(1, static_cast<int>(spec.step)));
                connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, [this, spec](int v) {
                    if (!updating_) setParam(spec.key, v);
                });
                pull_.push_back([this, spec, spin] {
                    const StageEntry* e = model_.stageEntry(stage_);
                    const nlohmann::json v = e && e->params.contains(spec.key) && e->params[spec.key].is_number() ? e->params[spec.key] : spec.def;
                    const QSignalBlocker b(spin);
                    spin->setValue(v.is_number() ? static_cast<int>(std::lround(v.get<double>())) : 0);
                });
                return spin;
            }
            auto* spin = new QDoubleSpinBox(form_);
            spin->setRange(spec.min, spec.max > spec.min ? spec.max : 1e9);
            spin->setSingleStep(spec.step > 0 ? spec.step : 0.1);
            spin->setDecimals(3);
            connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, spec](double v) {
                if (!updating_) setParam(spec.key, v);
            });
            pull_.push_back([this, spec, spin] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) && e->params[spec.key].is_number() ? e->params[spec.key] : spec.def;
                const QSignalBlocker b(spin);
                spin->setValue(v.is_number() ? v.get<double>() : 0.0);
            });
            return spin;
        }
        case ParamWidget::Check: {
            auto* check = new QCheckBox(form_);
            connect(check, &QCheckBox::toggled, this, [this, spec](bool on) {
                if (!updating_) setParam(spec.key, on);
            });
            pull_.push_back([this, spec, check] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) ? e->params[spec.key] : spec.def;
                const QSignalBlocker b(check);
                check->setChecked(v.is_boolean() && v.get<bool>());
            });
            return check;
        }
        case ParamWidget::Text:
        default: {
            auto* edit = new QLineEdit(form_);
            connect(edit, &QLineEdit::editingFinished, this, [this, spec, edit] {
                if (!updating_) setParam(spec.key, edit->text().toStdString());
            });
            pull_.push_back([this, spec, edit] {
                const StageEntry* e = model_.stageEntry(stage_);
                const nlohmann::json v = e && e->params.contains(spec.key) ? e->params[spec.key] : spec.def;
                const QSignalBlocker b(edit);
                edit->setText(v.is_string() ? QString::fromStdString(v.get<std::string>()) : QString::fromStdString(v.dump()));
            });
            return edit;
        }
    }
}

void StageCard::setParam(const std::string& key, const nlohmann::json& value) { model_.setStageParam(stage_, key, value); }

void StageCard::refresh() {
    updating_ = true;
    const StageEntry* e = model_.stageEntry(stage_);
    enabled_->setChecked(e && e->enabled);
    form_->setEnabled(e && e->enabled);
    for (const auto& pull : pull_) pull();
    const AppModel::StageCardState st = model_.cardState(stage_);
    state_->setText(st.state);
    SetRole(state_, kToneRole[ToneIndex(st.tone)]);
    sub_->setText(st.sub);
    sub_->setVisible(!st.sub.isEmpty());
    version_->setText(st.version);
    version_->setVisible(!st.version.isEmpty());
    warning_->setText(st.warning);
    warning_->setVisible(!st.warning.isEmpty());
    force_->setChecked(model_.forced(stage_));
    force_->setVisible(e && e->enabled);
    engineer_->setVisible(model_.engineerMode());
    if (e && !json_->hasFocus()) json_->setPlainText(QString::fromStdString(e->params.dump()));
    updating_ = false;
}

}  // namespace dlssvid
