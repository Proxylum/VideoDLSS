#include "StageCard.h"

#include <QAbstractButton>
#include <QAbstractSpinBox>
#include <QButtonGroup>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <set>

#include "AppModel.h"
#include "Theme.h"
#include "stages/ParamSchema.h"

namespace dlssvid {

namespace {

// A value from a slider or a spin box, rounded to the decimals the control shows: 3 x 0.1 is 0.30000000000000004, which
// read as another value than the 0.3 typed into the JSON (the plan said «Интенсивность 1.4 → 1.4», TASK-0045)
double QuantizeParam(double v, int decimals) {
    const double k = std::pow(10.0, decimals);
    return std::round(v * k) / k;
}

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
    auto* column = new QVBoxLayout(form_);
    column->setContentsMargins(0, 0, 0, 0);
    column->setSpacing(4);
    auto makeForm = [](QWidget* parent) {
        auto* f = new QFormLayout(parent);
        f->setContentsMargins(0, 0, 0, 0);
        f->setHorizontalSpacing(8);
        f->setVerticalSpacing(6);
        f->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
        return f;
    };
    auto* basicBox = new QWidget(form_);
    QFormLayout* basic = makeForm(basicBox);
    column->addWidget(basicBox);
    // the rows of the form in their groups; the `more` ones in «Ещё настройки», collapsed (TASK-0046)
    auto addRows = [this](QFormLayout* f, bool more) {
        std::string group;
        for (const ParamSpec& spec : schema_->params) {
            if (spec.advanced || spec.more != more) continue;
            if (!spec.group.empty() && spec.group != group) {
                group = spec.group;
                auto* header = new QLabel(QString::fromStdString(group), f->parentWidget());
                SetRole(header, "group");
                f->addRow(header);
            }
            Row row;
            row.spec = &spec;
            row.form = f;
            row.label = new QLabel(QString::fromStdString(spec.label), f->parentWidget());
            const QString tip = QString::fromStdString(spec.hint.empty() ? spec.key : spec.hint + " (" + spec.key + ")");
            row.label->setToolTip(tip);
            // the cell: the control and ↺ on one line, the note of a disabled row under them (inline it squeezed the
            // control of «Структура кожи» to nothing)
            auto* cell = new QWidget(f->parentWidget());
            auto* lines = new QVBoxLayout(cell);
            lines->setContentsMargins(0, 0, 0, 0);
            lines->setSpacing(2);
            auto* h = new QHBoxLayout();
            h->setSpacing(6);
            lines->addLayout(h);
            row.field = MakeParamWidget(spec);
            row.field->setToolTip(tip);
            // the label is no buddy of the control inside the cell: name the controls for screen readers
            const QString name = QString::fromStdString(spec.label);
            row.field->setAccessibleName(name);
            for (QWidget* c : row.field->findChildren<QWidget*>()) {
                if (qobject_cast<QSlider*>(c) || qobject_cast<QAbstractSpinBox*>(c)) c->setAccessibleName(name);
                else if (auto* b = qobject_cast<QAbstractButton*>(c)) b->setAccessibleName(name + ": " + b->text());
            }
            h->addWidget(row.field, 1);
            row.note = new QLabel(cell);
            SetRole(row.note, "hint");
            row.note->setWordWrap(true);
            row.note->hide();
            lines->addWidget(row.note);
            row.reset = new QToolButton(cell);
            row.reset->setText(QString::fromUtf8("↺"));
            row.reset->setAutoRaise(true);
            row.reset->setFocusPolicy(Qt::TabFocus);  // reachable from the keyboard
            row.reset->setToolTip(tr("Сбросить: %1").arg(QString::fromStdString(ParamValueLabel(spec, spec.def))));
            row.reset->setAccessibleName(tr("Сбросить «%1»").arg(QString::fromStdString(spec.label)));
            QSizePolicy keep = row.reset->sizePolicy();  // hidden until the value changes: the row keeps its width
            keep.setRetainSizeWhenHidden(true);
            row.reset->setSizePolicy(keep);
            const std::string key = spec.key;
            connect(row.reset, &QToolButton::clicked, this, [this, key] { resetParam(key); });
            h->addWidget(row.reset);
            row.resetAction = new QAction(row.reset->toolTip(), row.label);
            connect(row.resetAction, &QAction::triggered, this, [this, key] { resetParam(key); });
            row.label->addAction(row.resetAction);
            row.label->setContextMenuPolicy(Qt::ActionsContextMenu);
            f->addRow(row.label, cell);
            row.index = f->rowCount() - 1;
            rows_.push_back(row);
        }
    };
    if (schema_) {
        addRows(basic, false);
        bool anyMore = false;
        for (const ParamSpec& spec : schema_->params) anyMore = anyMore || (!spec.advanced && spec.more);
        if (anyMore) {
            moreToggle_ = new QToolButton(form_);
            moreToggle_->setCheckable(true);
            moreToggle_->setAutoRaise(true);
            moreToggle_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
            SetRole(moreToggle_, "more");
            column->addWidget(moreToggle_, 0, Qt::AlignLeft);
            more_ = new QWidget(form_);
            addRows(makeForm(more_), true);
            column->addWidget(more_);
            // one label column for both forms: the rows of «Ещё настройки» line up with the basic ones
            int labelWidth = 0;
            for (const Row& r : rows_) labelWidth = std::max(labelWidth, r.label->sizeHint().width());
            for (const Row& r : rows_) {
                r.label->setMinimumWidth(labelWidth);
                r.label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);  // the widened label keeps the text at the field
            }
            const QString settingsKey = QString("card/%1/more").arg(QString::fromStdString(stage));
            const bool open = QSettings().value(settingsKey, false).toBool();
            moreToggle_->setChecked(open);
            more_->setVisible(open);
            connect(moreToggle_, &QToolButton::toggled, this, [this, settingsKey](bool on) {
                more_->setVisible(on);
                QSettings().setValue(settingsKey, on);
                refresh();  // the summary of the changed values shows on the collapsed toggle only
            });
        }
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
    resetCard_ = new QPushButton(tr("Сбросить"), this);
    SetRole(resetCard_, "small");
    resetCard_->setToolTip(tr("Вернуть настройки стадии к умолчаниям (бэкенд и инженерные ключи остаются)"));
    side->addWidget(resetCard_, 0, Qt::AlignLeft);
    connect(resetCard_, &QPushButton::clicked, this, [this] { resetCard(); });
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

std::string StageCard::currentBackend() const {
    const StageEntry* e = model_.stageEntry(stage_);
    return StageBackend(stage_, e ? e->params : nlohmann::json::object());
}

QWidget* StageCard::MakeParamWidget(const ParamSpec& spec) {
    const std::string key = spec.key;
    if (!spec.formChoices.empty()) {  // the list of the current backend (depth.model: the VDA models), rebuilt on refresh
        auto* combo = new QComboBox(form_);
        combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        combo->setMinimumContentsLength(10);
        connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, spec, combo](int i) {
            if (!updating_ && i >= 0) setParam(spec.key, combo->itemData(i).toString().toStdString());
        });
        pull_.push_back([this, spec, combo] {
            const nlohmann::json v = valueOf(spec);
            const QString value = QString::fromStdString(v.is_string() ? v.get<std::string>() : v.dump());
            const QSignalBlocker b(combo);
            combo->clear();
            if (const auto* list = FormChoicesFor(spec, currentBackend()))
                for (const ParamChoice& c : *list) {
                    combo->addItem(QString::fromStdString(c.label), QString::fromStdString(c.value));
                    combo->setItemData(combo->count() - 1, QString::fromStdString(c.hint), Qt::ToolTipRole);
                }
            int idx = combo->findData(value);
            if (idx < 0) {  // an id from the JSON / CLI that the list does not name (vda-small): shown as it is
                combo->addItem(value + tr(" (из JSON)"), value);
                idx = combo->count() - 1;
            }
            combo->setCurrentIndex(idx);
        });
        return combo;
    }
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
                const int idx = ChoiceIndex(spec, valueOf(spec));
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
            bool separated = false;
            for (const ParamChoice& c : spec.choices) {
                if (c.experimental && !separated) {  // the experimental ones after a separator (nr.style 3..6, TASK-0046)
                    combo->insertSeparator(combo->count());
                    separated = true;
                }
                combo->addItem(QString::fromStdString(c.label) + (c.experimental ? tr(" (эксп.)") : QString()), QString::fromStdString(c.value));
                combo->setItemData(combo->count() - 1, QString::fromStdString(c.hint), Qt::ToolTipRole);
            }
            connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, spec, combo](int i) {
                if (!updating_ && i >= 0 && combo->itemData(i).isValid()) setParam(spec.key, ChoiceValue(spec, combo->itemData(i).toString().toStdString()));
            });
            pull_.push_back([this, spec, combo] {
                const int idx = ChoiceIndex(spec, valueOf(spec));  // folded: nr.style 1 / "1" -> natural
                const QSignalBlocker b(combo);
                combo->setCurrentIndex(std::max(0, idx >= 0 ? combo->findData(QString::fromStdString(spec.choices[idx].value)) : 0));
            });
            return combo;
        }
        case ParamWidget::Slider: {
            auto* row = new QWidget(form_);
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(0, 0, 0, 0);
            // created before the slider: the Tab order follows the order on screen («отдельно», slider, number)
            QCheckBox* own = spec.sentinel ? new QCheckBox(tr("отдельно"), row) : nullptr;
            auto* slider = new QSlider(Qt::Horizontal, row);
            slider->setMinimumWidth(80);
            const double step = spec.step > 0 ? spec.step : 0.1;
            const double lo = spec.sentinel ? spec.rangeMin : spec.min;  // skin_structure: -1 is the check box, the slider 0..2
            slider->setRange(static_cast<int>(std::lround(lo / step)), static_cast<int>(std::lround(spec.max / step)));
            auto* spin = new QDoubleSpinBox(row);
            spin->setRange(lo, spec.max);
            spin->setSingleStep(step);
            const int decimals = ParamDecimals(spec);
            spin->setDecimals(decimals);
            if (!spec.zeroLabel.empty()) spin->setSpecialValueText(QString::fromStdString(spec.zeroLabel));  // «выкл» at the minimum
            if (!spec.suffix.empty()) spin->setSuffix(QString::fromStdString(spec.suffix));
            if (spec.sentinel) {
                // «отдельно» off = the special value (skin structure as the local structure); on = its own strength, starting
                // from the value it followed
                own->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);  // the slider gives way, not the word
                own->setToolTip(tr("Выключено — %1").arg(QString::fromStdString(spec.sentinelLabel)));
                h->addWidget(own);
                connect(own, &QCheckBox::toggled, this, [this, spec](bool on) {
                    if (updating_) return;
                    if (!on) {
                        setParam(spec.key, *spec.sentinel);
                        return;
                    }
                    setParam(spec.key, followedValue(spec));
                });
                pull_.push_back([this, spec, own, slider, spin] {
                    const nlohmann::json v = valueOf(spec);
                    const bool isOwn = !(v.is_number() && v.get<double>() == *spec.sentinel);
                    const QSignalBlocker b(own);
                    own->setChecked(isOwn);
                    slider->setEnabled(isOwn);
                    spin->setEnabled(isOwn);
                });
            }
            h->addWidget(slider, 1);
            h->addWidget(spin);
            connect(slider, &QSlider::valueChanged, this, [this, spec, spin, step, decimals](int v) {
                if (updating_) return;
                const QSignalBlocker b(spin);
                const double value = QuantizeParam(v * step, decimals);
                spin->setValue(value);
                setParam(spec.key, spec.type == ParamType::Int ? nlohmann::json(static_cast<int>(std::lround(value))) : nlohmann::json(value));
            });
            connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, spec, slider, step, decimals](double v) {
                if (updating_) return;
                const QSignalBlocker b(slider);
                slider->setValue(static_cast<int>(std::lround(v / step)));
                setParam(spec.key, spec.type == ParamType::Int ? nlohmann::json(static_cast<int>(std::lround(v))) : nlohmann::json(QuantizeParam(v, decimals)));
            });
            pull_.push_back([this, spec, slider, spin, step] {
                const nlohmann::json raw = valueOf(spec);
                const nlohmann::json v = raw.is_number() ? raw : spec.def;
                double d = v.is_number() ? v.get<double>() : 0.0;
                // the special value: the check box says it, the disabled slider shows the value followed (not 0)
                if (spec.sentinel && d == *spec.sentinel) d = followedValue(spec);
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
                    const nlohmann::json raw = valueOf(spec);
                    const nlohmann::json v = raw.is_number() ? raw : spec.def;
                    const QSignalBlocker b(spin);
                    spin->setValue(v.is_number() ? static_cast<int>(std::lround(v.get<double>())) : 0);
                });
                return spin;
            }
            auto* spin = new QDoubleSpinBox(form_);
            spin->setRange(spec.min, spec.max > spec.min ? spec.max : 1e9);
            spin->setSingleStep(spec.step > 0 ? spec.step : 0.1);
            const int decimals = ParamDecimals(spec);
            spin->setDecimals(decimals);
            if (!spec.zeroLabel.empty()) spin->setSpecialValueText(QString::fromStdString(spec.zeroLabel));
            if (!spec.suffix.empty()) spin->setSuffix(QString::fromStdString(spec.suffix));
            connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, spec, decimals](double v) {
                if (!updating_) setParam(spec.key, QuantizeParam(v, decimals));
            });
            pull_.push_back([this, spec, spin] {
                const nlohmann::json raw = valueOf(spec);
                const nlohmann::json v = raw.is_number() ? raw : spec.def;
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
                const nlohmann::json v = valueOf(spec);  // folded: no_guides true -> guides false
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
                const nlohmann::json v = valueOf(spec);
                const QSignalBlocker b(edit);
                edit->setText(v.is_string() ? QString::fromStdString(v.get<std::string>()) : QString::fromStdString(v.dump()));
            });
            return edit;
        }
    }
}

nlohmann::json StageCard::stageParams() const {
    const StageEntry* e = model_.stageEntry(stage_);
    return e && e->params.is_object() ? NormalizeStageParams(stage_, e->params) : nlohmann::json::object();
}

nlohmann::json StageCard::valueOf(const ParamSpec& spec) const {
    const nlohmann::json p = stageParams();
    return p.contains(spec.key) ? p[spec.key] : spec.def;
}

double StageCard::followedValue(const ParamSpec& spec) const {
    const StageEntry* e = model_.stageEntry(stage_);
    const nlohmann::json eff = EffectiveStageParams(stage_, e ? e->params : nlohmann::json::object());
    const std::string& seed = spec.sentinelFollows;
    const double v = !seed.empty() && eff.contains(seed) && eff[seed].is_number() ? eff[seed].get<double>() : std::max(spec.rangeMin, 1.0);
    return std::clamp(v, spec.rangeMin, spec.max);
}

void StageCard::setParam(const std::string& key, const nlohmann::json& value) {
    // an alias of the key (skin-structure, no_guides) would stay beside it and lose to it: the card writes the schema's
    // spelling only
    if (const StageEntry* e = model_.stageEntry(stage_); e && e->params.is_object()) {
        const nlohmann::json folded = NormalizeStageParams(stage_, e->params);
        if (folded != e->params) model_.setStageParams(stage_, folded);
    }
    model_.setStageParam(stage_, key, value);
}

void StageCard::resetParam(const std::string& key) {
    nlohmann::json params = stageParams();
    if (!params.contains(key)) return;
    // back to the default backend: a model from the old backend's list goes too, as on a switch in the list (VDA Large
    // under DA3 is an error of the plan)
    if (key == "backend" && schema_)
        if (const ParamSpec* b = schema_->Find("backend"); b && b->def.is_string()) params = ParamsForBackendChange(stage_, params, b->def.get<std::string>());
    params.erase(key);
    model_.setStageParams(stage_, params);
}

void StageCard::resetCard() {
    if (!model_.stageEntry(stage_) || !schema_) return;
    // the form's keys go; the backend and the keys of the JSON editor (format, ...) stay - setStageParams replaces the
    // whole object, so they are carried over (TASK-0046). Folded first: an alias of a form key goes with it.
    nlohmann::json params = nlohmann::json::object();
    const nlohmann::json current = stageParams();  // named: items() of a temporary dangles in the range-for (C++20)
    for (const auto& [key, value] : current.items()) {
        const ParamSpec* spec = schema_->Find(key);
        if (!spec || spec->advanced || key == "backend") params[key] = value;
    }
    model_.setStageParams(stage_, params);
}

bool StageCard::changed(const ParamSpec& spec) const {
    const nlohmann::json p = stageParams();
    if (!p.contains(spec.key)) return false;
    const nlohmann::json& v = p[spec.key];
    if (v.is_number() && spec.def.is_number()) return std::fabs(v.get<double>() - spec.def.get<double>()) > 1e-9;
    return v != spec.def;
}

void StageCard::refresh() {
    updating_ = true;
    const StageEntry* e = model_.stageEntry(stage_);
    enabled_->setChecked(e && e->enabled);
    form_->setEnabled(e && e->enabled);
    for (const auto& pull : pull_) pull();
    // rows: the backend's own, RowState (conditions on the other parameters, the plan's inputs), changed values in accent
    const nlohmann::json params = e ? e->params : nlohmann::json::object();
    const nlohmann::json effective = EffectiveStageParams(stage_, params);
    std::set<std::string> inputs;
    for (const auto& d : model_.plan().stages)
        if (d.name == stage_)
            for (const auto& [pass, fp] : d.inputs) inputs.insert(pass);
    const std::string backend = currentBackend();
    const nlohmann::json folded = NormalizeStageParams(stage_, params);
    QStringList moreChanged, hiddenChanged;
    bool anyChanged = false;
    for (const Row& row : rows_) {
        const ParamSpec& spec = *row.spec;
        const bool ofBackend = spec.backends.empty() || std::find(spec.backends.begin(), spec.backends.end(), backend) != spec.backends.end();
        const bool hasList = spec.formChoices.empty() || FormChoicesFor(spec, backend) != nullptr;
        const ParamRowState st = RowState(spec, effective, inputs);
        const bool visible = ofBackend && hasList && st.visible;
        row.form->setRowVisible(row.index, visible);
        row.field->setEnabled(st.enabled);
        row.note->setText(QString::fromStdString(st.note));
        row.note->setVisible(!st.enabled && !st.note.empty());
        // a value of a row hidden by its condition (the threshold with the filter off) is still in the fingerprint:
        // the card names it and «Сбросить» drops it; a key of another backend is not (EffectiveStageParams drops it)
        const bool differs = ofBackend && hasList && changed(spec);
        const bool isChanged = visible && differs;
        SetRole(row.label, isChanged ? "changed" : "");
        row.reset->setVisible(isChanged);
        row.resetAction->setEnabled(isChanged);
        anyChanged = anyChanged || differs;
        const QString item = QString("%1 %2").arg(QString::fromStdString(spec.label).toLower(),
                                                  QString::fromStdString(ParamValueLabel(spec, folded.contains(spec.key) ? folded[spec.key] : spec.def, backend)));
        if (differs && !visible) hiddenChanged << item;
        else if (isChanged && spec.more) moreChanged << item;
    }
    auto list = [](const QStringList& l) { return l.size() > 3 ? l.mid(0, 3).join(", ") + ", …" : l.join(", "); };
    if (moreToggle_) {
        const bool open = moreToggle_->isChecked();
        QString text = open ? tr("Ещё настройки ▾") : tr("Ещё настройки ▸");
        if (!open && !moreChanged.isEmpty()) text += tr(" · изменено: %1").arg(list(moreChanged));  // the collapsed block says what it hides
        if (!hiddenChanged.isEmpty()) text += tr(" · не действует: %1").arg(list(hiddenChanged));
        moreToggle_->setText(text);
    }
    resetCard_->setVisible(anyChanged);
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
