#include "CompareBar.h"

#include <QButtonGroup>
#include <QFrame>
#include <QLabel>
#include <QMenu>
#include <QSignalBlocker>

#include "AppModel.h"
#include "SourceNames.h"
#include "Theme.h"

namespace dlssvid {

namespace {

QToolButton* ModeButton(QWidget* parent, const QString& text, const QString& tip) {
    auto* b = new QToolButton(parent);
    b->setText(text);
    b->setToolTip(tip);
    b->setCheckable(true);
    b->setAutoRaise(true);
    SetRole(b, "segment");
    return b;
}

QFrame* Separator(QWidget* parent) {
    auto* f = new QFrame(parent);
    f->setFrameShape(QFrame::VLine);
    f->setFrameShadow(QFrame::Sunken);
    return f;
}

}  // namespace

CompareBar::CompareBar(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    setObjectName("compareBar");
    setAttribute(Qt::WA_StyledBackground, true);
    SetRole(this, "bar");
    setMinimumHeight(48);
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(16, 6, 16, 6);
    layout->setSpacing(10);
    auto* modeLabel = new QLabel(tr("Режим"), this);
    SetRole(modeLabel, "muted");
    layout->addWidget(modeLabel);
    beforeAfter_ = ModeButton(this, tr("До | После"), tr("Шторка между исходником и результатом (W)"));
    afterOnly_ = ModeButton(this, tr("Только после"), tr("Один источник на весь вьюпорт"));
    grid_ = ModeButton(this, tr("Сетка 2×2"), tr("Четыре источника рядом (Ctrl+3)"));
    auto* modes = new QButtonGroup(this);
    modes->setExclusive(true);
    auto* modeGroup = new QFrame(this);
    SetRole(modeGroup, "segments");
    auto* modeLayout = new QHBoxLayout(modeGroup);
    modeLayout->setContentsMargins(0, 0, 0, 0);
    modeLayout->setSpacing(0);
    for (QToolButton* b : {beforeAfter_, afterOnly_, grid_}) {
        modes->addButton(b);
        modeLayout->addWidget(b);
    }
    layout->addWidget(modeGroup);
    layout->addWidget(Separator(this));
    auto* layerLabel = new QLabel(tr("Слой"), this);
    SetRole(layerLabel, "muted");
    layout->addWidget(layerLabel);
    chipsLayout_ = new QHBoxLayout();
    chipsLayout_->setSpacing(8);
    layout->addLayout(chipsLayout_);
    layout->addStretch(1);
    engineer_ = new QCheckBox(tr("Инженерный режим"), this);
    engineer_->setToolTip(tr("Стек слоёв, настройки слоя, шторки и сетки в инспекторе (Ctrl+E)"));
    layout->addWidget(engineer_);

    connect(beforeAfter_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::BeforeAfter); });
    connect(afterOnly_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::AfterOnly); });
    connect(grid_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::Grid); });
    connect(engineer_, &QCheckBox::toggled, this, [this](bool on) { model_.setEngineerMode(on); });
    connect(&model_, &AppModel::engineerModeChanged, this, [this](bool on) {
        const QSignalBlocker b(engineer_);
        engineer_->setChecked(on);
    });
    connect(&model_, &AppModel::sourcesChanged, this, [this] {
        rebuildChips();
        refresh();
    });
    connect(&model_, &AppModel::stateChanged, this, &CompareBar::refresh);
    rebuildChips();
    refresh();
}

void CompareBar::rebuildChips() {
    for (QToolButton* c : chips_) {
        chipsLayout_->removeWidget(c);
        c->deleteLater();
    }
    chips_.clear();
    chipSources_.clear();
    for (const AppModel::Chip& chip : model_.chips()) {
        auto* b = new QToolButton(this);
        SetRole(b, "chip");
        b->setText(chip.label);
        QString tip = chip.tooltip;
        if (chip.hotkey > 0) tip += "\n" + tr("клавиша %1").arg(chip.hotkey);
        b->setToolTip(tip);
        b->setCheckable(true);
        b->setAutoRaise(true);
        const std::string source = chip.source;
        connect(b, &QToolButton::clicked, this, [this, source] { model_.showSource(source); });
        if (!chip.versions.empty()) {
            auto* menu = new QMenu(b);
            menu->addAction(tr("Текущая версия"), this, [this, source] { model_.showSource(source); });
            for (const ViewportSource& v : chip.versions) {
                QString text = VersionLabel(v.version);
                if (v.manifest && v.manifest->paramsCanonical.is_object() && !v.manifest->paramsCanonical.empty())
                    text += "   " + QString::fromStdString(v.manifest->paramsCanonical.dump());
                const std::string name = v.name;
                menu->addAction(text, this, [this, name] { model_.showSource(name); })->setToolTip(SourceTooltip(v));
            }
            b->setMenu(menu);
            b->setPopupMode(QToolButton::MenuButtonPopup);
            b->setToolTip(tip + "\n" + tr("%1: меню справа").arg(Plural(static_cast<int>(chip.versions.size()), tr("предыдущая версия"), tr("предыдущие версии"), tr("предыдущих версий"))));
        }
        chipsLayout_->addWidget(b);
        chips_.push_back(b);
        chipSources_.push_back(source);
    }
}

void CompareBar::refresh() {
    const QSignalBlocker b1(beforeAfter_), b2(afterOnly_), b3(grid_), b4(engineer_);
    const AppModel::CompareView view = model_.compareView();
    beforeAfter_->setChecked(view == AppModel::CompareView::BeforeAfter);
    afterOnly_->setChecked(view == AppModel::CompareView::AfterOnly);
    grid_->setChecked(view == AppModel::CompareView::Grid);
    engineer_->setChecked(model_.engineerMode());
    // the chip of the source on the «after» side (or of the single view) is lit, versions light their pass
    std::string shown;
    const ViewportState& st = model_.state();
    if (const LayerState* after = st.CompareSide(true)) shown = after->source;
    else if (view == AppModel::CompareView::AfterOnly && !st.layers.empty()) shown = st.layers[0].source;
    const std::string shownPass = SplitSourceVersion(shown).first;
    for (size_t i = 0; i < chips_.size(); ++i) {
        const QSignalBlocker b(chips_[i]);
        chips_[i]->setChecked(!shown.empty() && chipSources_[i] == shownPass);
    }
}

}  // namespace dlssvid
