#include "CompareBar.h"

#include <QButtonGroup>
#include <QFrame>
#include <QLabel>
#include <QMenu>
#include <QSignalBlocker>

#include "AppModel.h"
#include "SourceNames.h"

namespace dlssvid {

namespace {

QToolButton* ModeButton(QWidget* parent, const QString& text, const QString& tip) {
    auto* b = new QToolButton(parent);
    b->setText(text);
    b->setToolTip(tip);
    b->setCheckable(true);
    b->setAutoRaise(true);
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
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 2, 6, 2);
    layout->setSpacing(4);
    layout->addWidget(new QLabel(tr("Режим:"), this));
    beforeAfter_ = ModeButton(this, tr("До | После"), tr("Шторка между исходником и результатом (W)"));
    afterOnly_ = ModeButton(this, tr("Только после"), tr("Один источник на весь вьюпорт"));
    grid_ = ModeButton(this, tr("Сетка 2×2"), tr("Четыре источника рядом (Ctrl+3)"));
    auto* modes = new QButtonGroup(this);
    modes->setExclusive(true);
    for (QToolButton* b : {beforeAfter_, afterOnly_, grid_}) {
        modes->addButton(b);
        layout->addWidget(b);
    }
    layout->addWidget(Separator(this));
    layout->addWidget(new QLabel(tr("Слой:"), this));
    chipsLayout_ = new QHBoxLayout();
    chipsLayout_->setSpacing(4);
    layout->addLayout(chipsLayout_);
    layout->addStretch(1);
    engineer_ = ModeButton(this, tr("Инженерный режим"), tr("Стек слоёв, настройки слоя, шторки и сетки в инспекторе (Ctrl+E)"));
    layout->addWidget(engineer_);

    connect(beforeAfter_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::BeforeAfter); });
    connect(afterOnly_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::AfterOnly); });
    connect(grid_, &QToolButton::clicked, this, [this] { model_.setCompareView(AppModel::CompareView::Grid); });
    connect(engineer_, &QToolButton::toggled, this, [this](bool on) { model_.setEngineerMode(on); });
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
        b->setText(chip.label);
        b->setToolTip(chip.tooltip);
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
            b->setToolTip(chip.tooltip + "\n" + tr("%n предыдущих версий: меню справа", nullptr, static_cast<int>(chip.versions.size())));
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
