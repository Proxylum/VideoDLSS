#include "InspectorPanel.h"

#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

#include "AppModel.h"
#include "SourceNames.h"
#include "ViewportWindow.h"

namespace dlssvid {

namespace {
const char* kDisplayNames[] = {"color", "grayscale", "viridis", "turbo", "mv_hsv", "mv_arrows", "mv_magnitude", "mask_fill", "mask_contour"};
const char* kBlendNames[] = {"normal", "difference", "multiply", "screen"};
}  // namespace

InspectorPanel::InspectorPanel(AppModel& model, ViewportWindow& viewport, QWidget* parent) : QWidget(parent), model_(model), viewport_(viewport) {
    auto* layout = new QVBoxLayout(this);

    // mode + layer stack
    auto* modeBox = new QGroupBox(tr("Режим"), this);
    auto* modeLayout = new QVBoxLayout(modeBox);
    mode_ = new QComboBox(modeBox);
    mode_->addItems({tr("Одиночный"), tr("Наложение"), tr("Сетка 2x2")});
    modeLayout->addWidget(mode_);
    layers_ = new QListWidget(modeBox);
    layers_->setMaximumHeight(110);
    modeLayout->addWidget(layers_);
    auto* layerButtons = new QHBoxLayout();
    auto* addLayer = new QPushButton(tr("+ слой"), modeBox);
    auto* removeLayer = new QPushButton(tr("− слой"), modeBox);
    layerButtons->addWidget(addLayer);
    layerButtons->addWidget(removeLayer);
    modeLayout->addLayout(layerButtons);
    layout->addWidget(modeBox);

    // layer settings
    auto* layerBox = new QGroupBox(tr("Слой"), this);
    auto* form = new QFormLayout(layerBox);
    source_ = new QComboBox(layerBox);
    display_ = new QComboBox(layerBox);
    for (const char* n : kDisplayNames) display_->addItem(n);
    autoRange_ = new QCheckBox(tr("авто"), layerBox);
    minValue_ = new QDoubleSpinBox(layerBox);
    minValue_->setRange(-1e9, 1e9);
    minValue_->setDecimals(4);
    maxValue_ = new QDoubleSpinBox(layerBox);
    maxValue_->setRange(-1e9, 1e9);
    maxValue_->setDecimals(4);
    invert_ = new QCheckBox(tr("инверсия"), layerBox);
    opacity_ = new QSlider(Qt::Horizontal, layerBox);
    opacity_->setRange(0, 100);
    opacityValue_ = new QSpinBox(layerBox);
    opacityValue_->setRange(0, 100);
    opacityValue_->setSuffix(" %");
    blend_ = new QComboBox(layerBox);
    for (const char* n : kBlendNames) blend_->addItem(n);
    visible_ = new QCheckBox(tr("видимый"), layerBox);
    solo_ = new QCheckBox(tr("соло"), layerBox);
    arrowStep_ = new QSpinBox(layerBox);
    arrowStep_->setRange(4, 64);
    mvScale_ = new QDoubleSpinBox(layerBox);
    mvScale_->setRange(0, 1e4);
    mvScale_->setSpecialValueText(tr("авто"));
    auto* rangeRow = new QHBoxLayout();
    rangeRow->addWidget(autoRange_);
    rangeRow->addWidget(minValue_);
    rangeRow->addWidget(maxValue_);
    auto* opacityRow = new QHBoxLayout();
    opacityRow->addWidget(opacity_, 1);
    opacityRow->addWidget(opacityValue_);
    auto* flagsRow = new QHBoxLayout();
    flagsRow->addWidget(visible_);
    flagsRow->addWidget(solo_);
    flagsRow->addWidget(invert_);
    form->addRow(tr("Источник"), source_);
    form->addRow(tr("Отображение"), display_);
    form->addRow(tr("Диапазон"), rangeRow);
    form->addRow(tr("Прозрачность"), opacityRow);
    form->addRow(tr("Blend"), blend_);
    form->addRow("", flagsRow);
    form->addRow(tr("Шаг стрелок"), arrowStep_);
    form->addRow(tr("Масштаб MV"), mvScale_);
    layout->addWidget(layerBox);

    // wipe
    auto* wipeBox = new QGroupBox(tr("Шторка (wipe)"), this);
    auto* wipeForm = new QFormLayout(wipeBox);
    wipeEnabled_ = new QCheckBox(tr("включена"), wipeBox);
    wipeAxis_ = new QComboBox(wipeBox);
    wipeAxis_->addItems({tr("вертикальная"), tr("горизонтальная")});
    wipePos_ = new QSlider(Qt::Horizontal, wipeBox);
    wipePos_->setRange(0, 1000);
    wipeA_ = new QComboBox(wipeBox);
    wipeB_ = new QComboBox(wipeBox);
    wipeForm->addRow(wipeEnabled_, wipeAxis_);
    wipeForm->addRow(tr("Позиция"), wipePos_);
    wipeForm->addRow(tr("Слой A"), wipeA_);
    wipeForm->addRow(tr("Слой B"), wipeB_);
    layout->addWidget(wipeBox);

    // grid
    auto* gridBox = new QGroupBox(tr("Сетка 2x2"), this);
    auto* gridForm = new QFormLayout(gridBox);
    for (int i = 0; i < 4; ++i) {
        grid_[static_cast<size_t>(i)] = new QComboBox(gridBox);
        gridForm->addRow(tr("Ячейка %1").arg(i + 1), grid_[static_cast<size_t>(i)]);
    }
    layout->addWidget(gridBox);
    sections_ = {modeBox, layerBox, wipeBox, gridBox};

    probe_ = new QLabel(tr("Пробник: наведите на кадр"), this);
    probe_->setWordWrap(true);
    probe_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(probe_);
    layout->addStretch(1);

    // ---- wiring ----
    connect(mode_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int i) {
        if (updating_) return;
        model_.setMode(static_cast<ViewMode>(std::clamp(i, 0, 2)));
    });
    connect(layers_, &QListWidget::currentRowChanged, this, [this](int row) {
        if (updating_ || row < 0) return;
        model_.state().selectedLayer = row;
        refresh();
    });
    connect(addLayer, &QPushButton::clicked, this, [this] {
        auto& st = model_.state();
        if (st.mode != ViewMode::Overlay) model_.setMode(ViewMode::Overlay);
        std::vector<std::string> names;
        for (const QString& n : model_.sourceNames()) names.push_back(n.toStdString());
        st.AddOverlayLayer(st.NextUnusedSource(names));  // 100 %, the next source not on the stack yet
        model_.notifyStateChanged();
        refresh();
    });
    connect(removeLayer, &QPushButton::clicked, this, [this] {
        model_.state().RemoveLayer(model_.state().selectedLayer);
        model_.notifyStateChanged();
        refresh();
    });
    for (QComboBox* c : {source_, display_, blend_}) connect(c, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { applyLayer(); });
    for (QCheckBox* c : {autoRange_, invert_, visible_, solo_}) connect(c, &QCheckBox::toggled, this, [this](bool) { applyLayer(); });
    connect(minValue_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { applyLayer(); });
    connect(maxValue_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { applyLayer(); });
    connect(mvScale_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { applyLayer(); });
    connect(arrowStep_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { applyLayer(); });
    connect(opacity_, &QSlider::valueChanged, this, [this](int v) {
        if (updating_) return;
        const QSignalBlocker b(opacityValue_);
        opacityValue_->setValue(v);
        applyLayer();
    });
    connect(opacityValue_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int v) {
        if (updating_) return;
        const QSignalBlocker b(opacity_);
        opacity_->setValue(v);
        applyLayer();
    });
    connect(wipeEnabled_, &QCheckBox::toggled, this, [this](bool) { applyWipe(); });
    connect(wipeAxis_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { applyWipe(); });
    connect(wipePos_, &QSlider::valueChanged, this, [this](int) { applyWipe(); });
    connect(wipeA_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { applyWipe(); });
    connect(wipeB_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { applyWipe(); });
    for (QComboBox* g : grid_) connect(g, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { applyGrid(); });
    connect(&model_, &AppModel::stateChanged, this, &InspectorPanel::refresh);
    connect(&model_, &AppModel::sourcesChanged, this, &InspectorPanel::refresh);
    connect(&viewport_, &ViewportWindow::probe, this, &InspectorPanel::showProbe);
    connect(&model_, &AppModel::engineerModeChanged, this, &InspectorPanel::setEngineerMode);
    setEngineerMode(model_.engineerMode());
    refresh();
}

void InspectorPanel::setEngineerMode(bool on) {
    for (QGroupBox* g : sections_) g->setVisible(on);
}

void InspectorPanel::refresh() {
    updating_ = true;
    const ViewportState& st = model_.state();
    const QStringList names = model_.sourceNames();
    auto fillSources = [&](QComboBox* c, const std::string& current) {
        c->clear();
        c->addItems(names);
        if (!names.contains(QString::fromStdString(current))) c->addItem(QString::fromStdString(current));
        c->setCurrentText(QString::fromStdString(current));
    };
    mode_->setCurrentIndex(static_cast<int>(st.mode));
    layers_->clear();
    for (size_t i = 0; i < st.layers.size(); ++i) {
        const LayerState& l = st.layers[i];
        auto* item = new QListWidgetItem(
            QString("%1  %2  %3%").arg(i == 0 ? tr("база") : tr("слой %1").arg(i)).arg(HumanSourceName(l.source)).arg(static_cast<int>(std::lround(l.opacity * 100))));
        item->setToolTip(QString::fromStdString(l.source));
        layers_->addItem(item);
    }
    const int sel = std::clamp(st.selectedLayer, 0, static_cast<int>(st.layers.size()) - 1);
    layers_->setCurrentRow(sel);
    const LayerState& l = st.layers[static_cast<size_t>(sel)];
    fillSources(source_, l.source);
    display_->setCurrentIndex(static_cast<int>(l.display));
    autoRange_->setChecked(l.autoRange);
    minValue_->setEnabled(!l.autoRange);
    maxValue_->setEnabled(!l.autoRange);
    minValue_->setValue(l.minValue);
    maxValue_->setValue(l.maxValue);
    invert_->setChecked(l.invert);
    opacity_->setValue(static_cast<int>(std::lround(l.opacity * 100)));
    opacityValue_->setValue(static_cast<int>(std::lround(l.opacity * 100)));
    blend_->setCurrentIndex(static_cast<int>(l.blend));
    visible_->setChecked(l.visible);
    solo_->setChecked(l.solo);
    arrowStep_->setValue(l.arrowStep);
    mvScale_->setValue(l.mvScale);

    wipeEnabled_->setChecked(st.wipe.enabled);
    wipeAxis_->setCurrentIndex(st.wipe.vertical ? 0 : 1);
    wipePos_->setValue(static_cast<int>(std::lround(st.wipe.position * 1000)));
    for (QComboBox* c : {wipeA_, wipeB_}) {
        c->clear();
        for (size_t i = 0; i < st.layers.size(); ++i) c->addItem(QString("%1: %2").arg(i).arg(QString::fromStdString(st.layers[i].source)));
    }
    wipeA_->setCurrentIndex(std::clamp(st.wipe.layerA, 0, static_cast<int>(st.layers.size()) - 1));
    wipeB_->setCurrentIndex(std::clamp(st.wipe.layerB, 0, static_cast<int>(st.layers.size()) - 1));
    for (size_t i = 0; i < 4; ++i) fillSources(grid_[i], st.gridSources[i]);
    updating_ = false;
}

void InspectorPanel::applyLayer() {
    if (updating_) return;
    ViewportState& st = model_.state();
    const int sel = std::clamp(st.selectedLayer, 0, static_cast<int>(st.layers.size()) - 1);
    LayerState& l = st.layers[static_cast<size_t>(sel)];
    const std::string newSource = source_->currentText().toStdString();
    if (newSource != l.source && !newSource.empty()) {
        l.source = newSource;
        l.display = LayerState::DefaultDisplayFor(newSource);
        l.autoRange = true;
        model_.notifyStateChanged();
        refresh();
        return;
    }
    l.display = static_cast<DisplayMode>(std::clamp(display_->currentIndex(), 0, 8));
    l.autoRange = autoRange_->isChecked();
    l.minValue = static_cast<float>(minValue_->value());
    l.maxValue = static_cast<float>(maxValue_->value());
    l.invert = invert_->isChecked();
    l.opacity = static_cast<float>(opacityValue_->value()) / 100.f;
    l.blend = static_cast<BlendMode>(std::clamp(blend_->currentIndex(), 0, 3));
    l.visible = visible_->isChecked();
    l.solo = solo_->isChecked();
    l.arrowStep = arrowStep_->value();
    l.mvScale = static_cast<float>(mvScale_->value());
    minValue_->setEnabled(!l.autoRange);
    maxValue_->setEnabled(!l.autoRange);
    model_.notifyStateChanged();
}

void InspectorPanel::applyWipe() {
    if (updating_) return;
    WipeState& w = model_.state().wipe;
    w.enabled = wipeEnabled_->isChecked();
    w.vertical = wipeAxis_->currentIndex() == 0;
    w.position = static_cast<float>(wipePos_->value()) / 1000.f;
    w.layerA = std::max(0, wipeA_->currentIndex());
    w.layerB = std::max(0, wipeB_->currentIndex());
    model_.notifyStateChanged();
}

void InspectorPanel::applyGrid() {
    if (updating_) return;
    for (size_t i = 0; i < 4; ++i) {
        const std::string s = grid_[i]->currentText().toStdString();
        if (!s.empty()) model_.state().gridSources[i] = s;
    }
    model_.notifyStateChanged();
}

void InspectorPanel::showProbe(int cell, float imageX, float imageY, bool inside) {
    if (!inside) {
        probe_->setText(tr("Пробник: вне кадра"));
        return;
    }
    const uint32_t x = static_cast<uint32_t>(imageX), y = static_cast<uint32_t>(imageY);
    QString text = tr("Пробник (%1, %2)%3").arg(x).arg(y).arg(model_.state().mode == ViewMode::Grid ? tr(" ячейка %1").arg(cell + 1) : "");
    const FrameTextures frame = model_.store().TexturesAt(model_.state().time);
    for (const auto& [name, tex] : frame) {
        // pass textures may have another resolution (mv_dlss at target size): scale the probe
        const uint32_t tx = std::min(tex.width - 1, static_cast<uint32_t>(imageX * tex.width / std::max(1u, model_.store().ImageWidth())));
        const uint32_t ty = std::min(tex.height - 1, static_cast<uint32_t>(imageY * tex.height / std::max(1u, model_.store().ImageHeight())));
        const auto v = viewport_.renderer().ReadTexel(tex, tx, ty);
        QString line;
        switch (tex.kind) {
            case TextureKind::Color: line = QString("%1: R %2 G %3 B %4").arg(QString::fromStdString(name)).arg(v[0], 0, 'f', 4).arg(v[1], 0, 'f', 4).arg(v[2], 0, 'f', 4); break;
            case TextureKind::Mv: line = QString("%1: u %2 v %3").arg(QString::fromStdString(name)).arg(v[0], 0, 'f', 3).arg(v[1], 0, 'f', 3); break;
            default: line = QString("%1: %2").arg(QString::fromStdString(name)).arg(v[0], 0, 'g', 6); break;
        }
        text += "\n" + line;
    }
    probe_->setText(text);
}

}  // namespace dlssvid
