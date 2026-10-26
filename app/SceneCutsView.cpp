#include "SceneCutsView.h"

#include <QApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QToolButton>
#include <QVBoxLayout>

#include "AppModel.h"
#include "SourceNames.h"
#include "Theme.h"
#include "passes/ScenesPass.h"
#include "viewport/ViewportState.h"

namespace dlssvid {
namespace {

constexpr int kChips = 8;  // chips in a row; the rest in the «ещё N…» menu

void Clear(QHBoxLayout* row) {
    while (QLayoutItem* item = row->takeAt(0)) {
        if (QWidget* w = item->widget()) {  // out of the view now, deleted when the event loop runs (a refresh may come from a signal of it)
            w->hide();
            w->setParent(nullptr);
            w->deleteLater();
        }
        delete item;
    }
}

// what the stages below are called in «пересчитаются …»
QString StageWord(const std::string& stage) {
    if (stage == "flow") return QCoreApplication::translate("SceneCutsView", "векторы");
    if (stage == "upscale") return QCoreApplication::translate("SceneCutsView", "апскейл");
    if (stage == "nr") return QCoreApplication::translate("SceneCutsView", "улучшение");
    if (stage == "fg") return QCoreApplication::translate("SceneCutsView", "генерация кадров");
    return QString::fromStdString(stage);
}

}  // namespace

SceneCutsView::SceneCutsView(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    auto* column = new QVBoxLayout(this);
    column->setContentsMargins(0, 4, 0, 0);
    column->setSpacing(4);
    summary_ = new QLabel(this);
    summary_->setWordWrap(true);
    column->addWidget(summary_);
    chips_ = new QWidget(this);
    chipRow_ = new QHBoxLayout(chips_);
    chipRow_->setContentsMargins(0, 0, 0, 0);
    chipRow_->setSpacing(4);
    column->addWidget(chips_);
    near_ = new QWidget(this);
    nearRow_ = new QHBoxLayout(near_);
    nearRow_->setContentsMargins(0, 0, 0, 0);
    nearRow_->setSpacing(4);
    column->addWidget(near_);
    change_ = new QLabel(this);
    change_->setWordWrap(true);
    SetRole(change_, "hint");
    column->addWidget(change_);
    connect(&model_, &AppModel::planChanged, this, &SceneCutsView::refresh);
    refresh();
}

QToolButton* SceneCutsView::Chip(int64_t frame, const QString& text, const QString& tip, QWidget* parent) {
    auto* chip = new QToolButton(parent);
    SetRole(chip, "chip");
    chip->setText(text);
    chip->setToolTip(tip);
    connect(chip, &QToolButton::clicked, this, [this, frame] {
        // Alt: the frame before the cut - the last one of the old shot
        model_.seekToSourceFrame(QApplication::keyboardModifiers() & Qt::AltModifier ? frame - 1 : frame);
    });
    return chip;
}

void SceneCutsView::refresh() {
    Clear(chipRow_);
    Clear(nearRow_);
    const StageDecision* d = nullptr;
    for (const auto& s : model_.plan().stages)
        if (s.name == kScenesPass) d = &s;
    const StageEntry* e = model_.stageEntry(kScenesPass);
    const bool show = e && e->enabled && d && d->sceneCuts.known;
    setVisible(show);
    if (!show) return;
    const SceneCutsPlan& sc = d->sceneCuts;
    const double fps = model_.sourceInfo().fps > 0 ? model_.sourceInfo().fps : 24.0;
    auto at = [fps](int64_t f) { return QString::fromStdString(FormatTimecode(static_cast<double>(f) / fps)); };  // the timeline's timecode
    if (sc.off) summary_->setText(tr("Порог 0: стадии работают без сбросов на склейках"));
    else if (sc.cuts.empty()) summary_->setText(tr("Склеек не найдено — стадии работают без сбросов"));
    else summary_->setText(tr("Найдено %1").arg(Plural(static_cast<int>(sc.count), tr("склейка"), tr("склейки"), tr("склеек"))));
    const QString tipOf = tr("кадр %1 · клик — показать, Alt+клик — кадр до склейки");
    for (size_t i = 0; i < sc.cuts.size() && i < static_cast<size_t>(kChips); ++i)
        chipRow_->addWidget(Chip(sc.cuts[i], at(sc.cuts[i]), tipOf.arg(sc.cuts[i]), chips_));
    if (sc.cuts.size() > static_cast<size_t>(kChips)) {
        auto* more = new QToolButton(chips_);
        SetRole(more, "chip");
        more->setText(tr("ещё %1…").arg(sc.cuts.size() - kChips));
        more->setPopupMode(QToolButton::InstantPopup);
        auto* menu = new QMenu(more);
        for (size_t i = kChips; i < sc.cuts.size(); ++i) {
            const int64_t f = sc.cuts[i];
            QAction* a = menu->addAction(at(f) + " · " + tr("кадр %1").arg(f));
            connect(a, &QAction::triggered, this, [this, f] { model_.seekToSourceFrame(QApplication::keyboardModifiers() & Qt::AltModifier ? f - 1 : f); });
        }
        more->setMenu(menu);
        chipRow_->addWidget(more);
    }
    chipRow_->addStretch(1);
    chips_->setVisible(!sc.cuts.empty());
    // the highest scores just below the threshold: where a missed cut would be
    if (!sc.off && !sc.nearMisses.empty()) {
        auto* label = new QLabel(tr("Почти склейки:"), near_);
        SetRole(label, "hint");
        nearRow_->addWidget(label);
        for (const auto& [f, score] : sc.nearMisses)
            nearRow_->addWidget(Chip(f, at(f) + " · " + QString::number(score, 'f', 2),
                                     tr("кадр %1 · разница %2 при пороге %3 · клик — показать").arg(f).arg(score, 0, 'f', 2).arg(sc.threshold, 0, 'f', 2), near_));
        nearRow_->addStretch(1);
    }
    near_->setVisible(!sc.off && !sc.nearMisses.empty());
    // a new threshold over the stored scores: what changes below
    QString change;
    if (sc.rederive) {
        QStringList again;
        for (const auto& s : model_.plan().stages)
            if (s.name != kScenesPass && s.action == "run" && s.diff.contains("inputs") && s.diff["inputs"].is_array())
                for (const auto& in : s.diff["inputs"])
                    if (in == kScenesPass) again << StageWord(s.name);
        const QString now = Plural(static_cast<int>(sc.count), tr("склейка"), tr("склейки"), tr("склеек"));
        if (!again.isEmpty())
            change = (sc.before >= 0 ? tr("При этих настройках: %1 (сейчас %2)").arg(now).arg(sc.before) : tr("При этих настройках: %1").arg(now)) + " — " +
                     tr("пересчитаются %1").arg(again.join(", "));
        else change = tr("Список не изменится — стадии ниже не пересчитываются");
    }
    change_->setText(change);
    change_->setVisible(!change.isEmpty());
}

}  // namespace dlssvid
