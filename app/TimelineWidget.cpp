#include "TimelineWidget.h"

#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QStyle>
#include <algorithm>
#include <cmath>

#include "AppModel.h"
#include "Theme.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

TimelineWidget::TimelineWidget(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(16, 6, 16, 6);
    layout->setSpacing(8);
    auto* toStart = new QToolButton(this);
    toStart->setIcon(ThemeIcon(Glyph::ToStart));
    toStart->setToolTip(tr("В начало"));
    auto* prev = new QToolButton(this);
    prev->setIcon(ThemeIcon(Glyph::Prev));
    prev->setToolTip(tr("Кадр назад (,)"));
    play_ = new QToolButton(this);
    play_->setIcon(ThemeIcon(Glyph::Play));
    play_->setToolTip(tr("Play / Pause (пробел)"));
    auto* next = new QToolButton(this);
    next->setIcon(ThemeIcon(Glyph::Next));
    next->setToolTip(tr("Кадр вперёд (.)"));
    loop_ = new QToolButton(this);
    loop_->setIcon(ThemeIcon(Glyph::Loop));
    loop_->setCheckable(true);
    loop_->setChecked(model_.loop());
    loop_->setToolTip(tr("Зациклить воспроизведение (L)"));
    slider_ = new QSlider(Qt::Horizontal, this);
    slider_->setTracking(true);
    frame_ = new QSpinBox(this);
    frame_->setMinimumWidth(80);
    frame_->setToolTip(tr("Кадр таймлайна (в частоте справа)"));
    info_ = new QLabel(this);
    fps_ = new QComboBox(this);
    fps_->setToolTip(tr("Частота кадров таймлайна: источники с разной частотой сопоставляются по времени"));
    fps_->setVisible(false);
    for (QToolButton* b : {toStart, prev, play_, next, loop_}) SetRole(b, "icon");
    layout->addWidget(toStart);
    layout->addWidget(prev);
    layout->addWidget(play_);
    layout->addWidget(next);
    layout->addWidget(loop_);
    layout->addWidget(slider_, 1);
    layout->addWidget(frame_);
    layout->addWidget(info_);
    layout->addWidget(fps_);

    connect(toStart, &QToolButton::clicked, this, [this] { model_.setTime(0.0); });
    connect(prev, &QToolButton::clicked, this, [this] { model_.stepFrame(-1); });
    connect(next, &QToolButton::clicked, this, [this] { model_.stepFrame(+1); });
    connect(play_, &QToolButton::clicked, this, [this] { model_.setPlaying(!model_.playing()); });
    connect(loop_, &QToolButton::toggled, this, [this](bool on) { model_.setLoop(on); });
    connect(&model_, &AppModel::loopChanged, this, [this](bool on) {
        const QSignalBlocker b(loop_);
        loop_->setChecked(on);
    });
    connect(slider_, &QSlider::valueChanged, this, [this](int v) { model_.setFrame(v); });
    connect(frame_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int v) { model_.setFrame(v); });
    connect(fps_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int i) {
        const auto choices = model_.fpsChoices();
        if (i >= 0 && static_cast<size_t>(i) < choices.size()) model_.setTimelineFps(choices[static_cast<size_t>(i)]);
    });
    connect(&model_, &AppModel::timeChanged, this, [this](double) { refresh(); });
    connect(&model_, &AppModel::timelineFpsChanged, this, [this] { refresh(); });
    connect(&model_, &AppModel::stateChanged, this, [this] { refresh(); });
    connect(&model_, &AppModel::sourcesChanged, this, [this] {
        refreshRates();
        refresh();
    });
    connect(&model_, &AppModel::playingChanged, this, [this](bool p) { play_->setIcon(ThemeIcon(p ? Glyph::Pause : Glyph::Play)); });
    refreshRates();
    refresh();
}

void TimelineWidget::refreshRates() {
    const auto choices = model_.fpsChoices();
    const QSignalBlocker b(fps_);
    fps_->clear();
    for (const auto& r : choices) fps_->addItem(tr("%1 fps").arg(QString::number(r.ToDouble(), 'g', 5)));
    fps_->setVisible(choices.size() > 1);
}

void TimelineWidget::refresh() {
    const int64_t count = model_.timelineFrameCount();
    const int max = static_cast<int>(std::max<int64_t>(0, count - 1));
    const QSignalBlocker b1(slider_), b2(frame_), b3(fps_);
    slider_->setRange(0, max);
    frame_->setRange(0, max);
    const int cur = static_cast<int>(model_.timelineFrame());
    slider_->setValue(cur);
    frame_->setValue(cur);
    const Rational tl = model_.timelineFps();
    const auto choices = model_.fpsChoices();
    for (size_t i = 0; i < choices.size(); ++i)
        if (std::abs(choices[i].ToDouble() - tl.ToDouble()) < 1e-6) fps_->setCurrentIndex(static_cast<int>(i));
    info_->setText(tr("%1 / %2   %3 #%4   %5 fps")
                       .arg(QString::fromStdString(FormatTimecode(model_.time())))
                       .arg(QString::fromStdString(FormatTimecode(model_.lastTime())))
                       .arg(QString::fromStdString(model_.baseSource()))
                       .arg(model_.baseFrame())
                       .arg(QString::number(tl.ToDouble(), 'g', 5)));
}

}  // namespace dlssvid
