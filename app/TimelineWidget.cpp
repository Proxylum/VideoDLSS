#include "TimelineWidget.h"

#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QStyle>
#include <algorithm>

#include "AppModel.h"

namespace dlssvid {

TimelineWidget::TimelineWidget(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(4, 2, 4, 2);
    auto* toStart = new QToolButton(this);
    toStart->setIcon(style()->standardIcon(QStyle::SP_MediaSkipBackward));
    auto* prev = new QToolButton(this);
    prev->setIcon(style()->standardIcon(QStyle::SP_MediaSeekBackward));
    prev->setToolTip(tr("Кадр назад (,)"));
    play_ = new QToolButton(this);
    play_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    play_->setToolTip(tr("Play / Pause (пробел)"));
    auto* next = new QToolButton(this);
    next->setIcon(style()->standardIcon(QStyle::SP_MediaSeekForward));
    next->setToolTip(tr("Кадр вперёд (.)"));
    slider_ = new QSlider(Qt::Horizontal, this);
    slider_->setTracking(true);
    frame_ = new QSpinBox(this);
    frame_->setMinimumWidth(80);
    info_ = new QLabel(this);
    layout->addWidget(toStart);
    layout->addWidget(prev);
    layout->addWidget(play_);
    layout->addWidget(next);
    layout->addWidget(slider_, 1);
    layout->addWidget(frame_);
    layout->addWidget(info_);

    connect(toStart, &QToolButton::clicked, this, [this] { model_.setFrame(0); });
    connect(prev, &QToolButton::clicked, this, [this] { model_.stepFrame(-1); });
    connect(next, &QToolButton::clicked, this, [this] { model_.stepFrame(+1); });
    connect(play_, &QToolButton::clicked, this, [this] { model_.setPlaying(!model_.playing()); });
    connect(slider_, &QSlider::valueChanged, this, [this](int v) { model_.setFrame(v); });
    connect(frame_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int v) { model_.setFrame(v); });
    connect(&model_, &AppModel::frameChanged, this, [this](qint64) { refresh(); });
    connect(&model_, &AppModel::sourcesChanged, this, [this] { refresh(); });
    connect(&model_, &AppModel::playingChanged, this, [this](bool p) { play_->setIcon(style()->standardIcon(p ? QStyle::SP_MediaPause : QStyle::SP_MediaPlay)); });
    refresh();
}

void TimelineWidget::refresh() {
    const int64_t count = model_.store().FrameCount();
    const int max = static_cast<int>(std::max<int64_t>(0, count - 1));
    const QSignalBlocker b1(slider_), b2(frame_);
    slider_->setRange(0, max);
    frame_->setRange(0, max);
    slider_->setValue(static_cast<int>(model_.state().frame));
    frame_->setValue(static_cast<int>(model_.state().frame));
    const Rational fps = model_.store().Fps();
    info_->setText(tr("/ %1  %2 fps").arg(count).arg(fps.num > 0 ? QString::number(fps.ToDouble(), 'f', 2) : "?"));
}

}  // namespace dlssvid
