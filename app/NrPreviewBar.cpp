#include "NrPreviewBar.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <vector>

#include "AppModel.h"
#include "PresetMenu.h"
#include "PreviewRunner.h"
#include "SourceNames.h"
#include "Theme.h"

namespace dlssvid {

namespace {
QPushButton* Small(const QString& text, QWidget* parent) {
    auto* b = new QPushButton(text, parent);
    SetRole(b, "small");
    ClaimButtonKeys(b);
    return b;
}
}  // namespace

NrPreviewBar::NrPreviewBar(AppModel& model, PreviewRunner& runner, QWidget* parent) : QWidget(parent), model_(model), runner_(runner) {
    auto* column = new QVBoxLayout(this);
    column->setContentsMargins(0, 6, 0, 0);
    column->setSpacing(4);

    auto* header = new QLabel(tr("Превью"), this);  // as the form's groups
    SetRole(header, "group");
    column->addWidget(header);
    auto* row1 = new QHBoxLayout;
    row1->setSpacing(6);
    frame_ = new SplitButton(this);
    SetRole(frame_, "split");
    frame_->setText(tr("Превью кадра"));
    frameMenu_ = new QMenu(frame_);
    frameMenu_->setToolTipsVisible(true);
    QAction* history = frameMenu_->addAction(tr("Кадр с историей"));
    frameMenu_->addSeparator();
    QAction* clear = frameMenu_->addAction(tr("Очистить превью…"));
    frame_->setMenu(frameMenu_);
    row1->addWidget(frame_);
    segment_ = new PresetButton(tr("Превью отрезка"), this);
    SetRole(segment_, "small");
    segmentMenu_ = new QMenu(segment_);
    segmentMenu_->setToolTipsVisible(true);
    for (const int s : {1, 3, 5, 10}) {
        QAction* a = segmentMenu_->addAction(tr("%1 с").arg(s));
        a->setData(s);
        connect(a, &QAction::triggered, this, [this, s] {
            QSettings().setValue("card/nr/previewSeconds", s);
            runner_.start({PreviewRunner::Kind::Segment, s});
        });
    }
    segment_->setMenu(segmentMenu_);
    row1->addWidget(segment_);
    auto_ = new QCheckBox(tr("обновлять при изменении"), this);
    ClaimButtonKeys(auto_);
    auto_->setToolTip(tr("Через 0,6 с после изменения настройки (ползунок — после отпускания) последнее превью кадра считается заново. "
                         "Модель каждый раз создаётся заново: ≈ 2 с"));
    row1->addStretch(1);
    column->addLayout(row1);
    column->addWidget(auto_);

    status_ = new QLabel(this);
    status_->setWordWrap(true);
    SetRole(status_, "hint");
    column->addWidget(status_);

    auto* row2 = new QHBoxLayout;
    row2->setSpacing(6);
    cancel_ = Small(tr("Отменить"), this);
    prepare_ = Small(tr("Посчитать этапы до улучшения"), this);
    beforeLabel_ = new QLabel(tr("До:"), this);
    SetRole(beforeLabel_, "hint");
    before_ = new QComboBox(this);
    before_->addItem(tr("Текущее улучшение"), "nr");
    before_->addItem(tr("Апскейл"), "sr");
    before_->setToolTip(tr("Что показать на стороне «до» шторки"));
    save_ = Small(tr("Сохранить кадр…"), this);
    save_->setToolTip(tr("Сохранить кадр превью под курсором в PNG 16 бит или EXR"));
    hide_ = Small(tr("Убрать"), this);
    hide_->setToolTip(tr("Убрать превью из вьюпорта; посчитанное остаётся в кэше и покажется сразу"));
    toPreview_ = Small(tr("К превью"), this);
    toPreview_->setToolTip(tr("Перейти к первому кадру превью"));
    for (QWidget* w : std::initializer_list<QWidget*>{cancel_, prepare_, beforeLabel_, before_, save_, hide_, toPreview_}) row2->addWidget(w);
    row2->addStretch(1);
    column->addLayout(row2);

    connect(frame_, &QToolButton::clicked, this, [this] { runner_.start({PreviewRunner::Kind::Frame, 0}); });
    connect(history, &QAction::triggered, this, [this] { runner_.start({PreviewRunner::Kind::FrameHistory, 0}); });
    connect(clear, &QAction::triggered, this, [this] { runner_.clearCache(); });
    connect(auto_, &QCheckBox::toggled, this, [this](bool on) { runner_.setAutoUpdate(on); });
    connect(cancel_, &QPushButton::clicked, this, [this] {
        if (runner_.running()) runner_.cancel();
        else runner_.cancelUpstream();
    });
    connect(prepare_, &QPushButton::clicked, this, [this] { runner_.prepareInputs(); });
    connect(before_, &QComboBox::currentIndexChanged, this, [this](int) { runner_.setBefore(before_->currentData().toString()); });
    connect(save_, &QPushButton::clicked, this, [this] {
        QString file = saveFile_;
        if (file.isEmpty()) {
            const int64_t f = model_.store().FrameAt("source", model_.time());
            file = QFileDialog::getSaveFileName(this, tr("Сохранить кадр превью"), QDir(model_.lastDir("preview")).filePath(QString("nr_preview_%1.png").arg(f)),
                                                tr("PNG 16 бит (*.png);;EXR (*.exr)"));
            if (file.isEmpty()) return;
            model_.rememberDir("preview", file);
        }
        runner_.saveFrame(file);
    });
    connect(hide_, &QPushButton::clicked, this, [this] { runner_.hide(); });
    connect(toPreview_, &QPushButton::clicked, this, [this] {
        if (const auto& p = model_.nrPreview()) model_.seekToSourceFrame(p->first);
    });

    timeTimer_ = new QTimer(this);  // the cursor's frame and «вне превью»: not on every frame of playback
    timeTimer_->setSingleShot(true);
    timeTimer_->setInterval(200);
    connect(timeTimer_, &QTimer::timeout, this, &NrPreviewBar::refresh);
    connect(&model_, &AppModel::timeChanged, timeTimer_, [this](double) { timeTimer_->start(); });
    connect(&runner_, &PreviewRunner::changed, this, &NrPreviewBar::refresh);
    connect(&model_, &AppModel::planChanged, this, &NrPreviewBar::refresh);
    connect(&model_, &AppModel::nrPreviewChanged, this, &NrPreviewBar::refresh);
    connect(&model_, &AppModel::controlMaskChanged, this, &NrPreviewBar::refresh);
    refresh();
}

void NrPreviewBar::refresh() {
    const PreviewRunner::Availability frame = runner_.availability({PreviewRunner::Kind::Frame, 0});
    const PreviewRunner::Availability history = runner_.availability({PreviewRunner::Kind::FrameHistory, 0});
    const QString p = tr("P");
    const NrPreviewUsage usage = runner_.usage();
    const bool clearable = usage.entries > 0 && !runner_.running() && !runner_.housekeeping();
    // the menu («Кадр с историей», «Очистить превью…») stays reachable while the frame itself is refused (a click says why)
    frame_->setEnabled(frame.ok || history.ok || clearable);
    frame_->setToolTip(frame.ok ? tr("Посчитать улучшение только для кадра под курсором с текущими настройками, без истории модели, и показать «До | После». "
                                     "Пассы и версии не меняются. %1 · %2")
                                      .arg(frame.cached ? tr("уже посчитан") : PreviewDuration(frame.seconds), p)
                                : frame.text);
    if (QAction* h = frameMenu_->actions().value(0)) {
        h->setEnabled(history.ok);
        h->setToolTip(history.ok ? tr("Тот же кадр, но модель сначала проходит 6 предыдущих кадров — так он выйдет при обработке. %1").arg(PreviewDuration(history.seconds))
                                 : history.text);
    }
    if (QAction* c = frameMenu_->actions().value(2)) {
        c->setText(usage.entries > 0 ? tr("Очистить превью (%1)…").arg(HumanBytes(usage.bytes)) : tr("Очистить превью…"));
        c->setEnabled(clearable);
        c->setToolTip(tr("Удалить с диска все посчитанные превью этого проекта; пассы и версии не трогаются"));
    }
    bool anySegment = false;
    std::vector<PreviewRunner::Availability> segments;
    for (QAction* a : segmentMenu_->actions()) {
        const int s = a->data().toInt();
        const PreviewRunner::Availability& seg = segments.emplace_back(runner_.availability({PreviewRunner::Kind::Segment, s}));
        a->setEnabled(seg.ok);
        anySegment = anySegment || seg.ok;
        const long long frames = std::max<long long>(1, std::llround(s * (model_.sourceInfo().fps > 0 ? model_.sourceInfo().fps : 24.0)));
        if (seg.ok)
            a->setToolTip(tr("%1 кадров от курсора и 6 кадров разгона: видно мерцание и устойчивость. %2, на диске ≈ %3")
                              .arg(frames)
                              .arg(PreviewDuration(seg.seconds), HumanBytes(seg.bytes)));
        else a->setToolTip(seg.text);
    }
    segment_->setEnabled(anySegment);
    segment_->setToolTip(anySegment ? tr("Превью отрезка от курсора") : !segments.empty() ? segments.front().text : frame.text);  // the shortest's reason
    {
        const QSignalBlocker b(auto_);
        auto_->setChecked(runner_.autoUpdate());
    }
    // the status: the runner's when it has one, else what a frame would do
    QString text = runner_.statusText(), role = runner_.statusRole(), tip = runner_.statusTip();
    const auto& shown = model_.nrPreview();
    const int64_t cursor = model_.store().FrameAt("source", model_.time());
    const bool outside = shown && (cursor < shown->first || cursor > shown->last);
    if (text.isEmpty()) {
        text = frame.text;
        role = frame.ok ? "hint" : "warn";
    }
    if (outside && !runner_.running()) text += (text.isEmpty() ? QString() : QString("\n")) + tr("Курсор вне превью (кадры %1–%2)").arg(shown->first).arg(shown->last);
    status_->setText(text);
    status_->setToolTip(tip);
    SetRole(status_, role.toUtf8().constData());
    cancel_->setVisible(runner_.running() || runner_.upstreamRunning());
    cancel_->setToolTip(runner_.running() ? tr("Отменить превью") : tr("Отменить расчёт этапов до улучшения"));
    // the frame goes without the vectors and the scenes of a stage not reused, history and segments need them: the first that can
    const PreviewRunner::Availability* prep = frame.canPrepare ? &frame : history.canPrepare ? &history : nullptr;
    for (const auto& seg : segments)
        if (!prep && seg.canPrepare) prep = &seg;
    prepare_->setVisible(!runner_.running() && prep);
    prepare_->setEnabled(prep != nullptr);
    if (prep) {
        prepare_->setText(prep->upstreamSeconds > 0 ? tr("Посчитать этапы до улучшения · %1").arg(PreviewDuration(prep->upstreamSeconds))
                                                     : tr("Посчитать этапы до улучшения"));
        prepare_->setToolTip(tr("Посчитать %1 по всему клипу (dlssvid process --no-encode): результат не перезаписывается, отметки «пересчитать» остаются")
                                 .arg(prep->upstream.join(", ")));
    }
    const bool has = shown.has_value();
    beforeLabel_->setVisible(has);
    before_->setVisible(has);
    {
        const QSignalBlocker b(before_);
        before_->setCurrentIndex(model_.nrPreviewBeforeChoice() == "sr" ? 1 : 0);
    }
    save_->setVisible(has);
    save_->setEnabled(has && !outside && !runner_.housekeeping());
    hide_->setVisible(has);
    toPreview_->setVisible(has && outside);
}

}  // namespace dlssvid
