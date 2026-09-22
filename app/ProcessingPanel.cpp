#include "ProcessingPanel.h"

#include <QFontDatabase>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QRegularExpression>
#include <QVBoxLayout>
#include <cmath>

#include "SourceNames.h"

namespace dlssvid {

namespace {
constexpr int kTailLines = 8;
}

ProcessingPanel::ProcessingPanel(TaskQueue& tasks, QWidget* parent) : QWidget(parent), tasks_(tasks) {
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* column = new QWidget(this);
    column->setMaximumWidth(820);
    auto* layout = new QVBoxLayout(column);
    layout->setContentsMargins(32, 32, 32, 24);
    layout->setSpacing(12);

    headline_ = new QLabel(column);
    headline_->setStyleSheet("font-size: 20px; font-weight: 600;");
    layout->addWidget(headline_);
    what_ = new QLabel(column);
    what_->setStyleSheet("color: #a7abb3;");
    what_->setWordWrap(true);
    layout->addWidget(what_);

    bar_ = new QProgressBar(column);
    bar_->setRange(0, 100);
    bar_->setMinimumHeight(22);
    layout->addWidget(bar_);

    rowsWidget_ = new QWidget(column);
    auto* grid = new QGridLayout(rowsWidget_);
    grid->setContentsMargins(0, 4, 0, 4);
    grid->setHorizontalSpacing(16);
    grid->setVerticalSpacing(6);
    grid->setColumnStretch(0, 1);
    grid->setColumnMinimumWidth(1, 180);
    grid->setColumnMinimumWidth(2, 64);
    grid->setColumnMinimumWidth(3, 140);
    layout->addWidget(rowsWidget_);

    auto* buttons = new QHBoxLayout();
    cancel_ = new QPushButton(tr("Отменить"), column);
    cancel_->setObjectName("cancel");
    cancel_->setToolTip(tr("Остановить обработку; готовые стадии останутся на диске и переиспользуются в следующий раз"));
    minimize_ = new QPushButton(tr("Свернуть в фон"), column);
    minimize_->setObjectName("minimize");
    minimize_->setToolTip(tr("Свернуть окно; по завершении придёт системное уведомление"));
    result_ = new QPushButton(tr("Показать результат"), column);
    result_->setObjectName("showResult");
    result_->setMinimumHeight(32);
    back_ = new QPushButton(tr("К проекту"), column);
    back_->setObjectName("back");
    buttons->addWidget(cancel_);
    buttons->addWidget(minimize_);
    buttons->addStretch(1);
    buttons->addWidget(result_);
    buttons->addWidget(back_);
    layout->addLayout(buttons);
    connect(cancel_, &QPushButton::clicked, this, [this] {
        if (taskId_ >= 0) tasks_.cancel(taskId_);
    });
    connect(minimize_, &QPushButton::clicked, this, &ProcessingPanel::minimizeRequested);
    connect(result_, &QPushButton::clicked, this, &ProcessingPanel::showResultRequested);
    connect(back_, &QPushButton::clicked, this, &ProcessingPanel::backRequested);

    note_ = new QLabel(column);
    note_->setWordWrap(true);
    note_->setStyleSheet("color: #8a8f98;");
    layout->addWidget(note_);

    auto* logHead = new QHBoxLayout();
    auto* logTitle = new QLabel(tr("Лог"), column);
    logTitle->setStyleSheet("font-weight: 600;");
    fullLog_ = new QPushButton(tr("Показать полный лог"), column);
    fullLog_->setObjectName("fullLog");
    fullLog_->setFlat(true);
    logHead->addWidget(logTitle);
    logHead->addStretch(1);
    logHead->addWidget(fullLog_);
    layout->addLayout(logHead);
    connect(fullLog_, &QPushButton::clicked, this, &ProcessingPanel::fullLogRequested);
    tail_ = new QPlainTextEdit(column);
    tail_->setReadOnly(true);
    tail_->setMaximumBlockCount(kTailLines);
    tail_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    tail_->setLineWrapMode(QPlainTextEdit::NoWrap);
    tail_->setFixedHeight(tail_->fontMetrics().lineSpacing() * kTailLines + 14);
    layout->addWidget(tail_);

    outer->addWidget(column, 0, Qt::AlignHCenter | Qt::AlignTop);
    outer->addStretch(1);

    connect(&tasks_, &TaskQueue::taskProgress, this, &ProcessingPanel::onProgress);
    connect(&tasks_, &TaskQueue::taskOutput, this, &ProcessingPanel::onOutput);
    connect(&tasks_, &TaskQueue::taskFinished, this, &ProcessingPanel::onFinished);
    tick_.setInterval(1000);
    connect(&tick_, &QTimer::timeout, this, &ProcessingPanel::render);
    render();
}

void ProcessingPanel::watch(int taskId, const QString& what) {
    taskId_ = taskId;
    what_->setText(what);
    tail_->clear();
    last_ = tasks_.progress(taskId);
    if (!tasks_.finished(taskId)) tick_.start();
    else tick_.stop();
    render();
}

void ProcessingPanel::onProgress(int id, const TaskQueue::Progress& p) {
    if (id != taskId_) return;
    last_ = p;
    render();
}

void ProcessingPanel::onOutput(int id, const QString& line) {
    if (id != taskId_) return;
    // the CLI's lines carry a timestamp: the tail keeps the level and the message
    static const QRegularExpression stamp(R"(^\[[^\]]*\d:\d\d[^\]]*\]\s*)");
    QString shown = line;
    shown.remove(stamp);
    tail_->appendPlainText(shown);
}

void ProcessingPanel::onFinished(int id, bool) {
    if (id != taskId_) return;
    tick_.stop();
    last_ = tasks_.progress(id);
    render();
}

void ProcessingPanel::rebuildRows(const TaskQueue::Progress& p) {
    auto* grid = static_cast<QGridLayout*>(rowsWidget_->layout());
    for (Row& r : rows_) {
        delete r.title;
        delete r.bar;
        delete r.time;
        delete r.state;
    }
    rows_.clear();
    int row = 0;
    for (const auto& s : p.stages) {
        Row r;
        r.name = s.name;
        r.title = new QLabel(s.title, rowsWidget_);
        r.bar = new QProgressBar(rowsWidget_);
        r.bar->setRange(0, 100);
        r.bar->setTextVisible(false);
        r.bar->setFixedHeight(8);
        r.time = new QLabel(rowsWidget_);
        r.time->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        r.state = new QLabel(rowsWidget_);
        grid->addWidget(r.title, row, 0);
        grid->addWidget(r.bar, row, 1);
        grid->addWidget(r.time, row, 2);
        grid->addWidget(r.state, row, 3);
        rows_.push_back(r);
        ++row;
    }
}

void ProcessingPanel::render() {
    const TaskQueue::Progress& p = last_;
    const bool known = taskId_ >= 0;
    const bool done = known && tasks_.finished(taskId_);
    const bool ok = done && tasks_.succeeded(taskId_);
    const bool cancelled = done && tasks_.cancelled(taskId_);
    const double elapsed = done ? p.elapsed : (known ? tasks_.elapsed(taskId_) : 0.0);

    QString head;
    if (!known) head = tr("Обработка");
    else if (!done) head = tr("Обработка · Прошло %1 · Осталось %2").arg(FormatClock(elapsed), p.eta >= 0.0 ? tr("≈ %1").arg(FormatClock(p.eta)) : tr("оцениваем…"));
    else if (ok) head = tr("Готово за %1").arg(FormatClock(elapsed));
    else if (cancelled) head = tr("Отменено через %1 · готовые стадии сохранены").arg(FormatClock(elapsed));
    else head = tr("Ошибка через %1 · подробности в логе").arg(FormatClock(elapsed));
    headline_->setText(head);

    bar_->setValue(static_cast<int>(std::lround(p.fraction * 100)));
    bar_->setFormat(p.stagesTotal > 0 ? tr("%1 из %2 стадий · %p%").arg(p.stagesDone).arg(p.stagesTotal) : QString("%p%"));

    bool same = rows_.size() == p.stages.size();
    for (size_t i = 0; same && i < rows_.size(); ++i) same = rows_[i].name == p.stages[i].name;
    if (!same) rebuildRows(p);
    for (size_t i = 0; i < rows_.size(); ++i) {
        const auto& s = p.stages[i];
        Row& r = rows_[i];
        r.bar->setVisible(s.state == "running");
        r.bar->setValue(s.percent);
        QString state, time, tone = "color: #a7abb3;";
        if (s.state == "reused") {
            state = tr("переиспользуется");
            time = "—";
        } else if (s.state == "done") {
            state = tr("готово");
            time = FormatClock(s.seconds);
        } else if (s.state == "running") {
            state = tr("%1 %").arg(s.percent);
            time = FormatClock(s.seconds);
            tone = "color: #7aa2ff;";
        } else if (s.state == "queued") {
            state = tr("в очереди");
            time = s.seconds > 0.0 ? tr("≈ %1").arg(FormatClock(s.seconds)) : "—";
        } else if (s.state == "cancelled") {
            state = tr("отменена");
            time = s.seconds > 0.0 ? FormatClock(s.seconds) : "—";
            tone = "color: #e0a458;";
        } else {
            state = tr("ошибка");
            time = s.seconds > 0.0 ? FormatClock(s.seconds) : "—";
            tone = "color: #ff7a7a;";
        }
        r.state->setText(state);
        r.state->setStyleSheet(tone);
        r.time->setText(time);
    }

    cancel_->setVisible(known && !done);
    minimize_->setVisible(known && !done);
    result_->setVisible(ok);
    result_->setDefault(ok);
    back_->setVisible(done && !ok);
    if (!known) note_->setText(QString());
    else if (!done) note_->setText(tr("Отмена сохранит уже готовые стадии — в следующий раз они переиспользуются. Свёрнутое окно сообщит о завершении уведомлением."));
    else if (ok) note_->setText(tr("Вьюпорт показывает «До | После»: исходник слева, результат справа."));
    else if (cancelled) note_->setText(tr("Готовые стадии остались на диске: следующая обработка продолжит с прерванной."));
    else note_->setText(tr("Что случилось — в логе ниже; готовые стадии сохранены."));
}

QStringList ProcessingPanel::stageLines() const {
    QStringList out;
    for (const Row& r : rows_) out << QString("%1 · %2 · %3").arg(r.title->text(), r.state->text(), r.time->text());
    return out;
}

}  // namespace dlssvid
