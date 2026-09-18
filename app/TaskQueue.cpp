#include "TaskQueue.h"

#include <QHeaderView>
#include <QProgressBar>
#include <QRegularExpression>
#include <QVBoxLayout>

#include "util/Log.h"

namespace dlssvid {

TaskQueue::TaskQueue(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    table_ = new QTableWidget(0, 3, this);
    table_->setHorizontalHeaderLabels({tr("Задача"), tr("Прогресс"), tr("Статус")});
    table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table_->verticalHeader()->setVisible(false);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(table_);
}

TaskQueue::~TaskQueue() {
    if (process_ && process_->state() != QProcess::NotRunning) {
        process_->kill();
        process_->waitForFinished(2000);
    }
}

int TaskQueue::enqueue(const QString& title, const QString& program, const QStringList& args) {
    Task t;
    t.id = nextId_++;
    t.title = title;
    t.program = program;
    t.args = args;
    t.row = table_->rowCount();
    table_->insertRow(t.row);
    table_->setItem(t.row, 0, new QTableWidgetItem(title));
    auto* bar = new QProgressBar(table_);
    bar->setRange(0, 100);
    bar->setValue(0);
    table_->setCellWidget(t.row, 1, bar);
    table_->setItem(t.row, 2, new QTableWidgetItem(tr("в очереди")));
    tasks_.push_back(t);
    if (!busy()) startNext();
    return t.id;
}

void TaskQueue::startNext() {
    current_ = -1;
    for (size_t i = 0; i < tasks_.size(); ++i) {
        if (!tasks_[i].done) {
            current_ = static_cast<int>(i);
            break;
        }
    }
    if (current_ < 0) return;
    Task& t = tasks_[static_cast<size_t>(current_)];
    table_->item(t.row, 2)->setText(tr("выполняется"));
    process_ = std::make_unique<QProcess>(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    connect(process_.get(), &QProcess::readyRead, this, &TaskQueue::onOutput);
    connect(process_.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, &TaskQueue::onFinished);
    buffer_.clear();
    Log()->info("task {}: {} {}", t.id, t.program.toStdString(), t.args.join(' ').toStdString());
    process_->start(t.program, t.args);
}

void TaskQueue::onOutput() {
    if (!process_ || current_ < 0) return;
    buffer_ += QString::fromUtf8(process_->readAll());
    Task& t = tasks_[static_cast<size_t>(current_)];
    // progress lines look like "\r12/90 frames"
    static const QRegularExpression re(R"((\d+)/(\d+) frames)");
    QStringList lines = buffer_.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);
    if (!buffer_.endsWith('\n') && !buffer_.endsWith('\r') && !lines.isEmpty()) {
        buffer_ = lines.takeLast();
    } else {
        buffer_.clear();
    }
    for (const QString& line : lines) {
        const auto m = re.match(line);
        if (m.hasMatch()) {
            const int done = m.captured(1).toInt(), total = m.captured(2).toInt();
            if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, 1))) bar->setValue(total > 0 ? done * 100 / total : 0);
        } else if (!line.trimmed().isEmpty()) {
            emit taskOutput(t.id, line);
        }
    }
}

void TaskQueue::onFinished(int exitCode, QProcess::ExitStatus status) {
    if (current_ < 0) return;
    onOutput();
    Task& t = tasks_[static_cast<size_t>(current_)];
    t.done = true;
    t.ok = status == QProcess::NormalExit && exitCode == 0;
    table_->item(t.row, 2)->setText(t.ok ? tr("готово") : tr("ошибка (%1)").arg(exitCode));
    if (auto* bar = qobject_cast<QProgressBar*>(table_->cellWidget(t.row, 1))) bar->setValue(t.ok ? 100 : bar->value());
    const int id = t.id;
    const bool ok = t.ok;
    process_.reset();
    current_ = -1;
    emit taskFinished(id, ok);
    startNext();
}

}  // namespace dlssvid
