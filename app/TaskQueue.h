#pragma once

#include <QProcess>
#include <QStringList>
#include <QTableWidget>
#include <QWidget>
#include <memory>
#include <vector>

namespace dlssvid {

// Runs dlssvid CLI commands one after another and shows their progress (ТЗ §6: «очередь задач с
// прогрессом»). Progress is parsed from the CLI's "N/M frames" lines.
class TaskQueue : public QWidget {
    Q_OBJECT
public:
    explicit TaskQueue(QWidget* parent = nullptr);
    ~TaskQueue() override;

    // Queues `program args...`; returns the task id.
    int enqueue(const QString& title, const QString& program, const QStringList& args);
    bool busy() const { return current_ >= 0; }

signals:
    void taskFinished(int id, bool ok);
    void taskOutput(int id, const QString& line);

private:
    struct Task {
        int id;
        QString title, program;
        QStringList args;
        int row;
        bool done = false;
        bool ok = false;
    };
    void startNext();
    void onOutput();
    void onFinished(int exitCode, QProcess::ExitStatus status);

    QTableWidget* table_;
    std::vector<Task> tasks_;
    int current_ = -1;
    std::unique_ptr<QProcess> process_;
    QString buffer_;
    int nextId_ = 1;
};

}  // namespace dlssvid
