#pragma once

#include <QPlainTextEdit>
#include <QWidget>
#include <memory>

#include <spdlog/sinks/base_sink.h>

namespace dlssvid {

// spdlog sink -> QPlainTextEdit (thread-safe through a queued signal).
class LogPanel : public QWidget {
    Q_OBJECT
public:
    explicit LogPanel(QWidget* parent = nullptr);
    ~LogPanel() override;
public slots:
    void append(const QString& line);
signals:
    void lineReady(const QString& line);

private:
    QPlainTextEdit* text_;
    std::shared_ptr<spdlog::sinks::sink> sink_;
};

}  // namespace dlssvid
