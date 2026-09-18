#include "LogPanel.h"

#include <QVBoxLayout>
#include <algorithm>
#include <mutex>
#include <spdlog/details/null_mutex.h>

#include "util/Log.h"

namespace dlssvid {

namespace {
class QtSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    explicit QtSink(LogPanel* panel) : panel_(panel) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        base_sink<std::mutex>::formatter_->format(msg, formatted);
        QString line = QString::fromUtf8(formatted.data(), static_cast<int>(formatted.size()));
        while (line.endsWith('\n') || line.endsWith('\r')) line.chop(1);
        emit panel_->lineReady(line);
    }
    void flush_() override {}

private:
    LogPanel* panel_;
};
}  // namespace

LogPanel::LogPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    text_ = new QPlainTextEdit(this);
    text_->setReadOnly(true);
    text_->setMaximumBlockCount(5000);
    layout->addWidget(text_);
    connect(this, &LogPanel::lineReady, this, &LogPanel::append, Qt::QueuedConnection);
    auto sink = std::make_shared<QtSink>(this);
    sink->set_pattern("[%H:%M:%S] [%l] %v");
    sink_ = sink;
    Log()->sinks().push_back(sink_);
}

LogPanel::~LogPanel() {
    auto& sinks = Log()->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
}

void LogPanel::append(const QString& line) { text_->appendPlainText(line); }

}  // namespace dlssvid
