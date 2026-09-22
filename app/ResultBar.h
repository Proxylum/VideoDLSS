#pragma once

#include <QLabel>
#include <QPushButton>
#include <QWidget>

namespace dlssvid {

class AppModel;

// The result line of the work area (TASK-0013, the footer of the «Результат» mockup): what the result video is
// («3840×1600 · 48 fps · 479 кадров · со звуком · обновлён сегодня 19:23 · готов за 11:22») and what to do with it —
// «Сохранить как…», «Открыть папку», «Экспорт пассов…», «Другое видео». Hidden while the project has no result.
class ResultBar : public QWidget {
    Q_OBJECT
public:
    explicit ResultBar(AppModel& model, QWidget* parent = nullptr);
    QString text() const { return summary_->text(); }

public slots:
    void refresh();

signals:
    void saveAsRequested();
    void openFolderRequested();
    void exportPassesRequested();
    void anotherVideoRequested();

private:
    AppModel& model_;
    QLabel* summary_;
    QPushButton* saveAs_;
    QPushButton* openFolder_;
    QPushButton* exportPasses_;
    QPushButton* another_;
};

}  // namespace dlssvid
