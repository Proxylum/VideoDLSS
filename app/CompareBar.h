#pragma once

#include <QHBoxLayout>
#include <QCheckBox>
#include <QToolButton>
#include <QWidget>
#include <string>
#include <vector>

namespace dlssvid {

class AppModel;

// The one-action comparison bar above the viewport (stage 9, MR C; docs/ux-guidelines.md, principle 8): the view
// («До | После» wipe, «Только после», «Сетка 2x2»), one chip per source with its human name (a menu of previous
// versions when the pass has any) and the «Инженерный режим» switch that reveals the layer stack and the inspector.
class CompareBar : public QWidget {
    Q_OBJECT
public:
    explicit CompareBar(AppModel& model, QWidget* parent = nullptr);

private:
    void rebuildChips();
    void refresh();

    AppModel& model_;
    QToolButton* beforeAfter_;
    QToolButton* afterOnly_;
    QToolButton* grid_;
    QCheckBox* engineer_;
    QHBoxLayout* chipsLayout_;
    std::vector<QToolButton*> chips_;
    std::vector<std::string> chipSources_;
};

}  // namespace dlssvid
