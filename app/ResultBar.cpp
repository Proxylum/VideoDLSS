#include "ResultBar.h"

#include <QHBoxLayout>

#include "AppModel.h"
#include "Theme.h"

namespace dlssvid {

ResultBar::ResultBar(AppModel& model, QWidget* parent) : QWidget(parent), model_(model) {
    setAttribute(Qt::WA_StyledBackground, true);
    SetRole(this, "bar");
    setMinimumHeight(48);
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(16, 6, 16, 6);
    layout->setSpacing(10);
    summary_ = new QLabel(this);
    summary_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(summary_, 1);
    saveAs_ = new QPushButton(tr("Сохранить как…"), this);
    saveAs_->setObjectName("saveResultAs");
    saveAs_->setToolTip(tr("Скопировать видео результата в другое место"));
    openFolder_ = new QPushButton(tr("Открыть папку"), this);
    openFolder_->setObjectName("openResultFolder");
    openFolder_->setToolTip(tr("Показать папку с результатом в Проводнике"));
    exportPasses_ = new QPushButton(tr("Экспорт пассов…"), this);
    exportPasses_->setObjectName("exportPasses");
    exportPasses_->setToolTip(tr("Выгрузить текущие пассы (глубина, векторы, цвет) EXR-последовательностями с manifest.json — задачи в очереди"));
    another_ = new QPushButton(tr("Другое видео"), this);
    another_->setObjectName("anotherVideo");
    another_->setToolTip(tr("Закрыть проект и вернуться к стартовому экрану (Ctrl+W)"));
    SetRole(saveAs_, "primary");
    for (QPushButton* b : {openFolder_, exportPasses_, another_, saveAs_}) layout->addWidget(b);
    connect(saveAs_, &QPushButton::clicked, this, &ResultBar::saveAsRequested);
    connect(openFolder_, &QPushButton::clicked, this, &ResultBar::openFolderRequested);
    connect(exportPasses_, &QPushButton::clicked, this, &ResultBar::exportPassesRequested);
    connect(another_, &QPushButton::clicked, this, &ResultBar::anotherVideoRequested);
    connect(&model_, &AppModel::projectChanged, this, &ResultBar::refresh);
    connect(&model_, &AppModel::sourcesChanged, this, &ResultBar::refresh);
    connect(&model_, &AppModel::resultChanged, this, &ResultBar::refresh);
    refresh();
}

void ResultBar::refresh() {
    const QString summary = model_.resultSummary();
    setVisible(!summary.isEmpty());
    summary_->setText(summary.isEmpty() ? QString() : tr("<b>Результат</b> · %1").arg(summary));
    summary_->setToolTip(model_.resultInfo().path);
    exportPasses_->setEnabled(!model_.exportablePasses().empty());
}

}  // namespace dlssvid
