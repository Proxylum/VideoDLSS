#include "ProjectPanel.h"

#include <QCheckBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include "AppModel.h"
#include "TaskQueue.h"
#include "util/Log.h"

namespace dlssvid {

ProjectPanel::ProjectPanel(AppModel& model, TaskQueue& tasks, QWidget* parent) : QWidget(parent), model_(model), tasks_(tasks) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    tree_ = new QTreeWidget(this);
    tree_->setColumnCount(3);
    tree_->setHeaderLabels({tr("Элемент"), tr("Статус"), tr("Параметры")});
    tree_->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    tree_->header()->setStretchLastSection(true);
    layout->addWidget(tree_);
    connect(&model_, &AppModel::projectChanged, this, &ProjectPanel::refresh);
    connect(&model_, &AppModel::sourcesChanged, this, &ProjectPanel::refresh);
    connect(&tasks_, &TaskQueue::taskFinished, this, [this](int, bool ok) {
        if (ok) model_.reloadSources();
    });
    connect(tree_, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem* item, int) {
        if (item && item->parent() == passesItem_) model_.setSingleSource(item->text(0));
    });
    refresh();
}

void ProjectPanel::refresh() {
    tree_->clear();
    const Project& p = model_.project();
    sourceItem_ = new QTreeWidgetItem(tree_, {tr("Источник"), p.sourceVideo.empty() ? tr("не задан") : QString::fromStdWString(p.sourceVideo.filename().wstring()),
                                              QString::fromStdWString(p.passesRoot.wstring())});
    passesItem_ = new QTreeWidgetItem(tree_, {tr("Пассы")});
    const int64_t frameCount = model_.store().FrameCount();
    for (const auto& s : model_.store().Sources()) {
        QString status;
        if (s.isVideo) {
            status = tr("видео, %1 кадров").arg(s.lastFrame + 1);
        } else {
            const int64_t n = s.lastFrame - s.firstFrame + 1;
            const bool complete = frameCount == 0 || n >= frameCount;
            bool imported = false;
            if (s.manifest && !s.manifest->sourceHash.empty() && !p.sourceVideo.empty()) {
                // imported = produced from another source (hash mismatch is checked lazily by name only here)
                imported = s.manifest->sourceFile != p.sourceVideo.filename().string();
            }
            status = imported ? tr("импортирован, %1 кадров").arg(n) : complete ? tr("готов, %1 кадров").arg(n) : tr("частично: %1 из %2").arg(n).arg(frameCount);
        }
        QString params;
        if (s.manifest) params = QString("%1x%2 %3").arg(s.width).arg(s.height).arg(QString::fromStdString(std::string(ToString(s.manifest->format))));
        new QTreeWidgetItem(passesItem_, {QString::fromStdString(s.name), status, params});
    }
    if (model_.store().Sources().empty()) new QTreeWidgetItem(passesItem_, {tr("(нет)"), tr("не посчитаны"), ""});

    stagesItem_ = new QTreeWidgetItem(tree_, {tr("Стадии")});
    for (size_t i = 0; i < p.stages.size(); ++i) {
        const StageEntry& st = p.stages[i];
        auto* item = new QTreeWidgetItem(stagesItem_, {QString::fromStdString(st.name), "", ""});
        auto* enabled = new QCheckBox(tree_);
        enabled->setChecked(st.enabled);
        connect(enabled, &QCheckBox::toggled, this, [this, i](bool on) {
            if (i < model_.project().stages.size()) model_.project().stages[i].enabled = on;
        });
        tree_->setItemWidget(item, 1, enabled);
        auto* row = new QWidget(tree_);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        auto* params = new QLineEdit(QString::fromStdString(st.params.dump()), row);
        params->setToolTip(tr("JSON параметров: ключи становятся опциями CLI (--backend da3 ...)"));
        connect(params, &QLineEdit::editingFinished, this, [this, i, params] {
            try {
                if (i < model_.project().stages.size()) model_.project().stages[i].params = nlohmann::json::parse(params->text().toStdString());
                params->setStyleSheet("");
            } catch (...) {
                params->setStyleSheet("background: #553333");
            }
        });
        auto* run = new QPushButton(tr("Запустить"), row);
        connect(run, &QPushButton::clicked, this, [this, i] { runStage(static_cast<int>(i)); });
        h->addWidget(params, 1);
        h->addWidget(run);
        if (st.name == "nr") {
            auto* patch = new QPushButton(tr("Пропатчить DLL…"), row);
            patch->setToolTip(tr("Пропатчить вашу nvngx_dlssnr.dll для RTX 20/30/40 (dlssnr-patcher, CUDA Toolkit 13.3) и положить результат в bin/nvidia/"));
            connect(patch, &QPushButton::clicked, this, &ProjectPanel::patchDll);
            h->addWidget(patch);
        }
        tree_->setItemWidget(item, 2, row);
    }
    resultItem_ = new QTreeWidgetItem(tree_, {tr("Результат"), p.resultVideo.empty() ? tr("не задан") : QString::fromStdWString(p.resultVideo.filename().wstring()), ""});
    auto* processButton = new QPushButton(tr("Обработать → result"), tree_);
    processButton->setToolTip(tr("Сохранить проект и прогнать весь пайплайн (dlssvid process --project): готовые пассы переиспользуются, результат — видео с аудио"));
    processButton->setEnabled(!p.sourceVideo.empty());
    connect(processButton, &QPushButton::clicked, this, &ProjectPanel::processAll);
    tree_->setItemWidget(resultItem_, 2, processButton);
    tree_->expandAll();
}

void ProjectPanel::processAll() {
    Project& p = model_.project();
    if (p.sourceVideo.empty()) return;
    if (p.resultVideo.empty()) p.resultVideo = p.sourceVideo.parent_path() / (p.sourceVideo.stem().string() + "_result.mp4");
    if (p.file.empty()) {
        const auto file = p.sourceVideo.parent_path() / (p.sourceVideo.stem().string() + ".dlssvid.json");
        if (!model_.saveProjectAs(QString::fromStdWString(file.wstring()))) return;
    } else if (!model_.saveProject()) {
        return;
    }
    tasks_.enqueue(tr("process"), model_.cliPath(), {"process", "--project", QString::fromStdWString(model_.project().file.wstring()), "--disable-unavailable"});
}

void ProjectPanel::runStage(int index) {
    const Project& p = model_.project();
    if (index < 0 || index >= static_cast<int>(p.stages.size()) || p.sourceVideo.empty()) return;
    const StageEntry& st = p.stages[static_cast<size_t>(index)];
    QStringList args{QString::fromStdString(st.name), "-i", QString::fromStdWString(p.sourceVideo.wstring()), "-o", QString::fromStdWString(p.passesRoot.wstring())};
    for (auto it = st.params.begin(); it != st.params.end(); ++it) {
        const QString key = "--" + QString::fromStdString(it.key());
        if (it.value().is_boolean()) {
            if (it.value().get<bool>()) args << key;
        } else if (it.value().is_string()) {
            args << key << QString::fromStdString(it.value().get<std::string>());
        } else {
            args << key << QString::fromStdString(it.value().dump());
        }
    }
    tasks_.enqueue(QString::fromStdString(st.name), model_.cliPath(), args);
}

void ProjectPanel::patchDll() {
    const QString dll = QFileDialog::getOpenFileName(this, tr("Оригинальная nvngx_dlssnr.dll"), QString(), tr("nvngx_dlssnr.dll (nvngx_dlssnr.dll);;DLL (*.dll)"));
    if (dll.isEmpty()) return;
    QStringList args{"nr-patch", "--input", dll};
    // the CLI finds the patcher through DLSSNR_PATCHER_ROOT or tools/dlssnr-patcher near the executable; otherwise ask
    bool patcherKnown = !qEnvironmentVariableIsEmpty("DLSSNR_PATCHER_ROOT");
    QDir dir(QCoreApplication::applicationDirPath());
    for (int up = 0; up < 4 && !patcherKnown; ++up) {
        patcherKnown = QFileInfo::exists(dir.filePath("tools/dlssnr-patcher/dlssnr_patcher.py"));
        if (!dir.cdUp()) break;
    }
    if (!patcherKnown) {
        const QString patcher = QFileDialog::getExistingDirectory(this, tr("Папка dlssnr-patcher (git clone https://github.com/dev-camo/dlssnr-patcher)"));
        if (patcher.isEmpty()) return;
        args << "--patcher" << patcher;
    }
    if (qEnvironmentVariableIsEmpty("CUDA_PATH_V13_3")) {
        const QString cuda = QFileDialog::getExistingDirectory(this, tr("Папка bin CUDA Toolkit 13.3 (ptxas, fatbinary, cuobjdump); Отмена — искать в PATH"));
        if (!cuda.isEmpty()) args << "--cuda-bin" << cuda;
    }
    tasks_.enqueue(tr("nr-patch"), model_.cliPath(), args);
}

}  // namespace dlssvid
