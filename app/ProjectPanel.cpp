#include "ProjectPanel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QScrollArea>
#include <QShortcut>
#include <QSignalBlocker>
#include <QVBoxLayout>
#include <cmath>
#include <optional>
#include <map>

#include "AppModel.h"
#include "SourceNames.h"
#include "StageCard.h"
#include "Theme.h"
#include "TaskQueue.h"
#include "io/EncodeDefaults.h"
#include "pipeline/PassFingerprint.h"
#include "pipeline/PassVersions.h"
#include "stages/ParamSchema.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

QLabel* Muted(const QString& text, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    SetRole(l, "muted");
    return l;
}

QLabel* Value(QWidget* parent) {
    auto* l = new QLabel(parent);
    l->setWordWrap(true);
    l->setTextInteractionFlags(Qt::TextSelectableByMouse);
    return l;
}

QFrame* Card(QWidget* parent) {
    auto* f = new QFrame(parent);
    f->setFrameShape(QFrame::NoFrame);
    SetRole(f, "card");
    return f;
}

QLabel* Title(const QString& text, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    SetRole(l, "title");
    return l;
}

// «Интенсивность 1.4 · Стиль естественный»: the visible parameters of a pass version in the schema's words.
QString HumanParams(const std::string& stage, const nlohmann::json& params) {
    QStringList parts;
    if (const StageSchema* schema = FindStageSchema(stage))
        for (const auto& spec : schema->params) {
            if (spec.advanced || !params.is_object() || !params.contains(spec.key)) continue;
            parts << QString("%1 %2").arg(QString::fromStdString(spec.label), QString::fromStdString(ParamValueLabel(spec, params[spec.key])));
        }
    return parts.join(" · ");
}

QString ShortHash(const std::string& hash) {
    const size_t colon = hash.find(':');
    const std::string hex = colon == std::string::npos ? hash : hash.substr(colon + 1);
    if (hex.size() < 12) return QString::fromStdString(hash);
    return "sha256 " + QString::fromStdString(hex.substr(0, 4)) + "…" + QString::fromStdString(hex.substr(hex.size() - 4));
}

}  // namespace

ProjectPanel::ProjectPanel(AppModel& model, TaskQueue& tasks, QWidget* parent) : QWidget(parent), model_(model), tasks_(tasks) {
    setMinimumWidth(840);  // the cards need the form and the state column side by side (the dock follows)
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);  // the content adapts to the dock's width
    auto* content = new QWidget(scroll);
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(12, 10, 12, 10);
    layout->setSpacing(10);
    scroll->setWidget(content);
    outer->addWidget(scroll);

    // ---- source ----
    auto* sourceCard = Card(content);
    auto* sg = new QGridLayout(sourceCard);
    sg->setContentsMargins(14, 9, 14, 9);
    sg->setHorizontalSpacing(12);
    sg->setVerticalSpacing(3);
    sg->addWidget(Title(tr("Источник"), sourceCard), 0, 0, 1, 2);
    sourceFile_ = Value(sourceCard);
    sourceFrame_ = Value(sourceCard);
    sourceAudio_ = Value(sourceCard);
    sourcePasses_ = Value(sourceCard);
    sourceHash_ = Value(sourceCard);
    int row = 1;
    for (auto [name, value] : std::initializer_list<std::pair<QString, QLabel*>>{
             {tr("Файл"), sourceFile_}, {tr("Кадр"), sourceFrame_}, {tr("Звук"), sourceAudio_}, {tr("Пассы"), sourcePasses_}, {tr("Хэш"), sourceHash_}}) {
        sg->addWidget(Muted(name, sourceCard), row, 0, Qt::AlignTop);
        sg->addWidget(value, row, 1);
        ++row;
    }
    changePasses_ = new QPushButton(tr("изменить…"), sourceCard);
    SetRole(changePasses_, "small");
    changePasses_->setToolTip(tr("Другая папка для пассов: промежуточные кадры стадий занимают десятки гигабайт"));
    connect(changePasses_, &QPushButton::clicked, this, &ProjectPanel::choosePassesRoot);
    sg->addWidget(changePasses_, 4, 2, Qt::AlignTop);  // the «Пассы» row
    sg->setColumnStretch(1, 1);
    layout->addWidget(sourceCard);

    // ---- what the run gives ----
    auto* outCard = Card(content);
    auto* og = new QGridLayout(outCard);
    og->setContentsMargins(14, 9, 14, 9);
    og->setHorizontalSpacing(12);
    og->setVerticalSpacing(3);
    og->addWidget(Title(tr("Что получится"), outCard), 0, 0, 1, 2);
    outVideo_ = Value(outCard);
    outTime_ = Value(outCard);
    outReused_ = Value(outCard);
    outDisk_ = Value(outCard);
    row = 1;
    for (auto [name, value] : std::initializer_list<std::pair<QString, QLabel*>>{{tr("Видео"), outVideo_}, {tr("Время"), outTime_}, {tr("Готово"), outReused_}, {tr("Диск"), outDisk_}}) {
        og->addWidget(Muted(name, outCard), row, 0, Qt::AlignTop);
        og->addWidget(value, row, 1);
        ++row;
    }
    outError_ = new QLabel(outCard);
    outError_->setWordWrap(true);
    SetRole(outError_, "warn");
    og->addWidget(outError_, row++, 0, 1, 2);
    process_ = new QPushButton(tr("Обработать"), outCard);
    process_->setDefault(true);
    SetRole(process_, "primary-big");
    process_->setToolTip(tr("Сохранить проект и прогнать пайплайн (dlssvid process --project): готовые пассы переиспользуются (Ctrl+Enter)"));
    og->addWidget(process_, row, 0, 1, 2);
    og->setColumnStretch(1, 1);
    layout->addWidget(outCard);
    connect(process_, &QPushButton::clicked, this, &ProjectPanel::processAll);
    new QShortcut(QKeySequence("Ctrl+Return"), this, [this] { processAll(); });

    // ---- stages ----
    auto* stagesHead = new QHBoxLayout();
    stagesHead->addWidget(Title(tr("Стадии"), content));
    stagesHead->addWidget(Muted(tr("порядок выполнения сверху вниз · готовые пассы переиспользуются"), content), 1);
    layout->addLayout(stagesHead);
    for (const char* stage : {"depth", "flow", "upscale", "nr", "fg"}) {
        auto* card = new StageCard(model_, stage, content);
        connect(card, &StageCard::runRequested, this, &ProjectPanel::runStage);
        connect(card, &StageCard::patchDllRequested, this, &ProjectPanel::patchDll);
        layout->addWidget(card);
        cards_.push_back(card);
    }
    // encode: always last, the project's encoder
    auto* encodeCard = Card(content);
    SetRole(encodeCard, "card-dashed");
    auto* eg = new QGridLayout(encodeCard);
    eg->setContentsMargins(14, 9, 14, 9);
    eg->setHorizontalSpacing(12);
    auto* encodeHead = new QVBoxLayout();
    encodeHead->setSpacing(0);
    encodeHead->addWidget(Title(tr("Кодирование"), encodeCard));
    encodeHead->addWidget(Muted(tr("всегда последним"), encodeCard));
    eg->addLayout(encodeHead, 0, 0, Qt::AlignTop);
    auto* encodeForm = new QHBoxLayout();
    codec_ = new QComboBox(encodeCard);
    codec_->addItem("H.264 NVENC", "h264_nvenc");
    codec_->addItem("HEVC NVENC", "hevc_nvenc");
    codec_->addItem("AV1 NVENC", "av1_nvenc");
    codec_->addItem(tr("FFV1 (без потерь)"), "ffv1");
    bitrate_ = new QComboBox(encodeCard);
    bitrate_->addItem(tr("Авто"), -1);
    for (int mbps : {20, 35, 50, 80, 120, 200}) bitrate_->addItem(tr("%1 Мбит/с").arg(mbps), mbps);
    bitrate_->addItem(tr("Свой…"), 0);
    bitrate_->setToolTip(tr("Битрейт видео (опция кодера b). «Авто» — по разрешению, частоте и кодеку результата: для H.264 1080p30 ≈ 16, 4K30 ≈ 45, "
                            "4K60 ≈ 70 Мбит/с; HEVC ≈ 0,65×, AV1 ≈ 0,55× от этого"));
    bitrateCustom_ = new QSpinBox(encodeCard);
    bitrateCustom_->setRange(1, 400);
    bitrateCustom_->setSuffix(tr(" Мбит/с"));
    bitrateCustom_->setValue(50);
    bitrateCustom_->setVisible(false);
    encodeAudio_ = Muted("", encodeCard);
    encodeAudio_->setMinimumWidth(60);
    encodeAudio_->setWordWrap(true);
    codec_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    codec_->setMinimumContentsLength(10);
    encodeForm->addWidget(codec_);
    encodeForm->addWidget(bitrate_);
    encodeForm->addWidget(bitrateCustom_);
    encodeForm->addWidget(encodeAudio_, 1);  // «≈ 45 Мбит/с · файл ≈ 54 МБ · звук копируется»
    eg->addLayout(encodeForm, 0, 1);
    encodeState_ = new QLabel(encodeCard);
    encodeState_->setWordWrap(true);
    SetRole(encodeState_, "warn");
    eg->addWidget(encodeState_, 0, 2, Qt::AlignTop);
    eg->setColumnStretch(1, 1);
    eg->setColumnMinimumWidth(2, 200);
    layout->addWidget(encodeCard);
    auto applyEncode = [this] {
        if (updating_) return;
        std::map<std::string, std::string> options = model_.project().codecOptions;
        const int choice = bitrate_->currentData().toInt();  // -1 auto, 0 custom, else Mbit/s
        if (choice < 0) options.erase("b");
        else options["b"] = BitrateOption(choice > 0 ? choice : bitrateCustom_->value());
        model_.setEncode(codec_->currentData().toString(), options);
    };
    connect(codec_, qOverload<int>(&QComboBox::currentIndexChanged), this, [applyEncode](int) { applyEncode(); });
    connect(bitrate_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, applyEncode](int) {
        bitrateCustom_->setVisible(bitrate_->currentData().toInt() == 0);
        applyEncode();
    });
    connect(bitrateCustom_, &QSpinBox::editingFinished, this, applyEncode);
    footer_ = Muted(tr("Улучшение и Генерация кадров используют глубину и векторы: без них качество ниже. Пасс переиспользуется при совпадении "
                       "отпечатка (исходник, параметры, входы, версия инструмента)."),
                    content);
    footer_->setWordWrap(true);
    layout->addWidget(footer_);

    // ---- versions on disk ----
    auto* versionsCard = Card(content);
    auto* vl = new QVBoxLayout(versionsCard);
    vl->setContentsMargins(10, 6, 10, 6);
    vl->setSpacing(4);
    auto* vh = new QHBoxLayout();
    vh->addWidget(Title(tr("Версии пассов на диске"), versionsCard));
    versionsInfo_ = Muted("", versionsCard);
    versionsInfo_->setWordWrap(true);
    versionsInfo_->setMinimumWidth(120);
    vh->addWidget(versionsInfo_, 1);
    gc_ = new QPushButton(tr("Очистить старые"), versionsCard);
    SetRole(gc_, "small");
    gc_->setToolTip(tr("Удалить предыдущие версии сверх хранимых (pass_versions_keep); версии, на которые ссылаются другие, остаются"));
    vh->addWidget(gc_);
    vl->addLayout(vh);
    versionsRows_ = new QWidget(versionsCard);
    versionsGrid_ = new QGridLayout(versionsRows_);
    versionsGrid_->setContentsMargins(0, 0, 0, 0);
    versionsGrid_->setHorizontalSpacing(10);
    versionsGrid_->setVerticalSpacing(2);
    vl->addWidget(versionsRows_);
    layout->addWidget(versionsCard);
    connect(gc_, &QPushButton::clicked, this, [this] {
        try {
            const PassGcReport r = GcPassVersions(model_.project().passesRoot, model_.project().passVersionsKeep);
            Log()->info("gc: {} version(s) removed, {} bytes", r.removed.size(), r.bytes);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, tr("Очистка версий"), QString::fromUtf8(e.what()));
        }
        model_.reloadSources();
    });
    layout->addStretch(1);

    connect(&model_, &AppModel::projectChanged, this, &ProjectPanel::refresh);
    connect(&model_, &AppModel::sourcesChanged, this, &ProjectPanel::refresh);
    connect(&model_, &AppModel::planChanged, this, &ProjectPanel::refresh);
    connect(&model_, &AppModel::engineerModeChanged, this, [this](bool) { refresh(); });
    connect(&tasks_, &TaskQueue::taskStarted, this, [this](int) { refresh(); });  // «Обработать» waits for the running task
    refresh();
}

void ProjectPanel::refresh() {
    updating_ = true;
    const Project& p = model_.project();
    const SourceInfo& src = model_.sourceInfo();
    const RunOutcome& out = model_.outcome();
    const bool hasSource = !p.sourceVideo.empty();
    sourceFile_->setText(hasSource ? QString::fromStdWString(p.sourceVideo.filename().wstring()) : tr("не задан — откройте видео (Ctrl+O)"));
    sourceFile_->setToolTip(hasSource ? QString::fromStdWString(p.sourceVideo.wstring()) : QString());
    if (src.width > 0)
        sourceFrame_->setText(tr("%1×%2 · %3 fps · %4 кадров · %5 с")
                                  .arg(src.width)
                                  .arg(src.height)
                                  .arg(QString::number(src.fps, 'g', 5))
                                  .arg(src.frames)
                                  .arg(QString::number(src.fps > 0 ? src.frames / src.fps : 0.0, 'f', 1)));
    else
        sourceFrame_->setText("—");
    sourceAudio_->setText(!hasSource ? "—" : src.audio ? tr("есть — будет скопирован") : tr("нет"));
    sourcePasses_->setText(hasSource ? QString::fromStdWString(p.passesRoot.wstring()) : "—");
    changePasses_->setEnabled(hasSource && !tasks_.busy());
    const std::string hash = model_.plan().sourceHash;
    sourceHash_->setText(hash.empty() ? "—" : ShortHash(hash) + " · " + model_.sourceHashStatus());
    sourceHash_->setToolTip(QString::fromStdString(hash));

    if (src.width > 0 && model_.planError().isEmpty()) {
        outVideo_->setText(tr("%1×%2 · %3 fps · %4").arg(out.width).arg(out.height).arg(QString::number(out.fps, 'g', 5)).arg(out.audio ? tr("со звуком") : tr("без звука")));
        outTime_->setText(tr("%1 · к запуску %2 из %3 (полный прогон — %4)")
                              .arg(FormatDuration(out.seconds))
                              .arg(out.stagesToRun)
                              .arg(out.stagesTotal)
                              .arg(FormatDuration(out.fullSeconds)));
        QStringList reused;
        for (const auto& s : out.reused) reused << (FindStageSchema(s) ? QString::fromStdString(FindStageSchema(s)->title).toLower() : QString::fromStdString(s));
        outReused_->setText(reused.isEmpty() ? tr("— (всё будет посчитано)") : tr("%1 — совпали по отпечатку").arg(reused.join(", ")));
        QString disk = out.newBytes > 0 ? tr("+%1 (предыдущие версии остаются рядом)").arg(HumanBytes(out.newBytes)) : tr("без новых пассов");
        if (const unsigned long long free = model_.freeSpace()) disk += tr(" · свободно %1").arg(HumanBytes(free));
        outDisk_->setText(disk);
    } else {
        for (QLabel* l : {outVideo_, outTime_, outReused_, outDisk_}) l->setText("—");
    }
    outError_->setText(model_.planError());
    outError_->setVisible(!model_.planError().isEmpty() && hasSource);
    const int toRun = std::max(0, out.stagesToRun - 1);  // the encode is not a «stage» for the button
    process_->setText(toRun > 0 ? tr("Обработать · %1").arg(Plural(toRun, tr("стадия"), tr("стадии"), tr("стадий"))) : tr("Обработать · только кодирование"));
    process_->setEnabled(hasSource && model_.planError().isEmpty() && !tasks_.busy());

    for (StageCard* c : cards_) c->refresh();
    {
        const QSignalBlocker b1(codec_), b2(bitrate_), b3(bitrateCustom_);
        const int idx = codec_->findData(QString::fromStdString(p.codec));
        codec_->setCurrentIndex(idx >= 0 ? idx : 0);
        const auto b = p.codecOptions.find("b");
        const std::optional<double> mbps = b == p.codecOptions.end() ? std::nullopt : ParseBitrateMbps(b->second);
        int choice = -1;  // auto
        if (mbps) {
            choice = 0;  // custom unless a preset matches
            for (int i = 0; i < bitrate_->count(); ++i)
                if (bitrate_->itemData(i).toInt() > 0 && std::fabs(bitrate_->itemData(i).toInt() - *mbps) < 0.01) choice = bitrate_->itemData(i).toInt();
            if (choice == 0) bitrateCustom_->setValue(std::max(1, static_cast<int>(std::lround(*mbps))));
        }
        bitrate_->setCurrentIndex(std::max(0, bitrate_->findData(choice)));
        bitrateCustom_->setVisible(choice == 0);
        const bool lossy = LossyCodec(p.codec);
        bitrate_->setEnabled(lossy && hasSource);
        bitrateCustom_->setEnabled(lossy && hasSource);
        // what the run will use and roughly how big the file gets
        QString info;
        if (!lossy) info = tr("без потерь");
        else if (hasSource && out.width > 0) {
            const double used = mbps ? *mbps : RecommendedBitrateMbps(p.codec, out.width, out.height, out.fps);
            const double seconds = src.fps > 0.0 ? static_cast<double>(src.frames) / src.fps : 0.0;
            if (!mbps) info = tr("≈ %1 Мбит/с").arg(QString::number(used, 'g', 3));
            if (seconds > 0.0 && used > 0.0) info += (info.isEmpty() ? QString() : QString(" · ")) + tr("файл ≈ %1").arg(HumanBytes(static_cast<unsigned long long>(used * 1e6 / 8.0 * seconds)));
        }
        const QString audio = src.audio ? tr("звук копируется") : (hasSource ? tr("без звука") : QString());
        encodeAudio_->setText(info.isEmpty() ? audio : (audio.isEmpty() ? info : info + " · " + audio));
    }
    const StageEstimate* enc = nullptr;
    for (const auto& e : out.stages)
        if (e.stage == "encode") enc = &e;
    const QString result = p.resultVideo.empty() ? tr("результат") : QString::fromStdWString(p.resultVideo.filename().wstring());
    encodeState_->setText(hasSource ? tr("пересчёт · %1 · %2 обновится").arg(enc ? FormatDuration(enc->seconds) : "—").arg(result) : QString());
    refreshVersions();
    updating_ = false;
}

void ProjectPanel::refreshVersions() {
    while (QLayoutItem* item = versionsGrid_->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    const Project& p = model_.project();
    std::vector<PassVersionInfo> versions;
    if (!p.passesRoot.empty()) {
        try {
            versions = ListPassVersions(p.passesRoot);
        } catch (const std::exception& e) {
            Log()->warn("versions: {}", e.what());
        }
    }
    unsigned long long total = 0, history = 0;
    for (const auto& v : versions) {
        total += v.bytes;
        if (!v.current) history += v.bytes;
    }
    versionsInfo_->setText(versions.empty() ? tr("пока ничего не посчитано")
                                            : tr("%1 · %2 на диске, %3 в предыдущих версиях · хранятся %4 на стадию")
                                                  .arg(QString::fromStdWString(p.passesRoot.wstring()), HumanBytes(total), HumanBytes(history),
                                                       Plural(p.passVersionsKeep, tr("последняя"), tr("последние"), tr("последних"))));
    gc_->setEnabled(history > 0);
    if (versions.empty()) return;
    int row = 0, col = 0;
    for (const QString& h : {tr("Стадия"), tr("Версия"), tr("Параметры"), tr("Когда"), tr("Размер"), tr("Состояние"), QString()}) versionsGrid_->addWidget(Muted(h, versionsRows_), row, col++);
    ++row;
    std::map<std::string, int> seen;  // pass -> versions listed so far (the current one is the newest)
    for (const auto& v : versions) {
        const std::string stage = StageForPass(v.pass);
        const int n = ++seen[v.pass];
        int total_n = 0;
        for (const auto& o : versions)
            if (o.pass == v.pass) ++total_n;
        versionsGrid_->addWidget(new QLabel(HumanSourceName(v.pass), versionsRows_), row, 0);
        auto* ver = new QLabel(QString("v%1").arg(total_n - n + 1), versionsRows_);
        SetRole(ver, "mono");
        ver->setToolTip(v.current ? tr("текущая версия") : QString::fromStdString(v.id));
        versionsGrid_->addWidget(ver, row, 1);
        auto* params = Muted(stage.empty() ? QString::fromStdString(v.params.dump()) : HumanParams(stage, v.params), versionsRows_);
        params->setToolTip(QString::fromStdString(v.params.dump()));
        params->setWordWrap(true);
        params->setMinimumWidth(90);
        versionsGrid_->addWidget(params, row, 2);
        versionsGrid_->addWidget(Muted(HumanWhen(v.created), versionsRows_), row, 3);
        versionsGrid_->addWidget(new QLabel(HumanBytes(v.bytes), versionsRows_), row, 4);
        QString state;
        const char* tone = "muted";
        if (v.current) {
            const AppModel::StageCardState st = stage.empty() ? AppModel::StageCardState{} : model_.cardState(stage);
            state = tr("текущая");
            if (st.runs) {
                state += tr(" · будет заменена");
                tone = "warn";
            } else if (st.tone == "ok") {
                state += tr(" · совпадает");
                tone = "ok";
            }
        } else {
            state = tr("предыдущая");
        }
        auto* stateLabel = new QLabel(state, versionsRows_);
        SetRole(stateLabel, tone);
        versionsGrid_->addWidget(stateLabel, row, 5);
        auto* actions = new QWidget(versionsRows_);
        auto* ah = new QHBoxLayout(actions);
        ah->setContentsMargins(0, 0, 0, 0);
        ah->setSpacing(4);
        const std::string name = v.current ? v.pass : v.pass + "@" + v.id;
        auto* compare = new QPushButton(tr("Сравнить"), actions);
        SetRole(compare, "small");
        compare->setToolTip(tr("Показать эту версию на стороне «после» шторки"));
        connect(compare, &QPushButton::clicked, this, [this, name] {
            if (model_.compareView() != AppModel::CompareView::BeforeAfter) model_.setCompareView(AppModel::CompareView::BeforeAfter);
            model_.showSource(name);
        });
        ah->addWidget(compare);
        if (!v.current) {
            const std::string pass = v.pass, id = v.id;
            auto* use = new QPushButton(tr("Вернуть"), actions);
            SetRole(use, "small");
            use->setToolTip(tr("Сделать эту версию текущей; нынешняя останется в истории"));
            connect(use, &QPushButton::clicked, this, [this, pass, id] {
                try {
                    UsePassVersion(model_.project().passesRoot, pass, id);
                } catch (const std::exception& e) {
                    QMessageBox::warning(this, tr("Версии пассов"), QString::fromUtf8(e.what()));
                }
                model_.reloadSources();
            });
            ah->addWidget(use);
            const auto dir = v.dir;
            auto* remove = new QPushButton(tr("Удалить"), actions);
            SetRole(remove, "small");
            connect(remove, &QPushButton::clicked, this, [this, dir, pass, id] {
                if (QMessageBox::question(this, tr("Удалить версию"), tr("Удалить %1 (%2) с диска?").arg(HumanSourceName(pass), VersionLabel(id))) != QMessageBox::Yes) return;
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
                if (ec) QMessageBox::warning(this, tr("Версии пассов"), QString::fromStdString(ec.message()));
                model_.reloadSources();
            });
            ah->addWidget(remove);
        }
        ah->addStretch(1);
        versionsGrid_->addWidget(actions, row, 6);
        ++row;
    }
    versionsGrid_->setColumnStretch(2, 1);
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
    std::vector<TaskQueue::StagePlan> plan;
    for (const StageEstimate& e : model_.outcome().stages) {
        TaskQueue::StagePlan s;
        s.name = QString::fromStdString(e.stage);
        s.title = model_.stageTitle(e.stage);
        s.seconds = e.seconds;
        s.reused = e.action != "run";
        plan.push_back(s);
    }
    const int id = tasks_.enqueue(tr("Обработка"), model_.cliPath(), model_.processArgs(), plan);
    emit processQueued(id, QString("%1 → %2").arg(QString::fromStdWString(p.sourceVideo.filename().wstring()), QString::fromStdWString(p.resultVideo.filename().wstring())));
}

void ProjectPanel::runStage(const std::string& stage) {
    const Project& p = model_.project();
    const StageEntry* st = nullptr;
    for (const auto& s : p.stages)
        if (s.name == stage) st = &s;
    if (!st || p.sourceVideo.empty()) return;
    QStringList args{QString::fromStdString(st->name), "-i", QString::fromStdWString(p.sourceVideo.wstring()), "-o", QString::fromStdWString(p.passesRoot.wstring())};
    for (auto it = st->params.begin(); it != st->params.end(); ++it) {
        const QString key = "--" + QString::fromStdString(it.key());
        if (it.value().is_boolean()) {
            if (it.value().get<bool>()) args << key;
        } else if (it.value().is_string()) {
            args << key << QString::fromStdString(it.value().get<std::string>());
        } else {
            args << key << QString::fromStdString(it.value().dump());
        }
    }
    tasks_.enqueue(QString::fromStdString(st->name), model_.cliPath(), args);
}

void ProjectPanel::choosePassesRoot() {
    if (!model_.hasProject()) return;
    if (tasks_.busy()) {
        QMessageBox::information(this, tr("Идёт обработка"), tr("Папку пассов можно сменить после окончания обработки."));
        return;
    }
    const QString current = QString::fromStdWString(model_.project().passesRoot.wstring());
    const QString dir = QFileDialog::getExistingDirectory(this, tr("Папка пассов (промежуточные кадры стадий, десятки ГБ)"), current.isEmpty() ? model_.lastDir("passes") : current);
    if (dir.isEmpty()) return;
    model_.rememberDir("passes", QDir(dir).filePath("."));
    model_.setPassesRoot(dir);
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
