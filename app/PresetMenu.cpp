#include "PresetMenu.h"

#include <QActionGroup>
#include <algorithm>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QKeyEvent>
#include <QUrl>

#include "AppModel.h"
#include "ParamText.h"
#include "SourceNames.h"
#include "pipeline/PassFingerprint.h"
#include "stages/PresetLibrary.h"

namespace dlssvid {

namespace {

// a name from a file is no mnemonic: «Rock & Roll» keeps its «&» and does not steal the menu's С, И, Э, У, Н, П
QString Escaped(QString s) { return s.replace("&", "&&"); }

QString U(const std::string& s) { return QString::fromStdString(s); }

// the new-projects mark when it is on a preset of the other kind than the menu's
struct OtherMark {
    QString item;     // the disabled checked line of «Для новых проектов ▸»
    QString phrase;   // «Сейчас для новых проектов: <phrase> — будет заменено»
    QString instead;  // «Новые проекты: … (вместо <instead>)»
};

}  // namespace

bool PresetButton::Opens(int key, Qt::KeyboardModifiers modifiers) {
    if ((key == Qt::Key_Space || key == Qt::Key_Return) && modifiers == Qt::NoModifier) return true;
    if (key == Qt::Key_Enter && (modifiers == Qt::NoModifier || modifiers == Qt::KeypadModifier)) return true;
    return key == Qt::Key_Down && modifiers == Qt::AltModifier;
}

bool PresetButton::event(QEvent* e) {
    if (e->type() == QEvent::ShortcutOverride) {
        const auto* k = static_cast<QKeyEvent*>(e);
        if (menu() && Opens(k->key(), k->modifiers())) {
            e->accept();  // the key comes to keyPressEvent, not to the window's shortcut
            return true;
        }
    }
    return QPushButton::event(e);
}

void PresetButton::keyPressEvent(QKeyEvent* e) {
    if (menu() && Opens(e->key(), e->modifiers())) {
        e->accept();
        showMenu();
        return;
    }
    QPushButton::keyPressEvent(e);
}

PresetMenu::PresetMenu(AppModel& model, std::string target, QWidget* parent)
    : QMenu(parent), model_(model), target_(std::move(target)), dialogs_(PresetDialogs::Default()) {
    setToolTipsVisible(true);
    connect(this, &QMenu::aboutToShow, this, &PresetMenu::rebuild);
}

bool PresetMenu::pipeline() const { return target_ == kPipelineTarget; }

QWidget* PresetMenu::dialogParent() const { return parentWidget() ? parentWidget()->window() : nullptr; }

QString PresetMenu::markText() const {
    const QString mark = PresetStore::NewProjectPreset();
    if (mark.isEmpty()) return {};
    const PresetLoad l = LoadPreset(PresetStore::Path(PresetStore::Dir()) / PresetStore::Path(mark));
    if (!l.scopeKnown) return {};
    const QString name = U(l.preset.name);
    if (pipeline() && l.preset.scope == PresetScope::Stage) return tr("«%1» из «%2»").arg(StageName(l.preset.stage()), name);
    if (!pipeline() && l.preset.scope == PresetScope::Pipeline) return name;
    return {};
}

namespace {

OtherMark MarkOfOtherKind(bool pipelineMenu, const QString& text) {
    if (text.isEmpty()) return {};
    if (pipelineMenu) return {text, text, text};
    return {QObject::tr("Пресет обработки «%1»").arg(text), QObject::tr("все настройки обработки из «%1»").arg(text), QObject::tr("пресета обработки «%1»").arg(text)};
}

}  // namespace

QString PresetMenu::preview(const Preset& preset, QString* statusLine) const {
    QStringList lines;
    if (!pipeline()) {
        const auto changes = model_.previewStagePreset(target_, preset);
        QStringList parts;
        for (const auto& c : changes) parts << ChangeText(target_, c);
        *statusLine = changes.empty() ? tr("Совпадает с текущими настройками") : tr("Изменит: %1").arg(parts.join("; "));
        lines << *statusLine;
    } else {
        const PipelineDiff d = model_.previewPipelinePreset(preset);
        if (d.empty()) {
            *statusLine = tr("Совпадает с текущими настройками");
            lines << *statusLine;
        } else {
            QStringList names, same;
            lines << tr("Изменит:");
            for (const StageDiff& sd : d.stages) {
                names << StageName(sd.stage);
                lines << QString("  %1: %2").arg(StageName(sd.stage), StageDiffText(sd));
            }
            if (!d.encode.empty()) {
                QStringList enc;
                for (const auto& c : d.encode) enc << EncodeChangeText(c);
                names << tr("Кодирование");
                lines << QString("  %1: %2").arg(tr("Кодирование"), enc.join("; "));
            }
            for (const auto& s : PipelineStageNames())
                if (std::none_of(d.stages.begin(), d.stages.end(), [&](const StageDiff& sd) { return sd.stage == s; })) same << StageName(s);
            if (d.encode.empty()) same << tr("Кодирование");
            if (!same.isEmpty()) lines << tr("Без изменений: %1").arg(same.join(", "));
            *statusLine = tr("Изменит: %1").arg(names.join(", "));
        }
    }
    return lines.join("\n");
}

void PresetMenu::rebuild() {
    clear();
    const bool pipe = pipeline();
    const PresetStore::Listing l = PresetStore::List(target_);
    const QString mark = PresetStore::NewProjectPreset();
    auto isMark = [&](const QString& file) { return !mark.isEmpty() && QFileInfo(file).fileName().compare(mark, Qt::CaseInsensitive) == 0; };
    auto describe = [&](QAction* a, const PresetStore::Listed& item) {
        QString status;
        QString tip = preview(item.preset, &status);
        QStringList skipped, slotErrors;
        for (const auto& i : item.notes) {
            if (i.fatal() || i.kind == PresetIssueKind::MissingStage || i.kind == PresetIssueKind::NoEncode) continue;
            (i.kind == PresetIssueKind::SlotValue ? slotErrors : skipped) << PresetStore::IssueText(i, pipe);
        }
        if (!skipped.isEmpty()) tip += "\n" + tr("Пропустит: %1").arg(skipped.join("; "));
        if (!slotErrors.isEmpty()) tip += "\n" + tr("В слоте есть и ошибки: %1").arg(slotErrors.join("; "));
        if (!item.when.isEmpty()) tip += "\n" + tr("Создан %1 · VideoDLSS %2").arg(item.when, U(item.preset.app));
        a->setToolTip(tip);
        a->setStatusTip(status);  // also while the arrows walk the menu
        return status == tr("Совпадает с текущими настройками");
    };
    for (const auto& item : l.items) {
        QString text = Escaped(item.name);
        if (isMark(item.file)) text += tr(" · для новых проектов");
        if (!item.when.isEmpty()) text += "\t" + item.when;
        QAction* a = addAction(text);
        a->setCheckable(true);
        a->setChecked(describe(a, item));  // the one that matches the current settings
        const QString file = item.file;
        connect(a, &QAction::triggered, this, [this, file] { applyFile(file); });
    }
    if (l.items.empty())
        addAction(pipe ? tr("(Пресетов обработки пока нет — «Сохранить текущие…»)") : tr("(Пресетов пока нет — «Сохранить текущие…»)"))->setEnabled(false);
    if (!l.autoSlots.empty()) {
        addSeparator();
        for (const auto& item : l.autoSlots) {
            QAction* a = addAction(Escaped(item.name) + (item.when.isEmpty() ? QString() : "\t" + item.when));
            describe(a, item);
            const QString file = item.file;
            connect(a, &QAction::triggered, this, [this, file] { applyFile(file); });
        }
    }
    addSeparator();
    connect(addAction(tr("&Сохранить текущие…")), &QAction::triggered, this, &PresetMenu::saveCurrent);
    connect(addAction(tr("&Импорт…")), &QAction::triggered, this, &PresetMenu::importFile);
    QMenu* exp = addMenu(tr("&Экспорт"));
    exp->setToolTipsVisible(true);
    connect(exp->addAction(tr("Текущие настройки…")), &QAction::triggered, this, [this] { exportFile({}); });
    if (!l.items.empty() || !l.autoSlots.empty()) exp->addSeparator();
    for (const auto* list : {&l.items, &l.autoSlots})
        for (const auto& item : *list) {
            const QString file = item.file;
            connect(exp->addAction(Escaped(item.name)), &QAction::triggered, this, [this, file] { exportFile(file); });
        }
    QMenu* del = addMenu(tr("&Удалить"));
    for (const auto& item : l.items) {
        const QString file = item.file, name = item.name;
        connect(del->addAction(Escaped(item.name)), &QAction::triggered, this, [this, file, name] { removeFile(file, name); });
    }
    del->setEnabled(!l.items.empty());  // the slots rewrite themselves: not deleted from here
    QMenu* mk = addMenu(tr("Для &новых проектов"));
    mk->setToolTipsVisible(true);
    auto* group = new QActionGroup(mk);
    group->setExclusive(true);
    QAction* builtin = mk->addAction(tr("Встроенные умолчания"));
    builtin->setCheckable(true);
    builtin->setChecked(mark.isEmpty());
    group->addAction(builtin);
    connect(builtin, &QAction::triggered, this, [this] { markForNewProjects({}); });
    for (const auto& item : l.items) {
        QAction* a = mk->addAction(Escaped(item.name));
        a->setCheckable(true);
        a->setChecked(isMark(item.file));
        group->addAction(a);
        const QString file = item.file;
        connect(a, &QAction::triggered, this, [this, file] { markForNewProjects(file); });
    }
    const OtherMark other = MarkOfOtherKind(pipe, markText());
    if (!other.item.isEmpty()) {
        QAction* o = mk->addAction(Escaped(other.item));
        o->setCheckable(true);
        o->setChecked(true);
        o->setEnabled(false);
        o->setToolTip(pipe ? tr("Новые проекты берут «Улучшение» из этого пресета. Выберите здесь пресет обработки — метка перейдёт на него")
                           : tr("Новые проекты берут все настройки обработки из «%1». Выберите здесь вариант для «Улучшения» — метка перейдёт на него")
                                 .arg(markText()));
        group->addAction(o);
    }
    connect(addAction(tr("Открыть &папку пресетов")), &QAction::triggered, this, [] {
        QDir().mkpath(PresetStore::Dir());
        QDesktopServices::openUrl(QUrl::fromLocalFile(PresetStore::Dir()));
    });
    if (!l.refused.empty()) {
        QStringList why;
        for (const auto& r : l.refused) why << QString("%1: %2").arg(QFileInfo(r.file).fileName(), r.error);
        QAction* a = addAction(tr("Не прочитаны: %1").arg(Plural(static_cast<int>(l.refused.size()), tr("файл"), tr("файла"), tr("файлов"))));
        a->setEnabled(false);
        a->setToolTip(why.join("\n"));
    }
    if (l.hidden > 0) addAction(tr("Не прочитаны: ещё %1 (в папке больше 500)").arg(Plural(l.hidden, tr("файл"), tr("файла"), tr("файлов"))))->setEnabled(false);
}

void PresetMenu::applyFile(const QString& file) {
    auto apply = [&](bool withoutBackup) {
        return pipeline() ? model_.applyPipelinePreset(file, withoutBackup) : model_.applyStagePreset(target_, file, withoutBackup);
    };
    AppModel::PresetResult r = apply(false);
    if (r.backupFailed) {
        if (!dialogs_.confirm(dialogParent(), tr("Пресет не применён"), r.problems.join("\n") + "\n" + tr("Применить без автосохранения?"), tr("Применить"))) return;
        r = apply(true);
    }
    if (!r.ok && !r.problems.isEmpty()) dialogs_.warn(dialogParent(), tr("Пресет не применён"), r.problems.join("\n"));
}

void PresetMenu::saveCurrent() {
    const bool pipe = pipeline();
    const PresetScope scope = pipe ? PresetScope::Pipeline : PresetScope::Stage;
    PresetSaveDialog::Setup s;
    s.title = pipe ? tr("Сохранить настройки обработки") : tr("Сохранить настройки «%1»").arg(StageNameGenitive(target_));
    s.leftOut = model_.presetLeftOut(target_);
    const OtherMark other = MarkOfOtherKind(pipe, markText());
    if (!other.phrase.isEmpty()) s.markNote = tr("Сейчас для новых проектов: %1 — будет заменено").arg(other.phrase);
    s.problem = [](const QString& name) -> QString {
        const std::string n = name.trimmed().toStdString();
        if (!PresetNameProblem(n).empty()) return QObject::tr("Имя пресета — от 1 до 120 символов, без управляющих.");
        if (ReservedPresetName(n)) return QObject::tr("Так называются автосохранения — выберите другое имя");
        return {};
    };
    s.exists = [scope](const QString& name) { return PresetStore::FindByName(name, scope).has_value(); };
    const auto answer = dialogs_.saveName(dialogParent(), s);
    if (!answer) return;
    const QString replacing = PresetStore::FindByName(answer->name, scope).value_or(QString());  // the dialog said «будет заменён»
    const AppModel::PresetResult r = pipe ? model_.savePipelinePreset(answer->name, replacing, answer->forNewProjects)
                                          : model_.saveStagePreset(target_, answer->name, replacing, answer->forNewProjects);
    if (!r.ok) dialogs_.warn(dialogParent(), tr("Пресет не сохранён"), r.problems.join("\n"));
}

void PresetMenu::importFile() {
    const QString file = dialogs_.openFile(dialogParent(), tr("Импорт пресета"), model_.lastDir("preset"), tr("Пресеты VideoDLSS (*.json)"));
    if (file.isEmpty()) return;
    model_.rememberDir("preset", file);
    PresetStore::Result r = PresetStore::Import(target_, file);
    if (r.exists) {
        switch (dialogs_.clash(dialogParent(), r.name, r.name + " (2)")) {
            case PresetDialogs::Clash::Replace: r = PresetStore::Import(target_, file, r.file); break;
            case PresetDialogs::Clash::Copy: r = PresetStore::Import(target_, file, {}, true); break;
            case PresetDialogs::Clash::Cancel: return;
        }
    }
    if (!r.ok) {
        dialogs_.warn(dialogParent(), tr("Пресет не импортирован"), r.problems.join("\n"));
        return;
    }
    const bool fromPipeline = !pipeline() && r.message.startsWith(tr("Из пресета обработки"));
    if (!r.notes.isEmpty() || fromPipeline) {
        QString text = r.message;
        if (!r.notes.isEmpty()) text += "\n" + tr("Пропущено: %1").arg(r.notes.join("; "));
        dialogs_.inform(dialogParent(), tr("Пресет импортирован"), text);
    }
    emit model_.message(r.message);
}

void PresetMenu::exportFile(const QString& file) {
    const bool pipe = pipeline();
    QString name;
    if (file.isEmpty()) name = U(PresetNameFor(pipe ? std::string("Обработка") : StageName(target_).toStdString(), model_.project().sourceVideo));
    else name = U(LoadPreset(PresetStore::Path(file), {}, PresetRead::Slot).preset.name);
    const QString suggested = QDir(model_.lastDir("preset")).filePath(U(PresetFileStem(name.toStdString())) + ".json");
    const QString to = dialogs_.saveFile(dialogParent(), tr("Экспорт пресета"), suggested, tr("Пресеты VideoDLSS (*.json)"));
    if (to.isEmpty()) return;
    model_.rememberDir("preset", to);
    QStringList problems;
    QString message;
    if (file.isEmpty()) {
        const AppModel::PresetResult r = pipe ? model_.exportCurrentPipelinePreset(to) : model_.exportCurrentStagePreset(target_, to);
        problems = r.problems;  // the model says it in the status bar itself
    } else {
        const PresetStore::Result r = PresetStore::Export(file, to);
        problems = r.problems;
        message = r.message;
    }
    if (!problems.isEmpty()) dialogs_.warn(dialogParent(), tr("Пресет не экспортирован"), problems.join("\n"));
    else if (!message.isEmpty()) emit model_.message(message);
}

void PresetMenu::removeFile(const QString& file, const QString& name) {
    const QString mark = PresetStore::NewProjectPreset();
    const bool marked = !mark.isEmpty() && QFileInfo(file).fileName().compare(mark, Qt::CaseInsensitive) == 0;
    QString text = tr("Удалить пресет «%1»? Файл будет удалён, вернуть его нельзя.").arg(name);
    if (marked) text += " " + tr("Новые проекты будут создаваться со встроенными умолчаниями.");
    if (!dialogs_.confirm(dialogParent(), tr("Удалить пресет"), text, tr("Удалить"))) return;
    const PresetStore::Result r = PresetStore::Remove(file);
    if (!r.ok) dialogs_.warn(dialogParent(), tr("Пресет не удалён"), r.problems.join("\n"));
    else emit model_.message(r.message);
}

void PresetMenu::markForNewProjects(const QString& file) {
    const OtherMark other = MarkOfOtherKind(pipeline(), markText());
    PresetStore::SetNewProjectPreset(file);
    QString msg;
    if (file.isEmpty()) {
        msg = tr("Новые проекты: встроенные умолчания");
    } else {
        const QString name = U(LoadPreset(PresetStore::Path(file)).preset.name);
        msg = pipeline() ? tr("Новые проекты: все настройки обработки из «%1»").arg(name) : tr("Новые проекты: «%1» из «%2»").arg(StageName(target_), name);
        if (!other.instead.isEmpty()) msg += tr(" (вместо %1)").arg(other.instead);
    }
    emit model_.message(msg);
}

}  // namespace dlssvid
