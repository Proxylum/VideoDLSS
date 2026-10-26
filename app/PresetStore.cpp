#include "PresetStore.h"

#include <QCollator>
#include <QCoreApplication>
#include <QFileInfo>
#include <QLocale>
#include <QSettings>
#include <QStandardPaths>
#include <algorithm>
#include <cmath>

#include "ParamText.h"
#include "SourceNames.h"
#include "io/EncodeDefaults.h"
#include "io/OutputFormat.h"
#include "stages/ParamSchema.h"

namespace dlssvid {

namespace {

QString g_dir;  // SetDir

QString Tr(const char* s) { return QCoreApplication::translate("PresetStore", s); }
QString U(const std::string& s) { return QString::fromStdString(s); }

const ParamSpec* SpecOf(const std::string& stage, const std::string& key) {
    const StageSchema* schema = FindStageSchema(stage);
    return schema ? schema->Find(key) : nullptr;
}

QString Num(double d) { return QLocale().toString(d, 'g', 9); }

// a name cut to `limit` code points with «…»
std::string CutName(const std::string& s, size_t limit = kPresetNameMax) {
    size_t points = 0;
    for (const unsigned char c : s) points += (c & 0xC0) != 0x80;
    if (points <= limit) return s;
    size_t i = 0, seen = 0;
    for (; i < s.size(); ++i)
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80 && seen++ == limit - 1) break;
    return s.substr(0, i) + "…";
}

// the first «<name> (n)» no preset of the scope has; the name is cut before the suffix, so a long one still gets it
std::string FreeCopyName(const std::string& name, PresetScope scope, const std::string& stage) {
    for (int n = 2; n < 10000; ++n) {
        const std::string suffix = " (" + std::to_string(n) + ")";  // ASCII: bytes = code points
        const std::string candidate = CutName(name, kPresetNameMax - suffix.size()) + suffix;
        if (!PresetStore::FindByName(QString::fromStdString(candidate), scope, stage)) return candidate;
    }
    return name;
}

QString ListText(const std::vector<std::string>& items, const char* sep) {
    QStringList l;
    for (const auto& x : items) l << U(x);
    return l.join(QString::fromUtf8(sep));
}

// the Russian text of a value problem: a stage's (ParamProblem) or the encoder's
QString ValueProblemText(const PresetIssue& i) {
    if (i.stage == "encode") {
        const CodecCaps* caps = FindCodecCaps(i.codec);
        const QString codec = caps ? U(caps->title) : U(i.codec);
        if (i.key == "codec") {
            if (i.got.is_null()) return Tr("Кодек: нет в пресете");
            QStringList all;
            for (const auto& c : KnownCodecs()) all << U(c.title);
            return Tr("Кодек: «%1» — нет в списке (есть: %2)").arg(U(i.got.is_string() ? i.got.get<std::string>() : i.got.dump()), all.join(", "));
        }
        if (i.key == "bit_depth") {
            std::optional<int> d = i.got.is_number_integer() ? std::optional<int>(i.got.get<int>())
                                   : i.got.is_string()        ? ParseBitDepthChoice(i.got.get<std::string>())
                                                              : std::nullopt;
            if (d && (*d == 8 || *d == 10 || *d == 12) && caps) {
                std::vector<std::string> takes;
                for (const int x : caps->bitDepths)
                    if (x <= 12) takes.push_back(std::to_string(x));
                const QString what = takes.size() == 1 ? Tr("только %1 бит").arg(U(takes[0])) : Tr("%1 бит").arg(ListText(takes, " и "));
                return Tr("Разрядность: %1 бит — %2 кодирует %3").arg(*d).arg(codec, what);
            }
            return Tr("Разрядность: %1 — допустимо: авто, 8, 10, 12 бит").arg(U(i.got.is_string() ? i.got.get<std::string>() : i.got.dump()));
        }
        if (i.key == "chroma") {
            const std::string v = i.got.is_string() ? i.got.get<std::string>() : i.got.dump();
            if (i.got.is_string() && ValidChromaChoice(v) && caps) {  // a value the codec cannot encode (a number is a type error)
                QStringList takes;
                for (const auto& c : caps->chromas) takes << EncodeValueLabel("chroma", c);
                return Tr("Субдискретизация: %1 — %2 кодирует %3").arg(EncodeValueLabel("chroma", v), codec, takes.join(Tr(" и ")));
            }
            return Tr("Субдискретизация: %1 — в файле допустимо строкой \"auto\", \"420\", \"422\" или \"444\"").arg(U(v));
        }
        if (i.key == "b") {
            const std::string s = i.got.is_string() ? i.got.get<std::string>() : i.got.is_number() ? i.got.dump() : std::string();
            if (const auto mbps = s.empty() ? std::nullopt : ParseBitrateMbps(s))
                return Tr("Битрейт: %1 Мбит/с — карточка принимает целые 1–400 Мбит/с").arg(QLocale().toString(*mbps, 'g', 6));
            return Tr("Битрейт: «%1» — нужно число Мбит/с, например 50M").arg(U(s.empty() ? i.got.dump() : s));
        }
        return Tr("%1: значение в пресете повреждено").arg(EncodeFieldLabel(i.key));
    }
    if (!i.problem) return U(i.text);
    const ParamProblem& p = *i.problem;
    const ParamSpec* spec = SpecOf(p.stage, p.key);
    if (p.kind == ParamProblem::Kind::Shape || !spec) return Tr("настройки «%1» в пресете повреждены").arg(StageNameGenitive(p.stage));
    const QString label = U(spec->label);
    const QString got = FileValueLabel(*spec, p.got);
    switch (p.kind) {
        case ParamProblem::Kind::Range:
            if (spec->sentinel)
                return Tr("%1: %2 — допустимо «%3» или %4–%5").arg(label, got, U(spec->sentinelLabel), Num(spec->rangeMin), Num(spec->max));
            return Tr("%1: %2 — допустимо %3–%4").arg(label, got, Num(spec->min), Num(spec->max));
        case ParamProblem::Kind::Type:
            if (spec->type == ParamType::Bool) return Tr("%1: нужно «да» или «нет», в файле %2").arg(label, got);
            if (spec->type == ParamType::Int && p.got.is_number()) return Tr("%1: нужно целое число, в файле %2").arg(label, got);
            if (spec->type == ParamType::Int || spec->type == ParamType::Float) return Tr("%1: нужно число, в файле %2").arg(label, got);
            return Tr("%1: нужен текст, в файле %2").arg(label, got);
        case ParamProblem::Kind::Choice: {
            QStringList all;
            for (const auto& c : spec->choices) all << U(c.label);
            return Tr("%1: %2 — нет такого варианта (есть: %3)").arg(label, got, all.join(", "));
        }
        case ParamProblem::Kind::Moved:
            if (const MovedParam* m = FindMovedParam(p.stage, p.key)) return Tr("%1 теперь в карточке «Склейки»").arg(U(m->label));
            break;
        case ParamProblem::Kind::Shape:
            break;
    }
    return U(p.text);
}

PresetStore::Listed ListedOf(const PresetFile& f, bool slot) {
    PresetStore::Listed l;
    l.file = PresetStore::Text(f.path);
    l.name = !f.load.preset.name.empty() ? U(f.load.preset.name) : QFileInfo(l.file).fileName();
    l.when = f.load.preset.created.empty() ? QString() : HumanWhen(f.load.preset.created);
    l.preset = f.load.preset;
    l.slot = slot;
    l.notes = f.load.issues;
    for (const auto& i : f.load.issues)
        if (i.fatal()) {
            l.error = PresetStore::IssueText(i);
            break;
        }
    return l;
}

}  // namespace

QString PresetStore::Dir() { return !g_dir.isEmpty() ? g_dir : QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/presets"; }

void PresetStore::SetDir(const QString& dir) { g_dir = dir; }

PresetLibrary PresetStore::Library() { return PresetLibrary(Path(Dir())); }

std::filesystem::path PresetStore::Path(const QString& file) { return std::filesystem::path(file.toStdWString()); }

QString PresetStore::Text(const std::filesystem::path& p) { return QString::fromStdWString(p.wstring()); }

PresetStore::Listing PresetStore::List(const std::string& target) {
    Listing out;
    const bool pipeline = target == kPipelineTarget;
    const PresetLibrary lib = Library();
    const PresetListing l = lib.List(pipeline ? PresetScope::Pipeline : PresetScope::Stage, pipeline ? std::string() : target);
    for (const auto& f : l.presets) out.items.push_back(ListedOf(f, false));
    QCollator collator{QLocale(QLocale::Russian)};
    collator.setNumericMode(true);
    std::stable_sort(out.items.begin(), out.items.end(), [&](const Listed& a, const Listed& b) { return collator.compare(a.name, b.name) < 0; });
    for (const PresetSlot slot : {PresetSlot::BeforeApply, PresetSlot::BeforeReset})
        if (const auto s = lib.Slot(slot, target)) (s->load.ok() ? out.autoSlots : out.refused).push_back(ListedOf(*s, true));
    for (const auto& f : l.refused) out.refused.push_back(ListedOf(f, false));
    out.hidden = static_cast<int>(l.hidden);
    return out;
}

std::optional<QString> PresetStore::FindByName(const QString& name, PresetScope scope, const std::string& stage) {
    const QString wanted = name.trimmed();
    for (const PresetName& n : Library().Names())
        if (n.scope == scope && U(n.name).compare(wanted, Qt::CaseInsensitive) == 0 && (stage.empty() || n.stage.empty() || n.stage == stage))
            return Text(n.path);
    return std::nullopt;
}

QString PresetStore::IssueText(const PresetIssue& i, bool withStage) {
    using K = PresetIssueKind;
    const std::string& st = i.stage;
    const QString label = LabelOf(st, i.key);
    QString text;
    switch (i.kind) {
        case K::Unreadable: return Tr("Не удалось прочитать файл.");
        case K::TooLarge: return Tr("Файл больше 1 МиБ — это не пресет.");
        case K::NotJson: return Tr("Файл повреждён: это не JSON-объект.");
        case K::NotPreset: return Tr("Это не пресет VideoDLSS.");
        case K::BadSchema: return Tr("Неверный номер формата пресета (schema).");
        case K::NewerSchema:
            return Tr("Пресет создан более новой версией VideoDLSS (формат %1, эта версия читает формат %2). Обновите программу.").arg(i.schema).arg(kPresetSchemaVersion);
        case K::BadScope:
            return i.stages.empty() ? Tr("Неизвестный вид пресета — возможно, он создан более новой версией VideoDLSS. Обновите программу.")
                                    : Tr("В пресете карточки должна быть ровно одна стадия.");
        case K::BadName: return Tr("Имя пресета — от 1 до 120 символов, без управляющих.");
        case K::BadStages:
            if (st.empty()) return Tr("В пресете нет настроек стадий.");
            if (i.text.find("no settings") != std::string::npos) return Tr("В пресете нет настроек «%1».").arg(StageNameGenitive(st));
            return Tr("Настройки «%1» в пресете повреждены.").arg(StageNameGenitive(st));
        case K::NoStage: return Tr("В пресете нет настроек «%1».").arg(StageNameGenitive(st));
        case K::BadValue: text = ValueProblemText(i); break;
        case K::SlotValue: text = Tr("в слоте сохранено и значение с ошибкой: %1").arg(ValueProblemText(i)); break;
        case K::UnknownKey: text = Tr("нет в этой версии: «%1»").arg(U(i.key)); break;
        case K::EngineerKey: {
            const ParamSpec* spec = SpecOf(st, i.key);
            text = spec && spec->tuning   ? Tr("инженерная настройка не входит в пресет карточки: %1").arg(label)
                   : spec && IsMachineKey(*spec) ? Tr("настройки этой машины не входят в пресеты: %1").arg(label)
                   : spec && spec->advanced && spec->choices.size() <= 1 ? Tr("выбирать нечего — не входит в пресеты: %1").arg(label)
                                                                       : Tr("инженерная настройка не входит в пресеты: %1").arg(label);
            break;
        }
        case K::MovedKey: {
            const MovedParam* m = FindMovedParam(st, i.key);
            text = Tr("%1 теперь в карточке «Склейки»").arg(m ? U(m->label) : label);
            break;
        }
        case K::ForeignKey:
            if (st == "encode") {
                const CodecCaps* caps = FindCodecCaps(i.codec);
                text = Tr("битрейт не относится к кодеку «%1» — пропущено").arg(caps ? U(caps->title) : U(i.codec));
            } else {
                QString method = U(i.to);
                if (const ParamSpec* b = SpecOf(st, "backend"))
                    for (const auto& c : b->choices)
                        if (c.value == i.to) method = U(c.label);
                const ParamSpec* spec = SpecOf(st, i.key);
                text = spec && IsListlessKey(*spec, i.to) ? Tr("%1: у метода «%2» нет списка — пресет её не задаёт, оставлено значение проекта").arg(label, method)
                                                          : Tr("%1 не относится к методу «%2» — пропущено").arg(label, method);
            }
            break;
        case K::UnknownStage: return Tr("стадии «%1» нет в этой версии").arg(U(st));
        case K::OtherStages: {
            QStringList titles;
            for (const auto& s : i.stages) titles << QString::fromUtf8("«%1»").arg(StageName(s));
            return Tr("настройки стадий %1 здесь не применяются").arg(titles.join(", "));
        }
        case K::PipelineFields:  // a card preset's enabled / encode carry the key; a pipeline preset read for one stage does not
            return st.empty() || !i.key.empty()
                       ? Tr("включение и кодирование в пресете карточки не хранятся — пропущено")
                       : Tr("здесь применяются только настройки «%1»: остальные стадии, включение и кодирование пропущены").arg(StageNameGenitive(st));
        case K::MissingStage: return Tr("в пресете нет «%1» — оставлены текущие настройки").arg(StageNameGenitive(st));
        case K::NoEncode: return Tr("в пресете нет кодирования — оставлено текущее");
        case K::EncodeField:
            if (i.key == "settings_note") return Tr("Заметка о настройках не входит в пресеты — пропущено");
            if (i.to == "options") return Tr("параметр кодера «%1» не входит в пресеты — пропущено").arg(U(i.key));
            return Tr("поле кодирования «%1» пропущено").arg(U(i.key));
        case K::UnknownField: return Tr("поле «%1» пропущено").arg(U(i.key.empty() ? i.text : i.key));
    }
    if (withStage && !st.empty() && st != "encode") text = StageName(st) + ": " + text;
    if (withStage && st == "encode" && i.kind != K::BadValue && i.kind != K::SlotValue) text = StageName(st) + ": " + text;
    return text;
}

QStringList PresetStore::IssueTexts(const std::vector<PresetIssue>& issues, bool fatal, bool withStage) {
    QStringList out;
    for (const auto& i : issues)
        if (i.fatal() == fatal) out << IssueText(i, withStage);
    out.removeDuplicates();
    return out;
}

QString PresetStore::LeftOutText(const std::vector<LeftOut>& l, bool withStage) {
    if (l.empty()) return {};
    QStringList items;
    for (const LeftOut& x : l) {
        const QString prefix = withStage && x.stage != "encode" ? StageName(x.stage) + ": " : QString();
        const ParamSpec* spec = SpecOf(x.stage, x.key);
        const QString label = spec ? U(spec->label) : U(x.key);
        const QString value = spec ? CardValueLabel(*spec, x.value, {}) : U(x.value.is_string() ? x.value.get<std::string>() : x.value.dump());
        switch (x.why) {
            case LeftOut::Why::Engineer: items << prefix + Tr("%1 %2 (инженерная настройка)").arg(label, value); break;
            case LeftOut::Why::Machine: items << prefix + Tr("%1 %2 (под видеопамять этой машины)").arg(label, value); break;
            case LeftOut::Why::Path: items << prefix + Tr("%1 (путь этой машины)").arg(label); break;
            case LeftOut::Why::NoRow: items << prefix + Tr("%1 %2 (нет в списке этого метода)").arg(label, value); break;
            case LeftOut::Why::Unknown: items << prefix + Tr("%1 (нет в схеме)").arg(U(x.key)); break;
            case LeftOut::Why::CodecOption: items << Tr("параметр кодера %1=%2").arg(U(x.key), value); break;
            case LeftOut::Why::SettingsNote: items << Tr("заметка о настройках выкл."); break;
        }
    }
    return Tr("Не войдут в пресет: %1 — их переносит только файл проекта").arg(items.join(", "));
}

QString PresetStore::NewProjectPreset() { return QSettings().value("presets/newProject").toString(); }

void PresetStore::SetNewProjectPreset(const QString& file) {
    if (file.isEmpty()) QSettings().remove("presets/newProject");
    else QSettings().setValue("presets/newProject", QFileInfo(file).fileName());
}

PresetStore::Result PresetStore::Import(const std::string& target, const QString& from, const QString& replacing, bool asCopy) {
    Result r;
    const bool pipeline = target == kPipelineTarget;
    const PresetLoad l = LoadPreset(Path(from), pipeline ? std::string() : target);
    if (pipeline && l.scopeKnown && l.preset.scope == PresetScope::Stage && !l.preset.stage().empty()) {  // else the reason below
        r.problems << Tr("Это пресет карточки «%1» — импортируйте его в её меню «Пресет ▾».").arg(StageName(l.preset.stage()));
        return r;
    }
    if (!l.ok()) {
        r.problems = IssueTexts(l.issues, true, pipeline);
        return r;
    }
    Preset p = l.preset;
    std::vector<PresetIssue> notes = l.issues;
    if (!pipeline && p.scope == PresetScope::Pipeline) {  // a pipeline preset: its entry of the stage becomes a card preset
        nlohmann::json card = {{"format", kPresetFormat}, {"schema", kPresetSchemaVersion}, {"scope", "stage"}, {"name", p.name}, {"stages", nlohmann::json::object()}};
        if (!p.created.empty()) card["created"] = p.created;
        if (!p.app.empty()) card["app"] = p.app;
        card["stages"][target] = {{"params", p.stages.at(target).params}};
        const PresetLoad c = ValidatePreset(card);
        if (!c.ok()) {
            r.problems = IssueTexts(c.issues, true);
            return r;
        }
        p = c.preset;
        // the pipeline's switches, other stages and fields are said by the message; the notes of the NR entry's own keys stay
        notes.erase(std::remove_if(notes.begin(), notes.end(),
                                   [](const PresetIssue& i) {
                                       return i.kind == PresetIssueKind::OtherStages || i.kind == PresetIssueKind::PipelineFields ||
                                              i.kind == PresetIssueKind::UnknownStage || i.kind == PresetIssueKind::UnknownField;
                                   }),
                    notes.end());
        notes.insert(notes.end(), c.issues.begin(), c.issues.end());
        r.message = Tr("Из пресета обработки «%1» взяты настройки «%2».").arg(U(p.name), StageNameGenitive(target));
    }
    std::string name = p.name;
    if (ReservedPresetName(name)) name = CutName("Импорт — " + name);  // in front: a suffix would leave the form of a slot's name
    std::filesystem::path replace;
    const std::string cardStage = p.scope == PresetScope::Stage ? p.stage() : std::string();
    if (!replacing.isEmpty()) {
        replace = Path(replacing);
    } else if (const auto taken = FindByName(U(name), p.scope, cardStage)) {
        if (!asCopy) {
            r.exists = true;
            r.name = U(name);
            r.file = *taken;
            r.copyName = U(FreeCopyName(name, p.scope, cardStage));  // the button names what «as a copy» will save
            return r;
        }
        name = FreeCopyName(name, p.scope, cardStage);
    }
    p.name = name;
    try {
        const auto path = Library().Save(p, replace);
        r.ok = true;
        r.file = Text(path);
        r.name = U(name);
    } catch (const std::exception& e) {
        r.problems << Tr("Не удалось сохранить пресет: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    // a stage or the encoder a partial pipeline preset lacks is no skip of the import (nothing is applied here)
    notes.erase(std::remove_if(notes.begin(), notes.end(),
                               [](const PresetIssue& i) { return i.kind == PresetIssueKind::MissingStage || i.kind == PresetIssueKind::NoEncode; }),
                notes.end());
    r.notes = IssueTexts(notes, false, pipeline);
    if (r.message.isEmpty()) r.message = Tr("Пресет «%1» импортирован").arg(r.name);
    return r;
}

PresetStore::Result PresetStore::Export(const QString& file, const QString& to) {
    Result r;
    const PresetLibrary lib = Library();
    const auto path = Path(file);
    const PresetLoad l = LoadPreset(path, {}, lib.IsSlotFile(path) ? PresetRead::Slot : PresetRead::Strict);
    r.name = U(l.preset.name);
    if (!l.ok()) {
        r.problems = IssueTexts(l.issues, true);
        return r;
    }
    try {
        SavePreset(Path(to), l.preset);
        r.ok = true;
        r.file = to;
        r.message = Tr("Пресет сохранён в %1").arg(QFileInfo(to).fileName());
    } catch (const PresetError& e) {
        r.problems << Tr("В «%1» есть ошибка — исправьте настройки и экспортируйте текущие: %2")
                          .arg(r.name, IssueTexts(e.issues, true, l.preset.scope == PresetScope::Pipeline).join("; "));
    } catch (const std::exception& e) {
        r.problems << Tr("Не удалось записать файл: %1").arg(QString::fromUtf8(e.what()));
    }
    return r;
}

PresetStore::Result PresetStore::Remove(const QString& file) {
    Result r;
    r.name = U(LoadPreset(Path(file)).preset.name);
    try {
        Library().Remove(Path(file));
    } catch (const std::exception& e) {
        r.problems << Tr("Не удалось удалить пресет: %1").arg(QString::fromUtf8(e.what()));
        return r;
    }
    r.ok = true;
    if (!NewProjectPreset().isEmpty() && NewProjectPreset().compare(QFileInfo(file).fileName(), Qt::CaseInsensitive) == 0) {
        SetNewProjectPreset({});
        r.message = Tr("Пресет «%1» удалён · новые проекты будут создаваться со встроенными умолчаниями").arg(r.name);
    } else {
        r.message = Tr("Пресет «%1» удалён").arg(r.name);
    }
    return r;
}

}  // namespace dlssvid
