#include "ParamText.h"

#include <QCoreApplication>
#include <QLocale>
#include <cmath>

#include "io/EncodeDefaults.h"
#include "io/OutputFormat.h"
#include "stages/ParamSchema.h"

namespace dlssvid {

namespace {

QString Tr(const char* s) { return QCoreApplication::translate("ParamText", s); }

const ParamSpec* SpecOf(const std::string& stage, const std::string& key) {
    const StageSchema* schema = FindStageSchema(stage);
    return schema ? schema->Find(key) : nullptr;
}

}  // namespace

QString CardValueLabel(const ParamSpec& spec, const nlohmann::json& v, const std::string& backend) {
    if (v.is_null()) return spec.omitDefault ? Tr("по умолчанию") : QString::fromUtf8("—");
    if (v.is_number() && spec.choices.empty() && spec.formChoices.empty()) {
        const double d = v.get<double>();
        const bool special = (spec.sentinel && d == *spec.sentinel && !spec.sentinelLabel.empty()) || (!spec.zeroLabel.empty() && d == spec.min);
        if (!special) return QLocale().toString(d, 'f', spec.type == ParamType::Int ? 0 : ParamDecimals(spec)) + QString::fromStdString(spec.suffix);
    }
    return QString::fromStdString(ParamValueLabel(spec, v, backend));
}

QString FileValueLabel(const ParamSpec& spec, const nlohmann::json& v) {
    if (v.is_number()) {
        const double d = v.get<double>();
        const bool special = (spec.sentinel && d == *spec.sentinel && !spec.sentinelLabel.empty());
        if (!special && spec.choices.empty()) return QLocale().toString(d, 'g', 9);
    }
    if (v.is_boolean() || (v.is_number() && !spec.choices.empty())) return QString::fromStdString(ParamValueLabel(spec, v));
    if (v.is_string()) return QString::fromUtf8("«%1»").arg(QString::fromStdString(v.get<std::string>()));
    return QString::fromStdString(v.dump());
}

QString EncodeFieldLabel(const std::string& field) {
    if (field == "codec") return Tr("Кодек");
    if (field == "bit_depth") return Tr("Разрядность");
    if (field == "chroma") return Tr("Субдискретизация");
    if (field == "bitrate" || field == "b") return Tr("Битрейт");
    return QString::fromStdString(field);
}

QString EncodeValueLabel(const std::string& field, const std::string& value) {
    if (field == "codec") {
        const CodecCaps* caps = FindCodecCaps(value);
        return caps ? QString::fromStdString(caps->title) : QString::fromStdString(value);
    }
    if (field == "bit_depth") return value.empty() || value == "auto" || value == "0" ? Tr("Авто") : Tr("%1 бит").arg(QString::fromStdString(value));
    if (field == "chroma") {
        if (value.empty() || value == "auto") return Tr("Авто");
        if (value.size() == 3) return QString("%1:%2:%3").arg(value[0]).arg(value[1]).arg(value[2]);
        return QString::fromStdString(value);
    }
    if (field == "bitrate" || field == "b") {
        if (value.empty()) return Tr("Авто");
        if (const auto mbps = ParseBitrateMbps(value)) return Tr("%1 Мбит/с").arg(QLocale().toString(*mbps, 'g', 6));
        return QString::fromStdString(value);
    }
    return QString::fromStdString(value);
}

QString StageName(const std::string& stage) {
    if (stage == "encode") return Tr("Кодирование");
    const StageSchema* schema = FindStageSchema(stage);
    return schema ? QString::fromStdString(schema->title) : QString::fromStdString(stage);
}

QString StageNameGenitive(const std::string& stage) {
    if (stage == "scenes") return Tr("Склеек");
    if (stage == "depth") return Tr("Глубины");
    if (stage == "flow") return Tr("Векторов движения");
    if (stage == "upscale") return Tr("Апскейла");
    if (stage == "nr") return Tr("Улучшения");
    if (stage == "fg") return Tr("Генерации кадров");
    if (stage == "encode") return Tr("Кодирования");
    return StageName(stage);
}

QString LabelOf(const std::string& stage, const std::string& key) {
    if (stage == "encode") return EncodeFieldLabel(key);
    const ParamSpec* spec = SpecOf(stage, key);
    return spec ? QString::fromStdString(spec->label) : QString::fromStdString(key);
}

QString ChangeText(const std::string& stage, const ParamChange& c) {
    const ParamSpec* spec = SpecOf(stage, c.key);
    if (!spec) return QString::fromStdString(c.key);
    return QString("%1 %2 → %3")
        .arg(QString::fromStdString(spec->label), CardValueLabel(*spec, c.was, c.wasBackend), CardValueLabel(*spec, c.now, c.nowBackend));
}

QString StageDiffText(const StageDiff& d, int max) {
    QStringList parts;
    if (d.enabledWas != d.enabledNow) parts << (d.enabledNow ? Tr("включится") : Tr("выключится"));
    std::vector<ParamChange> ordered;
    for (const auto& c : d.params)
        if (c.key == "backend") ordered.push_back(c);  // the method first: the rest follows from it
    for (const auto& c : d.params)
        if (c.key != "backend") ordered.push_back(c);
    int shown = 0;
    for (const auto& c : ordered) {
        if (shown == max) {
            parts << Tr("ещё %1").arg(static_cast<int>(ordered.size()) - max);
            break;
        }
        parts << ChangeText(d.stage, c);
        ++shown;
    }
    QString line = parts.join("; ");
    if (!d.enabledWas && !d.enabledNow) line += Tr(" (выключена)");
    return line;
}

QString EncodeChangeText(const EncodeChange& c) {
    return QString("%1 %2 → %3").arg(EncodeFieldLabel(c.field), EncodeValueLabel(c.field, c.was), EncodeValueLabel(c.field, c.now));
}

}  // namespace dlssvid
