#include "ControlMaskText.h"

#include <QCoreApplication>
#include <QStringList>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("ControlMask", s); }
}  // namespace

QString ControlMaskIssueText(const ControlMaskIssue& i) {
    using K = ControlMaskIssue::Kind;
    switch (i.kind) {
        case K::NoRoot: return Tr("Сначала откройте видео: маска хранится в папке пассов проекта.");
        case K::NotAFile: return Tr("Это не файл: %1.").arg(QString::fromStdString(i.detail).section(": ", 1));
        case K::Unsupported: return Tr("Формат не поддерживается. Подходят PNG, TIFF и EXR.");
        case K::TooLarge:
            return Tr("Изображение %1×%2 слишком большое: сторона — не больше 16384 точек, всего — не больше 67 Мп.").arg(i.width).arg(i.height);
        case K::Empty: return Tr("В изображении нет точек.");
        case K::Unreadable: return Tr("Файл не читается: он повреждён или это не изображение.");
        case K::Channels: {
            const QString d = QString::fromStdString(i.detail);
            const int open = d.lastIndexOf('('), close = d.lastIndexOf(')');
            const QString names = open >= 0 && close > open ? d.mid(open + 1, close - open - 1) : d;
            return Tr("Каналы изображения не подходят (%1): нужны R, G, B или один серый.").arg(names);
        }
        case K::Tiff: return Tr("Этот TIFF не поддерживается (палитра, CMYK, 1 бит, 32-битные целые, плитки или раздельные плоскости). Сохраните маску как PNG.");
        case K::Busy: return Tr("Идёт другой импорт маски в эту папку пассов — дождитесь его конца и повторите.");
        case K::Locked:
            return i.previousKept ? Tr("Папку маски держит другая программа — закройте её и повторите. Прежняя маска осталась на месте.")
                                  : Tr("Папку маски держит другая программа — закройте её и повторите. Маски сейчас нет.");
        case K::Write:
            return Tr("Не удалось записать маску в папку пассов: %1").arg(QString::fromStdString(i.detail)) +
                   (i.previousKept ? QString() : Tr(" Маски сейчас нет."));
    }
    return QString::fromStdString(i.detail);
}

QString ControlMaskWhyText(const std::string& why) {
    // short: the second line of the row's state, in the field column
    if (why == "changed outside dlssvid") return Tr("файл изменён вне программы");
    if (why == "not stamped") return Tr("нет отпечатка");
    if (why == "not static one-frame") return Tr("не статичная маска");
    if (why.rfind("not a mask_control pass", 0) == 0) return Tr("в папке другой пасс");
    if (why.rfind("channels", 0) == 0) return Tr("не те каналы");
    if (why.rfind("type", 0) == 0) return Tr("файл не PNG16");
    if (why.rfind("unreadable", 0) == 0) return Tr("манифест не читается");
    return QString::fromStdString(why);
}

QString HumanControlMaskParams(const nlohmann::json& p) {
    if (!p.is_object()) return {};
    QStringList parts;
    const std::string name = p.value("source_name", std::string());
    if (!name.empty()) parts << QString::fromStdString(name);
    if (p.contains("source_size") && p["source_size"].is_array() && p["source_size"].size() == 2 && p["source_size"][0].is_number() && p["source_size"][1].is_number())
        parts << QString("%1×%2").arg(p["source_size"][0].get<int64_t>()).arg(p["source_size"][1].get<int64_t>());
    parts << (p.value("grayscale", false) ? Tr("серая") : QStringLiteral("RGB"));
    return parts.join(QStringLiteral(" · "));
}

}  // namespace dlssvid
