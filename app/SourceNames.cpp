#include "SourceNames.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QTimeZone>
#include <cmath>

#include "viewport/FrameStore.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

namespace {

QString Tr(const char* s) { return QCoreApplication::translate("SourceNames", s); }

QString BaseName(const std::string& pass) {
    if (pass == "source") return Tr("Исходник");
    if (pass == "result") return Tr("Результат");
    if (pass == "depth_raw") return Tr("Глубина");
    if (pass == "depth_dlss") return Tr("Глубина (DLSS)");
    if (pass == "mv_raw") return Tr("Векторы");
    if (pass == "mv_dlss") return Tr("Векторы (DLSS)");
    if (pass == "color_sr") return Tr("Апскейл");
    if (pass == "color_nr") return Tr("Улучшение");
    if (pass == "color_fg") return Tr("Генерация");
    if (pass.rfind("mask_", 0) == 0) return Tr("Маска %1").arg(QString::fromStdString(pass.substr(5)));
    return QString::fromStdString(pass);
}

}  // namespace

QString VersionLabel(const std::string& id) {
    // "YYYYMMDD-HHMMSS_fp8" (PassVersions.h) -> "DD.MM HH:MM · fp8"
    if (id.size() >= 15 && id[8] == '-') {
        const QString q = QString::fromStdString(id);
        QString label = q.mid(6, 2) + "." + q.mid(4, 2) + " " + q.mid(9, 2) + ":" + q.mid(11, 2);
        const int us = q.indexOf('_');
        if (us >= 0 && us + 1 < q.size()) label += " · " + q.mid(us + 1);
        return label;
    }
    return QString::fromStdString(id);
}

QString HumanSourceName(const std::string& source) {
    const auto [pass, version] = SplitSourceVersion(source);
    const QString base = BaseName(pass);
    return version.empty() ? base : base + " · " + VersionLabel(version);
}

QString SourceTooltip(const ViewportSource& s) {
    QString t = QString::fromStdString(s.name);
    if (s.width && s.height) t += QString("  %1x%2").arg(s.width).arg(s.height);
    if (s.lastFrame >= 0) t += Tr("  %1 кадров").arg(s.lastFrame - s.firstFrame + 1);
    if (s.fps.num > 0) t += QString("  %1 fps").arg(QString::number(s.fps.ToDouble(), 'g', 5));
    if (s.manifest && s.manifest->paramsCanonical.is_object() && !s.manifest->paramsCanonical.empty())
        t += "\n" + QString::fromStdString(s.manifest->paramsCanonical.dump());
    if (s.manifest && !s.manifest->created.empty()) t += "\n" + QString::fromStdString(s.manifest->created);
    return t;
}

QString FormatDuration(double seconds) {
    if (seconds < 0) seconds = 0;
    if (seconds < 90) return Tr("≈ %1 с").arg(static_cast<int>(std::lround(seconds)));
    const int minutes = static_cast<int>(std::lround(seconds / 60.0));
    if (minutes < 60) return Tr("≈ %1 мин").arg(minutes);
    return Tr("≈ %1 ч %2 мин").arg(minutes / 60).arg(minutes % 60);
}

QString FormatClock(double seconds) {
    const int total = std::max(0, static_cast<int>(std::lround(seconds)));
    const int h = total / 3600, m = (total / 60) % 60, s = total % 60;
    if (h > 0) return QString("%1:%2:%3").arg(h).arg(m, 2, 10, QChar('0')).arg(s, 2, 10, QChar('0'));
    return QString("%1:%2").arg(m).arg(s, 2, 10, QChar('0'));
}

QString HumanWhen(const std::string& iso8601) {
    const QDateTime t = QDateTime::fromString(QString::fromStdString(iso8601), Qt::ISODate).toLocalTime();
    if (!t.isValid()) return QString::fromStdString(iso8601);
    const QDate today = QDate::currentDate();
    if (t.date() == today) return Tr("сегодня %1").arg(t.toString("HH:mm"));
    if (t.date() == today.addDays(-1)) return Tr("вчера %1").arg(t.toString("HH:mm"));
    return t.toString("dd.MM HH:mm");
}

QString HumanBytes(unsigned long long b) {
    if (b >= (1ull << 30)) return QString::number(static_cast<double>(b) / (1ull << 30), 'f', 1) + Tr(" ГБ");
    if (b >= (1ull << 20)) return QString::number(static_cast<double>(b) / (1ull << 20), 'f', 0) + Tr(" МБ");
    if (b >= (1ull << 10)) return QString::number(static_cast<double>(b) / (1ull << 10), 'f', 0) + Tr(" КБ");
    return QString::number(b) + Tr(" Б");
}

QString Plural(int n, const QString& one, const QString& few, const QString& many) {
    const int m10 = n % 10, m100 = n % 100;
    const QString& word = (m10 == 1 && m100 != 11) ? one : (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) ? few : many;
    return QString::number(n) + " " + word;
}

int SourceRank(const std::string& source) {
    const std::string pass = SplitSourceVersion(source).first;
    static const char* order[] = {"source", "depth_raw", "depth_dlss", "mv_raw", "mv_dlss", "color_sr", "color_nr", "color_fg", "result"};
    for (int i = 0; i < 9; ++i)
        if (pass == order[i]) return i;
    return 9;
}

}  // namespace dlssvid
