#pragma once

#include <QString>
#include <string>

namespace dlssvid {

struct ViewportSource;

// Human names of the sources for the main UI (docs/ux-guidelines.md, principle 2): «Исходник», «Глубина», «Апскейл»,
// «Улучшение», «Генерация», «Результат»; the technical pass names stay in tooltips and in the engineer mode. A previous
// version of a pass (ViewportState.h: "<pass>@<id>") carries the date of its id.
QString HumanSourceName(const std::string& source);  // "color_nr@20260922-140200_3f2a9c1d" -> «Улучшение · 22.09 14:02 · 3f2a9c1d»
QString VersionLabel(const std::string& versionId);  // "20260922-140200_3f2a9c1d" -> «22.09 14:02 · 3f2a9c1d»
QString SourceTooltip(const ViewportSource& s);      // the technical side: name, size, frames, rate, version parameters
// Pipeline order for chips and lists: source, depth, vectors, upscale, nr, fg, result, then the rest by name.
int SourceRank(const std::string& source);
// «≈ 2 мин», «≈ 45 с», «≈ 1 ч 10 мин»; and «сегодня 14:02» / «вчера 14:02» / «22.09 14:02» from an ISO-8601 UTC time.
QString FormatDuration(double seconds);
// «4:12», «1:02:05»: a clock for elapsed time and the time left on the processing screen.
QString FormatClock(double seconds);
QString HumanWhen(const std::string& iso8601);
// «12.0 ГБ», «60 МБ».
QString HumanBytes(unsigned long long bytes);
// «1 стадия», «3 стадии», «5 стадий»: n with the Russian form of the noun.
QString Plural(int n, const QString& one, const QString& few, const QString& many);

}  // namespace dlssvid
