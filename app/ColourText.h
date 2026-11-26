#pragma once

#include <QString>

#include "io/SourceColor.h"

namespace dlssvid {

// TASK-0078 (HDR TB-1): the «Цвет» row of the source card - «HDR10 (PQ) · BT.2020 · мастеринг 1000 нит · MaxCLL 1000/400», «HDR HLG ·
// BT.2020 · Dolby Vision 8.4 (будет снят)», «BT.2020 SDR», «лог-кривая · BT.2020», what cannot be read right («ICtCp: цвета при
// обработке будут неверны»). Empty for a source on the legacy path with nothing to say (an SDR video: the row is hidden). `tip`: the tags
// with where each was read, and every note in Russian.
QString SourceColourText(const SourceColor& colour, QString* tip = nullptr);
// A note in Russian, by its key (ColourNote::key); the English detail when the key is not known here.
QString ColourNoteText(const SourceColor& colour, const ColourNote& note);

}  // namespace dlssvid
