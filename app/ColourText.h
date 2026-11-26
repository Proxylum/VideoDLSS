#pragma once

#include <QString>

#include "convert/ColorPlan.h"
#include "io/SourceColor.h"

namespace dlssvid {

// TASK-0078 (HDR TB-1): the «Цвет» row of the source card - «HDR10 (PQ) · BT.2020 · мастеринг 1000 нит · MaxCLL 1000/400», «HDR HLG ·
// BT.2020 · Dolby Vision 8.4 (будет снят)», «BT.2020 SDR», «лог-кривая · BT.2020», what cannot be read right («ICtCp: цвета при
// обработке будут неверны»). Empty for a source on the legacy path with nothing to say (an SDR video: the row is hidden). `tip`: the tags
// with where each was read, and every note in Russian.
QString SourceColourText(const SourceColor& colour, QString* tip = nullptr);
// A note in Russian, by its key (ColourNote::key); the English detail when the key is not known here.
QString ColourNoteText(const SourceColor& colour, const ColourNote& note);

// TASK-0079 (HDR step TB-2): the stop of a source's processing (or of everything) in the plan's words - what the source is, why, what
// still works («Обработать · только кодирование» writes it as it is) - and the short line of a card or a button.
QString ColourStopText(ColourStop stop, const SourceColor& colour);
QString ColourStopShortText(ColourStop stop);
// A note of the plan (PlanNote: the source's by ColourNoteText, the encoding's by its key) in Russian.
QString PlanNoteText(const SourceColor& colour, const PlanNote& note);

}  // namespace dlssvid
