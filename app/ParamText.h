#pragma once

#include <QString>
#include <QStringList>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "stages/StagePreset.h"
#include "viewport/PipelinePreset.h"

namespace dlssvid {

// Values and changes of the settings in the words of the GUI (PR-9): the card's fields, the menus of presets, their messages.
// One formatter per kind of message (docs/ux-guidelines.md principle 2): the difference as the card's fields show the values, a
// file's problem as the file has the value, the encoder as its card names it.

// A value as the card's fields show it: the locale's decimal separator, the field's decimals, the suffix; the special values and the
// lists keep their words (ParamValueLabel); a value that does not act (another backend's) is «—», an omitDefault key «по умолчанию».
QString CardValueLabel(const ParamSpec& spec, const nlohmann::json& v, const std::string& backend);
// A value as a file has it: the shortest form with the locale's separator («5», «0,3»); the special values and the lists in words.
QString FileValueLabel(const ParamSpec& spec, const nlohmann::json& v);
// The «Кодирование» card's words: «Кодек» … and «H.264 NVENC», «Авто», «10 бит», «4:2:2», «50 Мбит/с».
QString EncodeFieldLabel(const std::string& field);  // codec | bit_depth | chroma | bitrate (b)
QString EncodeValueLabel(const std::string& field, const std::string& value);
// The stage's title (the schema's, «Кодирование» for "encode") and the genitive of it: «настройки «Улучшения»».
QString StageName(const std::string& stage);
QString StageNameGenitive(const std::string& stage);
QString LabelOf(const std::string& stage, const std::string& key);  // the schema's label, the key itself when it has none

// «Интенсивность 1,00 → 0,70»
QString ChangeText(const std::string& stage, const ParamChange& c);
// The changes of one stage: the switch and the method first, then the schema's order, at most `max` and «ещё N».
QString StageDiffText(const StageDiff& d, int max = 3);
QString EncodeChangeText(const EncodeChange& c);  // «Кодек H.264 NVENC → ProRes»

}  // namespace dlssvid
