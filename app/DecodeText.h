#pragma once

#include <QString>

#include "io/NvdecSupport.h"

namespace dlssvid {

// TASK-0076: why a video cannot be decoded here, in the GUI's words (AV1 without NVDEC AV1: this build has no software AV1 decoder).
// The core's English reason (DecodeSupport::why) stays in the log; empty for a stream that decodes.
QString SourceDecodeText(const DecodeSupport& support);
// TASK-0076: the short line of a card when the source cannot be decoded here (the whole paragraph is said once, under «Что получится»)
QString SourceRefusedShortText();
// The cell label's suffix for such a video: « · не декодируется».
QString SourceDecodeShort();
// A sample layout in the GUI's words: «10 бит 4:4:4», for VideoStreamInfo::chroma "other" by the pixel format «8 бит RGB», «8 бит,
// монохромный», «8 бит, палитра», «8 бит 4:1:1», else (a format not known) «8 бит, монохромный или особый формат»
QString DecodeLayoutText(int bitDepth, const std::string& chroma, const std::string& pixelFormat = {});

}  // namespace dlssvid
