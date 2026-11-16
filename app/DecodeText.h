#pragma once

#include <QString>

#include "io/NvdecSupport.h"

namespace dlssvid {

// TASK-0076: why a video cannot be decoded here, in the GUI's words (AV1 without NVDEC AV1: this build has no software AV1 decoder).
// The core's English reason (DecodeSupport::why) stays in the log; empty for a stream that decodes.
QString SourceDecodeText(const DecodeSupport& support);
// The cell label's suffix for such a video: « · не декодируется».
QString SourceDecodeShort();
// A sample layout in the GUI's words: «10 бит 4:4:4», «8 бит, монохромный или особый формат» (VideoStreamInfo::chroma "other")
QString DecodeLayoutText(int bitDepth, const std::string& chroma);

}  // namespace dlssvid
