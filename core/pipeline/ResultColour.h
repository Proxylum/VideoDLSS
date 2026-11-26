#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "convert/ColorPlan.h"
#include "io/OutputFormat.h"
#include "io/ResultFile.h"
#include "io/VideoDecoder.h"

namespace dlssvid {

// TASK-0079 (HDR step TB-2): what a passthrough writes of a source's colour, and what it loses. An SDR source on the legacy path: its
// tags as before, nothing else. Any other: the effective tags (WrittenTags: unspecified ones filled with the class's assumption), a PQ /
// HLG source's static HDR records, what the output container keeps of them checked back (ResultFile), and notes of what is lost.
struct PassthroughColour {
    ColorTags tags;
    HdrStatic meta;
    std::optional<ResultColor> expect;   // nullopt: an SDR source, checked as before
    std::vector<PlanNote> encodeNotes;   // scope "encode": encode.tags_filled, encode.hdr_8bit, encode.range_lost, encode.mdcv_lost, encode.mdcv_unfit, encode.mdcv_invalid, encode.cll_invalid, encode.no_amve
};

// Throws ColourUnsupported: a source refused in every mode (a Dolby Vision IPT base, RGB HDR; RequireColour), or a non-SDR class into a
// container without colour tags (AVI, the ipod muxer of .m4v, MPEG-TS, raw streams: NoContainerTags). `where` starts the message.
PassthroughColour PlanPassthroughColour(const VideoStreamInfo& info, const std::string& codec, const OutputFormatRequest& format,
                                        const std::filesystem::path& output, const std::string& where);

// The codec keeps the range in its own bitstream: the NVENC codecs (their VUI / sequence header) - and no other, an encoder outside the
// table included (`--codec` takes any): FFmpeg's ProRes header has no range (its decoder says limited to every frame), FFV1 writes none.
bool CodecKeepsRange(const std::string& codec);

}  // namespace dlssvid
