#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "convert/ColorConvert.h"
#include "io/SourceColor.h"
#include "io/VideoEncoder.h"
#include "util/Error.h"

namespace dlssvid {

// TASK-0079 (HDR step TB-2): what a source cannot get here. Processing (stages) of anything but an SDR source on the legacy path waits
// for the next step (TC: an SDR reading of HDR); passthrough writes such a source as it is, with its HDR records - except what no
// output can carry right.
enum class ColourStop : uint8_t {
    None,
    NotYet,           // HDR (PQ / HLG), wide gamut, a log curve, layers that disagree: processing comes with a later version
    Matrix,           // matrix 10 / 14 (BT.2020 CL, ICtCp): no processing
    DolbyVisionIpt,   // a Dolby Vision IPT base layer: neither processed nor passed through (wrong colours without the RPU)
    RgbHdr,           // RGB of HDR / BT.2020, or the RGB matrix on Y'CbCr: neither processed nor passed through
    NoContainerTags,  // passthrough of a non-SDR source into a container that carries no colour tags (AVI, M4V, MPEG-TS, raw streams)
};
std::string_view ToString(ColourStop s);  // "", "not yet", "matrix", "dolby vision ipt", "rgb hdr", "no container tags"
enum class ColourUse : uint8_t { Process, Passthrough };

// A note of the plan and the result: a source's (its SourceColor notes of plan level) or the encoding's (what the output loses).
struct PlanNote {
    std::string key;    // "source.dv_removed", "encode.hdr_8bit" ...
    std::string level;  // "note" | "warning"
    std::string scope;  // "source" | "encode"
    std::string text;   // English (the CLI, the logs); the GUI says it in Russian by `key`
    bool operator==(const PlanNote&) const = default;
};

// How one source is read, shown and written (TASK-0075): every consumer takes its colour from here, so all of them see the same thing.
// An SDR source on the legacy path (SourceColor::legacyPath) is read as before - every role ColorInfoFromStream, the result tagged
// with the matrix and range used and the source's primaries and transfer. Any other source (TB-2) has its processing stopped: the
// roles that would process it throw ColourUnsupported; the viewport, thumbnails, info and compare still read it (the SDR reading), and
// passthrough writes its effective tags and HDR records.
class ColorPlan {
public:
    SourceClass cls = SourceClass::Legacy;  // SourceColor::cls
    bool legacy = true;                     // SourceColor::legacyPath
    ColourStop process = ColourStop::None;  // processing (stages, export, a stage's preview)
    ColourStop all = ColourStop::None;      // processing and passthrough
    std::string why;                        // the stop in English: what the source is, why, what still works
    std::vector<PlanNote> notes;            // the source's notes of plan level (SourceColor::notes, plan = true), any class

    const ColorInfo& Colour() const;        // colour stages (upscale, nr, fg), export: throws ColourUnsupported when process != None
    const ColorInfo& Guide() const;         // guide stages (depth, flow, scenes): the same
    const ColorInfo& ResultEncode() const;  // RGB -> YUV of a processed result: the same
    const ColorInfo& View() const { return viewDecode_; }           // viewport, thumbnails, info: never throws (the SDR reading)
    const ColorInfo& MeasureDecode() const { return colourDecode_; }  // compare: a measure of two files, not a result
    const ColorTags& ResultTags() const { return resultTags_; }  // the result's tags: legacy as before; else WrittenTags
    const HdrStatic& ResultMeta() const { return resultMeta_; }  // a PQ / HLG source's MDCV / CLL / amve for passthrough; else empty

private:
    ColorInfo colourDecode_, guideDecode_, viewDecode_, resultEncode_;
    ColorTags resultTags_;
    HdrStatic resultMeta_;
    friend ColorPlan ResolveColor(const VideoStreamInfo& info);
};

ColorPlan ResolveColor(const VideoStreamInfo& info);

// A stop, thrown by RequireColour and by ColorPlan's processing roles: its kind and the source's class for the GUI's words.
class ColourUnsupported : public Error {
public:
    ColourUnsupported(ColourStop kind, SourceClass cls, const std::string& message) : Error(message), kind(kind), cls(cls) {}
    ColourStop kind;
    SourceClass cls;
};
// Throws ColourUnsupported ("<where>: <why>") when the plan stops `use`: Process - process != None; Passthrough - all != None.
void RequireColour(const ColorPlan& plan, ColourUse use, const std::string& where);

// The tags a non-legacy source is written with (passthrough): its effective tags, unspecified ones filled with the class's assumption
// (the user's decision 2026-11-26) - matrix and primaries BT.2020 for PQ / HLG / BT.2020 primaries, the transfer BT.709 under BT.2020
// primaries -, the range limited unless full; a source of mixed signalling (SDR frames over HDR codec parameters) its effective tags as
// they are. MP4 / MOV write colr only when all three are said (movenc).
ColorTags WrittenTags(const SourceColor& c);
// Whether WrittenTags filled a tag the source did not say.
bool TagsFilled(const SourceColor& c);

// The tags of a stream encoded with `encode` from a source with `info`: the matrix and range used, the source's
// primaries and transfer passed through (BT.601 is written as the source said, BT470BG, else SMPTE170M).
ColorTags ResultColorTags(const ColorInfo& encode, const VideoStreamInfo& info);

}  // namespace dlssvid
