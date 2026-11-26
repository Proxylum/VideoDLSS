#include "convert/ColorPlan.h"

namespace dlssvid {

std::string_view ToString(ColourStop s) {
    switch (s) {
        case ColourStop::None: return "";
        case ColourStop::NotYet: return "not yet";
        case ColourStop::Matrix: return "matrix";
        case ColourStop::DolbyVisionIpt: return "dolby vision ipt";
        case ColourStop::RgbHdr: return "rgb hdr";
        case ColourStop::NoContainerTags: return "no container tags";
    }
    return "";
}

namespace {

ColourStop StopOf(ColourRefusal r) {
    switch (r) {
        case ColourRefusal::None: return ColourStop::None;
        case ColourRefusal::Matrix: return ColourStop::Matrix;
        case ColourRefusal::DolbyVisionIpt: return ColourStop::DolbyVisionIpt;
        case ColourRefusal::RgbHdr: return ColourStop::RgbHdr;
    }
    return ColourStop::None;
}

[[noreturn]] void Stopped(const ColorPlan& p, const char* role) {
    throw ColourUnsupported(p.process, p.cls, std::string(role) + ": " + p.why);
}

std::string WhyOf(const ColorPlan& p, const SourceColor& c) {
    const std::string what = std::string(ClassWords(c.cls)) + " (" + ColourWords(c.effective) + ")";
    switch (p.all != ColourStop::None ? p.all : p.process) {
        case ColourStop::DolbyVisionIpt:
            return "a Dolby Vision IPT base layer" + std::string(c.iptAssumed ? " (taken for one: a base nobody names, no matrix nor transfer said)" : "") +
                   ": its colours are wrong without the RPU, which nothing here applies - it can be neither processed nor passed through";
        case ColourStop::RgbHdr:
            return c.rgbLayout ? "an RGB stream of HDR / BT.2020 (" + ColourWords(c.effective) +
                                     "): it is read as 8-bit SDR - it can be neither processed nor passed through"
                               : "the RGB matrix on a Y'CbCr stream of HDR / BT.2020 (" + ColourWords(c.effective) +
                                     "): no Y'CbCr matrix fits it - it can be neither processed nor passed through";
        case ColourStop::Matrix:
            return "matrix " + MatrixName(c.effective.matrix) + " (" + what +
                   "): its colours cannot be processed; passthrough (no stages) writes it as it is";
        case ColourStop::NotYet:
            if (c.Note("source.layers_differ"))  // layers that disagree (an SDR frame over HDR codec parameters ...)
                return "the source's layers disagree: " + std::string(LayersWords(c)) + " " + ColourWords(c.effective) + " (" +
                       std::string(ToString(c.cls)) + "), the codec parameters " + ColourWords(c.stream) + " (" + std::string(ToString(c.rawCls)) +
                       ") - processing such a source comes with a later version; passthrough (no stages) writes it as it is";
            return "processing of " + what + " comes with a later version; passthrough (no stages) writes it as it is, with its HDR";
        case ColourStop::NoContainerTags:
        case ColourStop::None: break;
    }
    return {};
}

}  // namespace

const ColorInfo& ColorPlan::Colour() const {
    if (process != ColourStop::None) Stopped(*this, "colour stages");
    return colourDecode_;
}

const ColorInfo& ColorPlan::Guide() const {
    if (process != ColourStop::None) Stopped(*this, "guide stages");
    return guideDecode_;
}

const ColorInfo& ColorPlan::ResultEncode() const {
    if (process != ColourStop::None) Stopped(*this, "the result");
    return resultEncode_;
}

void RequireColour(const ColorPlan& plan, ColourUse use, const std::string& where) {
    const ColourStop stop = use == ColourUse::Process ? plan.process : plan.all;
    if (stop != ColourStop::None) throw ColourUnsupported(stop, plan.cls, where + ": " + plan.why);
}

ColorTags WrittenTags(const SourceColor& c) {
    const RawTags& e = c.effective;
    ColorTags t;
    t.matrix = e.matrix;
    t.primaries = e.primaries;
    t.transfer = e.trc;
    t.range = e.range == 2 ? 2 : 1;  // AVCOL_RANGE_JPEG as said, else MPEG (unspecified: limited, as every reader takes it)
    const bool hdr = c.cls == SourceClass::Pq || c.cls == SourceClass::Hlg;
    const bool wide = c.cls == SourceClass::Wide || c.cls == SourceClass::Log;
    if (hdr || wide) {  // the class's assumption written down: MP4 / MOV write colr only when all three are said
        if (t.matrix == 2) t.matrix = 9;
        if (t.primaries == 2) t.primaries = 9;
        if (wide && t.transfer == 2) t.transfer = 1;
    }
    return t;
}

bool TagsFilled(const SourceColor& c) {
    const ColorTags t = WrittenTags(c);
    return t.matrix != c.effective.matrix || t.primaries != c.effective.primaries || t.transfer != c.effective.trc;
}

ColorPlan ResolveColor(const VideoStreamInfo& info) {
    ColorPlan p;
    const ColorInfo sdr = ColorInfoFromStream(info);
    p.colourDecode_ = sdr;
    p.guideDecode_ = sdr;
    p.viewDecode_ = sdr;
    p.resultEncode_ = sdr;
    const SourceColor& c = info.color;
    p.cls = c.cls;
    p.legacy = c.legacyPath;
    p.all = StopOf(c.refuseAll);
    p.process = p.all != ColourStop::None ? p.all : StopOf(c.refuseProcessing);
    // TB-2's stop: every source off the legacy path waits for TC's SDR reading (mixed signalling too: its codec parameters say HDR)
    if (!p.legacy && p.process == ColourStop::None) p.process = ColourStop::NotYet;
    p.why = WhyOf(p, c);
    for (const ColourNote& n : c.notes)
        if (n.plan) p.notes.push_back(PlanNote{n.key, "note", "source", n.detail});
    if (p.legacy) {
        p.resultTags_ = ResultColorTags(p.resultEncode_, info);
    } else {
        p.resultTags_ = WrittenTags(c);
        if (c.cls == SourceClass::Pq || c.cls == SourceClass::Hlg) p.resultMeta_ = c.hdr;  // an SDR one of mixed signalling: no mastering
    }
    return p;
}

ColorTags ResultColorTags(const ColorInfo& encode, const VideoStreamInfo& info) {
    ColorTags t;
    t.range = encode.range == ColorRange::Full ? 2 : 1;  // AVCOL_RANGE_JPEG / MPEG
    switch (encode.matrix) {
        case ColorMatrix::Bt709: t.matrix = 1; break;                                // AVCOL_SPC_BT709
        case ColorMatrix::Bt601: t.matrix = info.colorSpace == 5 ? 5 : 6; break;     // BT470BG as the source said, else SMPTE170M
        case ColorMatrix::Bt2020: t.matrix = 9; break;                               // AVCOL_SPC_BT2020_NCL
    }
    t.primaries = info.colorPrimaries;
    t.transfer = info.colorTrc;
    return t;
}

}  // namespace dlssvid
