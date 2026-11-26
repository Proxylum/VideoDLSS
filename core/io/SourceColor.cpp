#include "io/SourceColor.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/ambient_viewing_environment.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/frame.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
}

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>

#include "util/Error.h"

namespace dlssvid {

std::string_view ToString(TagFrom from) {
    switch (from) {
        case TagFrom::None: return "";
        case TagFrom::Container: return "container";
        case TagFrom::Stream: return "stream";
        case TagFrom::Frame: return "frame";
        case TagFrom::Packet: return "first packet";
    }
    return "";
}

std::string_view ToString(SourceClass c) {
    switch (c) {
        case SourceClass::Legacy: return "sdr";
        case SourceClass::Pq: return "pq";
        case SourceClass::Hlg: return "hlg";
        case SourceClass::Wide: return "wide";
        case SourceClass::Log: return "log";
    }
    return "sdr";
}

std::string_view ToString(ColourRefusal r) {
    switch (r) {
        case ColourRefusal::None: return "";
        case ColourRefusal::Matrix: return "matrix";
        case ColourRefusal::DolbyVisionIpt: return "dolby vision ipt";
        case ColourRefusal::RgbHdr: return "rgb hdr";
    }
    return "";
}

std::string_view ToString(FirstLook how) {
    switch (how) {
        case FirstLook::None: return "none";
        case FirstLook::Decode: return "first frame";
        case FirstLook::Packet: return "first packet";
    }
    return "none";
}

const ColourNote* SourceColor::Note(std::string_view key) const {
    for (const ColourNote& n : notes)
        if (n.key == key) return &n;
    return nullptr;
}

// ---- names ---------------------------------------------------------------------------------------------------------------------

namespace {
std::string NameOr(const char* name, int value) { return name ? std::string(name) : std::to_string(value); }
}  // namespace

std::string MatrixName(int matrix) { return NameOr(av_color_space_name(static_cast<AVColorSpace>(matrix)), matrix); }
std::string PrimariesName(int primaries) { return NameOr(av_color_primaries_name(static_cast<AVColorPrimaries>(primaries)), primaries); }
std::string TrcName(int trc) { return NameOr(av_color_transfer_name(static_cast<AVColorTransferCharacteristic>(trc)), trc); }
std::string RangeName(int range) { return NameOr(av_color_range_name(static_cast<AVColorRange>(range)), range); }

std::string ColourWords(const RawTags& t) {
    return "matrix " + MatrixName(t.matrix) + ", primaries " + PrimariesName(t.primaries) + ", transfer " + TrcName(t.trc) + ", range " +
           RangeName(t.range);
}

std::string_view ClassWords(SourceClass c) {
    switch (c) {
        case SourceClass::Legacy: return "SDR";
        case SourceClass::Pq: return "HDR PQ";
        case SourceClass::Hlg: return "HDR HLG";
        case SourceClass::Wide: return "wide gamut SDR";
        case SourceClass::Log: return "log curve";
    }
    return "SDR";
}

namespace {

std::string Num(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
    std::string s = buf;
    if (s.find('.') != std::string::npos) {  // 1000.0000 -> 1000, 0.0050 -> 0.005
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

std::string FieldWords(const char* label, const std::string& value, TagFrom from, const std::string& streamValue) {
    std::string w = std::string(label) + " " + value;
    if (from == TagFrom::Frame) w += " (frame; stream " + streamValue + ")";
    else if (from != TagFrom::None) w += " (" + std::string(ToString(from)) + ")";
    return w;
}

std::string Line(const char* label, const std::string& text) {
    std::string l = std::string(label) + ": ";
    while (l.size() < 13) l += ' ';  // the column of info's other lines ("video:       ")
    return l + text;
}

}  // namespace

std::vector<std::string> ColourInfoLines(const SourceColor& c) {
    std::vector<std::string> lines;
    bool planNote = false;
    for (const ColourNote& n : c.notes) planNote = planNote || n.plan;
    if (c.legacyPath && !c.doviConf && !c.doviRpu && c.hdr.Empty() && !c.hdr10Plus && !planNote) return lines;
    const RawTags& e = c.effective;
    lines.push_back(Line("colour", std::string(ClassWords(c.cls)) + " - " + FieldWords("matrix", MatrixName(e.matrix), c.matrixFrom, MatrixName(c.stream.matrix)) +
                                       ", " + FieldWords("primaries", PrimariesName(e.primaries), c.primariesFrom, PrimariesName(c.stream.primaries)) +
                                       ", " + FieldWords("transfer", TrcName(e.trc), c.trcFrom, TrcName(c.stream.trc)) + ", " +
                                       FieldWords("range", RangeName(e.range), c.rangeFrom, RangeName(c.stream.range))));
    if (c.hdr.mdcv) {
        const Mdcv& m = *c.hdr.mdcv;
        std::string t;
        if (m.hasPrimaries) {
            const char* names[3] = {"R", "G", "B"};
            for (int i = 0; i < 3; ++i) t += std::string(i ? " " : "") + names[i] + " " + Num(m.primaries[i][0].Value(), 4) + "," + Num(m.primaries[i][1].Value(), 4);
            t += " white " + Num(m.white[0].Value(), 4) + "," + Num(m.white[1].Value(), 4);
        }
        if (m.hasLuminance) t += std::string(t.empty() ? "" : ", ") + "luminance " + Num(m.minLum.Value(), 4) + "-" + Num(m.maxLum.Value(), 4) + " cd/m2";
        lines.push_back(Line("mastering", (t.empty() ? std::string("(empty)") : t) + " (" + std::string(ToString(c.mdcvFrom)) + ")"));
    }
    if (c.hdr.cll)
        lines.push_back(Line("light level", "MaxCLL " + std::to_string(c.hdr.cll->maxCll) + ", MaxFALL " + std::to_string(c.hdr.cll->maxFall) + " cd/m2 (" +
                                                std::string(ToString(c.cllFrom)) + ")"));
    if (c.hdr.amve)
        lines.push_back(Line("ambient", "illuminance " + Num(c.hdr.amve->illuminance.Value(), 4) + " lux, light " + Num(c.hdr.amve->lightX.Value(), 4) + "," +
                                            Num(c.hdr.amve->lightY.Value(), 4) + " (" + std::string(ToString(c.amveFrom)) + ")"));
    if (c.doviConf || c.doviRpu) {
        std::string t;
        if (c.doviConf)
            t = "profile " + std::to_string(c.doviConf->profile) + ", level " + std::to_string(c.doviConf->level) + ", compatibility " +
                std::to_string(c.doviConf->blCompat) + " (record)";
        else
            t = c.doviProfile ? "profile " + std::to_string(c.doviProfile) + " (guessed from the first frame's RPU)" : std::string("an RPU of an unknown profile");
        if (c.refuseAll == ColourRefusal::DolbyVisionIpt) t += ": IPT base - colours come out wrong without the RPU";
        else if (c.Note("source.dv_unknown") || c.Note("source.dv_record_short")) t += ": unknown base - removed in every output";
        else t += ": removed in every output";
        lines.push_back(Line("dolby vision", t));
    }
    if (c.hdr10Plus) lines.push_back(Line("hdr10+", "present (" + std::string(ToString(c.hdr10PlusFrom)) + "): not carried into results"));
    for (const ColourNote& n : c.notes) lines.push_back(Line("note", n.detail));
    return lines;
}

// ---- merge and classify --------------------------------------------------------------------------------------------------------

RawTags MergeTags(const RawTags& frame, const RawTags& stream, const RawTags& container, TagFrom from[4]) {
    TagFrom where[4] = {};
    auto pick = [](int f, int s, int c, int unspecified, bool useFrame, TagFrom& w) {
        if (useFrame && f != unspecified && f != s) {  // equal to the stream's: the stream's (FFmpeg filled the frame from it)
            w = TagFrom::Frame;
            return f;
        }
        if (s != unspecified) {
            w = TagFrom::Stream;
            return s;
        }
        if (c != unspecified) {
            w = TagFrom::Container;
            return c;
        }
        w = TagFrom::None;
        return unspecified;
    };
    RawTags out;
    out.matrix = pick(frame.matrix, stream.matrix, container.matrix, 2, true, where[0]);
    out.primaries = pick(frame.primaries, stream.primaries, container.primaries, 2, true, where[1]);
    out.trc = pick(frame.trc, stream.trc, container.trc, 2, true, where[2]);
    out.range = pick(frame.range, stream.range, container.range, 0, false, where[3]);  // a frame's range adds nothing (see the header)
    TagFrom unused;
    out.chromaLoc = pick(frame.chromaLoc, stream.chromaLoc, container.chromaLoc, 0, true, unused);
    if (from)
        for (int i = 0; i < 4; ++i) from[i] = where[i];
    return out;
}

SourceClass ClassOf(int matrix, int primaries, int trc) {
    if (trc == 16) return SourceClass::Pq;
    if (trc == 18) return SourceClass::Hlg;
    if (primaries == 9) {
        switch (trc) {
            case 1: case 6: case 13: case 14: case 15: case 2: return SourceClass::Wide;  // an SDR curve (or none said: taken as SDR)
            default: return SourceClass::Log;                                            // a curve not converted (a camera's log)
        }
    }
    if (matrix == 9) return SourceClass::Wide;  // BT.2020's matrix under other primaries
    return SourceClass::Legacy;
}

int GuessDoviProfile(const DoviRpu& rpu) {
    // libavcodec/dovi_rpu.c (n7.1.1) ff_dovi_guess_profile_hevc, as it is
    switch (rpu.vdrRpuProfile) {
        case 0:
            if (rpu.blFullRange) return 5;
            break;
        case 1:
            if (rpu.elResampling && !rpu.disableResidual) return rpu.vdrBitDepth == 12 ? 7 : 4;
            return 8;
        default: break;
    }
    return 0;
}

namespace {

std::string BaseWords(int blCompat) {
    switch (blCompat) {
        case 1: case 6: return "HDR10 base";
        case 2: return "SDR base";
        case 4: return "HLG base";
        default: return "base " + std::to_string(blCompat);
    }
}

std::string DoviWords(const DoviConf& d) {
    return "Dolby Vision profile " + std::to_string(d.profile) + ", level " + std::to_string(d.level) + ", compatibility " + std::to_string(d.blCompat);
}

std::string HdrRecordsWords(const HdrStatic& hdr) {
    std::string w;
    auto add = [&](const char* s) { w += (w.empty() ? "" : ", ") + std::string(s); };
    if (hdr.mdcv) add("mastering display");
    if (hdr.cll) add("light level");
    if (hdr.amve) add("ambient viewing environment");
    return w;
}

bool SaysSomething(const RawTags& t) { return t.matrix != 2 || t.primaries != 2 || t.trc != 2; }

// HLG (18) taken over the SDR curve an HEVC / H.264 VUI keeps beside an alternative transfer SEI - as it should be, no news
bool AltTransfer(int said, int taken) { return taken == 18 && (said == 1 || said == 6 || said == 14 || said == 15); }

std::string DifferenceWords(const std::vector<TagDifference>& d) {  // "transfer arib-std-b67 / bt2020-10, primaries ..."
    std::string w;
    for (const TagDifference& x : d)
        w += (w.empty() ? "" : ", ") + std::string(x.field) + " " + TagValueName(x.field, x.a) + " / " + TagValueName(x.field, x.b);
    return w;
}

}  // namespace

std::string TagValueName(std::string_view field, int value) {
    if (field == "matrix") return MatrixName(value);
    if (field == "primaries") return PrimariesName(value);
    if (field == "transfer") return TrcName(value);
    return RangeName(value);
}

std::vector<TagDifference> FrameDifferences(const SourceColor& c) {
    std::vector<TagDifference> d;
    const RawTags &f = c.frame, &s = c.stream;
    if (f.matrix != 2 && s.matrix != 2 && f.matrix != s.matrix) d.push_back({"matrix", f.matrix, s.matrix});
    if (f.primaries != 2 && s.primaries != 2 && f.primaries != s.primaries) d.push_back({"primaries", f.primaries, s.primaries});
    if (f.trc != 2 && s.trc != 2 && f.trc != s.trc && !AltTransfer(s.trc, f.trc)) d.push_back({"transfer", f.trc, s.trc});
    return d;
}

std::vector<TagDifference> ContainerDifferences(const SourceColor& c) {
    std::vector<TagDifference> d;
    const RawTags &k = c.container, &e = c.effective;
    auto taken = [](TagFrom from) { return from == TagFrom::Stream || from == TagFrom::Frame; };
    if (k.matrix != 2 && taken(c.matrixFrom) && k.matrix != e.matrix) d.push_back({"matrix", k.matrix, e.matrix});
    if (k.primaries != 2 && taken(c.primariesFrom) && k.primaries != e.primaries) d.push_back({"primaries", k.primaries, e.primaries});
    if (k.trc != 2 && taken(c.trcFrom) && k.trc != e.trc && !AltTransfer(k.trc, e.trc)) d.push_back({"transfer", k.trc, e.trc});
    return d;
}

void Classify(SourceColor& c, std::string_view layoutKind, std::string_view codec) {
    c.notes.clear();
    c.refuseProcessing = ColourRefusal::None;
    c.refuseAll = ColourRefusal::None;
    const RawTags& e = c.effective;
    const int M = e.matrix, P = e.primaries, T = e.trc;
    c.cls = ClassOf(e);
    c.rawCls = ClassOf(c.stream);
    auto note = [&](std::string key, std::string detail, bool plan = true) { c.notes.push_back({std::move(key), std::move(detail), plan}); };

    // 1. Dolby Vision: by its base layer's compatibility (any codec), then by the RPU of the first frame, then matrix 15 (IPT-C2)
    bool ipt = false;
    std::string iptWords;
    c.doviProfile = 0;
    if (c.doviConf) {
        const DoviConf& d = *c.doviConf;
        c.doviProfile = d.profile;
        if (d.profile == 1 || d.profile == 3 || d.profile == 5 || (d.profile == 10 && d.blCompat == 0)) {
            ipt = true;
            iptWords = DoviWords(d) + ": an IPT base layer";
        } else if (d.blCompat == 1 || d.blCompat == 6 || d.blCompat == 2 || d.blCompat == 4) {
            note("source.dv_removed", DoviWords(d) + " (" + BaseWords(d.blCompat) + "): the RPU is dropped in every output, the base layer stays");
        } else if (d.blCompat == 0 && (d.profile == 4 || d.profile == 7 || d.profile == 8 || d.profile == 9)) {
            note("source.dv_record_short", DoviWords(d) +
                                               ": a record too short to name its base (FFmpeg reads one under 5 bytes as compatibility 0) - the class "
                                               "follows the colour tags; the RPU is dropped in every output");
        } else if (M == 2 && T == 2) {
            ipt = true;
            iptWords = DoviWords(d) + " - a base FFmpeg 7.1.1 does not name, without colour tags: taken for an IPT base";
        } else {
            note("source.dv_unknown", DoviWords(d) + " - a base FFmpeg 7.1.1 does not name: the class follows the colour tags; the RPU is dropped "
                                                     "in every output");
        }
    } else if (c.doviRpu) {
        const int g = GuessDoviProfile(*c.doviRpu);
        c.doviProfile = g;
        const std::string words = g ? "Dolby Vision profile " + std::to_string(g) + " (guessed from the first frame's RPU, no record)"
                                    : std::string("Dolby Vision of an unknown profile (an RPU in the first frame, no record)");
        if (g == 5) {
            ipt = true;
            iptWords = words + ": an IPT base layer";
        } else if (g == 4 || g == 7 || g == 8) {
            note("source.dv_removed", words + ": the RPU is dropped in every output, the base layer stays");
        } else if (M == 2 && T == 2) {
            ipt = true;
            iptWords = words + ", without colour tags: taken for an IPT base";
        } else {
            note("source.dv_unknown", words + ": the class follows the colour tags; the RPU is dropped in every output");
        }
    }
    if (M == 15 && !ipt) {
        ipt = true;
        iptWords = "matrix ipt-c2 (a Dolby Vision IPT base layer)";
    }
    if (ipt) {
        c.refuseAll = ColourRefusal::DolbyVisionIpt;
        note("source.dv_ipt", iptWords + " - its colours are wrong without the RPU, and nothing here applies the RPU");
    }

    // 2. matrix 10 (BT.2020 CL) or 14 (ICtCp): no processing; the class from the rows below
    if (M == 10 || M == 14) {
        c.refuseProcessing = ColourRefusal::Matrix;
        note("source.matrix_unsupported", "matrix " + MatrixName(M) + ": read as BT.709 / BT.601 - its colours come out wrong when processed");
    }
    // 3. an RGB stream of HDR or BT.2020: swscale makes it 8-bit 4:2:0 SDR (VideoDecoder: the swscale path)
    if ((layoutKind == "rgb" || M == 0) && (T == 16 || T == 18 || P == 9)) {
        if (c.refuseAll == ColourRefusal::None) c.refuseAll = ColourRefusal::RgbHdr;
        note("source.rgb_hdr", "an RGB stream of " + TrcName(T) + " / " + PrimariesName(P) +
                                   ": read as 8-bit 4:2:0 SDR through swscale - its colours come out wrong");
    }

    // 4-9: what the class assumed
    if ((c.cls == SourceClass::Pq || c.cls == SourceClass::Hlg) && (M == 2 || P == 2)) {
        std::string which = M == 2 && P == 2 ? "matrix and primaries" : M == 2 ? "matrix" : "primaries";
        note("source.bt2020_assumed", which + " unspecified under " + TrcName(T) + ": BT.2020 assumed");
    }
    if (P == 9 && T == 2) note("source.trc_unspecified", "BT.2020 primaries without a transfer: taken as SDR");
    if (c.cls == SourceClass::Log)
        note("source.trc_not_converted", "transfer " + TrcName(T) + " under BT.2020 primaries (a camera's log curve?): not converted");
    if (M == 9 && P != 9 && T != 16 && T != 18)
        note("source.matrix_2020", "matrix BT.2020 under primaries " + PrimariesName(P) + ": taken as wide gamut");

    // HDR records
    if (codec == "av1" && (c.cls == SourceClass::Pq || c.cls == SourceClass::Hlg)) {
        const std::string records = HdrRecordsWords(c.hdr);
        note("source.av1_stream_metadata", "HDR metadata inside the AV1 stream (metadata OBUs) is not read here and not carried into results; " +
                                               (records.empty() ? std::string("the container has none") : "from the container: " + records));
    } else if (c.cls == SourceClass::Pq && !c.hdr.mdcv && !c.hdr.cll) {
        note("source.no_hdr_metadata", "PQ without mastering display or light level metadata");
    }
    if (c.hdr10Plus)
        note("source.hdr10plus", "HDR10+ dynamic metadata (in the " + std::string(ToString(c.hdr10PlusFrom)) + "): not carried into results");

    // the layers. HLG from an alternative transfer SEI over the VUI's SDR curve (the matrix and the primaries as the stream says) is no news
    const bool altTransfer = AltTransfer(c.stream.trc, e.trc) && (c.stream.matrix == 2 || c.stream.matrix == e.matrix) &&
                             (c.stream.primaries == 2 || c.stream.primaries == e.primaries);
    if (c.cls != c.rawCls && SaysSomething(c.stream) && !altTransfer) {
        note("source.layers_differ", "the frame or the container say " + ColourWords(e) + " (" + std::string(ToString(c.cls)) +
                                         "), the codec parameters " + ColourWords(c.stream) + " (" + std::string(ToString(c.rawCls)) + ")");
    } else if (const std::vector<TagDifference> d = FrameDifferences(c); !d.empty()) {
        note("source.frame_differs", "the first frame and the stream say " + DifferenceWords(d), false);
    }
    if (const std::vector<TagDifference> d = ContainerDifferences(c); !d.empty()) {
        // of consequence when the container alone would have given another class
        const RawTags k = c.container;
        const SourceClass containerCls = ClassOf(k.matrix != 2 ? k.matrix : M, k.primaries != 2 ? k.primaries : P, k.trc != 2 ? k.trc : T);
        note("source.container_differs", "the container says " + DifferenceWords(d) + " taken from the stream or the frame", containerCls != c.cls);
    }
    if (!c.probeError.empty())
        note("source.probe_failed", std::string(c.lookFailed == FirstLook::Packet ? "the first video packet" : "the first frame") +
                                        " could not be read (" + c.probeError + "): the colour comes from the stream and the container only");

    c.legacyPath = c.cls == SourceClass::Legacy && c.rawCls == SourceClass::Legacy && c.refuseProcessing == ColourRefusal::None &&
                   c.refuseAll == ColourRefusal::None;
}

FirstLook NeedsFirstLook(std::string_view codec, std::string_view demuxer, int bitDepth, const RawTags& t) {
    const bool hint = t.primaries == 9 || t.trc == 14 || t.trc == 15 || t.trc == 16 || t.trc == 18 || t.matrix == 9 || t.matrix == 10 ||
                      t.matrix == 14 || t.matrix == 15;
    const bool raw = demuxer == "hevc" || demuxer == "h264";
    if (codec == "h264" || codec == "hevc" || codec == "prores") {
        if (hint || raw || (codec == "hevc" && bitDepth >= 10 && t.matrix == 2 && t.trc == 2)) return FirstLook::Decode;
        // a ProRes frame header says its colour whatever the container: colr is optional (movenc writes it only when all three are said)
        if (codec == "prores" && (t.matrix == 2 || t.primaries == 2 || t.trc == 2)) return FirstLook::Decode;
    }
    if (demuxer == "matroska,webm" && t.trc == 16) return FirstLook::Packet;
    return FirstLook::None;
}

// ---- FFmpeg records ------------------------------------------------------------------------------------------------------------

namespace {
Ratio QOf(AVRational r) { return Ratio{r.num, r.den}; }

Mdcv MdcvOf(const AVMasteringDisplayMetadata& m) {
    Mdcv out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) out.primaries[i][j] = QOf(m.display_primaries[i][j]);
    out.white[0] = QOf(m.white_point[0]);
    out.white[1] = QOf(m.white_point[1]);
    out.minLum = QOf(m.min_luminance);
    out.maxLum = QOf(m.max_luminance);
    out.hasPrimaries = m.has_primaries != 0;
    out.hasLuminance = m.has_luminance != 0;
    return out;
}

Amve AmveOf(const AVAmbientViewingEnvironment& a) { return Amve{QOf(a.ambient_illuminance), QOf(a.ambient_light_x), QOf(a.ambient_light_y)}; }
}  // namespace

RawTags FrameTagsOf(const AVFrame* f) {
    RawTags t;
    t.matrix = static_cast<int>(f->colorspace);
    t.primaries = static_cast<int>(f->color_primaries);
    t.trc = static_cast<int>(f->color_trc);
    t.range = static_cast<int>(f->color_range);
    t.chromaLoc = static_cast<int>(f->chroma_location);
    return t;
}

void ReadCodedSideData(const AVPacketSideData* sd, int count, HdrStatic& hdr, std::optional<DoviConf>* dovi) {
    if (const AVPacketSideData* m = av_packet_side_data_get(sd, count, AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
        m && m->size >= sizeof(AVMasteringDisplayMetadata))
        hdr.mdcv = MdcvOf(*reinterpret_cast<const AVMasteringDisplayMetadata*>(m->data));
    if (const AVPacketSideData* l = av_packet_side_data_get(sd, count, AV_PKT_DATA_CONTENT_LIGHT_LEVEL); l && l->size >= sizeof(AVContentLightMetadata)) {
        const auto* c = reinterpret_cast<const AVContentLightMetadata*>(l->data);
        hdr.cll = Cll{c->MaxCLL, c->MaxFALL};
    }
    if (const AVPacketSideData* a = av_packet_side_data_get(sd, count, AV_PKT_DATA_AMBIENT_VIEWING_ENVIRONMENT);
        a && a->size >= sizeof(AVAmbientViewingEnvironment))
        hdr.amve = AmveOf(*reinterpret_cast<const AVAmbientViewingEnvironment*>(a->data));
    if (dovi)
        if (const AVPacketSideData* d = av_packet_side_data_get(sd, count, AV_PKT_DATA_DOVI_CONF); d && d->size >= 8) {
            // the record's fields up to dv_bl_signal_compatibility_id (the struct's size is not ABI - av_dovi_alloc - but its head is)
            const auto* r = reinterpret_cast<const AVDOVIDecoderConfigurationRecord*>(d->data);
            DoviConf c;
            c.major = r->dv_version_major;
            c.minor = r->dv_version_minor;
            c.profile = r->dv_profile;
            c.level = r->dv_level;
            c.rpu = r->rpu_present_flag != 0;
            c.el = r->el_present_flag != 0;
            c.bl = r->bl_present_flag != 0;
            c.blCompat = r->dv_bl_signal_compatibility_id;
            *dovi = c;
        }
}

void ReadFrameSideData(AVFrameSideData* const* sd, int count, HdrStatic& hdr, std::optional<DoviRpu>* rpu, bool* hdr10Plus) {
    if (const AVFrameSideData* m = av_frame_side_data_get(sd, count, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
        m && m->size >= sizeof(AVMasteringDisplayMetadata))
        hdr.mdcv = MdcvOf(*reinterpret_cast<const AVMasteringDisplayMetadata*>(m->data));
    if (const AVFrameSideData* l = av_frame_side_data_get(sd, count, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL); l && l->size >= sizeof(AVContentLightMetadata)) {
        const auto* c = reinterpret_cast<const AVContentLightMetadata*>(l->data);
        hdr.cll = Cll{c->MaxCLL, c->MaxFALL};
    }
    if (const AVFrameSideData* a = av_frame_side_data_get(sd, count, AV_FRAME_DATA_AMBIENT_VIEWING_ENVIRONMENT);
        a && a->size >= sizeof(AVAmbientViewingEnvironment))
        hdr.amve = AmveOf(*reinterpret_cast<const AVAmbientViewingEnvironment*>(a->data));
    if (rpu)
        if (const AVFrameSideData* d = av_frame_side_data_get(sd, count, AV_FRAME_DATA_DOVI_METADATA); d && d->size >= sizeof(AVDOVIMetadata)) {
            const AVDOVIRpuDataHeader* h = av_dovi_get_header(reinterpret_cast<const AVDOVIMetadata*>(d->data));
            DoviRpu r;
            r.vdrRpuProfile = h->vdr_rpu_profile;
            r.vdrBitDepth = h->vdr_bit_depth;
            r.blFullRange = h->bl_video_full_range_flag != 0;
            r.elResampling = h->el_spatial_resampling_filter_flag != 0;
            r.disableResidual = h->disable_residual_flag != 0;
            *rpu = r;
        }
    if (hdr10Plus && av_frame_side_data_get(sd, count, AV_FRAME_DATA_DYNAMIC_HDR_PLUS)) *hdr10Plus = true;
}

RawTags WithoutReserved(const RawTags& t, std::string_view codec) {
    RawTags out = t;
    if (out.primaries == 0 || out.primaries == 3) out.primaries = 2;
    if (out.trc == 0 || out.trc == 3) out.trc = 2;
    if (out.matrix == 3 || (out.matrix == 0 && codec == "prores")) out.matrix = 2;
    return out;
}

namespace {
long long Units(const Ratio& r, double per) { return std::llround(r.Value() * per); }
}  // namespace

bool SameRecord(const Mdcv& a, const Mdcv& b) {
    if (a.hasPrimaries != b.hasPrimaries || a.hasLuminance != b.hasLuminance) return false;
    if (a.hasPrimaries) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 2; ++j)
                if (Units(a.primaries[i][j], 50000) != Units(b.primaries[i][j], 50000)) return false;
        if (Units(a.white[0], 50000) != Units(b.white[0], 50000) || Units(a.white[1], 50000) != Units(b.white[1], 50000)) return false;
    }
    return !a.hasLuminance || (Units(a.minLum, 10000) == Units(b.minLum, 10000) && Units(a.maxLum, 10000) == Units(b.maxLum, 10000));
}

bool SameRecord(const Amve& a, const Amve& b) {
    return Units(a.illuminance, 10000) == Units(b.illuminance, 10000) && Units(a.lightX, 50000) == Units(b.lightX, 50000) &&
           Units(a.lightY, 50000) == Units(b.lightY, 50000);
}

SourceColor SourceColorOf(const ColourLayers& in, std::string_view layoutKind, std::string_view codec) {
    SourceColor c;
    c.container = WithoutReserved(in.container, codec);
    c.stream = WithoutReserved(in.stream, codec);
    c.doviConf = in.dovi;
    // the records: the container's, then the codec's (its extradata) where the container has none
    c.hdr = in.containerRecords;
    if (c.hdr.mdcv) c.mdcvFrom = TagFrom::Container;
    if (c.hdr.cll) c.cllFrom = TagFrom::Container;
    if (c.hdr.amve) c.amveFrom = TagFrom::Container;
    if (!c.hdr.mdcv && in.codecRecords.mdcv) c.hdr.mdcv = in.codecRecords.mdcv, c.mdcvFrom = TagFrom::Stream;
    if (!c.hdr.cll && in.codecRecords.cll) c.hdr.cll = in.codecRecords.cll, c.cllFrom = TagFrom::Stream;
    if (!c.hdr.amve && in.codecRecords.amve) c.hdr.amve = in.codecRecords.amve, c.amveFrom = TagFrom::Stream;
    if (in.first) {
        const FirstLookColor& first = *in.first;
        if (!first.error.empty()) {
            c.probeError = first.error;
            c.lookFailed = first.how;
        } else {
            c.looked = first.how;
            if (first.how == FirstLook::Decode) {
                c.frame = WithoutReserved(first.tags, codec);
                c.doviRpu = first.rpu;
                // the frame carries the stream's records as well (libavcodec copies them into every frame), an SEI's in other fractions:
                // the frame's own only where they say another value
                if (first.hdr.mdcv && !(c.hdr.mdcv && SameRecord(*first.hdr.mdcv, *c.hdr.mdcv))) c.hdr.mdcv = first.hdr.mdcv, c.mdcvFrom = TagFrom::Frame;
                if (first.hdr.cll && first.hdr.cll != c.hdr.cll) c.hdr.cll = first.hdr.cll, c.cllFrom = TagFrom::Frame;
                if (first.hdr.amve && !(c.hdr.amve && SameRecord(*first.hdr.amve, *c.hdr.amve))) c.hdr.amve = first.hdr.amve, c.amveFrom = TagFrom::Frame;
            }
            if (first.hdr10Plus) {
                c.hdr10Plus = true;
                c.hdr10PlusFrom = first.how == FirstLook::Decode ? TagFrom::Frame : TagFrom::Packet;
            }
        }
    }
    TagFrom from[4];
    c.effective = MergeTags(c.frame, c.stream, c.container, from);
    c.matrixFrom = from[0];
    c.primariesFrom = from[1];
    c.trcFrom = from[2];
    c.rangeFrom = from[3];
    Classify(c, layoutKind, codec);
    return c;
}

// ---- the first look ------------------------------------------------------------------------------------------------------------

namespace {

std::atomic<int64_t> g_frameDecodes{0};
std::atomic<int64_t> g_packetPeeks{0};

struct LookParts {  // what one look holds, freed whatever happens
    AVFormatContext* fmt = nullptr;
    AVCodecContext* ctx = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    ~LookParts() {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
    }
};

constexpr int kLookPackets = 64;      // video packets a look reads at most
constexpr int kLookReads = 4096;      // packets of any stream

FirstLookColor LookOnce(const std::filesystem::path& path, FirstLook how) {
    FirstLookColor r;
    r.how = how;
    LookParts p;
    const std::string url = path.string();
    CheckAv(avformat_open_input(&p.fmt, url.c_str(), nullptr, nullptr), "avformat_open_input");
    CheckAv(avformat_find_stream_info(p.fmt, nullptr), "avformat_find_stream_info");
    const AVCodec* decoder = nullptr;
    const int vs = av_find_best_stream(p.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (vs < 0) Throw("no video stream");
    p.pkt = av_packet_alloc();
    if (!p.pkt) Throw("av_packet_alloc failed");
    if (how == FirstLook::Packet) {
        ++g_packetPeeks;
        for (int reads = 0; reads < kLookReads; ++reads) {
            const int err = av_read_frame(p.fmt, p.pkt);
            if (err == AVERROR_EOF) break;
            CheckAv(err, "av_read_frame");
            const bool video = p.pkt->stream_index == vs;
            if (video) {
                size_t size = 0;
                r.hdr10Plus = av_packet_get_side_data(p.pkt, AV_PKT_DATA_DYNAMIC_HDR10_PLUS, &size) != nullptr;
            }
            av_packet_unref(p.pkt);
            if (video) return r;
        }
        Throw("no video packet");
    }
    ++g_frameDecodes;
    if (!decoder) Throw("no software decoder");
    p.ctx = avcodec_alloc_context3(decoder);
    p.frame = av_frame_alloc();
    if (!p.ctx || !p.frame) Throw("av_frame_alloc failed");
    CheckAv(avcodec_parameters_to_context(p.ctx, p.fmt->streams[vs]->codecpar), "avcodec_parameters_to_context");
    p.ctx->thread_count = 0;               // libavcodec's choice (its default is 1: no threads at all)
    p.ctx->thread_type = FF_THREAD_SLICE;  // frame threads would hold the first frame back by their count
    CheckAv(avcodec_open2(p.ctx, decoder, nullptr), "avcodec_open2");
    auto receive = [&] {
        const int err = avcodec_receive_frame(p.ctx, p.frame);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return false;
        CheckAv(err, "avcodec_receive_frame");
        return true;
    };
    bool got = false;
    int sent = 0;
    for (int reads = 0; !got && sent < kLookPackets && reads < kLookReads; ++reads) {
        const int err = av_read_frame(p.fmt, p.pkt);
        if (err == AVERROR_EOF) break;
        CheckAv(err, "av_read_frame");
        if (p.pkt->stream_index == vs) {
            const int s = avcodec_send_packet(p.ctx, p.pkt);
            if (s < 0 && s != AVERROR(EAGAIN)) CheckAv(s, "avcodec_send_packet");
            ++sent;
            got = receive();
        }
        av_packet_unref(p.pkt);
    }
    if (!got) {
        CheckAv(avcodec_send_packet(p.ctx, nullptr), "avcodec_send_packet (flush)");
        got = receive();
    }
    if (!got) Throw("no frame in the first " + std::to_string(sent) + " video packets");
    r.tags = FrameTagsOf(p.frame);
    ReadFrameSideData(p.frame->side_data, p.frame->nb_side_data, r.hdr, &r.rpu, &r.hdr10Plus);
    return r;
}

}  // namespace

int64_t FirstFrameDecodes() { return g_frameDecodes.load(); }
int64_t FirstPacketPeeks() { return g_packetPeeks.load(); }

FirstLookColor FirstLookCached(const std::filesystem::path& path, FirstLook how) {
    if (how == FirstLook::None) return {};
    struct Entry {
        uintmax_t size = 0;
        std::filesystem::file_time_type mtime, lookedAt;
        uint64_t used = 0;
        FirstLookColor look;
    };
    static std::mutex mutex;
    static std::map<std::wstring, Entry> cache;
    static uint64_t clock = 0;
    constexpr size_t kEntries = 64;
    std::error_code ec, sizeEc, timeEc;
    const auto abs = std::filesystem::absolute(path, ec);
    const uintmax_t size = std::filesystem::file_size(abs, sizeEc);
    const auto mtime = std::filesystem::last_write_time(abs, timeEc);
    if (!ec) ec = sizeEc ? sizeEc : timeEc;  // no key without all three: looked at every time (and the look says why it failed)
    // as Sha256FileCached: a look at a file written just before it says nothing about the next one (the clock tick of the file time)
    // (a time ahead of the clock - a camera card's local time, a NAS whose clock runs fast - was not written in the tick before the look)
    const auto settled = [](const Entry& e) { return e.lookedAt - e.mtime > std::chrono::seconds(2) || e.mtime - e.lookedAt > std::chrono::seconds(2); };
    if (!ec) {
        std::lock_guard<std::mutex> lock(mutex);
        if (auto it = cache.find(abs.wstring()); it != cache.end() && it->second.size == size && it->second.mtime == mtime && settled(it->second) &&
                                                 (it->second.look.how == how || it->second.look.how == FirstLook::Decode)) {
            it->second.used = ++clock;
            return it->second.look;
        }
    }
    const auto lookedAt = std::filesystem::file_time_type::clock::now();  // before the read: a write during it counts as after
    FirstLookColor look;
    try {
        look = LookOnce(abs.empty() ? path : abs, how);
    } catch (const std::exception& e) {
        look = FirstLookColor{};
        look.how = how;
        look.error = e.what();
    }
    if (!ec) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() >= kEntries && !cache.count(abs.wstring())) {
            auto oldest = cache.begin();
            for (auto it = cache.begin(); it != cache.end(); ++it)
                if (it->second.used < oldest->second.used) oldest = it;
            cache.erase(oldest);
        }
        cache[abs.wstring()] = Entry{size, mtime, lookedAt, ++clock, look};
    }
    return look;
}

}  // namespace dlssvid
