#include "pipeline/ResultColour.h"

#include "io/VideoEncoder.h"

extern "C" {
#include <libavutil/mathematics.h>
}

namespace dlssvid {

namespace {
bool NvencCodec(const std::string& codec) { return codec == "h264_nvenc" || codec == "hevc_nvenc" || codec == "av1_nvenc"; }
bool ProResCodec(const std::string& codec) { return codec.rfind("prores", 0) == 0; }  // prores_ks, prores_aw, prores

// What Matroska's reader keeps of a mastering display (matroskadec, mkv_parse_video_color): the primaries when every coordinate is above
// 0, the luminance when max > min >= 0 - a half that does not hold is written and dropped on reading. The values as matroskaenc writes
// them (av_q2d: a zero denominator is inf, 0/0 NaN - a value past INT_MAX in Matroska's floats comes as {1, 0} from av_d2q)
double Written(const Ratio& r) { return static_cast<double>(r.num) / r.den; }
bool PrimariesHold(const Mdcv& m) {
    for (const auto& xy : m.primaries)
        if (!(Written(xy[0]) > 0 && Written(xy[1]) > 0)) return false;
    return Written(m.white[0]) > 0 && Written(m.white[1]) > 0;
}
bool LuminanceHolds(const Mdcv& m) { return Written(m.maxLum) > Written(m.minLum) && Written(m.minLum) >= 0; }

// What MP4 / MOV's boxes hold (movenc: mdcv's coordinates in 16 bits of 1/50000, its luminance in 32 bits of 1/10000; clli in 16 bits):
// a value outside them is written as its low bits - a record from Matroska's floats in other units (x265's 34000 for 0.68) included.
// Counted as movenc counts it (rescale_rational: av_rescale, half away from zero; a zero denominator gives INT64_MIN). The luminance up to
// INT32_MAX only: mov.c reads the u32 through av_make_q(int, ...) - above it the value comes back negative (the result check would fail)
bool Fits(const Ratio& r, int64_t per, int64_t top) {
    const int64_t v = av_rescale(r.num, per, r.den);
    return v >= 0 && v <= top;
}
bool MdcvFitsBox(const Mdcv& m) {
    for (const auto& xy : m.primaries)
        if (!Fits(xy[0], 50000, 65535) || !Fits(xy[1], 50000, 65535)) return false;
    return Fits(m.white[0], 50000, 65535) && Fits(m.white[1], 50000, 65535) && Fits(m.maxLum, 10000, 2147483647LL) && Fits(m.minLum, 10000, 2147483647LL);
}

std::string TagWords(const ColorTags& t) {
    return "matrix " + MatrixName(t.matrix) + ", primaries " + PrimariesName(t.primaries) + ", transfer " + TrcName(t.transfer);
}
}  // namespace

bool CodecKeepsRange(const std::string& codec) { return NvencCodec(codec); }

PassthroughColour PlanPassthroughColour(const VideoStreamInfo& info, const std::string& codec, const OutputFormatRequest& format,
                                        const std::filesystem::path& output, const std::string& where) {
    const ColorPlan plan = ResolveColor(info);
    RequireColour(plan, ColourUse::Passthrough, where);
    PassthroughColour pc;
    pc.tags = plan.ResultTags();
    if (plan.legacy) return pc;  // SDR: written and checked as before
    const SourceColor& c = info.color;
    const VideoEncoder::ContainerColorCaps caps = VideoEncoder::ContainerColorCapsFor(output);
    const std::string container = caps.muxer.empty() ? output.extension().string() : caps.muxer;
    if (!output.empty() && c.cls != SourceClass::Legacy && !caps.tags)
        throw ColourUnsupported(ColourStop::NoContainerTags, c.cls,
                                where + ": " + std::string(ClassWords(c.cls)) + " needs its colour tags, and " + container +
                                    " has no place for them: players would show it as SDR - write it as .mp4, .mov or .mkv");
    pc.meta = plan.ResultMeta();
    auto note = [&](std::string key, std::string level, std::string text) {
        pc.encodeNotes.push_back(PlanNote{std::move(key), std::move(level), "encode", std::move(text)});
    };
    if (TagsFilled(c)) note("encode.tags_filled", "note", "the result is tagged " + TagWords(pc.tags) + ": the class's assumption where the source said none");
    const OutputFormat of = ResolveOutputFormat(codec, format, SourceLayout(info));
    if (c.Hdr() && of.desc.bitDepth < 10)
        note("encode.hdr_8bit", "warning",
             "HDR in 8 bits (" + codec + "): its gradations come out in steps (banding) - a 10-bit encoder keeps them (hevc_nvenc, av1_nvenc, prores_ks)");
    if (output.empty()) return pc;  // a plan without an output named: what the container keeps is not known yet

    ResultColor expect;
    // the tags read back: the container's colour (MP4 / MOV colr only with all three said) or the codec's own (NVENC's VUI)
    const bool colr = caps.tags && (!caps.allThreeOrNone || (pc.tags.matrix != 2 && pc.tags.primaries != 2 && pc.tags.transfer != 2));
    if (colr || NvencCodec(codec)) {
        expect.matrix = pc.tags.matrix;
        expect.primaries = pc.tags.primaries;
        expect.trc = pc.tags.transfer;
    }
    expect.range = pc.tags.range;
    // the container's range: Matroska's Colour, MP4's colr (written with the three tags only); else the codec's own
    expect.checkRange = (caps.range && colr) || CodecKeepsRange(codec);
    if (pc.tags.range == 2 && ProResCodec(codec))  // in any container: FFmpeg's ProRes decoder says limited to every frame
        note("encode.range_lost", "warning", "full range: ProRes keeps no range - decoders will read it as limited");
    else if (pc.tags.range == 2 && !expect.checkRange)
        note("encode.range_lost", "warning", "full range: " + container + " with " + codec + " keeps no range - players will read it as limited");
    const bool matroska = caps.muxer == "matroska" || caps.muxer == "webm", box = caps.muxer == "mp4" || caps.muxer == "mov";
    if (pc.meta.mdcv) {
        Mdcv m = *pc.meta.mdcv;
        const bool whole = m.hasPrimaries && m.hasLuminance;
        if (caps.mdcvNeedsBoth && !whole) {
            note("encode.mdcv_lost", "warning",
                 "the mastering display says only its " + std::string(m.hasPrimaries ? "primaries" : "luminance") + ": " + container +
                     " writes one only whole - not carried");
        } else if (caps.mdcv && box && !MdcvFitsBox(m)) {
            note("encode.mdcv_unfit", "warning",
                 "the mastering display has a value outside " + container + "'s mdcv fields (a coordinate above 1.31, a luminance below 0 or above "
                 "214748 cd/m2) - not carried");
            pc.meta.mdcv.reset();  // not written: the box would hold its low bits
        } else if (caps.mdcv) {
            if (matroska) {
                const bool primaries = m.hasPrimaries && !PrimariesHold(m), luminance = m.hasLuminance && !LuminanceHolds(m);
                if (primaries || luminance)
                    note("encode.mdcv_invalid", "warning",
                         "the mastering display's " + std::string(primaries && luminance ? "primaries and luminance do" : primaries ? "primaries do" : "luminance does") +
                             " not hold (a coordinate 0, a minimum below 0 or max not above min): " + container + "'s readers drop " + (primaries && luminance ? "them" : "it"));
                m.hasPrimaries = m.hasPrimaries && !primaries;
                m.hasLuminance = m.hasLuminance && !luminance;
            }
            if (m.hasPrimaries || m.hasLuminance) expect.mdcv = m;
        }
    }
    if (pc.meta.cll && caps.cll) {
        const Cll l = *pc.meta.cll;
        const std::string said = "the light level MaxCLL " + std::to_string(l.maxCll) + " / MaxFALL " + std::to_string(l.maxFall);
        if (box && (l.maxCll > 65535 || l.maxFall > 65535)) {
            note("encode.cll_invalid", "warning", said + " does not fit " + container + "'s clli fields (16 bits) - not carried");
            pc.meta.cll.reset();
        } else if (matroska && (l.maxCll == 0) != (l.maxFall == 0)) {  // both 0 say nothing: nothing lost
            note("encode.cll_invalid", "warning", said + ": " + container + "'s readers keep one only with both values - lost on reading");
        } else if (!(matroska && l.maxCll == 0)) {
            expect.cll = l;
        }
    }
    if (pc.meta.amve) {
        if (!caps.amve)
            note("encode.no_amve", "note", "the ambient viewing environment: " + container + " has no place for it - not carried");
        else if (pc.meta.amve->illuminance.num != 0)
            expect.amve = pc.meta.amve;
    }
    pc.expect = expect;
    return pc;
}

}  // namespace dlssvid
