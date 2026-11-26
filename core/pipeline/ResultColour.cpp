#include "pipeline/ResultColour.h"

#include "io/VideoEncoder.h"

namespace dlssvid {

bool CodecKeepsRange(const std::string& codec) { return codec != "prores_ks" && codec != "ffv1"; }

namespace {
bool NvencCodec(const std::string& codec) { return codec == "h264_nvenc" || codec == "hevc_nvenc" || codec == "av1_nvenc"; }

std::string TagWords(const ColorTags& t) {
    return "matrix " + MatrixName(t.matrix) + ", primaries " + PrimariesName(t.primaries) + ", transfer " + TrcName(t.transfer);
}
}  // namespace

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
    expect.checkRange = caps.range || CodecKeepsRange(codec);
    if (pc.tags.range == 2 && !expect.checkRange)
        note("encode.range_lost", "warning", "full range: " + container + " with " + codec + " keeps no range - players will read it as limited");
    if (pc.meta.mdcv) {
        const bool whole = pc.meta.mdcv->hasPrimaries && pc.meta.mdcv->hasLuminance;
        if (caps.mdcvNeedsBoth && !whole)
            note("encode.mdcv_lost", "warning",
                 "the mastering display says only its " + std::string(pc.meta.mdcv->hasPrimaries ? "primaries" : "luminance") + ": " + container +
                     " writes one only whole - not carried");
        else if (caps.mdcv)
            expect.mdcv = pc.meta.mdcv;
    }
    if (pc.meta.cll && caps.cll && !((caps.muxer == "matroska" || caps.muxer == "webm") && (pc.meta.cll->maxCll == 0 || pc.meta.cll->maxFall == 0)))
        expect.cll = pc.meta.cll;  // (Matroska's reader keeps a light level only when both values are said)
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
