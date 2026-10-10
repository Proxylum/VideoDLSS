#include "io/OutputFormat.h"

#include <algorithm>

#include "util/Error.h"

namespace dlssvid {

std::optional<int> ParseBitDepthChoice(std::string_view s) {
    if (s == "auto" || s.empty()) return 0;
    if (s == "8") return 8;
    if (s == "10") return 10;
    if (s == "12") return 12;
    return std::nullopt;
}

std::string BitDepthChoiceName(int bitDepth) { return bitDepth <= 0 ? "auto" : std::to_string(bitDepth); }

bool ValidChromaChoice(std::string_view s) { return s == "auto" || s == "420" || s == "422" || s == "444"; }

const std::vector<CodecCaps>& KnownCodecs() {
    static const std::vector<CodecCaps> codecs = {
        {"h264_nvenc", "H.264 NVENC", {8}, {"420", "444"}, true, ".mp4"},
        {"hevc_nvenc", "HEVC NVENC", {8, 10}, {"420", "444"}, true, ".mp4"},
        {"av1_nvenc", "AV1 NVENC", {8, 10}, {"420"}, true, ".mp4"},
        {"prores_ks", "ProRes", {10}, {"422", "444"}, false, ".mov"},
        {"ffv1", "FFV1", {8, 10, 12, 16}, {"420", "422", "444"}, false, ".mkv"},
    };
    return codecs;
}

const CodecCaps* FindCodecCaps(const std::string& codec) {
    for (const auto& c : KnownCodecs())
        if (c.name == codec) return &c;
    return nullptr;
}

bool HasBitrateControl(const std::string& codec) {
    const CodecCaps* c = FindCodecCaps(codec);
    return c ? c->bitrate : true;
}

namespace {

std::string Join(const std::vector<int>& v) {
    std::string s;
    for (const int x : v) s += (s.empty() ? "" : " | ") + std::to_string(x);
    return s;
}

std::string ChromaText(std::string_view c) { return std::string(1, c[0]) + ":" + std::string(1, c[1]) + ":" + std::string(1, c[2]); }

std::string Join(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : " | ") + ChromaText(x);
    return s;
}

}  // namespace

OutputFormat ResolveOutputFormat(const std::string& codec, const OutputFormatRequest& request, const FrameDesc& source) {
    if (request.bitDepth != 0 && request.bitDepth != 8 && request.bitDepth != 10 && request.bitDepth != 12)
        Throw("output bit depth must be auto | 8 | 10 | 12 (got " + std::to_string(request.bitDepth) + ")");
    if (!ValidChromaChoice(request.chroma)) Throw("output chroma must be auto | 420 | 422 | 444 (got '" + request.chroma + "')");
    const CodecCaps* caps = FindCodecCaps(codec);
    const int sourceDepth = source.HighBitDepth() ? source.bitDepth : 8;
    const std::string sourceChroma(ChromaName(source.format));
    OutputFormat out;
    out.autoBitDepth = request.bitDepth == 0;
    out.autoChroma = request.chroma == "auto";

    int depth = request.bitDepth;
    std::string chroma = request.chroma;
    std::string why;
    if (!caps) {  // an encoder outside the table: 8-bit 4:2:0 unless asked otherwise, the encoder itself validates
        if (depth == 0) depth = 8;
        if (chroma == "auto") chroma = "420";
        why = "unknown encoder " + codec + ": 8-bit 4:2:0 unless told otherwise";
    } else {
        if (depth == 0) {
            // the largest the codec does that is not above the source's; ProRes (10-bit only) takes an 8-bit source up
            depth = caps->bitDepths.front();
            for (const int d : caps->bitDepths)
                if (d <= sourceDepth) depth = d;
            why = "bit depth " + std::to_string(depth) + " for a " + std::to_string(sourceDepth) + "-bit source";
        } else if (std::find(caps->bitDepths.begin(), caps->bitDepths.end(), depth) == caps->bitDepths.end()) {
            std::string hint;
            if (codec == "h264_nvenc") hint = " — H.264 NVENC encodes 8-bit only; HEVC NVENC, AV1 NVENC, ProRes and FFV1 take 10-bit";
            else if (codec == "prores_ks") hint = " — ProRes is 10-bit only";
            else if (depth == 12) hint = " — NVENC encodes up to 10 bits; FFV1 keeps 12";
            Throw(caps->title + " does not encode " + std::to_string(depth) + "-bit (it takes " + Join(caps->bitDepths) + ")" + hint);
        }
        if (chroma == "auto") {
            if (codec == "prores_ks") chroma = sourceChroma == "444" ? "444" : "422";
            else if (codec == "ffv1") chroma = std::find(caps->chromas.begin(), caps->chromas.end(), sourceChroma) != caps->chromas.end() ? sourceChroma : "420";
            else chroma = "420";
            why += (why.empty() ? "" : "; ") + std::string("chroma ") + ChromaText(chroma) + (codec == "prores_ks" || codec == "ffv1" ? " from the source" : " (NVENC default)");
        } else if (std::find(caps->chromas.begin(), caps->chromas.end(), chroma) == caps->chromas.end()) {
            std::string hint;
            if (chroma == "422" && caps->name.find("nvenc") != std::string::npos) hint = " — NVENC on the RTX 20..40 generations has no 4:2:2; ProRes 422 HQ or FFV1 keep it";
            else if (codec == "av1_nvenc") hint = " — AV1 NVENC is 4:2:0 only; HEVC NVENC encodes 4:4:4";
            else if (codec == "prores_ks") hint = " — ProRes is 4:2:2 (422 HQ) or 4:4:4 (4444)";
            Throw(caps->title + " does not encode " + ChromaText(chroma) + " (it takes " + Join(caps->chromas) + ")" + hint);
        }
    }
    out.desc = FrameDesc{0, 0, PixelFormatFor(chroma, depth), static_cast<uint8_t>(depth)};
    out.reason = why;
    return out;
}

std::string ResultExtensionFor(const std::string& codec) {
    const CodecCaps* c = FindCodecCaps(codec);
    return c ? c->extension : ".mkv";
}

std::filesystem::path ResultPathFor(const std::filesystem::path& path, const std::string& codec) {
    if (path.empty()) return path;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext != ".mp4" && ext != ".mov" && ext != ".mkv") return path;
    const std::string wanted = ResultExtensionFor(codec);
    if (ext == wanted) return path;
    std::filesystem::path p = path;
    p.replace_extension(wanted);
    return p;
}

}  // namespace dlssvid
