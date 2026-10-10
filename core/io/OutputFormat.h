#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/Frame.h"

namespace dlssvid {

// What the user asked for the result's pixel layout. Bit depth 0 = automatic: the source's, within what the codec can
// do (a 10-bit source gives a 10-bit HEVC / AV1 / ProRes / FFV1 result and an 8-bit H.264 one). Chroma "auto": 4:2:0
// for the NVENC codecs, 4:2:2 for ProRes unless the source is 4:4:4, the source's own subsampling for FFV1.
struct OutputFormatRequest {
    int bitDepth = 0;             // 0 auto | 8 | 10 | 12
    std::string chroma = "auto";  // auto | 420 | 422 | 444
    bool operator==(const OutputFormatRequest&) const = default;
};
std::optional<int> ParseBitDepthChoice(std::string_view s);  // "auto" -> 0, "8" | "10" | "12" -> the number; nullopt otherwise
std::string BitDepthChoiceName(int bitDepth);                // 0 -> "auto", else the number
bool ValidChromaChoice(std::string_view s);                  // auto | 420 | 422 | 444

// The encoders the result can be written with and what they take (the NVENC limits are those of the RTX 20..40
// generations: 4:2:0 / 4:4:4, 10 bits for HEVC and AV1, 8 bits for H.264).
struct CodecCaps {
    std::string name;                  // FFmpeg encoder name
    std::string title;                 // "HEVC NVENC", "ProRes" ...
    std::vector<int> bitDepths;        // ascending
    std::vector<std::string> chromas;  // "420" | "422" | "444"
    bool bitrate = false;              // takes the `b` option (NVENC); ProRes and FFV1 do not
    std::string extension;             // the result's container: ".mp4" | ".mov" | ".mkv"
};
const std::vector<CodecCaps>& KnownCodecs();
const CodecCaps* FindCodecCaps(const std::string& codec);  // nullptr for other encoders
bool HasBitrateControl(const std::string& codec);          // the `b` option means something (NVENC and unknown encoders)

// The result's frame layout for a codec, a request and the source's decoded layout (VideoStreamInfo::decoded; width
// and height are ignored). Throws with a readable message when the request is outside what the codec does (10-bit
// H.264, 4:2:2 NVENC, 8-bit ProRes ...).
struct OutputFormat {
    FrameDesc desc;      // format + bitDepth, width / height 0
    bool autoBitDepth = false, autoChroma = false;
    std::string reason;  // how the automatic choices were made, for logs: "bit depth: the source's 10 bits; chroma: 4:2:0 (NVENC)"
};
OutputFormat ResolveOutputFormat(const std::string& codec, const OutputFormatRequest& request, const FrameDesc& source);

// ".mp4" for the NVENC codecs, ".mov" for ProRes, ".mkv" for FFV1 and unknown encoders.
std::string ResultExtensionFor(const std::string& codec);
// `path` with the container the codec needs when its extension is one of the result containers (.mp4 / .mov / .mkv);
// other extensions (and an empty path) are returned unchanged.
std::filesystem::path ResultPathFor(const std::filesystem::path& path, const std::string& codec);

}  // namespace dlssvid
