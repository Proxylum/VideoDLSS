#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "io/VideoDecoder.h"  // Rational
#include "passes/PassImage.h"

namespace dlssvid {

// File formats a pass sequence can be stored in (one file per frame).
enum class FileFormat : uint8_t { Exr, Png, Tiff, Npz, Raw };
std::string_view ToString(FileFormat f);
std::optional<FileFormat> ParseFileFormat(std::string_view s);
std::optional<FileFormat> FormatFromExtension(const std::filesystem::path& p);
// Extension for a format given the image layout (raw: .r32 / .rg16f / .r16f ...).
std::string ExtensionFor(FileFormat f, PixelType type, size_t channels);

struct DepthParams {
    std::string units = "meters";  // "meters" (metric) | "relative"
    bool relative = false;
    float zNear = 0.1f;   // metres, for raw <-> reverse-Z
    float zFar = 1000.f;  // metres
    float minValue = 0.f;  // relative: raw value range mapped to [near, far]
    float maxValue = 0.f;
};

struct MvParams {
    std::string direction = "forward";  // raw: "forward" (t -> t+1); dlss: "backward" (t -> t-1)
    bool yUp = false;                   // false: y grows downwards (image rows), as DLSS expects; true: v points up in the
                                        // files, PassReader::ReadFrame returns it negated (y down). dlssvid never writes true
    uint32_t refWidth = 0;              // resolution the vectors are expressed in (0 = image size)
    uint32_t refHeight = 0;
};

// manifest.json in the root of a pass folder (ТЗ §5 «Манифест»).
struct Manifest {
    static constexpr int kSchemaVersion = 1;
    static constexpr const char* kFileName = "manifest.json";

    int schemaVersion = kSchemaVersion;
    std::string pass;  // canonical pass name (PassKind) or custom
    uint32_t width = 0;
    uint32_t height = 0;
    Rational fps{0, 1};
    int64_t frameCount = 0;
    int64_t firstFrame = 0;
    int64_t lastFrame = -1;
    FileFormat format = FileFormat::Exr;
    std::string filePattern;  // printf-style, e.g. "depth_raw_%06d.exr"
    PixelType pixelType = PixelType::F32;
    std::vector<std::string> channels;
    std::string colorspace;  // "srgb" | "linear" | "bt709" | "" for non-colour passes
    Convention convention = Convention::Raw;
    std::string model;
    std::string modelVersion;
    std::string sourceHash;  // "sha256:<hex>" of the source video, may be empty
    std::string sourceFile;
    std::string sourceFormat;  // FrameDesc::Name() of the decoded source the pass was made from ("yuv444p10"); empty in
                               // passes from before the decoder kept the source's bit depth (they come from 8-bit 4:2:0)
    nlohmann::json stageParams = nlohmann::json::object();
    DepthParams depth;
    MvParams mv;
    // Stage 9 (MR A): written by ProcessRunner after a run — empty in passes made by a standalone `dlssvid <stage>`.
    std::string fingerprint;                    // "sha256:<hex>" of the run that produced the pass (PassFingerprint.h)
    std::map<std::string, std::string> inputs;  // input pass -> its fingerprint at the time of the run
    nlohmann::json tool = nlohmann::json::object();             // ToolInfo: app version, backend, model, DLL hash
    nlohmann::json paramsCanonical = nlohmann::json::object();  // canonical stage parameters the fingerprint hashed
    std::string created;                        // ISO-8601 UTC, when the pass was finished (or adopted)

    static Manifest ForPass(PassKind kind, uint32_t w, uint32_t h, FileFormat format);

    nlohmann::json ToJson() const;
    static Manifest FromJson(const nlohmann::json& j);
    void Save(const std::filesystem::path& dir) const;
    static Manifest Load(const std::filesystem::path& dir);
    static bool Exists(const std::filesystem::path& dir);

    std::string FrameFileName(int64_t frame) const;
    std::optional<PassKind> Kind() const { return ParsePassKind(pass); }
    // A data pass (PR-13: scenes): the manifest is the content, there are no frame files - no viewport source, never read frame by frame.
    bool IsDataPass() const { return channels.empty(); }
    // Motion vectors stored with y pointing up (mv.y_up on an mv_raw / mv_dlss pass): PassReader negates v on read
    // (docs/conventions.md §3).
    bool IsYUpMv() const {
        const auto k = Kind();
        return mv.yUp && k && (*k == PassKind::MvRaw || *k == PassKind::MvDlss);
    }
};

}  // namespace dlssvid
