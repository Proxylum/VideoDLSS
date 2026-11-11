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
    std::string sourceFormat;  // FrameDesc::Name() of the decoded source the pass was made from ("yuv444p10"), "+<turn>" when it is shown
                               // turned ("yuv420p+r90", TASK-0072: SourceFormatLabel); empty in passes from before the decoder kept the
                               // source's bit depth (they come from 8-bit 4:2:0)
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
    // A static pass (PR-8: mask_control): one frame 0 that stands for every frame of the clip - stage_params.static.
    bool IsStatic() const {
        if (!stageParams.is_object() || frameCount != 1 || firstFrame != 0) return false;
        const auto it = stageParams.find("static");  // a value of another type (a hand-made manifest) is no static pass, not an exception
        return it != stageParams.end() && it->is_boolean() && it->get<bool>();
    }
    // Motion vectors stored with y pointing up (mv.y_up on an mv_raw / mv_dlss pass): PassReader negates v on read
    // (docs/conventions.md §3).
    bool IsYUpMv() const {
        const auto k = Kind();
        return mv.yUp && k && (*k == PassKind::MvRaw || *k == PassKind::MvDlss);
    }
};

// A text a stage recorded in stage_params ("" when absent or of another type).
std::string StageParamText(const Manifest& m, const char* key);
// TASK-0073: the pixels of a colour pass of a source whose pixels are `sourceSar` (the axes of the frame the stages get, sourceW x sourceH):
// what the pass recorded (stage_params.pixel_aspect, PixelAspect::Stamp); a pass without the record (of the build before; depth, vectors,
// masks) - the source's when it has the grid's proportions, else {1, 1}: the build before moved a pass off them only by an explicit target
// of both sides, taken in square pixels, and a mask drawn over the frame as shown is square. A square source: {}.
PixelAspect PassPixelAspect(const Manifest& m, PixelAspect sourceSar, uint32_t sourceW, uint32_t sourceH);
// The pixels of the colour a stage reads: the pass in `colorDir` (unreadable: the source's), the source's frames when empty.
PixelAspect ColourPixelAspect(const std::filesystem::path& colorDir, PixelAspect sourceSar, uint32_t sourceW, uint32_t sourceH);
// TASK-0073: the frame as shown a colour pass of an anamorphic source records (stage_params.shown_size [w, h]: the upscale's target, which
// its grid stretched may miss by a rounding); nullopt when not recorded or malformed.
std::optional<std::pair<uint32_t, uint32_t>> RecordedShownSize(const Manifest& m);
// That record of the colour pass in `colorDir` when the pass is of size w x h - NR and FG pass it on (their frame is their colour's).
std::optional<std::pair<uint32_t, uint32_t>> ColourShownSize(const std::filesystem::path& colorDir, uint32_t w, uint32_t h);
// Whether a recorded frame as shown is this one's - the grid stretched - to the roundings of a grid (each side within 4 px and 1 %; a capped
// grid is rounded down by up to 2 source pixels): a record of another frame is not taken.
bool ShownSizeMatches(std::pair<uint32_t, uint32_t> recorded, std::pair<uint32_t, uint32_t> stretched);
// The frame as shown of a pass of pixels `px` (the viewport's labels and reference): its record when it matches, else the pass stretched
// (PixelAspect::DisplaySize); square pixels - the pass as it is.
std::pair<uint32_t, uint32_t> PassShownSize(const Manifest& m, PixelAspect px);

}  // namespace dlssvid
