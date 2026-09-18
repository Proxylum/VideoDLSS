#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "passes/Manifest.h"
#include "passes/PassImage.h"

namespace dlssvid {

// Writes a pass folder: one file per frame + manifest.json (ТЗ §5). Frames may arrive in
// any order; the manifest records count and range on Finish().
class PassWriter {
public:
    PassWriter(const std::filesystem::path& dir, Manifest manifest);
    ~PassWriter();

    void WriteFrame(int64_t frame, const PassImage& img);
    void Finish();
    const Manifest& Man() const { return manifest_; }
    const std::filesystem::path& Dir() const { return dir_; }
    int64_t FramesWritten() const { return count_; }

private:
    std::filesystem::path dir_;
    Manifest manifest_;
    int64_t count_ = 0;
    bool finished_ = false;
};

// Reads a pass folder (with a manifest, or without one when the caller names the pass).
class PassReader {
public:
    static PassReader Open(const std::filesystem::path& dir);

    struct ScanHints {
        PassKind kind = PassKind::DepthRaw;
        Convention convention = Convention::Raw;
        uint32_t width = 0;   // required for raw dumps
        uint32_t height = 0;
        std::optional<PixelType> pixelType;  // raw dumps: default = canonical for the kind
    };
    static PassReader OpenWithoutManifest(const std::filesystem::path& dir, const ScanHints& hints);

    const Manifest& Man() const { return manifest_; }
    Manifest& Man() { return manifest_; }
    const std::filesystem::path& Dir() const { return dir_; }
    bool HadManifest() const { return hadManifest_; }

    std::filesystem::path FramePath(int64_t frame) const { return dir_ / manifest_.FrameFileName(frame); }
    bool HasFrame(int64_t frame) const { return std::filesystem::exists(FramePath(frame)); }
    PassImage ReadFrame(int64_t frame) const;
    std::vector<int64_t> MissingFrames() const;  // inside [firstFrame, lastFrame]

    struct Expect {
        uint32_t width = 0;      // 0 = don't check
        uint32_t height = 0;
        int64_t frameCount = -1; // <0 = don't check
        std::optional<PassKind> kind;
    };
    // Throws dlssvid::Error listing exactly what does not match (ТЗ §5 «Импорт»).
    void Validate(const Expect& expect) const;

private:
    std::filesystem::path dir_;
    Manifest manifest_;
    bool hadManifest_ = false;
};

// ---- export ---------------------------------------------------------------------------

struct FrameRange {
    int64_t first = 0;
    int64_t last = -1;  // < 0: until the source runs out
    static std::optional<FrameRange> Parse(std::string_view s);  // "a-b" | "a" | "a-"
};

// Returns false when there is no frame `frame`.
using FrameSource = std::function<bool(int64_t frame, PassImage& out)>;
using ProgressFn = std::function<void(int64_t done, int64_t total)>;

// Exports frames [range] from `source` into `dir` as described by `manifest` (pass, format,
// pixel type). Images are converted to the manifest's pixel type. Returns frames written.
int64_t ExportPass(const std::filesystem::path& dir, Manifest manifest, const FrameSource& source, FrameRange range,
                   const ProgressFn& progress = {});

// FrameSource over an existing pass folder.
FrameSource SourceFromReader(const PassReader& reader);

// Preset "nuke": one multi-layer EXR per frame: R,G,B (+ depth.Z) (+ mv.u, mv.v).
int64_t ExportLayeredExr(const std::filesystem::path& dir, const std::string& name, Manifest base, const FrameSource& color,
                         const FrameSource* depth, const FrameSource* mv, FrameRange range, const ProgressFn& progress = {});

enum class ExportPreset { None, Nuke, ComfyUi, RawDlss };
std::optional<ExportPreset> ParseExportPreset(std::string_view s);
// Format and pixel type a preset uses for a pass kind.
FileFormat PresetFormat(ExportPreset preset, PassKind kind);
PixelType PresetPixelType(ExportPreset preset, PassKind kind);

}  // namespace dlssvid
