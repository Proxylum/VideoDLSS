#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "passes/ControlMaskPass.h"

namespace dlssvid {

struct Manifest;

// The control mask of NR (PR-8): import, reading, removal and the texels NR uploads. Format in docs/conventions.md §6: the folder
// <passes root>/mask_control holds mask_control_000000.png (PNG16: one channel Y for a grey mask, else R, G, B) and a manifest
// written last - static, frame 0, stamped with ContentFingerprint (the input id of NR).

struct ControlMaskIssue {
    enum class Kind : uint8_t { NoRoot, NotAFile, Unsupported, TooLarge, Empty, Unreadable, Channels, Tiff, Busy, Locked, Write };
    Kind kind = Kind::Unreadable;
    std::string detail;               // English: what() of the error, the CLI's text
    uint32_t width = 0, height = 0;   // TooLarge: the image's size
    bool previousKept = true;         // Locked / Write: the previous mask (if any) is still in place; false - there is no mask now
};
// what() is the English detail (CLI); the GUI writes its Russian text by issue.kind.
class ControlMaskError : public std::runtime_error {
public:
    explicit ControlMaskError(ControlMaskIssue i);
    ControlMaskIssue issue;
};

struct ControlMaskInfo {
    std::string fingerprint, sourceName, created;
    std::string pixelsSha256;  // stage_params.pixels_sha256: the same pixels whatever file they came from
    uint32_t width = 0, height = 0;
    bool grayscale = false, bandsDiffer = false, alphaIgnored = false;
    int64_t clamped = 0;  // float sources: samples clamped into [0, 1] (NaN included)
};

struct ControlMaskImport {
    enum class Outcome : uint8_t { Imported, Replaced, Unchanged } outcome = Outcome::Imported;
    ControlMaskInfo mask;
    std::string retiredVersion;  // Replaced: the id of the previous mask in mask_control.v ("" when a broken folder was deleted)
    // the source as read, for the CLI's report
    std::string sourceFormat, sourceType;  // png | tiff | exr; u8 | u16 | f16 | f32
    uint32_t sourceChannels = 0;
    bool exrPadded = false;
};

// One image (PNG, TIFF or EXR, by its signature) -> <passesRoot>/mask_control, under the import lock of the root (<root>/.import/lock):
// the previous mask goes to the history, the same pixels again are Unchanged and write nothing. Throws ControlMaskError.
ControlMaskImport ImportControlMask(const std::filesystem::path& image, const std::filesystem::path& passesRoot);
// «Убрать»: the mask goes to the history (its id), under the import lock. nullopt: there was no mask, or a folder without a loadable
// manifest was deleted (it could not be brought back from the history anyway). Throws ControlMaskError (Busy, Locked, Write).
std::optional<std::string> RemoveControlMask(const std::filesystem::path& passesRoot);
// Whether the folder without a loadable manifest that RemoveControlMask and a replacing import delete is there (the dialog says so).
bool ControlMaskFolderBroken(const std::filesystem::path& passesRoot);

// The folder as a control mask; «is there a mask» is ReadControlMaskInfo(dir).has_value(). why: "absent" (no manifest) | "not a
// mask_control pass" | "not static one-frame" | "channels ..." | "type ..." | "not stamped" | "changed outside dlssvid" (the frame file
// does not match the fingerprint: edited in place) | "unreadable: ...". The content check hashes the frame through Sha256FileCached.
std::optional<ControlMaskInfo> ReadControlMaskInfo(const std::filesystem::path& dir, std::string* why = nullptr);

// For NR: the mask at the frame size w x h, each axis on its own - area-averaged where it shrinks, bilinear (pixel centres as the GPU's)
// where it grows. Grey: one channel (R16_UNORM); colour: RGBA, A = 65535. split: G differs from B somewhere after the resize. After
// decoding, the folder's ContentFingerprint is checked against expectFingerprint (the one Init read): a mask replaced between the two
// throws "the control mask changed while nr started: run again".
struct ControlMaskTexels {
    uint32_t width = 0, height = 0;
    bool rgb = false, split = false;
    std::vector<uint16_t> data;  // width * height (grey) or width * height * 4 (RGBA)
};
ControlMaskTexels LoadControlMaskForFrame(const std::filesystem::path& dir, uint32_t w, uint32_t h, const std::string& expectFingerprint);

// Test seam of the import (U11): called with the new folder after the current mask went to the history and before the new one is
// put in place. Empty in the product.
struct ControlMaskTestHooks {
    std::function<void(const std::filesystem::path& newFolder)> beforePlace;
};
ControlMaskTestHooks& ControlMaskHooksForTests();

// stage_params.control_mask.fingerprint of a color_nr pass: "" when it was made without a control mask.
std::string ControlMaskLabelOf(const Manifest& nrPass);

}  // namespace dlssvid
