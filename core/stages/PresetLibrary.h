#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "stages/StagePreset.h"

namespace dlssvid {

// The folder of presets (PR-9): the person's files on the top level, both scopes side by side (the scope is read from the file), the
// automatic «Перед …» slots under auto/. Without Qt: the app gives the folder (AppDataLocation/presets) and Russian texts.
enum class PresetSlot : uint8_t { BeforeApply, BeforeReset };
inline constexpr size_t kPresetListMax = 500;  // files a listing parses; the rest are counted (a stranger's folder of JSON files)

struct PresetFile {
    std::filesystem::path path;
    PresetLoad load;  // always: a refused file carries its issues (Unreadable, NotJson, NewerSchema, BadValue ...)
};
struct PresetListing {
    std::vector<PresetFile> presets;  // ok() and of this scope (Stage: of this stage), in file name order
    std::vector<PresetFile> refused;  // !ok(): of this scope, or with no scope known (those are in both menus)
    size_t hidden = 0;                // files past the first kPresetListMax
};
struct PresetName {
    std::filesystem::path path;
    std::string name;
    PresetScope scope = PresetScope::Stage;
};

// A file name stem for a preset name, UTF-8 in and out: <>:"/\|?* and control characters become _, dots and spaces at the ends go,
// the device names of Windows (CON, NUL.txt ...) get a leading _, at most 80 bytes cut at a code point; "preset" when nothing is left.
std::string PresetFileStem(const std::string& name);
// Exactly a slot's name: «Перед применением · <…>» or «Перед сбросом · <…>» (« · » and something after it); «Перед сбросом в ч/б» is free.
bool ReservedPresetName(const std::string& name);
// «<prefix> · <the video's stem>» (the stem from u8string, not string(): the ANSI code page), the stem cut at a code point with «…» so
// that the whole name keeps to kPresetNameMax; the prefix alone without a video.
std::string PresetNameFor(const std::string& prefix, const std::filesystem::path& video);
// The slot's file name: before-apply-<target>.json | before-reset-<target>.json (target: a stage or kPipelineTarget).
std::string PresetSlotFileName(PresetSlot slot, const std::string& target);

class PresetLibrary {
public:
    explicit PresetLibrary(std::filesystem::path dir);
    const std::filesystem::path& dir() const { return dir_; }

    // the top level's *.json, the first kPresetListMax parsed (in file name order), the rest counted in `hidden`
    PresetListing List(PresetScope scope, const std::string& stage = {}) const;
    // the name of every top-level file whose envelope gives one, without the limit (a refused file too, when its name was read)
    std::vector<PresetName> Names() const;
    // exact, in the scope; over every file (Names)
    std::optional<std::filesystem::path> FindByName(const std::string& name, PresetScope scope) const;
    // target: a stage or kPipelineTarget; BeforeReset exists for stages only. Read with PresetRead::Slot.
    std::optional<PresetFile> Slot(PresetSlot slot, const std::string& target) const;
    std::filesystem::path SlotPath(PresetSlot slot, const std::string& target) const;
    bool IsSlotFile(const std::filesystem::path& file) const;  // one of auto/'s slot files: read it with PresetRead::Slot
    // `replacing` empty: a new file next to the others - never over an existing one, whatever it holds (a taken stem gets « (2)» ...);
    // throws when the name is a slot's or another preset of the scope has it (FindByName). `replacing` given: that file is overwritten.
    std::filesystem::path Save(const Preset& p, const std::filesystem::path& replacing = {}) const;
    // always overwrites the slot's file atomically, also a slot of a newer version; written in PresetRead::Slot
    std::filesystem::path SaveSlot(PresetSlot slot, const std::string& target, const Preset& p) const;
    void Remove(const std::filesystem::path& file) const;  // permanent

private:
    std::vector<std::filesystem::path> TopFiles() const;  // *.json of the top level, in file name order
    std::filesystem::path dir_;
};

}  // namespace dlssvid
