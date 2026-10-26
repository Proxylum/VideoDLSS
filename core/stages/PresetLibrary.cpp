#include "stages/PresetLibrary.h"

#include <algorithm>
#include <cstring>

#include "util/Error.h"

namespace dlssvid {

namespace {

const char* const kBeforeApply = "Перед применением · ";
const char* const kBeforeReset = "Перед сбросом · ";

std::filesystem::path U8Path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

size_t CodePoints(const std::string& s) {
    size_t n = 0;
    for (const unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

// the first `n` code points of a UTF-8 string
std::string FirstCodePoints(const std::string& s, size_t n) {
    size_t i = 0, seen = 0;
    for (; i < s.size(); ++i) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            if (seen == n) break;
            ++seen;
        }
    }
    return s.substr(0, i);
}

void TrimDotsAndSpaces(std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '.')) s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '.')) s.pop_back();
}

bool DeviceName(const std::string& stem) {
    std::string base = stem.substr(0, stem.find('.'));
    while (!base.empty() && base.back() == ' ') base.pop_back();
    std::transform(base.begin(), base.end(), base.begin(), [](unsigned char c) { return static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c); });
    if (base == "CON" || base == "PRN" || base == "AUX" || base == "NUL") return true;
    return base.size() == 4 && (base.compare(0, 3, "COM") == 0 || base.compare(0, 3, "LPT") == 0) && base[3] >= '1' && base[3] <= '9';
}

bool StartsWith(const std::string& s, const char* prefix) { return s.compare(0, std::strlen(prefix), prefix) == 0; }

}  // namespace

std::string PresetFileStem(const std::string& name) {
    std::string s;
    for (const char ch : name) {
        const unsigned char c = static_cast<unsigned char>(ch);
        s += c < 0x20 || c == 0x7F || std::strchr("<>:\"/\\|?*", ch) ? '_' : ch;
    }
    TrimDotsAndSpaces(s);
    if (s.size() > 80) {
        size_t n = 80;
        while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;  // at a code point
        s.resize(n);
        TrimDotsAndSpaces(s);
    }
    if (s.empty()) return "preset";
    if (DeviceName(s)) s = "_" + s;
    return s;
}

bool ReservedPresetName(const std::string& raw) {
    const std::string name = TrimPresetName(raw);
    for (const char* prefix : {kBeforeApply, kBeforeReset})
        if (StartsWith(name, prefix) && name.size() > std::strlen(prefix)) return true;
    return false;
}

std::string PresetNameFor(const std::string& prefix, const std::filesystem::path& video) {
    const std::string stem = PathText(video.stem());
    if (stem.empty()) return prefix;
    const std::string head = prefix + " · ";
    const std::string whole = head + stem;
    if (CodePoints(whole) <= kPresetNameMax) return whole;
    const size_t room = kPresetNameMax - CodePoints(head) - 1;  // and «…»
    return head + FirstCodePoints(stem, room) + "…";
}

std::string PresetSlotFileName(PresetSlot slot, const std::string& target) {
    return std::string(slot == PresetSlot::BeforeApply ? "before-apply-" : "before-reset-") + target + ".json";
}

PresetLibrary::PresetLibrary(std::filesystem::path dir) : dir_(std::move(dir)) {}

std::vector<std::filesystem::path> PresetLibrary::TopFiles() const {
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2)) continue;
        std::string ext = PathText(it->path().extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c); });
        if (ext == ".json") files.push_back(it->path());
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.filename().u8string() < b.filename().u8string(); });
    return files;
}

PresetListing PresetLibrary::List(PresetScope scope, const std::string& stage) const {
    PresetListing out;
    const auto files = TopFiles();
    for (size_t i = 0; i < files.size(); ++i) {
        if (i >= kPresetListMax) {
            ++out.hidden;
            continue;
        }
        PresetFile f{files[i], LoadPreset(files[i])};
        const PresetLoad& L = f.load;
        const bool ofStage = scope == PresetScope::Pipeline || stage.empty() || L.preset.stages.empty() || L.preset.stages.count(stage) > 0;
        if (L.ok()) {
            if (L.preset.scope == scope && (scope == PresetScope::Pipeline || stage.empty() || L.preset.stage() == stage)) out.presets.push_back(std::move(f));
        } else if (!L.scopeKnown || (L.preset.scope == scope && ofStage)) {
            out.refused.push_back(std::move(f));
        }
    }
    return out;
}

std::vector<PresetName> PresetLibrary::Names() const {
    std::vector<PresetName> out;
    for (const auto& file : TopFiles()) {
        const PresetLoad L = LoadPreset(file);
        if (L.scopeKnown && !L.preset.name.empty())
            out.push_back({file, L.preset.name, L.preset.scope, L.preset.scope == PresetScope::Stage ? L.preset.stage() : std::string()});
    }
    return out;
}

std::optional<std::filesystem::path> PresetLibrary::FindByName(const std::string& name, PresetScope scope, const std::string& stage) const {
    const std::string wanted = TrimPresetName(name);
    for (const PresetName& n : Names())
        if (n.scope == scope && n.name == wanted && (stage.empty() || n.stage.empty() || n.stage == stage)) return n.path;
    return std::nullopt;
}

std::filesystem::path PresetLibrary::SlotPath(PresetSlot slot, const std::string& target) const {
    return dir_ / "auto" / U8Path(PresetSlotFileName(slot, target));
}

std::optional<PresetFile> PresetLibrary::Slot(PresetSlot slot, const std::string& target) const {
    if (slot == PresetSlot::BeforeReset && target == kPipelineTarget) return std::nullopt;  // a pipeline preset has no reset
    const auto path = SlotPath(slot, target);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return std::nullopt;
    return PresetFile{path, LoadPreset(path, {}, PresetRead::Slot)};
}

bool PresetLibrary::IsSlotFile(const std::filesystem::path& file) const {
    const std::string name = PathText(file.filename());
    if (!(StartsWith(name, "before-apply-") || StartsWith(name, "before-reset-")) || name.size() < 5 || name.compare(name.size() - 5, 5, ".json") != 0)
        return false;
    std::error_code ec;
    const bool same = std::filesystem::equivalent(file.parent_path(), dir_ / "auto", ec);
    if (!ec) return same;
    return file.parent_path().lexically_normal() == (dir_ / "auto").lexically_normal();
}

std::filesystem::path PresetLibrary::Save(const Preset& p, const std::filesystem::path& replacing) const {
    if (!replacing.empty()) {
        SavePreset(replacing, p);
        return replacing;
    }
    if (ReservedPresetName(p.name)) Throw("the name '" + p.name + "' is an automatic slot's - give another one");
    if (const auto taken = FindByName(p.name, p.scope, p.scope == PresetScope::Stage ? p.stage() : std::string())) Throw("a " + std::string(PresetScopeName(p.scope)) + " preset '" + p.name + "' exists: " + PathText(*taken));
    const std::string stem = PresetFileStem(p.name);
    for (int n = 1; n < 10000; ++n) {
        const auto path = dir_ / U8Path(stem + (n > 1 ? " (" + std::to_string(n) + ")" : std::string()) + ".json");
        std::error_code ec;
        if (std::filesystem::exists(path, ec) || ec) continue;  // any file: another scope's, a broken one, a newer version's
        SavePreset(path, p);
        return path;
    }
    Throw("no free file name for the preset '" + p.name + "' in " + PathText(dir_));
}

std::filesystem::path PresetLibrary::SaveSlot(PresetSlot slot, const std::string& target, const Preset& p) const {
    const auto path = SlotPath(slot, target);
    SavePreset(path, p, PresetRead::Slot);
    return path;
}

void PresetLibrary::Remove(const std::filesystem::path& file) const {
    std::error_code ec;
    if (!std::filesystem::remove(file, ec) || ec) Throw(PathText(file) + ": cannot delete the preset" + (ec ? " (system error " + std::to_string(ec.value()) + ")" : std::string()));
}

}  // namespace dlssvid
