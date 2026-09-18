#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "viewport/FrameStore.h"
#include "viewport/ViewportState.h"

namespace dlssvid {

// Stage entry in the project panel: enabled flag + parameters passed to `dlssvid <stage>` (ТЗ §6 «стадии с включателями и параметрами»).
struct StageEntry {
    std::string name;  // "depth", "flow", ... (a dlssvid subcommand)
    bool enabled = true;
    nlohmann::json params = nlohmann::json::object();  // e.g. {"backend": "da3", "frames": 30}
    bool operator==(const StageEntry&) const = default;
};

// Project file (*.dlssvid.json): source video, passes root, optional result video, stage
// configs and the viewport state (ТЗ §6: «состояние вьюпорта сохраняется в проект»).
struct Project {
    std::filesystem::path file;         // where it was loaded from / saved to (may be empty)
    std::filesystem::path sourceVideo;
    std::filesystem::path passesRoot;
    std::filesystem::path resultVideo;
    std::vector<StageEntry> stages;
    ViewportState viewport;

    static Project Create(const std::filesystem::path& sourceVideo, const std::filesystem::path& passesRoot);
    static Project Load(const std::filesystem::path& file);
    void Save(const std::filesystem::path& file);
    void Save() { Save(file); }
    nlohmann::json ToJson(const std::filesystem::path& relativeTo) const;
    static Project FromJson(const nlohmann::json& j, const std::filesystem::path& relativeTo);

    // Sources for the FrameStore: "source", "result" (when set) and every pass folder found under passesRoot.
    std::vector<ViewportSource> Sources() const;
    static std::vector<StageEntry> DefaultStages();
};

}  // namespace dlssvid
