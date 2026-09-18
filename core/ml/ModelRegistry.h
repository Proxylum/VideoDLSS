#pragma once

#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace dlssvid {

// models/registry.json: what a model is, where to download it, how to verify it.
struct ModelEntry {
    std::string id;         // "da3metric-large"
    std::string stage;      // "depth" | "flow"
    std::string role;       // "primary" | "fallback" | ...
    std::string source;     // human URL
    std::string format;     // "onnx" | "pytorch" | "safetensors"
    std::string url;        // direct download (may be empty: produced locally by models/export)
    std::string sha256;     // expected hash of the file (empty = not verified)
    std::string license;
    std::string fileName;   // name in the cache dir; default = last URL path segment or id + ext
    nlohmann::json params;  // model-specific: input_size, window, metric, output ("depth"|"disparity"), mean/std ...
};

class ModelRegistry {
public:
    // Loads registry.json; `cacheDir` defaults to <registry dir>/cache (git-ignored).
    static ModelRegistry Load(const std::filesystem::path& registryJson, const std::filesystem::path& cacheDir = {});
    // Locates models/registry.json: DLSSVID_MODELS_DIR, next to the executable (../../models), project root.
    static std::filesystem::path DefaultRegistryPath();

    const std::vector<ModelEntry>& Entries() const { return entries_; }
    const ModelEntry* Find(const std::string& id) const;
    const ModelEntry& Get(const std::string& id) const;
    const std::filesystem::path& CacheDir() const { return cacheDir_; }

    std::filesystem::path LocalPath(const ModelEntry& e) const;
    bool IsCached(const ModelEntry& e) const;

    using ProgressFn = std::function<void(uint64_t done, uint64_t total)>;
    // Downloads (if needed) and verifies the hash. Returns the local path.
    std::filesystem::path Fetch(const ModelEntry& e, const ProgressFn& progress = {}) const;

private:
    std::vector<ModelEntry> entries_;
    std::filesystem::path cacheDir_;
};

// HTTP(S) GET to a file with redirects (WinHTTP). Throws dlssvid::Error.
void DownloadFile(const std::string& url, const std::filesystem::path& dest, const ModelRegistry::ProgressFn& progress = {});

}  // namespace dlssvid
