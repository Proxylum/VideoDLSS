#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "convert/ColorConvert.h"
#include "gpu/D3D12Device.h"
#include "io/VideoDecoder.h"
#include "passes/Manifest.h"
#include "passes/PassImage.h"
#include "passes/PassSequence.h"
#include "viewport/ViewportRenderer.h"

namespace dlssvid {

// A pass folder or video that the viewport can show.
struct ViewportSource {
    std::string name;             // "source", "result", pass name
    std::filesystem::path path;   // video file or pass folder
    bool isVideo = false;
    TextureKind kind = TextureKind::Color;
    int64_t firstFrame = 0;
    int64_t lastFrame = -1;       // inclusive; video: frameCount - 1 (or -1 when unknown)
    uint32_t width = 0, height = 0;
    Rational fps{0, 1};
    // Pass folders: shared, sources are copied around freely (Manifest is ~0.5 KB + JSON).
    std::shared_ptr<const Manifest> manifest;
    bool HasFrame(int64_t f) const { return f >= firstFrame && (lastFrame < 0 || f <= lastFrame); }
};

// GPU-resident cache of pass frames for the viewport (ТЗ §6: one frame cache for all modes,
// passes are not copied to the CPU for display). Loader threads read pass files / decode video
// into CPU images (one decoder per video source, frames on the way to a seek target are kept
// when the viewport wants them); Update() (render thread) uploads them into textures within a
// budget and evicts least-recently-used frames outside the prefetch window.
class FrameStore {
public:
    struct Options {
        size_t vramBudgetBytes = size_t{1536} << 20;  // 1.5 GiB
        int prefetch = 6;                             // frames before/after the current one
        int uploadsPerUpdate = 4;
        int loaderThreads = 0;                        // 0 = auto (2..4)
    };
    enum class Status { Missing, Queued, Ready };

    explicit FrameStore(D3D12Device& device, const Options& options = {});
    ~FrameStore();
    FrameStore(const FrameStore&) = delete;
    FrameStore& operator=(const FrameStore&) = delete;

    // Replaces all sources and clears the cache (waits for loads in flight).
    void SetSources(std::vector<ViewportSource> sources);
    const std::vector<ViewportSource>& Sources() const { return sources_; }
    const ViewportSource* FindSource(const std::string& name) const;
    static std::vector<ViewportSource> DiscoverPasses(const std::filesystem::path& passesRoot);
    static std::optional<ViewportSource> VideoSource(const std::string& name, const std::filesystem::path& file);

    // Reference image size (the "source" video or the first source) and frame range.
    uint32_t ImageWidth() const { return imageW_; }
    uint32_t ImageHeight() const { return imageH_; }
    int64_t FrameCount() const { return frameCount_; }
    Rational Fps() const { return fps_; }

    // Sets the current frame and the sources it needs; schedules current-first loads and prefetch.
    void SetCurrentFrame(int64_t frame, const std::vector<std::string>& neededSources);
    int64_t CurrentFrame() const { return current_; }

    // Render thread: uploads finished loads, evicts, returns whether anything changed.
    bool Update();
    // Blocks until the current frame's needed sources are ready (or missing), then uploads. CLI / tests.
    void WaitForCurrent(uint32_t timeoutMs = 60000);

    Status GetStatus(int64_t frame, const std::string& source) const;
    // Textures of a frame that are ready right now (missing entries are simply absent).
    FrameTextures Textures(int64_t frame) const;
    size_t ResidentBytes() const { return residentBytes_; }
    size_t ResidentCount() const;
    int LoaderThreads() const { return static_cast<int>(loaders_.size()); }

private:
    struct Key {
        int64_t frame;
        std::string source;
        bool operator<(const Key& o) const { return frame != o.frame ? frame < o.frame : source < o.source; }
    };
    struct Loaded {
        Key key;
        PassImage image;
        TextureKind kind = TextureKind::None;
        float minValue = 0.f, maxValue = 1.f, maxMagnitude = 1.f;
        bool missing = false;
    };
    struct Entry {
        ComPtr<ID3D12Resource> texture;
        LayerTexture desc;
        size_t bytes = 0;
        uint64_t lastUse = 0;
    };
    // Per-source loader state; created in SetSources, used by the loader threads only.
    struct SourceRuntime {
        std::mutex mutex;  // serialises the decoder / lazy reader creation
        std::unique_ptr<VideoDecoder> decoder;
        ColorInfo color;
        std::shared_ptr<const PassReader> reader;
    };

    void LoaderMain();
    std::vector<Loaded> LoadOne(const Key& key);
    std::vector<Loaded> LoadVideo(const Key& key, const ViewportSource& src, SourceRuntime& rt);
    Loaded LoadPass(const Key& key, const ViewportSource& src, SourceRuntime& rt);
    bool Wanted(const Key& key) const;  // needed by the viewport and not resident / known-missing
    void Evict();
    bool InWindow(int64_t frame) const;

    D3D12Device& device_;
    Options options_;
    std::vector<ViewportSource> sources_;
    std::map<std::string, std::unique_ptr<SourceRuntime>> runtimes_;
    uint32_t imageW_ = 0, imageH_ = 0;
    int64_t frameCount_ = 0;
    Rational fps_{0, 1};

    mutable std::mutex mutex_;
    std::condition_variable cv_;        // loaders wait for work
    std::condition_variable idleCv_;    // SetSources waits for loads in flight
    std::deque<Key> queue_;             // pending loads (front = most urgent)
    std::set<Key> queued_;              // keys in queue_ or in flight
    std::set<Key> inFlight_;            // keys a loader is working on
    std::deque<Loaded> loaded_;         // finished loads awaiting upload
    std::set<Key> missing_;             // known-missing (frame outside range / read error)
    std::map<Key, Entry> entries_;      // GPU-resident
    size_t residentBytes_ = 0;
    uint64_t useCounter_ = 0;
    int64_t current_ = 0;
    std::vector<std::string> needed_;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> loaders_;
};

}  // namespace dlssvid
