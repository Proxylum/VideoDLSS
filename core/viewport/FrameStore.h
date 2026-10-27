#pragma once

#include <array>
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
    std::string name;             // "source", "result", pass folder name
    std::filesystem::path path;   // video file or pass folder
    bool isVideo = false;
    TextureKind kind = TextureKind::Color;
    int64_t firstFrame = 0;
    int64_t lastFrame = -1;       // inclusive; video: frameCount - 1 (or -1 when unknown)
    uint32_t width = 0, height = 0;
    Rational fps{0, 1};
    // Pass folders: the folder name, as in PassVersions.h (the current version and its previous versions share it). The
    // manifest's `pass` is the kind and may differ: mask_ui, mask_ignore, mask_face and mask_skin are all "mask".
    std::string pass;
    std::string version;          // a previous version (PassVersions.h): its history id; empty for the current one
    // Videos: the scrubbing proxy (ViewportProxy.h, bit for bit the same frames, intra) - jumps decode one frame of it;
    // empty when there is none. The colour of the original (the proxy's tags are only a copy).
    std::filesystem::path proxy;
    ColorInfo color;
    bool hasColor = false;
    // Pass folders: shared, sources are copied around freely (Manifest is ~0.5 KB + JSON).
    std::shared_ptr<const Manifest> manifest;
    // A static pass (PR-8: mask_control, Manifest::IsStatic): its one frame stands at every time - no duration of its own
    bool isStatic = false;
    bool HasFrame(int64_t f) const { return f >= firstFrame && (lastFrame < 0 || f <= lastFrame); }
};

// GPU-resident cache of pass frames for the viewport (ТЗ §6: one frame cache for all modes,
// passes are not copied to the CPU for display). Loader threads read pass files / decode video
// into CPU images (one decoder per video source, frames on the way to a seek target are kept
// when the viewport wants them); Update() (render thread) uploads them into textures within a
// budget and evicts least-recently-used frames outside the prefetch window.
//
// Time axis (stage 9): the viewport position is a time in seconds, every source keeps its own
// frame rate and shows frame round(time × fps) — a 24 fps source and a 48 fps result stay in step
// over the whole clip. Sources without a rate (pass folders written without one, tests) follow the
// base rate: the "source" video's, else the slowest known one, else 1 fps (frame == second).
class FrameStore {
public:
    struct Options {
        size_t vramBudgetBytes = size_t{1536} << 20;  // 1.5 GiB
        int prefetch = 6;                             // frames of the fastest source before/after the current time
        int uploadsPerUpdate = 4;
        int loaderThreads = 0;                        // 0 = auto (2..4)
    };
    // Missing: outside the source or failed to read; Queued: loadable, not requested yet or waiting in the
    // queue; Loading: a loader reads it or it waits for the upload; Ready: resident on the GPU.
    enum class Status { Missing, Queued, Loading, Ready };

    explicit FrameStore(D3D12Device& device, const Options& options = {});
    ~FrameStore();
    FrameStore(const FrameStore&) = delete;
    FrameStore& operator=(const FrameStore&) = delete;

    // Replaces all sources and clears the cache (waits for loads in flight).
    void SetSources(std::vector<ViewportSource> sources);
    const std::vector<ViewportSource>& Sources() const { return sources_; }
    const ViewportSource* FindSource(const std::string& name) const;
    // Pass folders with a manifest directly under passesRoot, named by their folder (the texture kind comes from the manifest).
    static std::vector<ViewportSource> DiscoverPasses(const std::filesystem::path& passesRoot);
    // Previous versions under <pass>.v/ as sources named "<pass>@<id>" (stage 9: compare versions with the wipe).
    static std::vector<ViewportSource> DiscoverPassVersions(const std::filesystem::path& passesRoot);
    static std::optional<ViewportSource> VideoSource(const std::string& name, const std::filesystem::path& file);

    // Reference image size (the "source" video or the first source).
    uint32_t ImageWidth() const { return imageW_; }
    uint32_t ImageHeight() const { return imageH_; }

    // ---- time axis ----
    Rational Fps() const { return fps_; }       // base rate as declared ({0,1} when no source has one)
    Rational BaseFps() const;                    // effective base rate (1 fps when unknown)
    Rational SourceFps(const std::string& source) const;  // the source's rate, else the base rate
    double FpsOf(const std::string& source) const { return SourceFps(source).ToDouble(); }
    double MaxFps(const std::vector<std::string>& sources = {}) const;  // fastest of these (all sources when empty)
    Rational MaxRate(const std::vector<std::string>& sources = {}) const;  // the same, as the exact rational rate
    double Duration() const { return duration_; }   // seconds covered by the longest source
    int64_t FrameCount() const;                     // frames of the base rate over the duration
    int64_t FrameAt(const std::string& source, double time) const;  // round(time × fps); may lie outside the source
    double TimeOfFrame(const std::string& source, int64_t frame) const;
    // Frames a source needs to reach the reference end: the "source" video's last frame (else the longest source's).
    int64_t ExpectedFrames(const std::string& source) const;

    // Sets the current time and the sources it needs; schedules current-first loads and prefetch (a window of
    // `prefetch` frames of the fastest source on either side, fewer frames for slower sources).
    void SetCurrentTime(double time, const std::vector<std::string>& neededSources);
    double CurrentTime() const { return currentTime_; }
    // The same at the base rate: frame / base fps.
    void SetCurrentFrame(int64_t frame, const std::vector<std::string>& neededSources);
    int64_t CurrentFrame() const;

    // Render thread: uploads finished loads, evicts, returns whether anything changed. Frames of the current time go
    // first and alone (the target of a jump is not held back by the uploads of its neighbours); a finished frame
    // outside the window is uploaded only when it stands in better than what is resident (fast drag without a proxy).
    bool Update();
    // A video's scrubbing proxy appeared or went away (the frames are the same: nothing is evicted).
    void SetProxy(const std::string& source, const std::filesystem::path& proxy);
    // Called from a loader thread when a frame of the current time is ready for Update() (the GUI uploads it at once
    // instead of on its next poll). Must not block; not called under the store's lock.
    void SetOnCurrentLoaded(std::function<void()> callback);
    struct SourceStats {
        int64_t decodedFrames = 0;  // decoded by the original's and the proxy's decoders
        int64_t seeks = 0;
        int64_t aborted = 0;        // loads dropped before decoding because the time moved on
        int64_t proxyLoads = 0;     // jumps served by the proxy
    };
    SourceStats Stats(const std::string& source) const;
    // Blocks until the current time's needed sources are ready (or missing), then uploads. CLI / tests.
    void WaitForCurrent(uint32_t timeoutMs = 60000);

    Status GetStatus(int64_t frame, const std::string& source) const;
    Status StatusAt(double time, const std::string& source) const { return GetStatus(KeyFrame(source, time), source); }
    // Textures resident right now: of one frame index in every source (uniform rates: tests, benchmarks; a static source at any) ...
    FrameTextures Textures(int64_t frame) const;
    // ... or of every source's frame at `time`. While a loadable frame is not resident yet, the source's nearest
    // resident frame stands in for it (earlier frames first): the viewport holds the last frame instead of flashing a
    // plate on every tick of playback or scrubbing. Missing frames are never substituted; absent entries mean nothing
    // to show.
    FrameTextures TexturesAt(double time) const;
    // The stand-in TexturesAt(time) shows for `source`: nullopt when the exact frame is resident or missing, or when
    // nothing of the source is resident.
    std::optional<int64_t> HeldFrameAt(double time, const std::string& source) const;
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
        PassImage image;  // the pixels are dropped once staged (only the size stays)
        TextureKind kind = TextureKind::None;
        float minValue = 0.f, maxValue = 1.f, maxMagnitude = 1.f;
        bool missing = false;
        D3D12Device::StagedTexture staged;  // texture + upload buffer made by the loader thread (Publish)
        size_t bytes = 0;
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
        // the proxy's decoders: jumps run on them while the original walks forward; a pool, so the target of a jump does
        // not wait behind the prefetch of its neighbours (TASK-0040)
        struct ProxySlot {
            std::mutex mutex;
            std::unique_ptr<VideoDecoder> decoder;
            std::filesystem::path opened;
        };
        std::array<ProxySlot, 4> proxies;
        std::atomic<int64_t> aborted{0}, proxyLoads{0};
    };
    struct LoadResult {
        std::vector<Loaded> loaded;
        bool aborted = false;  // not loaded because nobody wants it any more; not missing
    };

    void LoaderMain();
    LoadResult LoadOne(const Key& key);
    LoadResult LoadVideo(const Key& key, const ViewportSource& src, SourceRuntime& rt);
    // Decodes the target (and keeps wanted frames on the way) with `dec`; false: dropped before decoding (not wanted).
    bool DecodeTo(VideoDecoder& dec, const Key& key, int64_t walkLimit, std::vector<CpuFrame>& kept, bool& missing);
    Loaded LoadPass(const Key& key, const ViewportSource& src, SourceRuntime& rt);
    void Publish(Loaded&& loaded);   // into loaded_ (and the current-frame callback)
    bool Wanted(const Key& key) const;  // needed by the viewport and not resident / known-missing
    bool WantedLocked(const Key& key) const;
    bool IsCurrentLocked(const Key& key) const;  // the needed frame of its source at the current time (mutex_ held)
    // A frame of the current time is being decoded or converted (mutex_ held): prefetch waits for it, so the target of
    // a jump gets the whole CPU (TASK-0040: four loaders decoding at once doubled its latency).
    bool CurrentPendingLocked() const;
    Status StatusLocked(const Key& key) const;  // GetStatus with mutex_ held
    // Nearest resident frame of the source (the nearest earlier one, else the nearest later); end() when none (mutex_ held).
    std::map<Key, Entry>::const_iterator NearestResident(int64_t frame, const std::string& source) const;
    void Evict();
    bool InWindow(const Key& key) const;              // within the prefetch window of its source (mutex_ held)
    int SpanOf(const std::string& source) const;      // prefetch frames of a source: window seconds × its rate
    double ReferenceLastTime() const;                 // time of the reference's last frame (ExpectedFrames)
    // The frame a source loads and keeps for `time`: FrameAt, or the one frame of a static source (PR-8) - the key of every load,
    // window and status; FrameAt itself stays the time's frame (stepping, the «#N» of a cell)
    int64_t KeyFrame(const std::string& source, double time) const;

    D3D12Device& device_;
    Options options_;
    std::vector<ViewportSource> sources_;
    std::map<std::string, std::unique_ptr<SourceRuntime>> runtimes_;
    uint32_t imageW_ = 0, imageH_ = 0;
    Rational fps_{0, 1};
    double duration_ = 0.0;

    mutable std::mutex mutex_;
    std::condition_variable cv_;        // loaders wait for work
    std::condition_variable idleCv_;    // SetSources waits for loads in flight
    std::condition_variable loadedCv_;  // WaitForCurrent: something reached loaded_
    std::set<Key> kept_;                // decoded on the way to another frame, converting: loaded, not queued
    std::function<void()> onCurrentLoaded_;
    std::vector<ComPtr<ID3D12Resource>> freeTextures_;  // evicted, reused by Publish for a frame of the same size and format
    std::deque<Key> queue_;             // pending loads (front = most urgent)
    std::set<Key> queued_;              // keys in queue_ or in flight
    std::set<Key> inFlight_;            // keys a loader is working on
    int prefetchActive_ = 0;            // of them, not of the current time: at most loaders - 2, the rest stays free for a jump
    std::deque<Loaded> loaded_;         // finished loads awaiting upload
    std::set<Key> missing_;             // known-missing (frame outside range / read error)
    std::map<Key, Entry> entries_;      // GPU-resident
    size_t residentBytes_ = 0;
    uint64_t useCounter_ = 0;
    double currentTime_ = 0.0;
    std::vector<std::string> needed_;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> loaders_;
};

}  // namespace dlssvid
