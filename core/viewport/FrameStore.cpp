#include "viewport/FrameStore.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "passes/ControlMaskPass.h"
#include "pipeline/PassVersions.h"
#include "util/Error.h"
#include "util/Half.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

TextureKind KindForManifest(const Manifest& m) {
    // PR-8: the control mask is colour - ToRgba16f scales U16 to [0, 1] and shows one grey channel grey (as a Scalar it would be raw
    // 0..65535, white everywhere); the general rule for mask_* stays
    if (m.pass == kControlMaskPass) return TextureKind::Color;
    if (const auto k = m.Kind()) {
        switch (*k) {
            case PassKind::DepthRaw:
            case PassKind::DepthDlss: return TextureKind::Scalar;
            case PassKind::MvRaw:
            case PassKind::MvDlss: return TextureKind::Mv;
            case PassKind::Mask: return TextureKind::Mask;
            default: return TextureKind::Color;
        }
    }
    if (m.channels.size() == 1) return m.pixelType == PixelType::U8 ? TextureKind::Mask : TextureKind::Scalar;
    if (m.channels.size() == 2) return TextureKind::Mv;
    return TextureKind::Color;
}

DXGI_FORMAT FormatFor(TextureKind k) {
    switch (k) {
        case TextureKind::Color: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case TextureKind::Scalar: return DXGI_FORMAT_R32_FLOAT;
        case TextureKind::Mv: return DXGI_FORMAT_R32G32_FLOAT;
        case TextureKind::Mask: return DXGI_FORMAT_R8_UNORM;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

// halves -> floats, four at a time with F16C (a 1080p depth pass: ~1 ms instead of ~7 ms through Get / Set)
void HalvesToFloats(const uint16_t* src, float* dst, size_t n) {
    static const bool f16c = HasF16C();
    size_t i = 0;
    if (f16c)
        for (; i + 4 <= n; i += 4) _mm_storeu_ps(dst + i, _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i))));
    for (; i < n; ++i) dst[i] = HalfToFloat(src[i]);
}

// Repack a pass image into the texture layout of its kind and compute statistics.
void Prepare(PassImage& img, TextureKind kind, float& minV, float& maxV, float& maxMag) {
    minV = 0.f;
    maxV = 1.f;
    maxMag = 1.f;
    switch (kind) {
        case TextureKind::Color: {
            if (img.type == PixelType::F16 && img.channels.size() == 4) break;  // already RGBA16F
            img = ToRgba16f(img);
            break;
        }
        case TextureKind::Scalar: {
            if (img.type == PixelType::F16 && img.channels.size() == 1) {  // EXR half: the common case, bulk
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::F32, {"Z"});
                HalvesToFloats(img.As<uint16_t>(), out.As<float>(), static_cast<size_t>(img.width) * img.height);
                img = std::move(out);
            } else if (img.type != PixelType::F32 || img.channels.size() != 1) {
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::F32, {"Z"});
                for (uint32_t y = 0; y < img.height; ++y)
                    for (uint32_t x = 0; x < img.width; ++x) out.Set(x, y, 0, img.Get(x, y, 0));
                img = std::move(out);
            }
            // min / max of the finite samples: v - v is 0 only for a finite v (NaN and inf give NaN), no branch to mispredict
            const float* d = img.As<float>();
            const size_t n = static_cast<size_t>(img.width) * img.height;
            float lo = std::numeric_limits<float>::infinity(), hi = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < n; ++i) {
                const float v = d[i];
                const bool finite = v - v == 0.f;
                lo = finite && v < lo ? v : lo;
                hi = finite && v > hi ? v : hi;
            }
            if (lo <= hi) {
                minV = lo;
                maxV = hi;
            } else {  // nothing finite
                minV = 0.f;
                maxV = 1.f;
            }
            break;
        }
        case TextureKind::Mv: {
            if (img.type == PixelType::F16 && img.channels.size() == 2) {  // EXR half u, v interleaved: bulk
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::F32, {"u", "v"});
                HalvesToFloats(img.As<uint16_t>(), out.As<float>(), static_cast<size_t>(img.width) * img.height * 2);
                img = std::move(out);
            } else if (img.type != PixelType::F32 || img.channels.size() != 2) {
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::F32, {"u", "v"});
                for (uint32_t y = 0; y < img.height; ++y)
                    for (uint32_t x = 0; x < img.width; ++x) {
                        out.Set(x, y, 0, img.Get(x, y, 0));
                        out.Set(x, y, 1, img.channels.size() > 1 ? img.Get(x, y, 1) : 0.f);
                    }
                img = std::move(out);
            }
            const float* d = img.As<float>();
            float m = 0.f;
            for (size_t i = 0; i < static_cast<size_t>(img.width) * img.height; ++i) {
                const float mag2 = d[2 * i] * d[2 * i] + d[2 * i + 1] * d[2 * i + 1];  // the max of squares, one sqrt
                if (std::isfinite(mag2)) m = std::max(m, mag2);
            }
            m = std::sqrt(m);
            maxMag = std::max(m, 1e-3f);
            minV = 0.f;
            maxV = maxMag;
            break;
        }
        case TextureKind::Mask: {
            if (img.type != PixelType::U8 || img.channels.size() != 1) {
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::U8, {"A"});
                const float scale = img.type == PixelType::U16 ? 255.f / 65535.f : (img.type == PixelType::U8 ? 1.f : 255.f);
                for (uint32_t y = 0; y < img.height; ++y)
                    for (uint32_t x = 0; x < img.width; ++x) out.Set(x, y, 0, img.Get(x, y, 0) * scale);
                img = std::move(out);
            }
            break;
        }
        default: break;
    }
}

constexpr int64_t kWalkLimit = 48;  // decode forward instead of seeking when the target is this close ahead

}  // namespace

// ---- sources -----------------------------------------------------------------------------------

std::vector<ViewportSource> FrameStore::DiscoverPasses(const std::filesystem::path& passesRoot) {
    std::vector<ViewportSource> out;
    if (!std::filesystem::is_directory(passesRoot)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(passesRoot)) {
        if (!entry.is_directory() || !Manifest::Exists(entry.path())) continue;
        try {
            ViewportSource s;
            s.manifest = std::make_shared<const Manifest>(Manifest::Load(entry.path()));
            if (s.manifest->IsDataPass()) continue;  // a data pass (scenes, PR-13): no frames to show
            s.name = s.manifest->pass.empty() ? entry.path().filename().string() : s.manifest->pass;
            s.pass = s.name;
            s.path = entry.path();
            s.kind = KindForManifest(*s.manifest);
            s.firstFrame = s.manifest->firstFrame;
            s.lastFrame = s.manifest->lastFrame;
            s.width = s.manifest->width;
            s.height = s.manifest->height;
            s.fps = s.manifest->fps;
            s.isStatic = s.manifest->IsStatic();
            out.push_back(std::move(s));
        } catch (const std::exception& e) {
            Log()->warn("pass folder {} skipped: {}", entry.path().string(), e.what());
        }
    }
    std::sort(out.begin(), out.end(), [](const ViewportSource& a, const ViewportSource& b) { return a.name < b.name; });
    return out;
}

std::vector<ViewportSource> FrameStore::DiscoverPassVersions(const std::filesystem::path& passesRoot) {
    std::vector<ViewportSource> out;
    if (!std::filesystem::is_directory(passesRoot)) return out;
    for (const auto& v : ListPassVersions(passesRoot)) {
        if (v.current) continue;
        try {
            ViewportSource s;
            s.manifest = std::make_shared<const Manifest>(Manifest::Load(v.dir));
            if (s.manifest->IsDataPass()) continue;  // a data pass (scenes, PR-13): no frames to show
            s.pass = v.pass;
            s.version = v.id;
            s.name = VersionSourceName(v.pass, v.id);
            s.path = v.dir;
            s.kind = KindForManifest(*s.manifest);
            s.firstFrame = s.manifest->firstFrame;
            s.lastFrame = s.manifest->lastFrame;
            s.width = s.manifest->width;
            s.height = s.manifest->height;
            s.fps = s.manifest->fps;
            s.isStatic = s.manifest->IsStatic();
            out.push_back(std::move(s));
        } catch (const std::exception& e) {
            Log()->warn("pass version {} skipped: {}", v.dir.string(), e.what());
        }
    }
    std::sort(out.begin(), out.end(), [](const ViewportSource& a, const ViewportSource& b) { return a.name < b.name; });
    return out;
}

std::optional<ViewportSource> FrameStore::VideoSource(const std::string& name, const std::filesystem::path& file) {
    if (file.empty() || !std::filesystem::exists(file)) return std::nullopt;
    try {
        VideoDecoder dec(file, VideoDecoder::Probe());
        const VideoStreamInfo& info = dec.Info();
        ViewportSource s;
        s.name = name;
        s.path = file;
        s.isVideo = true;
        s.kind = TextureKind::Color;
        s.firstFrame = 0;
        int64_t frames = info.frameCount;
        if (frames <= 0 && info.durationUs > 0 && info.frameRate.num > 0)  // MKV/MOV without nb_frames: estimate from the duration
            frames = static_cast<int64_t>(std::llround(static_cast<double>(info.durationUs) * info.frameRate.ToDouble() / 1e6));
        s.lastFrame = frames > 0 ? frames - 1 : -1;
        s.width = info.width;
        s.height = info.height;
        s.fps = info.frameRate;
        s.color = ColorInfoFromStream(info);
        s.hasColor = true;
        return s;
    } catch (const std::exception& e) {
        Log()->warn("video {} not usable as source '{}': {}", file.string(), name, e.what());
        return std::nullopt;
    }
}

FrameStore::FrameStore(D3D12Device& device, const Options& options) : device_(device), options_(options) {
    const int threads = options_.loaderThreads > 0 ? options_.loaderThreads : std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 2, 2, 4);
    for (int i = 0; i < threads; ++i) loaders_.emplace_back([this] { LoaderMain(); });
}

FrameStore::~FrameStore() {
    {
        // Under the mutex, or a loader between its predicate check and the wait misses the notification and the
        // join below never returns (incident 2026-09-22: a CI test process stuck after passing, one loader asleep).
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    loadedCv_.notify_all();
    for (auto& t : loaders_)
        if (t.joinable()) t.join();
}

void FrameStore::SetSources(std::vector<ViewportSource> sources) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.clear();
    queued_.clear();
    idleCv_.wait(lock, [&] { return inFlight_.empty(); });  // loaders may still use the old runtimes
    sources_ = std::move(sources);
    runtimes_.clear();
    for (const auto& s : sources_) runtimes_[s.name] = std::make_unique<SourceRuntime>();
    loaded_.clear();
    kept_.clear();
    freeTextures_.clear();
    prefetchActive_ = 0;  // inFlight_ is empty here
    missing_.clear();
    entries_.clear();
    residentBytes_ = 0;
    imageW_ = imageH_ = 0;
    fps_ = Rational{0, 1};
    duration_ = 0.0;
    bool fromSource = false;
    for (const auto& s : sources_) {
        if (s.name == "source" || imageW_ == 0) {
            imageW_ = s.width;
            imageH_ = s.height;
        }
        if (s.fps.num <= 0 || s.fps.den <= 0) continue;
        // base rate: the source video's; without it the slowest source (the source is the slowest, FG doubles it)
        if (s.name == "source") {
            fps_ = s.fps;
            fromSource = true;
        } else if (!fromSource && (fps_.num == 0 || s.fps.ToDouble() < fps_.ToDouble())) {
            fps_ = s.fps;
        }
    }
    for (const auto& s : sources_)  // a static source has no length: one frame would not make the time line one frame long
        if (s.lastFrame >= 0 && !s.isStatic) duration_ = std::max(duration_, static_cast<double>(s.lastFrame + 1) / FpsOf(s.name));
}

const ViewportSource* FrameStore::FindSource(const std::string& name) const {
    for (const auto& s : sources_)
        if (s.name == name) return &s;
    return nullptr;
}

// ---- time axis ----------------------------------------------------------------------------------------

Rational FrameStore::BaseFps() const { return fps_.num > 0 && fps_.den > 0 ? fps_ : Rational{1, 1}; }

Rational FrameStore::SourceFps(const std::string& source) const {
    const ViewportSource* s = FindSource(source);
    return s && s->fps.num > 0 && s->fps.den > 0 ? s->fps : BaseFps();
}

double FrameStore::MaxFps(const std::vector<std::string>& sources) const { return MaxRate(sources).ToDouble(); }

Rational FrameStore::MaxRate(const std::vector<std::string>& sources) const {
    Rational best = BaseFps();
    auto consider = [&](const std::string& name) {
        const Rational r = SourceFps(name);
        if (r.ToDouble() > best.ToDouble()) best = r;
    };
    if (sources.empty()) {
        for (const auto& s : sources_) consider(s.name);
    } else {
        for (const auto& n : sources) consider(n);
    }
    return best;
}

int64_t FrameStore::FrameCount() const { return static_cast<int64_t>(std::llround(duration_ * BaseFps().ToDouble())); }

int64_t FrameStore::FrameAt(const std::string& source, double time) const { return static_cast<int64_t>(std::llround(time * FpsOf(source))); }

double FrameStore::TimeOfFrame(const std::string& source, int64_t frame) const { return static_cast<double>(frame) / FpsOf(source); }

double FrameStore::ReferenceLastTime() const {
    if (const ViewportSource* src = FindSource("source"); src && src->lastFrame >= 0) return TimeOfFrame("source", src->lastFrame);
    double last = 0.0;
    for (const auto& s : sources_)
        if (s.lastFrame >= 0 && !s.isStatic) last = std::max(last, TimeOfFrame(s.name, s.lastFrame));
    return last;
}

int64_t FrameStore::ExpectedFrames(const std::string& source) const {
    if (sources_.empty()) return 0;
    if (const ViewportSource* s = FindSource(source); s && s->isStatic) return 1;
    return FrameAt(source, ReferenceLastTime()) + 1;
}

int64_t FrameStore::KeyFrame(const std::string& source, double time) const {
    if (const ViewportSource* s = FindSource(source); s && s->isStatic) return s->firstFrame;
    return FrameAt(source, time);
}

int FrameStore::SpanOf(const std::string& source) const {
    const double window = static_cast<double>(options_.prefetch) / MaxFps();  // seconds on either side
    return std::max(1, static_cast<int>(std::llround(window * FpsOf(source))));
}

int64_t FrameStore::CurrentFrame() const { return FrameAt("", currentTime_); }

// ---- scheduling -----------------------------------------------------------------------------------

bool FrameStore::InWindow(const Key& key) const { return std::llabs(key.frame - KeyFrame(key.source, currentTime_)) <= SpanOf(key.source); }

void FrameStore::SetCurrentTime(double time, const std::vector<std::string>& neededSources) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentTime_ = std::max(0.0, time);
    needed_ = neededSources;
    // Rebuild the queue: every source's current frame first, then prefetch outwards (forward first). Loads in
    // flight stay marked so they are not requested a second time.
    queue_.clear();
    queued_ = inFlight_;
    auto push = [&](const std::string& name, int64_t f) {
        const ViewportSource* s = FindSource(name);
        if (!s || !s->HasFrame(f)) return;
        const Key k{f, name};
        if (entries_.count(k) || missing_.count(k) || queued_.count(k) || kept_.count(k)) return;
        queue_.push_back(k);
        queued_.insert(k);
    };
    int maxSpan = 0;
    for (const auto& name : needed_) maxSpan = std::max(maxSpan, SpanOf(name));
    for (int d = 0; d <= maxSpan; ++d)
        for (const auto& name : needed_) {
            if (d > SpanOf(name)) continue;
            const int64_t f0 = KeyFrame(name, currentTime_);  // a static source: its one frame; the neighbours are outside it
            push(name, f0 + d);
            if (d > 0 && f0 - d >= 0) push(name, f0 - d);
        }
    cv_.notify_all();
}

void FrameStore::SetCurrentFrame(int64_t frame, const std::vector<std::string>& neededSources) { SetCurrentTime(TimeOfFrame("", frame), neededSources); }

bool FrameStore::Wanted(const Key& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return WantedLocked(key);
}

bool FrameStore::WantedLocked(const Key& key) const {
    if (!InWindow(key)) return false;
    if (std::find(needed_.begin(), needed_.end(), key.source) == needed_.end()) return false;
    if (entries_.count(key) || missing_.count(key) || kept_.count(key)) return false;
    for (const auto& l : loaded_)
        if (l.key.frame == key.frame && l.key.source == key.source) return false;
    return true;
}

FrameStore::Status FrameStore::GetStatus(int64_t frame, const std::string& source) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return StatusLocked(Key{frame, source});
}

FrameStore::Status FrameStore::StatusLocked(const Key& k) const {
    const int64_t frame = k.frame;
    const std::string& source = k.source;
    if (entries_.count(k)) return Status::Ready;
    if (inFlight_.count(k) || kept_.count(k)) return Status::Loading;
    for (const auto& l : loaded_)
        if (l.key.frame == frame && l.key.source == source) return l.missing ? Status::Missing : Status::Loading;
    if (queued_.count(k)) return Status::Queued;
    const ViewportSource* s = FindSource(source);
    if (!s || !s->HasFrame(frame) || missing_.count(k)) return Status::Missing;
    return Status::Queued;  // not loaded yet but loadable
}

FrameTextures FrameStore::Textures(int64_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameTextures out;
    for (const auto& [key, e] : entries_) {
        if (const ViewportSource* s = FindSource(key.source); key.frame != frame && !(s && s->isStatic)) continue;  // a static source: at any index
        out[key.source] = e.desc;
        const_cast<Entry&>(e).lastUse = ++const_cast<uint64_t&>(useCounter_);
    }
    return out;
}

std::map<FrameStore::Key, FrameStore::Entry>::const_iterator FrameStore::NearestResident(int64_t frame, const std::string& source) const {
    // entries_ is ordered by frame, then source: down from the first entry past `frame` for the nearest earlier frame
    // of the source, up from there for the nearest later one.
    const auto past = entries_.lower_bound(Key{frame + 1, std::string()});
    for (auto it = past; it != entries_.begin();) {
        --it;
        if (it->first.source == source) return it;
    }
    for (auto it = past; it != entries_.end(); ++it)
        if (it->first.source == source) return it;
    return entries_.end();
}

FrameTextures FrameStore::TexturesAt(double time) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameTextures out;
    for (const auto& s : sources_) {
        const Key key{KeyFrame(s.name, time), s.name};
        std::map<Key, Entry>::const_iterator it = entries_.find(key);
        if (it == entries_.end()) {
            if (StatusLocked(key) == Status::Missing) continue;  // nothing to show and nothing to hold
            it = NearestResident(key.frame, s.name);          // hold the last frame while this one loads
            if (it == entries_.end()) continue;
        }
        out[s.name] = it->second.desc;
        const_cast<Entry&>(it->second).lastUse = ++const_cast<uint64_t&>(useCounter_);
    }
    return out;
}

std::optional<int64_t> FrameStore::HeldFrameAt(double time, const std::string& source) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Key key{KeyFrame(source, time), source};
    if (entries_.count(key) || StatusLocked(key) == Status::Missing) return std::nullopt;
    const auto it = NearestResident(key.frame, source);
    if (it == entries_.end()) return std::nullopt;
    return it->first.frame;
}

size_t FrameStore::ResidentCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

// ---- loader threads ---------------------------------------------------------------------------------

void FrameStore::LoaderMain() {
    for (;;) {
        Key key;
        bool prefetch = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // the frame of the current time at once; prefetch only when no current frame is loading and with loaders to
            // spare (a jump finds a free loader and a free proxy decoder: TASK-0040)
            const int maxPrefetch = std::max(1, static_cast<int>(loaders_.size()) - 2);
            cv_.wait(lock, [&] {
                return stop_ || (!queue_.empty() && (IsCurrentLocked(queue_.front()) || (!CurrentPendingLocked() && prefetchActive_ < maxPrefetch)));
            });
            if (stop_) return;
            key = queue_.front();
            queue_.pop_front();
            inFlight_.insert(key);
            prefetch = !IsCurrentLocked(key);
            if (prefetch) ++prefetchActive_;
        }
        LoadResult result;
        try {
            result = LoadOne(key);
        } catch (const std::exception& e) {
            Log()->warn("frame {} of '{}' failed to load: {}", key.frame, key.source, e.what());
            Loaded l;
            l.key = key;
            l.missing = true;
            result.loaded.push_back(std::move(l));
        }
        for (auto& l : result.loaded) Publish(std::move(l));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            inFlight_.erase(key);
            if (prefetch) --prefetchActive_;
            cv_.notify_all();  // a current frame done: the prefetch waiting for it may start
            if (result.aborted) {
                // dropped before decoding (TASK-0040). In the same critical section as the erase: if the time came back to
                // this frame meanwhile, SetCurrentTime skipped it (it was in flight) - queue it again, or nobody loads it
                queued_.erase(key);
                if (WantedLocked(key)) {
                    queue_.push_front(key);
                    queued_.insert(key);
                    cv_.notify_one();
                }
            }
        }
        idleCv_.notify_all();
    }
}

void FrameStore::Publish(Loaded&& l) {
    if (!l.missing && !l.image.Empty() && !l.staged.texture) {
        // on the loader thread: the texture and the copy of the rows into an upload buffer (Update only records copies)
        l.bytes = l.image.ByteSize();
        const auto t0 = std::chrono::steady_clock::now();
        const DXGI_FORMAT format = FormatFor(l.kind);
        ComPtr<ID3D12Resource> reuse;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = freeTextures_.begin(); it != freeTextures_.end(); ++it) {
                const D3D12_RESOURCE_DESC d = (*it)->GetDesc();
                if (d.Width == l.image.width && d.Height == l.image.height && d.Format == format) {
                    reuse = std::move(*it);
                    freeTextures_.erase(it);
                    break;
                }
            }
        }
        l.staged = device_.StageTexture2D(l.image.width, l.image.height, format, l.image.data.data(), l.image.RowBytes(), std::move(reuse));
        if (Log()->should_log(spdlog::level::debug))
            Log()->debug("viewport stage {} {}: {:.1f} ms", l.key.source, l.key.frame, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        l.image.data.clear();
        l.image.data.shrink_to_fit();
    }
    bool current = false;
    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        kept_.erase(l.key);
        current = !l.missing && IsCurrentLocked(l.key);
        loaded_.push_back(std::move(l));
        if (current) callback = onCurrentLoaded_;
    }
    loadedCv_.notify_all();
    if (current) cv_.notify_all();
    if (callback) callback();
}

FrameStore::LoadResult FrameStore::LoadOne(const Key& key) {
    ViewportSource src;
    SourceRuntime* rt = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.count(key) || kept_.count(key)) {  // arrived meanwhile (frame kept on the way to another one)
            queued_.erase(key);
            return {};
        }
        for (const auto& l : loaded_)
            if (l.key.frame == key.frame && l.key.source == key.source) return {};
        const ViewportSource* s = FindSource(key.source);
        const auto it = runtimes_.find(key.source);
        if (!s || it == runtimes_.end()) {
            Loaded l;
            l.key = key;
            l.missing = true;
            LoadResult r;
            r.loaded.push_back(std::move(l));
            return r;
        }
        src = *s;
        rt = it->second.get();
    }
    if (src.isVideo) return LoadVideo(key, src, *rt);
    LoadResult r;
    r.loaded.push_back(LoadPass(key, src, *rt));
    return r;
}

FrameStore::Loaded FrameStore::LoadPass(const Key& key, const ViewportSource& src, SourceRuntime& rt) {
    Loaded l;
    l.key = key;
    std::shared_ptr<const PassReader> reader;
    {
        std::lock_guard<std::mutex> guard(rt.mutex);
        if (!rt.reader) rt.reader = std::make_shared<const PassReader>(PassReader::Open(src.path));
        reader = rt.reader;
    }
    if (!reader->HasFrame(key.frame)) {
        l.missing = true;
        return l;
    }
    const auto t0 = std::chrono::steady_clock::now();
    l.image = reader->ReadFrame(key.frame);
    const auto t1 = std::chrono::steady_clock::now();
    l.kind = src.kind;
    Prepare(l.image, l.kind, l.minValue, l.maxValue, l.maxMagnitude);
    if (Log()->should_log(spdlog::level::debug))
        Log()->debug("viewport load {} {}: read {:.1f} ms, prepare {:.1f} ms", key.source, key.frame,
                     std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    return l;
}

bool FrameStore::DecodeTo(VideoDecoder& dec, const Key& key, int64_t walkLimit, std::vector<CpuFrame>& kept, bool& missing) {
    // the time may have moved on while this load waited for the decoder: drop it before decoding anything
    if (!Wanted(key)) return false;
    std::set<int64_t> seen;
    auto keep = [&](const CpuFrame& f) {
        if (!seen.insert(f.index).second) return;
        kept.push_back(f);
        if (f.index != key.frame) {
            std::lock_guard<std::mutex> lock(mutex_);
            kept_.insert(Key{f.index, key.source});  // visible as pending: no second load walks the GOP for it
        }
    };
    // A short distance ahead is decoded through; otherwise seek to the keyframe before the target.
    const int64_t next = dec.NextIndex();
    bool afterSeek = false;
    if (dec.Drained() || !dec.PositionKnown() || key.frame < next || key.frame - next > walkLimit) {
        if (!dec.SeekToKeyframeBefore(key.frame)) missing = true;
        afterSeek = true;
    }
    // Walk forward. Frames on the way that the viewport wants (prefetch window) are kept: they are
    // decoded anyway, so stepping backwards afterwards finds them resident.
    CpuFrame f;
    int retries = 0;
    while (!missing) {
        // right after a seek the index of the next frame is unknown; otherwise decide before decoding
        const int64_t upcoming = dec.NextIndex();
        const bool want = afterSeek || upcoming == key.frame || Wanted(Key{upcoming, key.source});
        if (!dec.NextFrame(f, nullptr, want)) {  // end of stream before the target
            missing = true;
            break;
        }
        afterSeek = false;
        if (f.index == key.frame) {
            keep(f);
            break;
        }
        if (f.index > key.frame) {
            // landed after the target (timestamp rounding): retry from an earlier keyframe, then from the start
            if (retries++ >= 2 || !dec.SeekToKeyframeBefore(retries == 1 ? std::max<int64_t>(0, key.frame - 1) : 0)) {
                missing = true;
                break;
            }
            afterSeek = true;
            continue;
        }
        if (!f.data.empty() && Wanted(Key{f.index, key.source})) keep(f);
    }
    return true;
}

FrameStore::LoadResult FrameStore::LoadVideo(const Key& key, const ViewportSource& src, SourceRuntime& rt) {
    LoadResult result;
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();
    Clock::time_point tDecoder{}, tDecoded{};
    const char* path = "original";
    std::vector<CpuFrame> kept;  // decoded frames to convert once the decoder is released
    bool missing = false;
    ColorInfo color = src.color;
    auto openOriginal = [&] {
        if (!rt.decoder) {
            rt.decoder = std::make_unique<VideoDecoder>(src.path);
            rt.color = ColorInfoFromStream(rt.decoder->Info());
        }
        if (!src.hasColor) color = rt.color;
    };
    // frames marked in kept_ by DecodeTo leave it when published or dropped; an exception must not leave them "Loading"
    struct KeptCleanup {
        FrameStore& fs;
        const std::string& source;
        std::vector<CpuFrame>& frames;
        bool armed = true;
        ~KeptCleanup() {
            if (!armed) return;
            std::lock_guard<std::mutex> lock(fs.mutex_);
            for (const auto& f : frames) fs.kept_.erase(Key{f.index, source});
        }
    } cleanup{*this, key.source, kept};
    if (!src.HasFrame(key.frame)) {
        missing = true;
    } else {
        bool done = false;
        if (!src.proxy.empty()) {
            // hybrid (TASK-0040): a step forward walks the original decoder that stands just before the target; a jump
            // decodes one frame of the intra proxy - on its own decoder, so it does not wait for a walk in progress
            std::unique_lock<std::mutex> guard(rt.mutex, std::try_to_lock);
            if (guard.owns_lock() && rt.decoder && !rt.decoder->Drained() && rt.decoder->PositionKnown() && key.frame >= rt.decoder->NextIndex() &&
                key.frame - rt.decoder->NextIndex() <= kWalkLimit) {
                openOriginal();
                tDecoder = Clock::now();
                path = "walk";
                if (!DecodeTo(*rt.decoder, key, kWalkLimit, kept, missing)) result.aborted = true;
                tDecoded = Clock::now();
                done = true;
            }
        }
        if (!done && !src.proxy.empty()) {
            // a free decoder of the pool, the one standing at the target first (the next frame of a walk: no seek)
            SourceRuntime::ProxySlot* slot = nullptr;
            std::unique_lock<std::mutex> held;
            for (auto& s : rt.proxies) {
                std::unique_lock<std::mutex> l(s.mutex, std::try_to_lock);
                if (!l.owns_lock()) continue;
                const bool positioned = s.decoder && s.opened == src.proxy && !s.decoder->Drained() && s.decoder->PositionKnown() &&
                                        s.decoder->NextIndex() == key.frame;
                if (!slot || positioned) {
                    slot = &s;
                    held = std::move(l);  // releases the slot taken before
                }
                if (positioned) break;
            }
            if (!slot) {
                slot = &rt.proxies[static_cast<size_t>(key.frame) % rt.proxies.size()];
                held = std::unique_lock<std::mutex>(slot->mutex);
            }
            if (!slot->decoder || slot->opened != src.proxy) {
                VideoDecoder::Options po;
                po.probe = true;  // logged at debug level: a pool of decoders, not news
                po.sliceThreadsOnly = true;
                po.indexRate = src.fps;  // the original's rate: the proxy's container may guess another
                slot->decoder = std::make_unique<VideoDecoder>(src.proxy, po);
                slot->opened = src.proxy;
            }
            if (!src.hasColor) {
                std::lock_guard<std::mutex> g2(rt.mutex);
                openOriginal();
            }
            ++rt.proxyLoads;
            tDecoder = Clock::now();
            path = "proxy";
            if (!DecodeTo(*slot->decoder, key, 1, kept, missing)) result.aborted = true;
            tDecoded = Clock::now();
            done = true;
        }
        if (!done) {
            std::lock_guard<std::mutex> guard(rt.mutex);  // one decoder per video: its frames decode sequentially
            openOriginal();
            tDecoder = Clock::now();
            if (!DecodeTo(*rt.decoder, key, kWalkLimit, kept, missing)) result.aborted = true;
            tDecoded = Clock::now();
        }
    }
    if (result.aborted) {
        ++rt.aborted;
        return result;
    }
    // the target first, published as soon as it is converted; then the frames kept on the way, nearest first, each
    // only while it is still wanted (a stale neighbour must not take the CPU from the next jump)
    std::stable_sort(kept.begin(), kept.end(), [&](const CpuFrame& a, const CpuFrame& b) {
        if ((a.index == key.frame) != (b.index == key.frame)) return a.index == key.frame;
        return std::llabs(a.index - key.frame) < std::llabs(b.index - key.frame);
    });
    for (CpuFrame& f : kept) {
        const Key k{f.index, key.source};
        const bool target = f.index == key.frame;
        bool use = false;
        {
            // decided and dropped in one critical section: SetCurrentTime skips a key in kept_ (or the target, in flight),
            // so a drop the time came back to meanwhile would lose it; dropped, the target is re-queued by LoaderMain
            std::lock_guard<std::mutex> lock(mutex_);
            bool done = entries_.count(k) || missing_.count(k);
            for (const auto& l : loaded_) done = done || (l.key.frame == k.frame && l.key.source == k.source);
            const bool needed = std::find(needed_.begin(), needed_.end(), k.source) != needed_.end();
            // the target of a jump the time already left is still a stand-in for Update; a neighbour only in the window
            use = needed && !done && (target || InWindow(k));
            if (!use) {
                kept_.erase(k);
                if (target) result.aborted = true;
            }
        }
        if (!use) continue;
        Loaded l;
        l.key = k;
        const auto tc = Clock::now();
        l.image = YuvToRgba16f(f, color);
        l.kind = TextureKind::Color;
        Prepare(l.image, l.kind, l.minValue, l.maxValue, l.maxMagnitude);
        if (target && Log()->should_log(spdlog::level::debug)) {
            const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            Log()->debug("viewport load {} {} via {}: wait {:.1f} ms, decode {:.1f} ms ({} kept), convert {:.1f} ms", key.source, key.frame, path,
                         ms(tStart, tDecoder), ms(tDecoder, tDecoded), kept.size(), ms(tc, Clock::now()));
        }
        Publish(std::move(l));
    }
    cleanup.armed = false;
    if (result.aborted) ++rt.aborted;
    if (missing) {
        Loaded l;
        l.key = key;
        l.missing = true;
        result.loaded.push_back(std::move(l));
    }
    return result;
}

// ---- render thread ----------------------------------------------------------------------------------

bool FrameStore::CurrentPendingLocked() const {
    for (const auto& name : needed_) {
        const Key k{KeyFrame(name, currentTime_), name};
        if (entries_.count(k) || missing_.count(k)) continue;
        if (!inFlight_.count(k) && !kept_.count(k)) continue;  // queued or not loadable: a loader takes it first anyway
        bool published = false;
        for (const auto& l : loaded_) published = published || (l.key.frame == k.frame && l.key.source == k.source);
        if (!published) return true;
    }
    return false;
}

bool FrameStore::IsCurrentLocked(const Key& key) const {
    return std::find(needed_.begin(), needed_.end(), key.source) != needed_.end() && key.frame == KeyFrame(key.source, currentTime_);
}

bool FrameStore::Update() {
    std::vector<Loaded> batch;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 1) frames of the current time and missing marks: alone, so the target of a jump is not held back
        for (auto it = loaded_.begin(); it != loaded_.end();) {
            if (it->missing || IsCurrentLocked(it->key)) {
                batch.push_back(std::move(*it));
                it = loaded_.erase(it);
            } else {
                ++it;
            }
        }
        bool currentUploaded = false;
        for (const auto& l : batch) currentUploaded = currentUploaded || !l.missing;
        // 2) otherwise up to uploadsPerUpdate others: the window first; a frame outside it only as a better stand-in
        //    than the resident one (fast drag without a proxy: the screen follows), the rest is dropped
        if (!currentUploaded) {
            int n = 0;
            for (auto it = loaded_.begin(); it != loaded_.end() && n < options_.uploadsPerUpdate;) {
                const Key& k = it->key;
                const bool needed = std::find(needed_.begin(), needed_.end(), k.source) != needed_.end();
                bool take = needed && InWindow(k);
                if (!take && needed) {
                    const int64_t target = FrameAt(k.source, currentTime_);
                    const auto held = NearestResident(target, k.source);
                    take = held == entries_.end() || std::llabs(k.frame - target) < std::llabs(held->first.frame - target);
                    if (take)  // the closest of the finished ones of this source only
                        for (const auto& o : loaded_)
                            if (&o != &*it && !o.missing && o.key.source == k.source && std::llabs(o.key.frame - target) < std::llabs(k.frame - target)) take = false;
                }
                if (take) {
                    batch.push_back(std::move(*it));
                    it = loaded_.erase(it);
                    ++n;
                } else if (!needed || !InWindow(k)) {
                    queued_.erase(k);
                    it = loaded_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    if (batch.empty()) return false;
    const auto tUpload = std::chrono::steady_clock::now();
    // one command list and one wait for the whole batch (the textures and upload buffers were made by the loaders)
    std::vector<D3D12Device::StagedTexture*> copies;
    for (Loaded& l : batch)
        if (!l.missing && l.staged.texture) copies.push_back(&l.staged);
    const auto tCopy = std::chrono::steady_clock::now();
    device_.CopyStagedTextures(copies);
    if (Log()->should_log(spdlog::level::debug) && !copies.empty())
        Log()->debug("viewport copy {} texture(s): {:.1f} ms", copies.size(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tCopy).count());
    for (Loaded& l : batch) {
        if (l.missing) {
            std::lock_guard<std::mutex> lock(mutex_);
            missing_.insert(l.key);
            queued_.erase(l.key);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (entries_.count(l.key) || !l.staged.texture) {  // already resident (loaded twice): keep the first texture
                queued_.erase(l.key);
                continue;
            }
        }
        Entry e;
        e.texture = l.staged.texture;
        e.desc.texture = e.texture.Get();
        e.desc.kind = l.kind;
        e.desc.width = l.image.width;
        e.desc.height = l.image.height;
        e.desc.minValue = l.minValue;
        e.desc.maxValue = l.maxValue;
        e.desc.maxMagnitude = l.maxMagnitude;
        e.bytes = l.bytes;
        std::lock_guard<std::mutex> lock(mutex_);
        e.lastUse = ++useCounter_;
        residentBytes_ += e.bytes;
        entries_[l.key] = std::move(e);
        queued_.erase(l.key);
    }
    if (Log()->should_log(spdlog::level::debug))
        Log()->debug("viewport upload {} frame(s) in {:.1f} ms", batch.size(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tUpload).count());
    Evict();
    return true;
}

void FrameStore::Evict() {
    std::lock_guard<std::mutex> lock(mutex_);
    // a static source no longer shown (PR-8: a mask version compared once): its one frame is «in the window» and «current» at any time,
    // so it would stay resident until the next SetSources - it goes as a frame outside the window does
    auto idleStatic = [&](const Key& k) {
        const ViewportSource* s = FindSource(k.source);
        return s && s->isStatic && std::find(needed_.begin(), needed_.end(), k.source) == needed_.end();
    };
    while (residentBytes_ > options_.vramBudgetBytes && !entries_.empty()) {
        // least recently used entry outside the prefetch window; if all are inside, the oldest anyway
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (InWindow(it->first) && !idleStatic(it->first)) continue;
            if (victim == entries_.end() || it->second.lastUse < victim->second.lastUse) victim = it;
        }
        if (victim == entries_.end()) {
            for (auto it = entries_.begin(); it != entries_.end(); ++it)
                if ((it->first.frame != KeyFrame(it->first.source, currentTime_) || idleStatic(it->first)) &&
                    (victim == entries_.end() || it->second.lastUse < victim->second.lastUse))
                    victim = it;
        }
        if (victim == entries_.end()) break;
        residentBytes_ -= victim->second.bytes;
        if (freeTextures_.size() < 8) freeTextures_.push_back(victim->second.texture);  // the next frames reuse it
        entries_.erase(victim);
    }
}

void FrameStore::WaitForCurrent(uint32_t timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        Update();
        std::unique_lock<std::mutex> lock(mutex_);
        bool pending = false;
        for (const auto& name : needed_) {
            const Key k{KeyFrame(name, currentTime_), name};
            const ViewportSource* s = FindSource(name);
            if (!s || !s->HasFrame(k.frame) || missing_.count(k) || entries_.count(k)) continue;
            pending = true;
        }
        if (!pending) return;
        if (std::chrono::steady_clock::now() > deadline) Throw("FrameStore::WaitForCurrent timed out");
        // woken by Publish (TASK-0040: sleep_for(1 ms) is a 15.6 ms timer tick on Windows and quantised the benchmark)
        loadedCv_.wait_for(lock, std::chrono::milliseconds(5), [&] { return stop_ || !loaded_.empty(); });
    }
}

void FrameStore::SetProxy(const std::string& source, const std::filesystem::path& proxy) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : sources_)
        if (s.name == source && s.isVideo) s.proxy = proxy;  // the loaders open it lazily; resident frames stay (the same pixels)
}

void FrameStore::SetOnCurrentLoaded(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    onCurrentLoaded_ = std::move(callback);
}

FrameStore::SourceStats FrameStore::Stats(const std::string& source) const {
    SourceStats st;
    SourceRuntime* rt = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = runtimes_.find(source);
        if (it == runtimes_.end()) return st;
        rt = it->second.get();
    }
    {
        std::lock_guard<std::mutex> g(rt->mutex);
        if (rt->decoder) {
            st.decodedFrames += rt->decoder->FramesDecoded();
            st.seeks += rt->decoder->Seeks();
        }
    }
    for (auto& slot : rt->proxies) {
        std::lock_guard<std::mutex> g(slot.mutex);
        if (slot.decoder) {
            st.decodedFrames += slot.decoder->FramesDecoded();
            st.seeks += slot.decoder->Seeks();
        }
    }
    st.aborted = rt->aborted.load();
    st.proxyLoads = rt->proxyLoads.load();
    return st;
}

}  // namespace dlssvid
