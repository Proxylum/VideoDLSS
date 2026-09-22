#include "viewport/FrameStore.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

TextureKind KindForManifest(const Manifest& m) {
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
            if (img.type != PixelType::F32 || img.channels.size() != 1) {
                PassImage out;
                out.Allocate(img.width, img.height, PixelType::F32, {"Z"});
                for (uint32_t y = 0; y < img.height; ++y)
                    for (uint32_t x = 0; x < img.width; ++x) out.Set(x, y, 0, img.Get(x, y, 0));
                img = std::move(out);
            }
            bool first = true;
            const float* d = img.As<float>();
            for (size_t i = 0; i < static_cast<size_t>(img.width) * img.height; ++i) {
                if (!std::isfinite(d[i])) continue;
                if (first) {
                    minV = maxV = d[i];
                    first = false;
                } else {
                    minV = std::min(minV, d[i]);
                    maxV = std::max(maxV, d[i]);
                }
            }
            if (first) {
                minV = 0.f;
                maxV = 1.f;
            }
            break;
        }
        case TextureKind::Mv: {
            if (img.type != PixelType::F32 || img.channels.size() != 2) {
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
                const float mag = std::hypot(d[2 * i], d[2 * i + 1]);
                if (std::isfinite(mag)) m = std::max(m, mag);
            }
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
            s.name = s.manifest->pass.empty() ? entry.path().filename().string() : s.manifest->pass;
            s.path = entry.path();
            s.kind = KindForManifest(*s.manifest);
            s.firstFrame = s.manifest->firstFrame;
            s.lastFrame = s.manifest->lastFrame;
            s.width = s.manifest->width;
            s.height = s.manifest->height;
            s.fps = s.manifest->fps;
            out.push_back(std::move(s));
        } catch (const std::exception& e) {
            Log()->warn("pass folder {} skipped: {}", entry.path().string(), e.what());
        }
    }
    std::sort(out.begin(), out.end(), [](const ViewportSource& a, const ViewportSource& b) { return a.name < b.name; });
    return out;
}

std::optional<ViewportSource> FrameStore::VideoSource(const std::string& name, const std::filesystem::path& file) {
    if (file.empty() || !std::filesystem::exists(file)) return std::nullopt;
    try {
        VideoDecoder dec(file);
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
    stop_ = true;
    cv_.notify_all();
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
    for (const auto& s : sources_)
        if (s.lastFrame >= 0) duration_ = std::max(duration_, static_cast<double>(s.lastFrame + 1) / FpsOf(s.name));
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

double FrameStore::MaxFps(const std::vector<std::string>& sources) const {
    double best = BaseFps().ToDouble();
    if (sources.empty()) {
        for (const auto& s : sources_) best = std::max(best, FpsOf(s.name));
    } else {
        for (const auto& n : sources) best = std::max(best, FpsOf(n));
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
        if (s.lastFrame >= 0) last = std::max(last, TimeOfFrame(s.name, s.lastFrame));
    return last;
}

int64_t FrameStore::ExpectedFrames(const std::string& source) const {
    if (sources_.empty()) return 0;
    return FrameAt(source, ReferenceLastTime()) + 1;
}

int FrameStore::SpanOf(const std::string& source) const {
    const double window = static_cast<double>(options_.prefetch) / MaxFps();  // seconds on either side
    return std::max(1, static_cast<int>(std::llround(window * FpsOf(source))));
}

int64_t FrameStore::CurrentFrame() const { return FrameAt("", currentTime_); }

// ---- scheduling -----------------------------------------------------------------------------------

bool FrameStore::InWindow(const Key& key) const { return std::llabs(key.frame - FrameAt(key.source, currentTime_)) <= SpanOf(key.source); }

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
        if (entries_.count(k) || missing_.count(k) || queued_.count(k)) return;
        queue_.push_back(k);
        queued_.insert(k);
    };
    int maxSpan = 0;
    for (const auto& name : needed_) maxSpan = std::max(maxSpan, SpanOf(name));
    for (int d = 0; d <= maxSpan; ++d)
        for (const auto& name : needed_) {
            if (d > SpanOf(name)) continue;
            const int64_t f0 = FrameAt(name, currentTime_);
            push(name, f0 + d);
            if (d > 0 && f0 - d >= 0) push(name, f0 - d);
        }
    cv_.notify_all();
}

void FrameStore::SetCurrentFrame(int64_t frame, const std::vector<std::string>& neededSources) { SetCurrentTime(TimeOfFrame("", frame), neededSources); }

bool FrameStore::Wanted(const Key& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!InWindow(key)) return false;
    if (std::find(needed_.begin(), needed_.end(), key.source) == needed_.end()) return false;
    if (entries_.count(key) || missing_.count(key)) return false;
    for (const auto& l : loaded_)
        if (l.key.frame == key.frame && l.key.source == key.source) return false;
    return true;
}

FrameStore::Status FrameStore::GetStatus(int64_t frame, const std::string& source) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Key k{frame, source};
    if (entries_.count(k)) return Status::Ready;
    if (inFlight_.count(k)) return Status::Loading;
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
        if (key.frame != frame) continue;
        out[key.source] = e.desc;
        const_cast<Entry&>(e).lastUse = ++const_cast<uint64_t&>(useCounter_);
    }
    return out;
}

FrameTextures FrameStore::TexturesAt(double time) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameTextures out;
    for (const auto& s : sources_) {
        const auto it = entries_.find(Key{FrameAt(s.name, time), s.name});
        if (it == entries_.end()) continue;
        out[s.name] = it->second.desc;
        const_cast<Entry&>(it->second).lastUse = ++const_cast<uint64_t&>(useCounter_);
    }
    return out;
}

size_t FrameStore::ResidentCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

// ---- loader threads ---------------------------------------------------------------------------------

void FrameStore::LoaderMain() {
    for (;;) {
        Key key;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            key = queue_.front();
            queue_.pop_front();
            inFlight_.insert(key);
        }
        std::vector<Loaded> results;
        try {
            results = LoadOne(key);
        } catch (const std::exception& e) {
            Log()->warn("frame {} of '{}' failed to load: {}", key.frame, key.source, e.what());
            Loaded l;
            l.key = key;
            l.missing = true;
            results.push_back(std::move(l));
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& l : results) loaded_.push_back(std::move(l));
            inFlight_.erase(key);
        }
        idleCv_.notify_all();
    }
}

std::vector<FrameStore::Loaded> FrameStore::LoadOne(const Key& key) {
    ViewportSource src;
    SourceRuntime* rt = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.count(key)) {  // arrived meanwhile (frame kept on the way to another one)
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
            return {std::move(l)};
        }
        src = *s;
        rt = it->second.get();
    }
    if (src.isVideo) return LoadVideo(key, src, *rt);
    std::vector<Loaded> out;
    out.push_back(LoadPass(key, src, *rt));
    return out;
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
    l.image = reader->ReadFrame(key.frame);
    l.kind = src.kind;
    Prepare(l.image, l.kind, l.minValue, l.maxValue, l.maxMagnitude);
    return l;
}

std::vector<FrameStore::Loaded> FrameStore::LoadVideo(const Key& key, const ViewportSource& src, SourceRuntime& rt) {
    std::vector<Loaded> out;
    std::vector<CpuFrame> kept;  // decoded frames to convert once the decoder is released
    bool missing = false;
    if (!src.HasFrame(key.frame)) {
        missing = true;
    } else {
        std::lock_guard<std::mutex> guard(rt.mutex);  // one decoder per video: its frames decode sequentially
        if (!rt.decoder) {
            rt.decoder = std::make_unique<VideoDecoder>(src.path);
            rt.color = ColorInfoFromStream(rt.decoder->Info());
        }
        VideoDecoder& dec = *rt.decoder;
        std::set<int64_t> seen;
        auto keep = [&](const CpuFrame& f) {
            if (seen.insert(f.index).second) kept.push_back(f);
        };
        // A short distance ahead is decoded through; otherwise seek to the keyframe before the target.
        const int64_t next = dec.NextIndex();
        bool afterSeek = false;
        if (dec.Drained() || key.frame < next || key.frame - next > kWalkLimit) {
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
    }
    for (CpuFrame& f : kept) {
        Loaded l;
        l.key = Key{f.index, key.source};
        l.image = Yuv420pToRgba16f(f, rt.color);
        l.kind = TextureKind::Color;
        Prepare(l.image, l.kind, l.minValue, l.maxValue, l.maxMagnitude);
        out.push_back(std::move(l));
    }
    if (missing) {
        Loaded l;
        l.key = key;
        l.missing = true;
        out.push_back(std::move(l));
    }
    return out;
}

// ---- render thread ----------------------------------------------------------------------------------

bool FrameStore::Update() {
    std::vector<Loaded> batch;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < options_.uploadsPerUpdate && !loaded_.empty(); ++i) {
            batch.push_back(std::move(loaded_.front()));
            loaded_.pop_front();
        }
    }
    if (batch.empty()) return false;
    for (Loaded& l : batch) {
        if (l.missing) {
            std::lock_guard<std::mutex> lock(mutex_);
            missing_.insert(l.key);
            queued_.erase(l.key);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (entries_.count(l.key)) {  // already resident (loaded twice): keep the first texture
                queued_.erase(l.key);
                continue;
            }
        }
        Entry e;
        e.texture = device_.CreateTexture2D(l.image.width, l.image.height, FormatFor(l.kind));
        device_.UploadTexture2D(e.texture.Get(), l.image.data.data(), l.image.RowBytes());
        e.desc.texture = e.texture.Get();
        e.desc.kind = l.kind;
        e.desc.width = l.image.width;
        e.desc.height = l.image.height;
        e.desc.minValue = l.minValue;
        e.desc.maxValue = l.maxValue;
        e.desc.maxMagnitude = l.maxMagnitude;
        e.bytes = l.image.ByteSize();
        std::lock_guard<std::mutex> lock(mutex_);
        e.lastUse = ++useCounter_;
        residentBytes_ += e.bytes;
        entries_[l.key] = std::move(e);
        queued_.erase(l.key);
    }
    Evict();
    return true;
}

void FrameStore::Evict() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (residentBytes_ > options_.vramBudgetBytes && !entries_.empty()) {
        // least recently used entry outside the prefetch window; if all are inside, the oldest anyway
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (InWindow(it->first)) continue;
            if (victim == entries_.end() || it->second.lastUse < victim->second.lastUse) victim = it;
        }
        if (victim == entries_.end()) {
            for (auto it = entries_.begin(); it != entries_.end(); ++it)
                if (it->first.frame != FrameAt(it->first.source, currentTime_) && (victim == entries_.end() || it->second.lastUse < victim->second.lastUse))
                    victim = it;
        }
        if (victim == entries_.end()) break;
        residentBytes_ -= victim->second.bytes;
        entries_.erase(victim);
    }
}

void FrameStore::WaitForCurrent(uint32_t timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        Update();
        bool pending = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& name : needed_) {
                const Key k{FrameAt(name, currentTime_), name};
                const ViewportSource* s = FindSource(name);
                if (!s || !s->HasFrame(k.frame) || missing_.count(k) || entries_.count(k)) continue;
                pending = true;
            }
        }
        if (!pending) return;
        if (std::chrono::steady_clock::now() > deadline) Throw("FrameStore::WaitForCurrent timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace dlssvid
