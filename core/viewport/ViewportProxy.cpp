#include "viewport/ViewportProxy.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "io/VideoDecoder.h"
#include "io/VideoEncoder.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Sha256.h"

namespace dlssvid {

namespace {

// What ties a proxy to its source: a rewrite in place (a stage writing result.mp4 again) changes at least one of them.
struct SourceStamp {
    uint64_t size = 0;
    int64_t mtime = 0;  // file_time_type ticks (100 ns), full precision
    std::string head, tail;  // sha256 of the first and the last MiB
    bool operator==(const SourceStamp& o) const { return size == o.size && mtime == o.mtime && head == o.head && tail == o.tail; }
    nlohmann::json ToJson() const { return {{"size", size}, {"mtime", mtime}, {"head", head}, {"tail", tail}}; }
    static SourceStamp FromJson(const nlohmann::json& j) {
        SourceStamp s;
        s.size = j.value("size", uint64_t{0});
        s.mtime = j.value("mtime", int64_t{0});
        s.head = j.value("head", "");
        s.tail = j.value("tail", "");
        return s;
    }
};

SourceStamp Stamp(const std::filesystem::path& file) {
    SourceStamp s;
    s.size = std::filesystem::file_size(file);
    s.mtime = static_cast<int64_t>(std::filesystem::last_write_time(file).time_since_epoch().count());
    constexpr size_t kChunk = size_t{1} << 20;
    std::ifstream in(file, std::ios::binary);
    if (!in) Throw("proxy: cannot read " + file.string());
    std::vector<char> buf(static_cast<size_t>(std::min<uint64_t>(kChunk, s.size)));
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    s.head = Sha256Hex(buf.data(), static_cast<size_t>(in.gcount()));
    const uint64_t tailStart = s.size > kChunk ? s.size - kChunk : 0;
    in.clear();
    in.seekg(static_cast<std::streamoff>(tailStart));
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    s.tail = Sha256Hex(buf.data(), static_cast<size_t>(in.gcount()));
    return s;
}

std::filesystem::path SidecarOf(const std::filesystem::path& passesRoot, const std::string& name) {
    return ProxyDir(passesRoot) / (name + ".proxy.json");
}

// FFV1 slices by frame size: the encoder refuses a slice grid finer than the frame allows
std::string SlicesFor(uint32_t w, uint32_t h) {
    const uint64_t px = static_cast<uint64_t>(w) * h;
    return px >= 1280ull * 720 ? "16" : px >= 640ull * 360 ? "4" : "";
}

bool SameFrame(const CpuFrame& a, const CpuFrame& b) { return a.desc == b.desc && a.data == b.data; }

}  // namespace

std::string ProxyCodecFor(const FrameDesc& decoded) {
    for (const char* codec : {"magicyuv", "ffvhuff"})  // the encoder's pixel format must hold the layout exactly (lossless)
        if (!(decoded.HighBitDepth() && std::string(codec) == "magicyuv") && !VideoEncoder::PixelFormatFor(codec, decoded).empty()) return codec;
    return "ffv1";
}

std::filesystem::path ProxyDir(const std::filesystem::path& passesRoot) { return passesRoot / ".proxy"; }

std::filesystem::path ProxyFile(const std::filesystem::path& passesRoot, const std::string& name) { return ProxyDir(passesRoot) / (name + ".mkv"); }

ProxyPlan PlanProxy(const std::filesystem::path& video) {
    ProxyPlan p;
    const KeyframeScan scan = ScanKeyframes(video);
    p.frames = scan.frames;
    p.maxKeyframeInterval = scan.maxInterval;
    VideoDecoder probe(video, VideoDecoder::Probe());
    p.estimatedBytes = static_cast<uint64_t>(static_cast<double>(probe.Info().decoded.ByteSize()) * 0.45 * static_cast<double>(scan.frames));
    p.needed = scan.maxInterval > kProxyMaxGop;
    p.reason = p.needed ? "keyframes up to " + std::to_string(scan.maxInterval) + " frames apart: a jump decodes that many frames"
                        : "keyframes at most " + std::to_string(scan.maxInterval) + " frames apart: jumps are cheap without a proxy";
    return p;
}

std::optional<std::filesystem::path> ValidProxy(const std::filesystem::path& video, const std::filesystem::path& passesRoot, const std::string& name) {
    if (video.empty() || passesRoot.empty()) return std::nullopt;
    const auto file = ProxyFile(passesRoot, name);
    const auto side = SidecarOf(passesRoot, name);
    std::error_code ec;
    if (!std::filesystem::exists(file, ec) || !std::filesystem::exists(side, ec) || !std::filesystem::exists(video, ec)) return std::nullopt;
    try {
        nlohmann::json j;
        std::ifstream(side) >> j;
        if (j.value("format", 0) != kProxyFormat) return std::nullopt;
        if (!(SourceStamp::FromJson(j.value("source", nlohmann::json::object())) == Stamp(video))) return std::nullopt;
        return file;
    } catch (const std::exception& e) {
        Log()->debug("proxy {}: sidecar unreadable: {}", file.string(), e.what());
        return std::nullopt;
    }
}

void RemoveProxy(const std::filesystem::path& passesRoot, const std::string& name) {
    std::error_code ec;
    std::filesystem::remove(ProxyFile(passesRoot, name), ec);
    std::filesystem::remove(SidecarOf(passesRoot, name), ec);
}

std::filesystem::path BuildProxy(const std::filesystem::path& video, const std::filesystem::path& passesRoot, const std::string& name,
                                 const ProxyBuildOptions& options) {
    const SourceStamp before = Stamp(video);
    std::filesystem::create_directories(ProxyDir(passesRoot));
    const auto out = ProxyFile(passesRoot, name);
    const auto part = ProxyDir(passesRoot) / (name + ".building.mkv");  // the extension picks the container
    std::error_code ec;
    std::filesystem::remove(part, ec);
    const auto fail = [&](const std::string& why) -> std::filesystem::path {
        std::error_code e2;
        std::filesystem::remove(part, e2);
        Throw("proxy for " + video.filename().string() + ": " + why);
    };

    VideoDecoder dec(video);
    const VideoStreamInfo info = dec.Info();
    if (info.frameRate.num <= 0 || info.frameRate.den <= 0) return fail("the stream has no frame rate");
    // indices from pts as the viewport sees them after a seek
    if (!dec.SeekToKeyframeBefore(0)) return fail("cannot seek to the first frame");
    ColorTags tags;
    tags.range = info.colorRange;
    tags.matrix = info.colorSpace;
    tags.primaries = info.colorPrimaries;
    tags.transfer = info.colorTrc;
    std::string codec = ProxyCodecFor(info.decoded);
    auto makeEncoder = [&](bool slices) {
        VideoEncoder::Options eo;
        eo.codec = codec;
        eo.frameRate = info.frameRate;
        eo.timeBase = Rational{info.frameRate.den, info.frameRate.num};  // pts = frame index
        eo.color = tags;
        if (codec == "ffv1") {  // intra (every frame a keyframe), slices for a parallel decode
            eo.codecOptions = {{"g", "1"}, {"level", "3"}};
            const std::string s = SlicesFor(info.width, info.height);
            if (slices && !s.empty()) eo.codecOptions["slices"] = s;
        }
        auto enc = std::make_unique<VideoEncoder>(part, info.decoded, eo);
        enc->Open();
        return enc;
    };
    std::unique_ptr<VideoEncoder> enc;
    try {
        enc = makeEncoder(true);
    } catch (const std::exception& e) {
        // FFV1: a slice grid the frame size does not allow; MagicYUV / FFVHuff: a layout they do not take -> FFV1
        Log()->debug("proxy: {} refused ({}), falling back", codec, e.what());
        std::filesystem::remove(part, ec);
        if (codec != "ffv1") codec = "ffv1";
        try {
            enc = makeEncoder(true);
        } catch (const std::exception&) {
            std::filesystem::remove(part, ec);
            enc = makeEncoder(false);
        }
    }

    const int64_t total = info.frameCount > 0 ? info.frameCount : -1;
    std::vector<std::pair<int64_t, CpuFrame>> samples;  // frames checked against the proxy afterwards
    CpuFrame f, last;
    int64_t expected = 0;
    while (dec.NextFrame(f)) {
        if (options.cancel && options.cancel->load()) {
            enc.reset();
            return fail("cancelled");
        }
        if (f.index != expected)
            return fail("frame " + std::to_string(expected) + " came as index " + std::to_string(f.index) + " (variable rate or odd timestamps): no proxy");
        if (f.data.size() < f.desc.ByteSize()) return fail("frame " + std::to_string(f.index) + " has no data");
        f.pts = f.index;
        enc->WriteFrame(f);
        if (expected == 0 || (total > 0 && expected == total / 2)) samples.emplace_back(expected, f);
        last = f;
        ++expected;
        if (options.progress) options.progress(expected, total);
    }
    if (expected == 0) return fail("no frames decoded");
    if (samples.empty() || samples.back().first != last.index) samples.emplace_back(last.index, last);
    enc->Close();
    enc.reset();

    if (!(Stamp(video) == before)) return fail("the source changed while it was read");

    // the proxy decodes to the same frames under the same indices (the original's rate)
    {
        VideoDecoder::Options po;
        po.sliceThreadsOnly = true;
        po.indexRate = info.frameRate;
        VideoDecoder check(part, po);
        for (const auto& [index, frame] : samples) {
            CpuFrame got;
            if (!check.SeekToKeyframeBefore(index) || !check.NextFrame(got)) return fail("frame " + std::to_string(index) + " not found in the proxy");
            if (got.index != index) return fail("frame " + std::to_string(index) + " comes as " + std::to_string(got.index) + " from the proxy");
            if (!SameFrame(got, frame)) return fail("frame " + std::to_string(index) + " differs in the proxy");
        }
    }

    std::filesystem::rename(part, out);
    const nlohmann::json side = {{"format", kProxyFormat},
                                 {"source", before.ToJson()},
                                 {"source_file", video.filename().string()},
                                 {"frames", expected},
                                 {"codec", codec},
                                 {"rate", {info.frameRate.num, info.frameRate.den}},
                                 {"decoded", info.decoded.Name()}};
    const auto sidePath = SidecarOf(passesRoot, name);
    const auto sideTmp = sidePath.string() + ".tmp";
    std::ofstream(sideTmp) << side.dump(2) << "\n";
    std::filesystem::rename(sideTmp, sidePath);
    Log()->info("proxy {}: {} frames, {:.1f} MB ({} intra, bit-exact)", out.string(), expected, static_cast<double>(std::filesystem::file_size(out)) / 1e6, codec);
    return out;
}

}  // namespace dlssvid
