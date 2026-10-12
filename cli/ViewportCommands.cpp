#include "ViewportCommands.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <vector>

#include "gpu/D3D12Device.h"
#include "passes/formats/PngIO.h"
#include "util/Error.h"
#include "util/Log.h"
#include "viewport/FrameStore.h"
#include "viewport/Project.h"
#include "viewport/ViewportRenderer.h"
#include "viewport/ViewportProxy.h"
#include "viewport/ViewportState.h"

namespace dlssvid::cli {

namespace {

std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}

bool ParseSize(const std::string& s, uint32_t& w, uint32_t& h) {
    if (s.empty()) return false;
    const size_t x = s.find('x');
    if (x == std::string::npos) Throw("--size expects WxH (got '" + s + "')");
    w = static_cast<uint32_t>(std::stoul(s.substr(0, x)));
    h = static_cast<uint32_t>(std::stoul(s.substr(x + 1)));
    if (!w || !h) Throw("--size must be positive");
    return true;
}

// "name[:display[:opacity[:blend]]]"
void ApplyLayerSpec(LayerState& l, const std::string& spec) {
    const auto parts = Split(spec, ':');
    if (parts.empty() || parts[0].empty()) Throw("--layers: empty layer name in '" + spec + "'");
    l.source = parts[0];
    l.display = LayerState::DefaultDisplayFor(l.source);
    if (parts.size() > 1 && !parts[1].empty()) {
        const auto d = ParseDisplayMode(parts[1]);
        if (!d) Throw("--layers: unknown display '" + parts[1] + "'");
        l.display = *d;
    }
    if (parts.size() > 2 && !parts[2].empty()) l.opacity = std::clamp(std::stof(parts[2]), 0.f, 1.f);
    if (parts.size() > 3 && !parts[3].empty()) {
        const auto b = ParseBlendMode(parts[3]);
        if (!b) Throw("--layers: unknown blend '" + parts[3] + "'");
        l.blend = *b;
    }
}

void ApplyOverrides(ViewportState& st, const ViewportCommands::RenderArgs& a) {
    if (!a.mode.empty()) {
        const auto m = ParseViewMode(a.mode);
        if (!m) Throw("--mode must be single | overlay | grid");
        st.mode = *m;
    }
    if (!a.source.empty()) {
        st.SetSingleSource(a.source);
        if (st.mode == ViewMode::Grid && a.mode.empty()) st.mode = ViewMode::Single;
    }
    if (!a.layers.empty()) {
        const auto specs = Split(a.layers, ',');
        st.layers.clear();
        for (const auto& spec : specs) {
            LayerState l;
            ApplyLayerSpec(l, spec);
            if (st.layers.size() >= static_cast<size_t>(ViewportState::kMaxLayers)) Throw("--layers: at most 5 layers");
            st.layers.push_back(l);
        }
        if (st.layers.empty()) st.layers.push_back(LayerState{});
        if (a.mode.empty()) st.mode = st.layers.size() > 1 ? ViewMode::Overlay : ViewMode::Single;
    }
    if (!a.sources.empty()) {
        const auto names = Split(a.sources, ',');
        if (names.size() != 4) Throw("--sources expects four comma-separated names (grid cells)");
        for (size_t i = 0; i < 4; ++i) st.gridSources[i] = names[i];
        if (a.mode.empty()) st.mode = ViewMode::Grid;
    }
    if (!a.display.empty()) {
        const auto d = ParseDisplayMode(a.display);
        if (!d) Throw("--display: unknown display mode '" + a.display + "'");
        const int sel = std::clamp(st.selectedLayer, 0, static_cast<int>(st.layers.size()) - 1);
        st.layers[static_cast<size_t>(sel)].display = *d;
    }
    if (a.expand >= 0) st.expandedCell = std::min(a.expand, 3);
    if (a.zoom > 0.f) {
        st.view.zoom = std::clamp(a.zoom, kMinZoom, kMaxZoom);
        if (a.centerX >= 0.f) st.view.centerX = a.centerX;
        if (a.centerY >= 0.f) st.view.centerY = a.centerY;
    }
    if (!a.wipe.empty()) {
        if (a.wipe == "off") {
            st.wipe.enabled = false;
        } else {
            const auto parts = Split(a.wipe, ':');
            if (parts[0] != "v" && parts[0] != "h") Throw("--wipe expects off | v[:pos[:a:b]] | h[:pos[:a:b]]");
            st.wipe.enabled = true;
            st.wipe.vertical = parts[0] == "v";
            if (parts.size() > 1 && !parts[1].empty()) st.wipe.position = std::clamp(std::stof(parts[1]), 0.f, 1.f);
            if (parts.size() > 3) {
                st.wipe.layerA = std::stoi(parts[2]);
                st.wipe.layerB = std::stoi(parts[3]);
            }
        }
    }
    if (a.frame >= 0) st.legacyFrame = a.frame;  // a base-rate frame: becomes a time once the sources (and the rate) are known
    if (a.time >= 0) {
        st.time = a.time;
        st.legacyFrame = -1;
    }
}

int CmdProjectInit(const ViewportCommands::ProjectArgs& a) {
    if (a.input.empty()) Throw("project init: -i/--input is required");
    const std::filesystem::path video = std::filesystem::absolute(a.input);
    if (!std::filesystem::exists(video)) Throw("project init: no such file " + video.string());
    Project p = Project::Create(video, a.passes.empty() ? std::filesystem::path{} : std::filesystem::absolute(a.passes));
    if (!a.result.empty()) p.resultVideo = std::filesystem::absolute(a.result);
    const std::filesystem::path out = a.output.empty() ? video.parent_path() / (video.stem().string() + ".dlssvid.json") : std::filesystem::path(a.output);
    p.Save(out);
    std::printf("project:  %s\n", p.file.string().c_str());
    std::printf("source:   %s\n", p.sourceVideo.string().c_str());
    std::printf("passes:   %s\n", p.passesRoot.string().c_str());
    if (!p.resultVideo.empty()) std::printf("result:   %s\n", p.resultVideo.string().c_str());
    for (const auto& s : FrameStore::DiscoverPasses(p.passesRoot))
        std::printf("pass:     %s  %ux%u  frames %lld..%lld\n", s.name.c_str(), s.width, s.height, static_cast<long long>(s.firstFrame),
                    static_cast<long long>(s.lastFrame));
    return 0;
}

int CmdProjectShow(const ViewportCommands::ProjectArgs& a) {
    if (a.project.empty()) Throw("project show: --project is required");
    const Project p = Project::Load(a.project);
    std::printf("%s\n", p.ToJson(p.file.parent_path()).dump(2).c_str());
    for (const auto& s : p.Sources())
        std::printf("source:   %-12s %s  %ux%u  frames %lld..%lld  %g fps%s\n", s.name.c_str(), s.path.string().c_str(), s.width, s.height,
                    static_cast<long long>(s.firstFrame), static_cast<long long>(s.lastFrame), s.fps.num > 0 ? s.fps.ToDouble() : 0.0,
                    s.isVideo ? "  (video)" : "");
    return 0;
}

// The scrubbing proxies of the project's videos that are missing or stale and needed (long keyframe intervals).
void BuildMissingProxies(const Project& project, const std::vector<std::string>& names, bool force, double maxGb = 0.0) {
    if (project.passesRoot.empty()) Throw("proxy: the project has no passes folder (the proxy lives in <passes>/.proxy)");
    for (const auto& name : names) {
        if (name != "source" && name != "result") Throw("proxy: unknown source '" + name + "' (source | result)");
        const std::filesystem::path video = name == "source" ? project.sourceVideo : project.resultVideo;
        if (video.empty() || !std::filesystem::exists(video)) {
            std::printf("%s: no video\n", name.c_str());
            continue;
        }
        if (!force)
            if (const auto valid = ValidProxy(video, project.passesRoot, name)) {
                std::printf("%s: proxy up to date: %s\n", name.c_str(), valid->string().c_str());
                continue;
            }
        const ProxyPlan plan = PlanProxy(video);
        if (!plan.needed && !force) {
            std::printf("%s: no proxy needed (%s)\n", name.c_str(), plan.reason.c_str());
            continue;
        }
        if (maxGb > 0) {  // the disk policy of the GUI's automatic build: a proxy never fills the disk unasked
            std::error_code ec;
            std::filesystem::path probe = project.passesRoot;
            while (!probe.empty() && !std::filesystem::exists(probe, ec)) probe = probe.parent_path();
            const uint64_t free = probe.empty() ? 0 : std::filesystem::space(probe, ec).available;
            const double gb = static_cast<double>(plan.estimatedBytes) / 1e9;
            if (gb > maxGb || (free > 0 && plan.estimatedBytes > free / 4)) {
                std::printf("%s: proxy not built automatically: ~%.1f GB (limit %.1f GB, free %.1f GB); `dlssvid proxy` builds it on request\n", name.c_str(), gb,
                            maxGb, static_cast<double>(free) / 1e9);
                continue;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        ProxyBuildOptions o;
        o.progress = [&](int64_t done, int64_t total) {
            if (done % 25 == 0 || done == total) {
                std::printf("proxy %s: %lld/%lld frames\n", name.c_str(), static_cast<long long>(done), static_cast<long long>(total > 0 ? total : plan.frames));
                std::fflush(stdout);
            }
        };
        const auto path = BuildProxy(video, project.passesRoot, name, o);
        std::printf("%s: proxy %s (%.1f MB) in %.1f s\n", name.c_str(), path.string().c_str(), static_cast<double>(std::filesystem::file_size(path)) / 1e6,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

int CmdRender(const ViewportCommands::RenderArgs& a) {
    Project project;
    if (!a.project.empty()) {
        project = Project::Load(a.project);
    } else if (!a.input.empty()) {
        project = Project::Create(std::filesystem::absolute(a.input), a.passes.empty() ? std::filesystem::path{} : std::filesystem::absolute(a.passes));
    } else {
        Throw("render: --project or --input is required");
    }
    if (!a.passes.empty()) project.passesRoot = std::filesystem::absolute(a.passes);
    if (!a.result.empty()) project.resultVideo = std::filesystem::absolute(a.result);
    ViewportState& st = project.viewport;
    ApplyOverrides(st, a);

    if (a.proxy != "auto" && a.proxy != "off" && a.proxy != "build") Throw("render: --proxy must be auto | off | build");
    if (a.proxy == "build") BuildMissingProxies(project, {"source", "result"}, false);
    const bool useProxies = a.proxy != "off";
    D3D12Device device({a.warp, false});
    FrameStore store(device);
    store.SetSources(project.Sources(useProxies));
    if (store.Sources().empty()) Throw("render: no sources (video not readable and no pass folders under " + project.passesRoot.string() + ")");
    st.ResolveLegacyFrame(store.BaseFps().ToDouble());
    const std::vector<std::string> needed = st.NeededSources();
    for (const auto& n : needed)
        if (!store.FindSource(n)) Log()->warn("render: source '{}' is not available (shown empty)", n);
    uint32_t tw = 0, th = 0;
    if (!ParseSize(a.size, tw, th)) {
        tw = store.ImageWidth();
        th = store.ImageHeight();
    }
    ViewportRenderer renderer(device);
    auto renderAt = [&](double t) {
        store.SetCurrentTime(t, needed);
        store.WaitForCurrent();
        FrameStates states;  // after the wait a source is either on the GPU or has no frame here
        for (const auto& n : needed) states[n] = store.StatusAt(t, n) == FrameStore::Status::Ready ? FrameState::Ready : FrameState::Missing;
        return renderer.RenderToImage(st, store.TexturesAt(t), store.ImageWidth(), store.ImageHeight(), tw, th, &states);
    };

    if (a.bench > 0) {
        const int64_t n = store.FrameCount() > 0 ? std::min<int64_t>(a.bench, store.FrameCount()) : a.bench;
        if (a.benchMode != "all" && a.benchMode != "cold" && a.benchMode != "warm" && a.benchMode != "random")
            Throw("render: --bench-mode must be all | cold | warm | random");
        const double rate = store.BaseFps().ToDouble();
        // one jump: SetCurrentTime -> the exact frames of every needed source on the GPU -> the composite done (the
        // latency the viewport sees; TexturesAt by time, so sources of other rates show their own frame)
        auto jump = [&](int64_t f) {
            const double t = static_cast<double>(f) / rate;
            const auto t0 = std::chrono::steady_clock::now();
            store.SetCurrentTime(t, needed);
            store.WaitForCurrent();
            FrameStates states;
            for (const auto& nm : needed) states[nm] = store.StatusAt(t, nm) == FrameStore::Status::Ready ? FrameState::Ready : FrameState::Missing;
            renderer.RenderOffscreen(st, store.TexturesAt(t), store.ImageWidth(), store.ImageHeight(), tw, th, &states);
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        auto percentile = [](std::vector<double> v, double p) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[static_cast<size_t>(std::min<double>(static_cast<double>(v.size() - 1), std::ceil(p * static_cast<double>(v.size())) - 1))];
        };
        nlohmann::json report = {{"clip", project.sourceVideo.filename().string()},
                                 {"image", {store.ImageWidth(), store.ImageHeight()}},
                                 {"target", {tw, th}},
                                 {"mode", std::string(ToString(st.mode))},
                                 {"sources", needed},
                                 {"adapter", device.AdapterName()},
                                 {"loader_threads", store.LoaderThreads()},
                                 {"frames", n},
                                 {"proxy", nlohmann::json::object()},
                                 {"passes", nlohmann::json::object()}};
        for (const auto& s : store.Sources())
            if (s.isVideo) report["proxy"][s.name] = !s.proxy.empty();
        std::printf("bench: %lld frames, image %ux%u -> target %ux%u, mode %s, %s, %d loader threads, proxy %s\n", static_cast<long long>(n), store.ImageWidth(),
                    store.ImageHeight(), tw, th, std::string(ToString(st.mode)).c_str(), device.AdapterName().c_str(), store.LoaderThreads(),
                    report["proxy"].dump().c_str());
        auto sequential = [&](const char* key, const char* label) {
            std::vector<double> ms;
            for (int64_t i = 0; i < n; ++i) ms.push_back(jump(i));
            double sum = 0;
            for (double v : ms) sum += v;
            const double mean = sum / static_cast<double>(n);
            std::printf("%-46s %7.2f ms/frame = %6.1f fps   (composite GPU %.2f ms)\n", label, mean, 1000.0 / mean, renderer.LastRenderMs());
            report["passes"][key] = {{"ms_per_frame", mean}, {"fps", 1000.0 / mean}, {"p90_ms", percentile(ms, 0.9)}};
        };
        if (a.benchMode == "all" || a.benchMode == "cold") sequential("cold", "cold scrub (decode/read + upload + composite):");
        if (a.benchMode == "all" || a.benchMode == "warm") sequential("warm", "warm scrub (resident frames, composite only):");
        if (a.benchMode == "all" || a.benchMode == "random") {
            store.SetSources(project.Sources(useProxies));  // drop the cache: random jumps hit the decoder / disk
            std::mt19937_64 rng(7);
            const int64_t last = std::max<int64_t>(1, store.FrameCount() > 0 ? store.FrameCount() : n) - 1;
            std::uniform_int_distribution<int64_t> dist(0, last);
            auto resident = [&](int64_t f) {  // a hit: some needed source already has the frame on the GPU or loading
                const double t = static_cast<double>(f) / rate;
                for (const auto& nm : needed) {
                    const auto s = store.StatusAt(t, nm);
                    if (s == FrameStore::Status::Ready || s == FrameStore::Status::Loading) return true;
                }
                return false;
            };
            for (int i = 0; i < 5; ++i) jump(dist(rng));  // warm-up: decoders opened, first allocations done
            std::map<std::string, FrameStore::SourceStats> before;
            for (const auto& nm : needed) before[nm] = store.Stats(nm);
            // TASK-0040: only misses are timed - a jump into a resident frame is a cache hit, not a jump
            std::vector<double> ms;
            int64_t hits = 0;
            for (int64_t i = 0; i < n; ++i) {
                int64_t f = dist(rng);
                for (int tries = 0; tries < 50 && resident(f); ++tries, ++hits) f = dist(rng);
                ms.push_back(jump(f));
            }
            double sum = 0, worst = 0;
            for (double v : ms) {
                sum += v;
                worst = std::max(worst, v);
            }
            const double mean = sum / static_cast<double>(ms.size()), p50 = percentile(ms, 0.5), p90 = percentile(ms, 0.9);
            int64_t decoded = 0, seeks = 0, proxyLoads = 0, aborted = 0;
            for (const auto& nm : needed) {
                const auto now = store.Stats(nm);
                decoded += now.decodedFrames - before[nm].decodedFrames;
                seeks += now.seeks - before[nm].seeks;
                proxyLoads += now.proxyLoads - before[nm].proxyLoads;
                aborted += now.aborted - before[nm].aborted;
            }
            std::printf("%-46s %7.2f ms/frame = %6.1f fps   (misses %lld: p50 %.2f ms, p90 %.2f ms = %.1f fps, max %.1f ms; skipped hits %lld)\n",
                        "random jumps (seek + decode/read + composite):", mean, 1000.0 / mean, static_cast<long long>(ms.size()), p50, p90, 1000.0 / p90, worst,
                        static_cast<long long>(hits));
            std::printf("  per jump: %.1f frames decoded, %.2f seeks, %lld served by the proxy, %lld loads dropped\n",
                        static_cast<double>(decoded) / static_cast<double>(ms.size()), static_cast<double>(seeks) / static_cast<double>(ms.size()),
                        static_cast<long long>(proxyLoads), static_cast<long long>(aborted));
            report["passes"]["random"] = {{"misses", ms.size()}, {"skipped_hits", hits},   {"mean_ms", mean},
                                          {"p50_ms", p50},       {"p90_ms", p90},          {"max_ms", worst},
                                          {"fps_p90", 1000.0 / p90}, {"decoded_per_jump", static_cast<double>(decoded) / static_cast<double>(ms.size())},
                                          {"seeks_per_jump", static_cast<double>(seeks) / static_cast<double>(ms.size())},
                                          {"proxy_loads", proxyLoads}, {"dropped_loads", aborted}};
        }
        if (!a.benchJson.empty()) {
            std::filesystem::create_directories(std::filesystem::absolute(a.benchJson).parent_path());
            std::ofstream(a.benchJson) << report.dump(2) << "\n";
        }
        return 0;
    }

    if (store.Duration() > 0 && st.time > store.Duration() + 1e-9)
        Throw("render: time " + std::to_string(st.time) + " s is outside 0.." + std::to_string(store.Duration()) + " s (" + std::to_string(store.FrameCount()) +
              " frames at the base rate)");
    const auto t0 = std::chrono::steady_clock::now();
    const PassImage img = renderAt(st.time);
    bool any = false;
    for (const auto& n : needed) any = any || store.StatusAt(st.time, n) == FrameStore::Status::Ready;
    if (!any) Throw("render: " + FormatTimecode(st.time) + " has none of the requested sources");
    if (!a.output.empty()) {
        std::filesystem::create_directories(std::filesystem::absolute(a.output).parent_path());
        WritePng(a.output, img);
    }
    if (a.saveState && !project.file.empty()) project.Save();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("rendered %s (frame %lld at %g fps; %s, %ux%u) in %.1f ms%s%s\n", FormatTimecode(st.time).c_str(), static_cast<long long>(store.CurrentFrame()),
                store.BaseFps().ToDouble(), std::string(ToString(st.mode)).c_str(), tw, th, ms, a.output.empty() ? "" : " -> ", a.output.c_str());
    return 0;
}

int CmdProxy(const ViewportCommands::ProxyArgs& a) {
    if (a.background) SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);  // the GUI keeps the CPU for the viewport
    Project project;
    if (!a.project.empty()) {
        project = Project::Load(a.project);
    } else if (!a.input.empty()) {
        project = Project::Create(std::filesystem::absolute(a.input), a.passes.empty() ? std::filesystem::path{} : std::filesystem::absolute(a.passes));
    } else {
        Throw("proxy: --project or --input is required");
    }
    if (!a.passes.empty()) project.passesRoot = std::filesystem::absolute(a.passes);
    if (!a.result.empty()) project.resultVideo = std::filesystem::absolute(a.result);
    std::vector<std::string> names;
    std::stringstream ss(a.sources);
    for (std::string s; std::getline(ss, s, ',');)
        if (!s.empty()) names.push_back(s);
    if (a.check) {
        for (const auto& name : names) {
            const std::filesystem::path video = name == "source" ? project.sourceVideo : name == "result" ? project.resultVideo : std::filesystem::path{};
            if (video.empty() || !std::filesystem::exists(video)) {
                std::printf("%-8s no video\n", name.c_str());
                continue;
            }
            const ProxyPlan plan = PlanProxy(video);
            const auto valid = ValidProxy(video, project.passesRoot, name);
            std::printf("%-8s %s: %lld frames, keyframes up to %lld apart -> %s; ~%.0f MB; proxy %s\n", name.c_str(), video.filename().string().c_str(),
                        static_cast<long long>(plan.frames), static_cast<long long>(plan.maxKeyframeInterval), plan.needed ? "needs a proxy" : "no proxy needed",
                        static_cast<double>(plan.estimatedBytes) / 1e6, valid ? valid->string().c_str() : "none");
        }
        return 0;
    }
    BuildMissingProxies(project, names, a.force, a.maxGb);
    return 0;
}

}  // namespace

void ViewportCommands::Register(CLI::App& app) {
    project_ = app.add_subcommand("project", "create or inspect a viewport project (*.dlssvid.json)");
    project_->add_option("action", pa_.action, "init | show")->default_val("init");
    project_->add_option("-i,--input", pa_.input, "source video (init)");
    project_->add_option("--passes", pa_.passes, "passes root folder (default: <video>_passes next to the video)");
    project_->add_option("--result", pa_.result, "result video (optional)");
    project_->add_option("-o,--output", pa_.output, "project file to write (default: <video>.dlssvid.json)");
    project_->add_option("--project", pa_.project, "project file (show)");

    render_ = app.add_subcommand("render", "render one viewport frame to PNG (single / overlay / 2x2 grid), or benchmark scrubbing");
    render_->add_option("--project", ra_.project, "project file (*.dlssvid.json)");
    render_->add_option("-i,--input", ra_.input, "source video (instead of a project)");
    render_->add_option("--passes", ra_.passes, "passes root folder");
    render_->add_option("--result", ra_.result, "result video");
    render_->add_option("-o,--output", ra_.output, "output PNG");
    render_->add_option("--frame", ra_.frame, "frame index at the base rate (default: the project's viewport time)");
    render_->add_option("--time", ra_.time, "time in seconds (wins over --frame)");
    render_->add_option("--mode", ra_.mode, "single | overlay | grid");
    render_->add_option("--source", ra_.source, "single mode: source name (source, result, depth_raw, mv_raw, ...)");
    render_->add_option("--layers", ra_.layers, "overlay: base,name[:display[:opacity[:blend]]],...");
    render_->add_option("--sources", ra_.sources, "grid: four comma-separated source names");
    render_->add_option("--display", ra_.display, "color | grayscale | viridis | turbo | mv_hsv | mv_arrows | mv_magnitude | mask_fill | mask_contour");
    render_->add_option("--expand", ra_.expand, "grid: show only this cell (0..3)");
    render_->add_option("--size", ra_.size, "output size WxH (default: image size)");
    render_->add_option("--zoom", ra_.zoom, "zoom factor 0.25..8 (default: fit)");
    render_->add_option("--center-x", ra_.centerX, "image x shown at the centre (with --zoom)");
    render_->add_option("--center-y", ra_.centerY, "image y shown at the centre (with --zoom)");
    render_->add_option("--wipe", ra_.wipe, "off | v[:pos[:layerA:layerB]] | h[:pos[:layerA:layerB]]");
    render_->add_flag("--warp", ra_.warp, "use the WARP software adapter");
    render_->add_option("--bench", ra_.bench, "scrub N frames and print timings instead of writing a PNG");
    render_->add_flag("--save-state", ra_.saveState, "write the resulting viewport state back into the project");
    render_->add_option("--proxy", ra_.proxy, "scrubbing proxy of the videos: auto (use a valid one) | off | build (make the missing ones first)")->default_val("auto");
    render_->add_option("--bench-mode,--bench_mode", ra_.benchMode, "all | cold | warm | random (random: only jumps into frames not resident are timed)")
        ->default_val("all");
    render_->add_option("--bench-json,--bench_json", ra_.benchJson, "write the bench results to this JSON file");

    proxy_ = app.add_subcommand("proxy", "build the scrubbing proxy of the source / result video (FFV1 intra, bit-exact) in <passes>/.proxy");
    proxy_->add_option("--project", xa_.project, "project file (*.dlssvid.json)");
    proxy_->add_option("-i,--input", xa_.input, "source video (instead of a project)");
    proxy_->add_option("--passes", xa_.passes, "passes root folder");
    proxy_->add_option("--result", xa_.result, "result video");
    proxy_->add_option("--sources", xa_.sources, "comma-separated: source, result")->default_val("source,result");
    proxy_->add_flag("--force", xa_.force, "rebuild even a valid proxy");
    proxy_->add_flag("--check", xa_.check, "print what is needed and what exists; build nothing");
    proxy_->add_flag("--background", xa_.background, "below-normal priority (the GUI builds while you work)");
    proxy_->add_option("--max-gb,--max_gb", xa_.maxGb, "skip a proxy estimated above this size or above a quarter of the free disk (0 = no limit)");
}

int ViewportCommands::Dispatch() {
    if (project_ && project_->parsed()) {
        if (pa_.action == "init") return CmdProjectInit(pa_);
        if (pa_.action == "show") return CmdProjectShow(pa_);
        Throw("project: action must be init | show");
    }
    if (render_ && render_->parsed()) return CmdRender(ra_);
    if (proxy_ && proxy_->parsed()) return CmdProxy(xa_);
    return -1;
}

}  // namespace dlssvid::cli
