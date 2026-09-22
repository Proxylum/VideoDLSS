#include "ViewportCommands.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
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

    D3D12Device device({a.warp, false});
    FrameStore store(device);
    store.SetSources(project.Sources());
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
        auto pass = [&](const char* label, const std::function<int64_t(int64_t)>& frameOf) {
            double gpuMs = 0;
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t i = 0; i < n; ++i) {
                const int64_t f = frameOf(i);
                store.SetCurrentFrame(f, needed);
                store.WaitForCurrent();
                renderer.RenderOffscreen(st, store.Textures(f), store.ImageWidth(), store.ImageHeight(), tw, th);
                gpuMs += renderer.LastRenderMs();
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / static_cast<double>(n);
            std::printf("%-46s %7.2f ms/frame = %6.1f fps   (composite GPU %.2f ms)\n", label, ms, 1000.0 / ms, gpuMs / static_cast<double>(n));
        };
        std::printf("bench: %lld frames, image %ux%u -> target %ux%u, mode %s, %s, %d loader threads\n", static_cast<long long>(n), store.ImageWidth(),
                    store.ImageHeight(), tw, th, std::string(ToString(st.mode)).c_str(), device.AdapterName().c_str(), store.LoaderThreads());
        pass("cold scrub (decode/read + upload + composite):", [](int64_t i) { return i; });
        pass("warm scrub (resident frames, composite only):", [](int64_t i) { return i; });
        store.SetSources(project.Sources());  // drop the cache: random jumps hit the decoder / disk
        std::mt19937_64 rng(7);
        const int64_t last = std::max<int64_t>(1, store.FrameCount() > 0 ? store.FrameCount() : n) - 1;
        std::uniform_int_distribution<int64_t> dist(0, last);
        pass("random jumps (seek + decode/read + composite):", [&](int64_t) { return dist(rng); });
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
}

int ViewportCommands::Dispatch() {
    if (project_ && project_->parsed()) {
        if (pa_.action == "init") return CmdProjectInit(pa_);
        if (pa_.action == "show") return CmdProjectShow(pa_);
        Throw("project: action must be init | show");
    }
    if (render_ && render_->parsed()) return CmdRender(ra_);
    return -1;
}

}  // namespace dlssvid::cli
