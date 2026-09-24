#include "stages/upscale/WorkerUpscaler.h"

#include <windows.h>

#include <cstdlib>

#include "passes/formats/NpzIO.h"
#include "util/Error.h"
#include "util/Log.h"
#include "util/Subprocess.h"

namespace dlssvid {

namespace {

std::filesystem::path ExeDir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return n ? std::filesystem::path(buf).parent_path() : std::filesystem::path();
}

// RGBA16F frame -> RGB F16 (H, W, 3) for the .npz exchange, and back.
PassImage ToRgb(const PassImage& rgba) {
    PassImage rgb;
    rgb.Allocate(rgba.width, rgba.height, PixelType::F16, {"R", "G", "B"});
    const uint16_t* s = rgba.As<uint16_t>();
    uint16_t* d = rgb.As<uint16_t>();
    const size_t n = static_cast<size_t>(rgba.width) * rgba.height;
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) d[i * 3 + c] = s[i * 4 + c];
    return rgb;
}

PassImage ToRgba(const PassImage& rgb) {
    PassImage rgba;
    rgba.Allocate(rgb.width, rgb.height, PixelType::F16, {"R", "G", "B", "A"});
    const uint16_t* s = rgb.As<uint16_t>();
    uint16_t* d = rgba.As<uint16_t>();
    const size_t n = static_cast<size_t>(rgb.width) * rgb.height;
    const uint16_t one = 0x3C00;  // 1.0 in fp16
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) d[i * 4 + c] = s[i * 3 + c];
        d[i * 4 + 3] = one;
    }
    return rgba;
}

}  // namespace

WorkerUpscaler::WorkerUpscaler() = default;

WorkerUpscaler::~WorkerUpscaler() {
    try {
        Shutdown();
    } catch (...) {
    }
}

std::filesystem::path WorkerUpscaler::FindProjectRoot() {
    if (const char* e = std::getenv("DLSSVID_PROJECT_ROOT")) return e;
    const auto exe = ExeDir();
    for (const auto& base : {exe, exe.parent_path(), exe.parent_path().parent_path(), exe.parent_path().parent_path().parent_path()})
        if (!base.empty() && std::filesystem::exists(base / "sr_worker" / "worker.py")) return base;
    return std::filesystem::current_path();
}

std::filesystem::path WorkerUpscaler::FindPython(const UpscalerConfig& config, const std::filesystem::path& projectRoot) {
    if (config.extra.contains("python")) return config.extra["python"].get<std::string>();
    if (const char* e = std::getenv("DLSSVID_PYTHON")) return e;
    const auto venv = projectRoot / "models" / "export" / ".venv" / "Scripts" / "python.exe";
    if (std::filesystem::exists(venv)) return venv;
    return "python";
}

UpscalerAvailability WorkerUpscaler::Available() {
    const auto root = FindProjectRoot();
    if (!std::filesystem::exists(root / "sr_worker" / "worker.py"))
        return {false, "sr_worker/worker.py not found under " + root.string() + " (set DLSSVID_PROJECT_ROOT to the source tree)"};
    UpscalerConfig none;
    const auto python = FindPython(none, root);
    if (python == "python" && !std::getenv("DLSSVID_PYTHON"))
        return {false, "no Python for the worker: create models/export/.venv (models/export/README.md) or set DLSSVID_PYTHON"};
    return {true, {}};
}

void WorkerUpscaler::Init(D3D12Device&, const UpscalerConfig& config) {
    Shutdown();
    config_ = config;
    const UpscalerAvailability a = Available();
    if (!a.available) Throw("worker: " + a.reason);
    const auto root = FindProjectRoot();
    const auto worker = root / "sr_worker" / "worker.py";
    const auto python = FindPython(config, root);
    modelId_ = config.extra.value("model", "");
    backend_ = config.extra.value("worker_backend", modelId_ == "stub" ? "stub" : "realbasicvsr");  // --model stub: the worker's test backend
    if (backend_ == "stub") modelId_ = "stub";
    else if (modelId_.empty()) modelId_ = kDefaultModel;
    scratch_ = std::filesystem::temp_directory_path() / ("dlssvid_sr_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(this) & 0xFFFF));
    std::filesystem::create_directories(scratch_);
    proc_ = std::make_unique<Subprocess>();
    proc_->Start({python.string(), worker.string()}, root / "sr_worker");
    nlohmann::json extra = nlohmann::json::object();
    if (config.extra.contains("window")) extra["window"] = config.extra["window"];
    if (config.extra.contains("overlap")) extra["overlap"] = config.extra["overlap"];
    const nlohmann::json init = {{"cmd", "init"},
                                 {"backend", backend_},
                                 {"model", modelId_},
                                 {"fp16", config.extra.value("fp16", false)},
                                 {"models_dir", config.extra.value("models_dir", (root / "models").string())},
                                 {"extra", extra}};
    info_ = Call(init, 10 * 60 * 1000);  // the first run downloads the weights
    window_ = std::max(1, info_.value("window", 15));
    overlap_ = std::clamp(info_.value("overlap", 3), 0, window_ - 1);
    scale_ = info_.value("scale", 4);
    Log()->info("upscale worker ready: backend {} model {} (x{} native, window {} overlap {}) -> {}x{}", backend_, info_.value("model", ""), scale_, window_, overlap_,
                config.outputWidth, config.outputHeight);
}

nlohmann::json WorkerUpscaler::Call(const nlohmann::json& request, uint32_t timeoutMs) {
    if (!proc_) Throw("upscale worker not started");
    proc_->WriteLine(request.dump());
    for (;;) {
        const auto line = proc_->ReadLine(timeoutMs);
        if (!line) {
            const auto code = proc_->ExitCode();
            Throw("upscale worker " + (code ? "exited with code " + std::to_string(*code) : std::string("timed out")) + " during '" + request.value("cmd", "?") + "'");
        }
        if (line->empty() || (*line)[0] != '{') {
            Log()->debug("[sr_worker] {}", *line);
            continue;
        }
        nlohmann::json reply;
        try {
            reply = nlohmann::json::parse(*line);
        } catch (const std::exception&) {
            Log()->debug("[sr_worker] {}", *line);
            continue;
        }
        if (reply.contains("log")) {
            Log()->info("[sr_worker] {}", reply["log"].get<std::string>());
            continue;
        }
        if (!reply.value("ok", false)) Throw("upscale worker error: " + reply.value("error", "unknown"));
        return reply;
    }
}

void WorkerUpscaler::Evaluate(ID3D12GraphicsCommandList*, const UpscaleInputs&, ID3D12Resource*) { Throw("worker: a CPU-side upscaler — the stage calls EvaluateCpuWindow"); }

void WorkerUpscaler::EvaluateCpuWindow(const std::vector<const PassImage*>& rgbaIn, uint32_t targetW, uint32_t targetH, std::vector<PassImage>& rgbaOut) {
    if (rgbaIn.empty()) Throw("WorkerUpscaler: no frames");
    const auto dir = scratch_ / ("w" + std::to_string(seq_++));
    std::filesystem::create_directories(dir);
    nlohmann::json inputs = nlohmann::json::array();
    for (size_t i = 0; i < rgbaIn.size(); ++i) {
        const auto p = dir / ("rgb_" + std::to_string(i) + ".npz");
        WriteNpz(p, {{"rgb", ToRgb(*rgbaIn[i])}});
        inputs.push_back(p.string());
    }
    const nlohmann::json reply = Call({{"cmd", "infer"}, {"inputs", inputs}, {"out_dir", dir.string()}, {"width", targetW}, {"height", targetH}}, 60 * 60 * 1000);
    rgbaOut.clear();
    for (const auto& outPath : reply.value("outputs", nlohmann::json::array())) {
        auto arrays = ReadNpz(outPath.get<std::string>());
        auto it = arrays.find("rgb");
        if (it == arrays.end()) it = arrays.begin();
        if (it == arrays.end()) Throw("worker output has no arrays: " + outPath.get<std::string>());
        PassImage rgb = it->second.type == PixelType::F16 ? std::move(it->second) : it->second.ConvertTo(PixelType::F16);
        if (rgb.ChannelCount() < 3) Throw("worker output is not RGB: " + outPath.get<std::string>());
        rgb.channels = {"R", "G", "B"};
        if (rgb.width != targetW || rgb.height != targetH)
            Throw("worker frame is " + std::to_string(rgb.width) + "x" + std::to_string(rgb.height) + ", the target is " + std::to_string(targetW) + "x" + std::to_string(targetH));
        rgbaOut.push_back(ToRgba(rgb));
    }
    if (rgbaOut.size() != rgbaIn.size()) Throw("worker returned " + std::to_string(rgbaOut.size()) + " frames for " + std::to_string(rgbaIn.size()));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

nlohmann::json WorkerUpscaler::Describe() const {
    nlohmann::json j = info_;
    j["backend"] = "worker";
    j["worker_backend"] = backend_;
    j["runtime"] = "pytorch (sr_worker)";
    return j;
}

void WorkerUpscaler::Shutdown() {
    if (proc_) {
        try {
            if (proc_->Running()) {
                proc_->WriteLine(R"({"cmd":"quit"})");
                proc_->CloseStdin();
                proc_->Wait(5000);
            }
        } catch (...) {
        }
        proc_->Terminate();
        proc_.reset();
    }
    if (!scratch_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(scratch_, ec);
        scratch_.clear();
    }
}

}  // namespace dlssvid
