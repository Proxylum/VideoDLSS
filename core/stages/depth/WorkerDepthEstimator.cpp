#include "stages/depth/WorkerDepthEstimator.h"

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
}  // namespace

WorkerDepthEstimator::WorkerDepthEstimator(std::string backend) : backend_(std::move(backend)) {}

WorkerDepthEstimator::~WorkerDepthEstimator() {
    try {
        Shutdown();
    } catch (...) {
    }
}

std::filesystem::path WorkerDepthEstimator::FindProjectRoot() {
    if (const char* e = std::getenv("DLSSVID_PROJECT_ROOT")) return e;
    const auto exe = ExeDir();
    for (const auto& base : {exe, exe.parent_path(), exe.parent_path().parent_path(), exe.parent_path().parent_path().parent_path()}) {
        if (!base.empty() && std::filesystem::exists(base / "depth_worker" / "worker.py")) return base;
    }
    return std::filesystem::current_path();
}

std::filesystem::path WorkerDepthEstimator::FindPython(const DepthEstimatorConfig& config, const std::filesystem::path& projectRoot) {
    if (config.extra.contains("python")) return config.extra["python"].get<std::string>();
    if (const char* e = std::getenv("DLSSVID_PYTHON")) return e;
    const auto venv = projectRoot / "models" / "export" / ".venv" / "Scripts" / "python.exe";
    if (std::filesystem::exists(venv)) return venv;
    return "python";
}

void WorkerDepthEstimator::Init(const DepthEstimatorConfig& config) {
    config_ = config;
    const auto root = FindProjectRoot();
    const auto worker = root / "depth_worker" / "worker.py";
    if (!std::filesystem::exists(worker)) Throw("depth_worker/worker.py not found under " + root.string() + " (set DLSSVID_PROJECT_ROOT)");
    const auto python = FindPython(config, root);
    scratch_ = std::filesystem::temp_directory_path() / ("dlssvid_depth_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(this) & 0xFFFF));
    std::filesystem::create_directories(scratch_);

    proc_ = std::make_unique<Subprocess>();
    proc_->Start({python.string(), worker.string()}, root / "depth_worker");
    nlohmann::json init = {{"cmd", "init"},
                           {"backend", backend_},
                           {"model", config.modelId},
                           {"input_size", config.inputSize},
                           {"max_res", config.maxInputRes},
                           {"fp16", config.fp16},
                           {"models_dir", config.modelsDir.empty() ? (root / "models").string() : config.modelsDir},
                           {"extra", config.extra}};
    info_ = Call(init, 10 * 60 * 1000);  // model download + load can take a while
    metric_ = info_.value("metric", false);
    window_ = info_.value("window", 1);
    overlap_ = info_.value("overlap", 0);
    Log()->info("depth worker ready: backend {} model {} (window {}, metric {})", backend_, info_.value("model", ""), window_, metric_);
}

nlohmann::json WorkerDepthEstimator::Call(const nlohmann::json& request, uint32_t timeoutMs) {
    if (!proc_) Throw("depth worker not started");
    proc_->WriteLine(request.dump());
    for (;;) {
        const auto line = proc_->ReadLine(timeoutMs);
        if (!line) {
            const auto code = proc_->ExitCode();
            Throw("depth worker " + (code ? "exited with code " + std::to_string(*code) : std::string("timed out")) + " during '" + request.value("cmd", "?") + "'");
        }
        if (line->empty() || (*line)[0] != '{') {
            Log()->debug("[worker] {}", *line);
            continue;
        }
        nlohmann::json reply;
        try {
            reply = nlohmann::json::parse(*line);
        } catch (const std::exception&) {
            Log()->debug("[worker] {}", *line);
            continue;
        }
        if (reply.contains("log")) {
            Log()->info("[worker] {}", reply["log"].get<std::string>());
            continue;
        }
        if (!reply.value("ok", false)) Throw("depth worker error: " + reply.value("error", "unknown"));
        return reply;
    }
}

void WorkerDepthEstimator::Estimate(const std::vector<const PassImage*>& rgb, std::vector<PassImage>& depthOut) {
    if (rgb.empty()) Throw("WorkerDepthEstimator::Estimate: no frames");
    const auto dir = scratch_ / ("w" + std::to_string(seq_++));
    std::filesystem::create_directories(dir);
    nlohmann::json inputs = nlohmann::json::array();
    for (size_t i = 0; i < rgb.size(); ++i) {
        const auto p = dir / ("rgb_" + std::to_string(i) + ".npz");
        const PassImage* f = rgb[i];
        WriteNpz(p, {{"rgb", f->type == PixelType::F16 ? *f : f->ConvertTo(PixelType::F16)}});
        inputs.push_back(p.string());
    }
    const nlohmann::json reply = Call({{"cmd", "infer"}, {"inputs", inputs}, {"out_dir", dir.string()}}, 30 * 60 * 1000);
    depthOut.clear();
    for (const auto& outPath : reply.value("outputs", nlohmann::json::array())) {
        auto arrays = ReadNpz(outPath.get<std::string>());
        auto it = arrays.find("depth");
        if (it == arrays.end()) it = arrays.begin();
        if (it == arrays.end()) Throw("worker output has no arrays: " + outPath.get<std::string>());
        PassImage d = it->second.type == PixelType::F32 ? std::move(it->second) : it->second.ConvertTo(PixelType::F32);
        d.channels = {"Z"};
        depthOut.push_back(std::move(d));
    }
    if (depthOut.size() != rgb.size()) Throw("worker returned " + std::to_string(depthOut.size()) + " depth maps for " + std::to_string(rgb.size()) + " frames");
    for (size_t i = 0; i < rgb.size(); ++i) {
        if (depthOut[i].width != rgb[i]->width || depthOut[i].height != rgb[i]->height)
            Throw("worker depth map " + std::to_string(i) + " is " + std::to_string(depthOut[i].width) + "x" + std::to_string(depthOut[i].height) + ", frame is " +
                  std::to_string(rgb[i]->width) + "x" + std::to_string(rgb[i]->height));
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void WorkerDepthEstimator::Shutdown() {
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
    }
}

nlohmann::json WorkerDepthEstimator::Describe() const {
    nlohmann::json j = info_;
    j["backend"] = Name();
    j["runtime"] = "pytorch (depth_worker)";
    return j;
}

}  // namespace dlssvid
