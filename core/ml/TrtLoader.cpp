#include "ml/TrtLoader.h"

#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "util/Log.h"

namespace dlssvid::trt {

namespace {

constexpr const wchar_t* kInferDll = L"nvinfer_10.dll";
constexpr const wchar_t* kOnnxDll = L"nvonnxparser_10.dll";

struct State {
    std::mutex mutex;
    bool tried = false;
    HMODULE infer = nullptr;
    HMODULE onnx = nullptr;
    std::filesystem::path dir;
    std::filesystem::path override;
    std::string error;
};

State& S() {
    static State s;
    return s;
}

std::filesystem::path ExeDir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return n ? std::filesystem::path(buf).parent_path() : std::filesystem::path();
}

std::vector<std::filesystem::path> Candidates() {
    std::vector<std::filesystem::path> out;
    if (!S().override.empty()) out.push_back(S().override);
    if (const char* e = std::getenv("DLSSVID_TENSORRT_DIR")) out.emplace_back(e);
    if (const char* e = std::getenv("TENSORRT_ROOT")) {
        out.emplace_back(std::filesystem::path(e) / "lib");
        out.emplace_back(e);
    }
    // venv next to the executable or in a source checkout (build/<preset>/bin -> project root)
    const auto exe = ExeDir();
    for (const auto& base : {exe, exe.parent_path(), exe.parent_path().parent_path(), exe.parent_path().parent_path().parent_path()}) {
        if (base.empty()) continue;
        out.push_back(base / "models" / "export" / ".venv" / "Lib" / "site-packages" / "tensorrt_libs");
    }
    return out;
}

HMODULE LoadFrom(const std::filesystem::path& dir, const wchar_t* name) {
    if (dir.empty()) return LoadLibraryW(name);
    const auto full = dir / name;
    if (!std::filesystem::exists(full)) return nullptr;
    // TensorRT loads its builder resources (nvinfer_builder_resource_sm89_10.dll, ...) by bare
    // name at build time, so the directory must be on the default DLL search path too.
    SetDllDirectoryW(dir.c_str());
    return LoadLibraryExW(full.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
}

void EnsureLoaded() {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.tried) return;
    s.tried = true;
    for (const auto& dir : Candidates()) {
        if (!std::filesystem::is_directory(dir)) continue;
        s.infer = LoadFrom(dir, kInferDll);
        if (!s.infer) continue;
        s.onnx = LoadFrom(dir, kOnnxDll);
        s.dir = dir;
        if (s.onnx) break;
        FreeLibrary(s.infer);
        s.infer = nullptr;
    }
    if (!s.infer) {  // default loader search (PATH)
        s.infer = LoadLibraryW(kInferDll);
        s.onnx = s.infer ? LoadLibraryW(kOnnxDll) : nullptr;
        if (s.infer) {
            wchar_t buf[MAX_PATH];
            if (GetModuleFileNameW(s.infer, buf, MAX_PATH)) s.dir = std::filesystem::path(buf).parent_path();
        }
    }
    if (!s.infer || !s.onnx) {
        s.error = "TensorRT DLLs (nvinfer_10.dll, nvonnxparser_10.dll) not found: set TENSORRT_ROOT or DLSSVID_TENSORRT_DIR, "
                  "or `pip install tensorrt-cu12` into models/export/.venv";
        if (s.infer) FreeLibrary(s.infer);
        if (s.onnx) FreeLibrary(s.onnx);
        s.infer = s.onnx = nullptr;
        Log()->debug("{}", s.error);
    } else {
        Log()->debug("TensorRT loaded from {}", s.dir.string());
    }
}

template <class Fn>
Fn Resolve(HMODULE m, const char* name) {
    EnsureLoaded();
    if (!m) return nullptr;
    return reinterpret_cast<Fn>(GetProcAddress(m, name));
}

}  // namespace

std::filesystem::path LibraryDirectory() {
    EnsureLoaded();
    return S().dir;
}

bool Available(std::string* reason) {
    EnsureLoaded();
    if (!S().infer && reason) *reason = S().error;
    return S().infer != nullptr && S().onnx != nullptr;
}

void SetLibraryDirectory(const std::filesystem::path& dir) {
    std::lock_guard<std::mutex> lock(S().mutex);
    S().override = dir;
}

std::string LibraryVersion();

}  // namespace dlssvid::trt

// ---- C entry points expected by the TensorRT public headers ------------------------------

using dlssvid::trt::Available;

namespace {
template <class Fn>
Fn Entry(bool onnx, const char* name) {
    dlssvid::trt::Available();
    // Access the loaded modules through GetModuleHandle: they stay loaded for the process lifetime.
    HMODULE m = GetModuleHandleW(onnx ? L"nvonnxparser_10.dll" : L"nvinfer_10.dll");
    return m ? reinterpret_cast<Fn>(GetProcAddress(m, name)) : nullptr;
}
}  // namespace

extern "C" void* createInferBuilder_INTERNAL(void* logger, int32_t version) noexcept {
    auto fn = Entry<void* (*)(void*, int32_t)>(false, "createInferBuilder_INTERNAL");
    return fn ? fn(logger, version) : nullptr;
}

extern "C" void* createInferRuntime_INTERNAL(void* logger, int32_t version) noexcept {
    auto fn = Entry<void* (*)(void*, int32_t)>(false, "createInferRuntime_INTERNAL");
    return fn ? fn(logger, version) : nullptr;
}

extern "C" void* createInferRefitter_INTERNAL(void* engine, void* logger, int32_t version) noexcept {
    auto fn = Entry<void* (*)(void*, void*, int32_t)>(false, "createInferRefitter_INTERNAL");
    return fn ? fn(engine, logger, version) : nullptr;
}

extern "C" int32_t getInferLibVersion() noexcept {
    auto fn = Entry<int32_t (*)()>(false, "getInferLibVersion");
    return fn ? fn() : 0;
}

extern "C" int32_t getInferLibMajorVersion() noexcept {
    auto fn = Entry<int32_t (*)()>(false, "getInferLibMajorVersion");
    return fn ? fn() : 0;
}
extern "C" int32_t getInferLibMinorVersion() noexcept {
    auto fn = Entry<int32_t (*)()>(false, "getInferLibMinorVersion");
    return fn ? fn() : 0;
}
extern "C" int32_t getInferLibPatchVersion() noexcept {
    auto fn = Entry<int32_t (*)()>(false, "getInferLibPatchVersion");
    return fn ? fn() : 0;
}

extern "C" void* createNvOnnxParser_INTERNAL(void* network, void* logger, int version) noexcept {
    auto fn = Entry<void* (*)(void*, void*, int)>(true, "createNvOnnxParser_INTERNAL");
    return fn ? fn(network, logger, version) : nullptr;
}

extern "C" int getNvOnnxParserVersion() noexcept {
    auto fn = Entry<int (*)()>(true, "getNvOnnxParserVersion");
    return fn ? fn() : 0;
}

namespace dlssvid::trt {
std::string LibraryVersion() {
    if (!Available()) return {};
    return std::to_string(getInferLibMajorVersion()) + "." + std::to_string(getInferLibMinorVersion()) + "." +
           std::to_string(getInferLibPatchVersion());
}
}  // namespace dlssvid::trt
