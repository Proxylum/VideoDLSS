#include "gpu/Ngx.h"

#if defined(DLSSVID_WITH_DLSS)

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid::ngx {

namespace {

std::mutex g_mutex;
std::weak_ptr<Runtime> g_runtime;

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level level, NVSDK_NGX_Feature) {
    std::string m = message ? message : "";
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
    (void)level;
    Log()->debug("[ngx] {}", m);
}

}  // namespace

const char* ResultName(NVSDK_NGX_Result r) {
    switch (r) {
        case NVSDK_NGX_Result_Success: return "Success";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FAIL_FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "FAIL_PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FAIL_FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FAIL_FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "FAIL_InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "FAIL_ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "FAIL_NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "FAIL_UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "FAIL_RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "FAIL_MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "FAIL_UnableToInitializeFeature";
        case NVSDK_NGX_Result_FAIL_OutOfDate: return "FAIL_OutOfDate";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "FAIL_OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "FAIL_UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "FAIL_UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "FAIL_UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied: return "FAIL_Denied";
        case NVSDK_NGX_Result_FAIL_NotImplemented: return "FAIL_NotImplemented";
        default: return "FAIL";
    }
}

std::string ResultString(NVSDK_NGX_Result r) {
    char code[32];
    std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(r));
    return std::string(ResultName(r)) + " (" + code + ")";
}

std::string ResultString(int r) { return ResultString(static_cast<NVSDK_NGX_Result>(r)); }

void Check(NVSDK_NGX_Result r, const char* what) {
    if (NVSDK_NGX_FAILED(r)) Throw(std::string("NGX ") + what + " failed: " + ResultString(r));
}

std::filesystem::path AppDataPath() {
    std::filesystem::path base;
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local) base = local;
    else base = std::filesystem::temp_directory_path();
    const auto dir = base / "dlssvid" / "ngx";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

std::shared_ptr<Runtime> Runtime::Acquire(D3D12Device& device, const std::vector<std::filesystem::path>& searchPaths) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (auto existing = g_runtime.lock()) {
        if (existing->device_ != device.Get()) Throw("NGX is already initialised for another D3D12 device");
        return existing;
    }
    std::vector<std::wstring> wide;
    std::vector<const wchar_t*> paths;
    for (const auto& p : searchPaths) {
        wide.push_back(p.wstring());
    }
    for (const auto& w : wide) paths.push_back(w.c_str());
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths.empty() ? nullptr : paths.data();
    info.PathListInfo.Length = static_cast<unsigned>(paths.size());
    info.LoggingInfo.LoggingCallback = &NgxLog;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    info.LoggingInfo.DisableOtherLoggingSinks = true;
    const std::wstring appData = AppDataPath().wstring();
    const NVSDK_NGX_Result r =
        NVSDK_NGX_D3D12_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, DLSSVID_VERSION, appData.c_str(), device.Get(), &info, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        if (r == NVSDK_NGX_Result_FAIL_FeatureNotSupported) Throw("NGX is not supported on this system (NVIDIA RTX GPU and a driver with DLSS are required)");
        Check(r, "Init");
    }
    std::shared_ptr<Runtime> rt(new Runtime());
    rt->device_ = device.Get();
    rt->paths_ = searchPaths;
    g_runtime = rt;
    Log()->debug("ngx: core initialised for {} ({} search path(s))", device.AdapterName(), searchPaths.size());
    return rt;
}

Runtime::~Runtime() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device_) NVSDK_NGX_D3D12_Shutdown1(device_);
    device_ = nullptr;
}

}  // namespace dlssvid::ngx

#endif  // DLSSVID_WITH_DLSS
