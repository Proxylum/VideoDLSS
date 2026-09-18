#pragma once

// The NVIDIA NGX core (driver `_nvngx.dll`) shared by DLSS Super Resolution (stage 5) and Neural
// Rendering (stage 6). Compiled only with the DLSS SDK (DLSSVID_WITH_DLSS).

#if defined(DLSSVID_WITH_DLSS)

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

#include "gpu/D3D12Device.h"

namespace dlssvid::ngx {

const char* ResultName(NVSDK_NGX_Result r);
std::string ResultString(NVSDK_NGX_Result r);  // "FAIL_FeatureNotSupported (0xBAD00001)"
std::string ResultString(int r);
void Check(NVSDK_NGX_Result r, const char* what);  // throws dlssvid::Error on failure
std::filesystem::path AppDataPath();               // %LOCALAPPDATA%/dlssvid/ngx: NGX log and cache

// GUID-like project id required by NVSDK_NGX_*_Init_with_ProjectID (ТЗ: engine type CUSTOM).
inline constexpr const char* kProjectId = "6f3e2b9a-4c1d-4a8e-9b7f-2d5c8e1a4f60";

// The core initialised once per process for one D3D12 device (NVSDK_NGX_D3D12_Init_with_ProjectID).
// Every user holds a shared_ptr; the last one shuts the core down (NVSDK_NGX_D3D12_Shutdown1).
// `searchPaths` are the folders the core searches for snippets (bin/nvidia/); the first Acquire
// decides them for the process. Acquire for a different device throws.
class Runtime {
public:
    static std::shared_ptr<Runtime> Acquire(D3D12Device& device, const std::vector<std::filesystem::path>& searchPaths);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    ID3D12Device* Device() const { return device_; }
    const std::vector<std::filesystem::path>& SearchPaths() const { return paths_; }

private:
    Runtime() = default;
    ID3D12Device* device_ = nullptr;
    std::vector<std::filesystem::path> paths_;
};

}  // namespace dlssvid::ngx

#endif  // DLSSVID_WITH_DLSS
