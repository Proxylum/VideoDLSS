#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "gpu/D3D12Device.h"

namespace dlssvid {

// CUDA <-> D3D12 interop through external memory (cudaImportExternalMemory).
// A D3D12 buffer created on a D3D12_HEAP_FLAG_SHARED heap is mapped into CUDA address
// space; TensorRT / NVDEC (later stages) then read and write frames without CPU copies.
class CudaInterop {
public:
    struct ImportedBuffer {
        cudaExternalMemory_t memory = nullptr;
        void* devicePtr = nullptr;
        size_t size = 0;
        explicit operator bool() const { return devicePtr != nullptr; }
    };

    // Picks the CUDA device whose LUID matches the D3D12 adapter. Throws if none matches.
    explicit CudaInterop(const D3D12Device& device);
    ~CudaInterop();

    CudaInterop(const CudaInterop&) = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    static bool Available(std::string* reason = nullptr);

    int CudaDeviceId() const { return cudaDevice_; }
    std::string CudaDeviceName() const;

    ImportedBuffer ImportBuffer(ID3D12Resource* sharedBuffer, size_t size);
    void Release(ImportedBuffer& buffer);

    // Blocking copies used by stage-0 tests.
    void CopyToDevice(void* devicePtr, const void* host, size_t size);
    void CopyToHost(void* host, const void* devicePtr, size_t size);
    void Synchronize();

private:
    const D3D12Device& device_;
    int cudaDevice_ = -1;
};

std::string CudaErrorToString(cudaError_t err);
void CheckCuda(cudaError_t err, const char* what);

}  // namespace dlssvid
