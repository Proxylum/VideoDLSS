#include "gpu/CudaInterop.h"

#include <cstring>

#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

std::string CudaErrorToString(cudaError_t err) {
    return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

void CheckCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) Throw(std::string(what) + " failed: " + CudaErrorToString(err));
}

bool CudaInterop::Available(std::string* reason) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        if (reason) *reason = CudaErrorToString(err);
        return false;
    }
    if (count == 0) {
        if (reason) *reason = "no CUDA devices";
        return false;
    }
    return true;
}

CudaInterop::CudaInterop(const D3D12Device& device) : device_(device) {
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    const LUID luid = device.AdapterLuid();
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        CheckCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
        if (std::memcmp(prop.luid, &luid, sizeof(luid)) == 0) {
            cudaDevice_ = i;
            break;
        }
    }
    if (cudaDevice_ < 0) Throw("no CUDA device matches the D3D12 adapter LUID (" + device.AdapterName() + ")");
    CheckCuda(cudaSetDevice(cudaDevice_), "cudaSetDevice");
    Log()->info("CUDA device {}: {}", cudaDevice_, CudaDeviceName());
}

CudaInterop::~CudaInterop() = default;

std::string CudaInterop::CudaDeviceName() const {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, cudaDevice_) != cudaSuccess) return "?";
    return prop.name;
}

CudaInterop::ImportedBuffer CudaInterop::ImportBuffer(ID3D12Resource* sharedBuffer, size_t size) {
    HANDLE handle = nullptr;
    CheckHr(device_.Get()->CreateSharedHandle(sharedBuffer, nullptr, GENERIC_ALL, nullptr, &handle), "CreateSharedHandle");

    cudaExternalMemoryHandleDesc desc{};
    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = handle;
    desc.size = size;
    desc.flags = cudaExternalMemoryDedicated;

    ImportedBuffer out;
    out.size = size;
    const cudaError_t err = cudaImportExternalMemory(&out.memory, &desc);
    CloseHandle(handle);
    CheckCuda(err, "cudaImportExternalMemory");

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = size;
    bufDesc.flags = 0;
    const cudaError_t mapErr = cudaExternalMemoryGetMappedBuffer(&out.devicePtr, out.memory, &bufDesc);
    if (mapErr != cudaSuccess) {
        cudaDestroyExternalMemory(out.memory);
        out.memory = nullptr;
        CheckCuda(mapErr, "cudaExternalMemoryGetMappedBuffer");
    }
    return out;
}

void CudaInterop::Release(ImportedBuffer& buffer) {
    if (buffer.devicePtr) cudaFree(buffer.devicePtr);
    if (buffer.memory) cudaDestroyExternalMemory(buffer.memory);
    buffer = ImportedBuffer{};
}

void CudaInterop::CopyNv12(void* devicePtr, size_t capacity, const GpuFrame& frame) {
    if (!frame.Valid()) Throw("CudaInterop::CopyNv12: invalid GPU frame");
    const size_t yBytes = static_cast<size_t>(frame.width) * frame.height;
    const size_t uvBytes = static_cast<size_t>(frame.width) * ((frame.height + 1) / 2);
    if (capacity < yBytes + uvBytes) Throw("CudaInterop::CopyNv12: destination too small");
    auto* dst = static_cast<uint8_t*>(devicePtr);
    CheckCuda(cudaMemcpy2D(dst, frame.width, reinterpret_cast<const void*>(frame.y), frame.pitch, frame.width, frame.height, cudaMemcpyDeviceToDevice),
              "cudaMemcpy2D(Y)");
    CheckCuda(cudaMemcpy2D(dst + yBytes, frame.width, reinterpret_cast<const void*>(frame.uv), frame.pitch, frame.width, (frame.height + 1) / 2,
                           cudaMemcpyDeviceToDevice),
              "cudaMemcpy2D(UV)");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaInterop::CopyToDevice(void* devicePtr, const void* host, size_t size) {
    CheckCuda(cudaMemcpy(devicePtr, host, size, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
}

void CudaInterop::CopyToHost(void* host, const void* devicePtr, size_t size) {
    CheckCuda(cudaMemcpy(host, devicePtr, size, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
}

void CudaInterop::Synchronize() { CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

}  // namespace dlssvid
