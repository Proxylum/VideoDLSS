#include "stages/flow/OfaFlowEstimator.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <windows.h>

#include <nvOpticalFlowCommon.h>
#include <nvOpticalFlowCuda.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "stages/depth/DepthPreprocess.h"
#include "util/Error.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

void CheckCudaRt(cudaError_t err, const char* what) {
    if (err != cudaSuccess) Throw(std::string(what) + " failed: " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

using PfnCreateInstanceCuda = NV_OF_STATUS(NVOFAPI*)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);
using PfnMaxApiVersion = NV_OF_STATUS(NVOFAPI*)(uint32_t*);

struct Api {
    HMODULE dll = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST fn{};
    uint32_t maxVersion = 0;
    std::string error;
    bool ok = false;
};

Api& LoadApi() {
    static Api api = [] {
        Api a;
        a.dll = LoadLibraryW(L"nvofapi64.dll");
        if (!a.dll) {
            a.error = "nvofapi64.dll not found (needs an NVIDIA driver with the Optical Flow API)";
            return a;
        }
        auto create = reinterpret_cast<PfnCreateInstanceCuda>(GetProcAddress(a.dll, "NvOFAPICreateInstanceCuda"));
        auto maxVer = reinterpret_cast<PfnMaxApiVersion>(GetProcAddress(a.dll, "NvOFGetMaxSupportedApiVersion"));
        if (!create) {
            a.error = "NvOFAPICreateInstanceCuda not exported by nvofapi64.dll";
            return a;
        }
        if (maxVer) maxVer(&a.maxVersion);
        const NV_OF_STATUS st = create(NV_OF_API_VERSION, &a.fn);
        if (st != NV_OF_SUCCESS) {
            a.error = "NvOFAPICreateInstanceCuda failed with status " + std::to_string(st);
            return a;
        }
        a.ok = true;
        return a;
    }();
    return api;
}

const char* StatusName(NV_OF_STATUS s) {
    switch (s) {
        case NV_OF_SUCCESS: return "NV_OF_SUCCESS";
        case NV_OF_ERR_OF_NOT_AVAILABLE: return "NV_OF_ERR_OF_NOT_AVAILABLE";
        case NV_OF_ERR_UNSUPPORTED_DEVICE: return "NV_OF_ERR_UNSUPPORTED_DEVICE";
        case NV_OF_ERR_DEVICE_DOES_NOT_EXIST: return "NV_OF_ERR_DEVICE_DOES_NOT_EXIST";
        case NV_OF_ERR_INVALID_PTR: return "NV_OF_ERR_INVALID_PTR";
        case NV_OF_ERR_INVALID_PARAM: return "NV_OF_ERR_INVALID_PARAM";
        case NV_OF_ERR_INVALID_CALL: return "NV_OF_ERR_INVALID_CALL";
        case NV_OF_ERR_INVALID_VERSION: return "NV_OF_ERR_INVALID_VERSION";
        case NV_OF_ERR_OUT_OF_MEMORY: return "NV_OF_ERR_OUT_OF_MEMORY";
        case NV_OF_ERR_GENERIC: return "NV_OF_ERR_GENERIC";
        default: return "NV_OF_ERR_?";
    }
}

}  // namespace

struct OfaFlowEstimator::Impl {
    NvOFHandle session = nullptr;
    NvOFGPUBufferHandle in = nullptr, ref = nullptr, out = nullptr, cost = nullptr;
    NV_OF_BUFFER_FORMAT inputFormat = NV_OF_BUFFER_FORMAT_NV12;
    uint32_t width = 0, height = 0, grid = 1, outW = 0, outH = 0;
    NV_OF_PERF_LEVEL perf = NV_OF_PERF_LEVEL_SLOW;
    std::vector<int16_t> flowHost;  // outW*outH*2
    std::vector<uint8_t> costHost;
    std::vector<uint8_t> abgrHost;  // CPU RGB -> ABGR8 upload staging
    bool temporalHints = false;
    int64_t executions = 0;

    std::string LastError() {
        char buf[512] = {};
        uint32_t size = sizeof(buf);
        if (session && LoadApi().fn.nvOFGetLastError) LoadApi().fn.nvOFGetLastError(session, buf, &size);
        return buf;
    }
    void Check(NV_OF_STATUS st, const char* what) {
        if (st != NV_OF_SUCCESS) Throw(std::string(what) + " failed: " + StatusName(st) + " " + LastError());
    }
    void Destroy() {
        Api& api = LoadApi();
        if (!api.ok) return;
        for (NvOFGPUBufferHandle* h : {&in, &ref, &out, &cost}) {
            if (*h) api.fn.nvOFDestroyGPUBufferCuda(*h);
            *h = nullptr;
        }
        if (session) api.fn.nvOFDestroy(session);
        session = nullptr;
    }
    ~Impl() { Destroy(); }
};

OfaFlowEstimator::OfaFlowEstimator() : impl_(std::make_unique<Impl>()) {}
OfaFlowEstimator::~OfaFlowEstimator() = default;

bool OfaFlowEstimator::Available(std::string* reason) {
    Api& api = LoadApi();
    if (!api.ok) {
        if (reason) *reason = api.error;
        return false;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        if (reason) *reason = "no CUDA device";
        return false;
    }
    return true;
}

void OfaFlowEstimator::Init(const FlowEstimatorConfig& config, uint32_t width, uint32_t height) {
    std::string reason;
    if (!Available(&reason)) Throw("backend 'ofa' unavailable: " + reason);
    Api& api = LoadApi();
    Impl& im = *impl_;
    im.Destroy();
    im.width = width;
    im.height = height;
    const std::string perf = config.extra.value("perf_level", "slow");
    im.perf = perf == "fast" ? NV_OF_PERF_LEVEL_FAST : perf == "medium" ? NV_OF_PERF_LEVEL_MEDIUM : NV_OF_PERF_LEVEL_SLOW;
    im.temporalHints = config.extra.value("temporal_hints", false);
    im.inputFormat = config.extra.value("input", "nv12") == "abgr" ? NV_OF_BUFFER_FORMAT_ABGR8 : NV_OF_BUFFER_FORMAT_NV12;

    // Primary CUDA context (shared with FFmpeg's NVDEC frames and TensorRT).
    CheckCudaRt(cudaFree(nullptr), "cudaFree(0) (context init)");
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) Throw("cuCtxGetCurrent: no current CUDA context");
    im.Check(api.fn.nvCreateOpticalFlowCuda(ctx, &im.session), "nvCreateOpticalFlowCuda");

    // Smallest supported output grid (1 on Ada/Blackwell with recent drivers).
    uint32_t grids = 0, size = 1;
    if (api.fn.nvOFGetCaps(im.session, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, &grids, &size) != NV_OF_SUCCESS) grids = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    const int wanted = config.extra.value("grid", 1);
    im.grid = 4;
    for (uint32_t g : {1u, 2u, 4u})
        if ((grids & g) && static_cast<int>(g) >= wanted) {
            im.grid = g;
            break;
        }
    im.outW = (width + im.grid - 1) / im.grid;
    im.outH = (height + im.grid - 1) / im.grid;

    NV_OF_INIT_PARAMS init{};
    init.width = width;
    init.height = height;
    init.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(im.grid);
    init.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = im.perf;
    init.enableExternalHints = NV_OF_FALSE;
    init.enableOutputCost = NV_OF_TRUE;
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init.enableRoi = NV_OF_FALSE;
    im.Check(api.fn.nvOFInit(im.session, &init), "nvOFInit");

    NV_OF_BUFFER_DESCRIPTOR inDesc{};
    inDesc.width = width;
    inDesc.height = height;
    inDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    inDesc.bufferFormat = im.inputFormat;
    im.Check(api.fn.nvOFCreateGPUBufferCuda(im.session, &inDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &im.in), "nvOFCreateGPUBufferCuda(input)");
    im.Check(api.fn.nvOFCreateGPUBufferCuda(im.session, &inDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &im.ref), "nvOFCreateGPUBufferCuda(reference)");
    NV_OF_BUFFER_DESCRIPTOR outDesc{};
    outDesc.width = im.outW;
    outDesc.height = im.outH;
    outDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    outDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    im.Check(api.fn.nvOFCreateGPUBufferCuda(im.session, &outDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &im.out), "nvOFCreateGPUBufferCuda(output)");
    NV_OF_BUFFER_DESCRIPTOR costDesc = outDesc;
    costDesc.bufferUsage = NV_OF_BUFFER_USAGE_COST;
    costDesc.bufferFormat = NV_OF_BUFFER_FORMAT_UINT8;
    if (api.fn.nvOFCreateGPUBufferCuda(im.session, &costDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &im.cost) != NV_OF_SUCCESS) {
        costDesc.bufferFormat = NV_OF_BUFFER_FORMAT_UINT;
        if (api.fn.nvOFCreateGPUBufferCuda(im.session, &costDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &im.cost) != NV_OF_SUCCESS) im.cost = nullptr;
    }
    im.flowHost.assign(static_cast<size_t>(im.outW) * im.outH * 2, 0);
    Log()->info("OFA session: {}x{}, grid {}, perf {}, input {}, api max 0x{:x}", width, height, im.grid, perf,
                im.inputFormat == NV_OF_BUFFER_FORMAT_NV12 ? "nv12" : "abgr8", api.maxVersion);
}

namespace {

void CopyNv12(const Api& api, NvOFGPUBufferHandle dst, const GpuFrame& src, uint32_t width, uint32_t height) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (api.fn.nvOFGPUBufferGetStrideInfo(dst, &stride) != NV_OF_SUCCESS) Throw("nvOFGPUBufferGetStrideInfo failed");
    const CUdeviceptr base = api.fn.nvOFGPUBufferGetCUdeviceptr(dst);
    const size_t pitchY = stride.strideInfo[0].strideXInBytes;
    const size_t planeYBytes = static_cast<size_t>(stride.strideInfo[0].strideYInBytes) * pitchY;
    const size_t pitchUV = stride.numPlanes > 1 ? stride.strideInfo[1].strideXInBytes : pitchY;
    CheckCudaRt(cudaMemcpy2D(reinterpret_cast<void*>(base), pitchY, reinterpret_cast<const void*>(src.y), src.pitch, width, height, cudaMemcpyDeviceToDevice),
                "cudaMemcpy2D(Y)");
    CheckCudaRt(cudaMemcpy2D(reinterpret_cast<void*>(base + planeYBytes), pitchUV, reinterpret_cast<const void*>(src.uv), src.pitch, width, height / 2,
                             cudaMemcpyDeviceToDevice),
                "cudaMemcpy2D(UV)");
}

void CopyAbgr(const Api& api, NvOFGPUBufferHandle dst, const PassImage& rgb, std::vector<uint8_t>& staging) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (api.fn.nvOFGPUBufferGetStrideInfo(dst, &stride) != NV_OF_SUCCESS) Throw("nvOFGPUBufferGetStrideInfo failed");
    const CUdeviceptr base = api.fn.nvOFGPUBufferGetCUdeviceptr(dst);
    const size_t pitch = stride.strideInfo[0].strideXInBytes;
    staging.resize(static_cast<size_t>(rgb.width) * rgb.height * 4);
    for (uint32_t y = 0; y < rgb.height; ++y)
        for (uint32_t x = 0; x < rgb.width; ++x) {
            uint8_t* p = staging.data() + (static_cast<size_t>(y) * rgb.width + x) * 4;
            for (size_t c = 0; c < 3; ++c) p[c] = static_cast<uint8_t>(std::clamp(std::lround(rgb.Get(x, y, c) * 255.f), 0L, 255L));  // R G B A byte order
            p[3] = 255;
        }
    CheckCudaRt(cudaMemcpy2D(reinterpret_cast<void*>(base), pitch, staging.data(), static_cast<size_t>(rgb.width) * 4, static_cast<size_t>(rgb.width) * 4, rgb.height,
                             cudaMemcpyHostToDevice),
                "cudaMemcpy2D(ABGR)");
}

}  // namespace

void OfaFlowEstimator::Estimate(const FlowInput& a, const FlowInput& b, PassImage& flowOut, PassImage* confidenceOut) {
    Impl& im = *impl_;
    Api& api = LoadApi();
    if (!im.session) Throw("OfaFlowEstimator: Init first");
    if (im.inputFormat == NV_OF_BUFFER_FORMAT_NV12) {
        if (!a.gpu || !b.gpu || !a.gpu->Valid() || !b.gpu->Valid()) {
            if (a.rgb && b.rgb) {
                // fall back to ABGR uploads when frames were decoded on the CPU
                Log()->info("OFA: no GPU frames, re-initialising for ABGR8 input");
                FlowEstimatorConfig cfg;
                cfg.extra["input"] = "abgr";
                cfg.extra["perf_level"] = im.perf == NV_OF_PERF_LEVEL_FAST ? "fast" : im.perf == NV_OF_PERF_LEVEL_MEDIUM ? "medium" : "slow";
                cfg.extra["grid"] = static_cast<int>(im.grid);
                Init(cfg, im.width, im.height);
            } else {
                Throw("OfaFlowEstimator: NV12 input needs GPU frames (decode with --hwaccel cuda)");
            }
        }
    }
    if (im.inputFormat == NV_OF_BUFFER_FORMAT_NV12) {
        if (a.gpu->width != im.width || a.gpu->height != im.height) Throw("OfaFlowEstimator: frame size mismatch");
        CopyNv12(api, im.in, *a.gpu, im.width, im.height);
        CopyNv12(api, im.ref, *b.gpu, im.width, im.height);
    } else {
        if (!a.rgb || !b.rgb) Throw("OfaFlowEstimator: ABGR input needs CPU RGB frames");
        CopyAbgr(api, im.in, *a.rgb, im.abgrHost);
        CopyAbgr(api, im.ref, *b.rgb, im.abgrHost);
    }

    NV_OF_EXECUTE_INPUT_PARAMS inP{};
    inP.inputFrame = im.in;
    inP.referenceFrame = im.ref;
    inP.externalHints = nullptr;
    inP.disableTemporalHints = im.temporalHints ? NV_OF_FALSE : NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS outP{};
    outP.outputBuffer = im.out;
    outP.outputCostBuffer = im.cost;
    im.Check(api.fn.nvOFExecute(im.session, &inP, &outP), "nvOFExecute");
    CheckCudaRt(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ++im.executions;

    // Read back the S10.5 vectors (grid resolution) and upsample to the source resolution.
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    if (api.fn.nvOFGPUBufferGetStrideInfo(im.out, &stride) != NV_OF_SUCCESS) Throw("nvOFGPUBufferGetStrideInfo(output) failed");
    const CUdeviceptr outPtr = api.fn.nvOFGPUBufferGetCUdeviceptr(im.out);
    CheckCudaRt(cudaMemcpy2D(im.flowHost.data(), static_cast<size_t>(im.outW) * 4, reinterpret_cast<const void*>(outPtr), stride.strideInfo[0].strideXInBytes,
                             static_cast<size_t>(im.outW) * 4, im.outH, cudaMemcpyDeviceToHost),
                "cudaMemcpy2D(flow)");
    PassImage grid = MakePassImage(PassKind::MvRaw, im.outW, im.outH);
    float* g = grid.As<float>();
    for (size_t i = 0; i < static_cast<size_t>(im.outW) * im.outH; ++i) {
        g[2 * i] = static_cast<float>(im.flowHost[2 * i]) / 32.f;
        g[2 * i + 1] = static_cast<float>(im.flowHost[2 * i + 1]) / 32.f;
    }
    flowOut = im.grid == 1 ? std::move(grid) : ResizeBilinear(grid, im.width, im.height);
    flowOut.channels = Spec(PassKind::MvRaw).channels;

    if (confidenceOut) {
        confidenceOut->Allocate(im.width, im.height, PixelType::F32, {"A"});
        float* c = confidenceOut->As<float>();
        if (im.cost) {
            NV_OF_CUDA_BUFFER_STRIDE_INFO cs{};
            api.fn.nvOFGPUBufferGetStrideInfo(im.cost, &cs);
            const CUdeviceptr cp = api.fn.nvOFGPUBufferGetCUdeviceptr(im.cost);
            const size_t elem = cs.strideInfo[0].strideXInBytes / std::max<uint32_t>(1, im.outW) >= 4 ? 4 : 1;
            im.costHost.resize(static_cast<size_t>(im.outW) * im.outH * elem);
            CheckCudaRt(cudaMemcpy2D(im.costHost.data(), static_cast<size_t>(im.outW) * elem, reinterpret_cast<const void*>(cp), cs.strideInfo[0].strideXInBytes,
                                     static_cast<size_t>(im.outW) * elem, im.outH, cudaMemcpyDeviceToHost),
                        "cudaMemcpy2D(cost)");
            PassImage cg;
            cg.Allocate(im.outW, im.outH, PixelType::F32, {"A"});
            float* cgp = cg.As<float>();
            for (size_t i = 0; i < static_cast<size_t>(im.outW) * im.outH; ++i) {
                const float cost = elem == 4 ? static_cast<float>(reinterpret_cast<const uint32_t*>(im.costHost.data())[i]) / 255.f
                                             : static_cast<float>(im.costHost[i]);
                cgp[i] = std::clamp(1.f - cost / 255.f, 0.f, 1.f);  // lower cost = more confident
            }
            const PassImage up = im.grid == 1 ? cg : ResizeBilinear(cg, im.width, im.height);
            std::copy(up.As<float>(), up.As<float>() + static_cast<size_t>(im.width) * im.height, c);
        } else {
            std::fill(c, c + static_cast<size_t>(im.width) * im.height, 1.f);
        }
    }
}

void OfaFlowEstimator::Shutdown() { impl_->Destroy(); }

nlohmann::json OfaFlowEstimator::Describe() const {
    const Impl& im = *impl_;
    return {{"backend", "ofa"}, {"api", "nvofapi cuda"}, {"grid", im.grid}, {"perf_level", im.perf == NV_OF_PERF_LEVEL_SLOW ? "slow" : im.perf == NV_OF_PERF_LEVEL_MEDIUM ? "medium" : "fast"},
            {"input", im.inputFormat == NV_OF_BUFFER_FORMAT_NV12 ? "nv12" : "abgr8"}, {"temporal_hints", im.temporalHints}};
}

}  // namespace dlssvid
