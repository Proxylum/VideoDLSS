// nvngx.dll_dlssvid.dll — the module the Neural Rendering snippet accepts as its caller (stage 6).
//
// HACK (ТЗ §4, docs/architecture.md): `nvngx_dlssnr.dll` resolves the module that owns the return
// address of its exported calls and refuses any module whose path does not contain "nvngx.dll"
// (the driver core is `_nvngx.dll`), answering FAIL_PlatformError before it looks at an argument.
// This library exists to be named correctly; it forwards Init_Ext / CreateFeature / EvaluateFeature /
// ReleaseFeature to the function pointers the caller resolved from the snippet, the way
// ComfyUI-DLSS5-NR's caller shim and OptiScaler_DLSSNR's forwarder do. It links nothing but the CRT.
//
// Two details are load-bearing: the exports must not tail-call the snippet (a `jmp` would drop this
// module's frame and the snippet would see dlssvid.exe as the caller), hence the volatile store after
// every call and __declspec(noinline); and the snippet's Init_Ext ABI is not the public one — the
// two reference projects even disagree on it — so `argOrder` selects the argument order and the host
// tries them in turn.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

using InitInfoVersionFn = int(__cdecl*)(unsigned long long, const wchar_t*, void*, const void*, int);  // ComfyUI: (app, path, device, info, version)
using InitVersionInfoFn = int(__cdecl*)(unsigned long long, const wchar_t*, void*, int, const void*);  // public SDK / OptiScaler: (app, path, device, version, info)
using CreateFn = int(__cdecl*)(void*, int, void*, void**);
using EvaluateFn = int(__cdecl*)(void*, void*, void*, void*);
using ReleaseFn = int(__cdecl*)(void*);

volatile long g_sink = 0;

__forceinline int Finish(int r) {
    g_sink = r;  // observable side effect after the call: keeps a real CALL/RET through this module
    return r;
}

}  // namespace

extern "C" {

__declspec(dllexport) const char* __cdecl dlssvid_nr_forwarder_version() { return "dlssvid-nr-forwarder 1"; }

// argOrder: 0 = (app, path, device, info, version); 1 = (app, path, device, version, info).
__declspec(dllexport) __declspec(noinline) int __cdecl dlssvid_nr_call_init(void* fn, int argOrder, unsigned long long appId, const wchar_t* path, void* device,
                                                                            int version, const void* info) {
    if (!fn) return 0;
    int r;
    if (argOrder == 0) r = reinterpret_cast<InitInfoVersionFn>(fn)(appId, path, device, info, version);
    else r = reinterpret_cast<InitVersionInfoFn>(fn)(appId, path, device, version, info);
    return Finish(r);
}

__declspec(dllexport) __declspec(noinline) int __cdecl dlssvid_nr_call_create(void* fn, void* cmdList, int featureId, void* params, void** handle) {
    if (!fn) return 0;
    const int r = reinterpret_cast<CreateFn>(fn)(cmdList, featureId, params, handle);
    return Finish(r);
}

__declspec(dllexport) __declspec(noinline) int __cdecl dlssvid_nr_call_evaluate(void* fn, void* cmdList, void* handle, void* params, void* callback) {
    if (!fn) return 0;
    const int r = reinterpret_cast<EvaluateFn>(fn)(cmdList, handle, params, callback);
    return Finish(r);
}

__declspec(dllexport) __declspec(noinline) int __cdecl dlssvid_nr_call_release(void* fn, void* handle) {
    if (!fn) return 0;
    const int r = reinterpret_cast<ReleaseFn>(fn)(handle);
    return Finish(r);
}

}  // extern "C"

BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
