#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace dlssvid::trt {

// TensorRT is loaded at runtime from nvinfer_10.dll / nvonnxparser_10.dll (pip package
// `tensorrt-cu12` ships them, so does the SDK zip). No import libraries are needed: the
// C entry points the public headers call (createInferRuntime_INTERNAL, ...) are defined in
// TrtLoader.cpp and forward to the DLLs.
//
// Search order for the DLL directory: DLSSVID_TENSORRT_DIR, TENSORRT_ROOT(/lib), `tensorrt/` next
// to the executable (the full package bundles the runtime there, docs/release.md), the venv
// `models/export/.venv/Lib/site-packages/tensorrt_libs` next to the executable / project,
// then the normal loader search path.

// Directory the DLLs were (or would be) loaded from; empty if not found.
std::filesystem::path LibraryDirectory();
// True when both DLLs can be loaded. `reason` explains a failure.
bool Available(std::string* reason = nullptr);
// Override the search directory (tests, CLI --tensorrt-dir). Must be called before first use.
void SetLibraryDirectory(const std::filesystem::path& dir);
// "10.16.1" as reported by the loaded library, empty if unavailable.
std::string LibraryVersion();
// The directories that are tried, in order (without the final PATH fallback); for diagnostics and tests.
std::vector<std::filesystem::path> SearchDirectories();

}  // namespace dlssvid::trt
