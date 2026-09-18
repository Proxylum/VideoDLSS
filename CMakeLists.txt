cmake_minimum_required(VERSION 3.28)

project(dlss-video
  VERSION 0.1.0
  DESCRIPTION "Offline DLSS video pipeline (SR / NR / FG) for Windows + NVIDIA RTX"
  LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "dlss-video targets Windows 10/11 x64 only (D3D12 + NVIDIA).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DLSSVID_BUILD_TESTS "Build unit and integration tests" ON)
option(DLSSVID_WITH_CUDA "Enable CUDA <-> D3D12 interop (requires CUDA Toolkit 12.x)" ON)

# ---- Third-party (vcpkg manifest) ----------------------------------------
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(directx-headers CONFIG REQUIRED)
find_package(FFMPEG REQUIRED) # vcpkg wrapper: FFMPEG_INCLUDE_DIRS / FFMPEG_LIBRARY_DIRS / FFMPEG_LIBRARIES
find_package(OpenEXR CONFIG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)
find_package(ZLIB REQUIRED)

add_library(ffmpeg_libs INTERFACE)
target_include_directories(ffmpeg_libs INTERFACE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(ffmpeg_libs INTERFACE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(ffmpeg_libs INTERFACE ${FFMPEG_LIBRARIES})
add_library(ffmpeg::ffmpeg ALIAS ffmpeg_libs)

if(DLSSVID_WITH_CUDA)
  find_package(CUDAToolkit 12.0)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "CUDA Toolkit 12.x not found: building without CUDA interop (set CUDAToolkit_ROOT).")
    set(DLSSVID_WITH_CUDA OFF)
  else()
    message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}")
  endif()
endif()

# ---- Common compile settings ---------------------------------------------
add_library(dlssvid_settings INTERFACE)
target_compile_options(dlssvid_settings INTERFACE
  /W4 /permissive- /utf-8 /Zc:__cplusplus /EHsc /MP
  /wd4324 # structure padded due to alignment specifier (d3dx12)
)
target_compile_definitions(dlssvid_settings INTERFACE
  NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE
  _CRT_SECURE_NO_WARNINGS
  DLSSVID_VERSION="${PROJECT_VERSION}"
)
if(DLSSVID_WITH_CUDA)
  target_compile_definitions(dlssvid_settings INTERFACE DLSSVID_WITH_CUDA=1)
endif()

# ---- TensorRT (headers only at build time; DLLs are loaded at runtime, see core/ml/TrtLoader) ----
option(DLSSVID_WITH_TENSORRT "Build the TensorRT depth/flow backends (needs TensorRT headers)" ON)
if(DLSSVID_WITH_TENSORRT AND NOT DLSSVID_WITH_CUDA)
  message(WARNING "TensorRT backends need CUDA: disabling DLSSVID_WITH_TENSORRT")
  set(DLSSVID_WITH_TENSORRT OFF)
endif()
if(DLSSVID_WITH_TENSORRT)
  set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory with NvInfer.h (TensorRT SDK or NVIDIA/TensorRT OSS include/)")
  if(NOT TENSORRT_INCLUDE_DIR)
    foreach(cand "$ENV{TENSORRT_ROOT}/include" "$ENV{TENSORRT_INCLUDE_DIR}")
      if(cand AND EXISTS "${cand}/NvInfer.h")
        set(TENSORRT_INCLUDE_DIR "${cand}" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
  if(NOT TENSORRT_INCLUDE_DIR OR NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(WARNING "TensorRT headers not found (set TENSORRT_ROOT or TENSORRT_INCLUDE_DIR): building without the TensorRT backends")
    set(DLSSVID_WITH_TENSORRT OFF)
  else()
    message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
    target_compile_definitions(dlssvid_settings INTERFACE DLSSVID_WITH_TENSORRT=1)
  endif()
endif()

add_subdirectory(core)
add_subdirectory(cli)

if(DLSSVID_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
