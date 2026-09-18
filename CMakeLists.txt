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
# No C++20 modules: skip the per-TU dependency scan (restat step that let a header change slip through in stage 4).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
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

# ---- NVIDIA Optical Flow SDK headers (nvofapi64.dll comes with the driver) ----
if(DLSSVID_WITH_CUDA)
  set(NV_OPTICAL_FLOW_SDK_INCLUDE_DIR "" CACHE PATH "Directory with nvOpticalFlowCuda.h (NVIDIA/NVIDIAOpticalFlowSDK checkout)")
  if(NOT NV_OPTICAL_FLOW_SDK_INCLUDE_DIR)
    foreach(cand "$ENV{NV_OPTICAL_FLOW_SDK_ROOT}" "$ENV{NV_OPTICAL_FLOW_SDK_ROOT}/include" "${CMAKE_SOURCE_DIR}/third_party/OpticalFlowSDK")
      if(cand AND EXISTS "${cand}/nvOpticalFlowCuda.h")
        set(NV_OPTICAL_FLOW_SDK_INCLUDE_DIR "${cand}" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
  if(NOT NV_OPTICAL_FLOW_SDK_INCLUDE_DIR OR NOT EXISTS "${NV_OPTICAL_FLOW_SDK_INCLUDE_DIR}/nvOpticalFlowCuda.h")
    message(FATAL_ERROR "NVIDIA Optical Flow SDK headers not found: set NV_OPTICAL_FLOW_SDK_ROOT (git clone https://github.com/NVIDIA/NVIDIAOpticalFlowSDK) or build with -DDLSSVID_WITH_CUDA=OFF")
  endif()
  message(STATUS "Optical Flow SDK headers: ${NV_OPTICAL_FLOW_SDK_INCLUDE_DIR}")
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

# ---- Qt GUI (optional) ----
option(DLSSVID_BUILD_APP "Build the Qt 6 GUI (dlssvid-gui)" ON)
if(DLSSVID_BUILD_APP)
  if(DEFINED ENV{QT_ROOT} AND NOT "$ENV{QT_ROOT}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{QT_ROOT}")
  endif()
  find_package(Qt6 6.5 COMPONENTS Widgets Test QUIET)
  if(NOT Qt6_FOUND)
    message(WARNING "Qt6 Widgets not found (set QT_ROOT or CMAKE_PREFIX_PATH): building without the GUI")
    set(DLSSVID_BUILD_APP OFF)
  else()
    message(STATUS "Qt ${Qt6_VERSION}: ${Qt6_DIR}")
    get_target_property(_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(DLSSVID_QT_BIN "${_qmake}" DIRECTORY)
    find_program(DLSSVID_WINDEPLOYQT windeployqt HINTS "${DLSSVID_QT_BIN}")
    # Copies the Qt runtime (platform plugin, DLLs) next to a target so it runs from the build tree.
    function(dlssvid_deploy_qt target)
      if(DLSSVID_WINDEPLOYQT)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND "${DLSSVID_WINDEPLOYQT}" --no-translations --no-compiler-runtime --no-system-d3d-compiler --no-opengl-sw "$<TARGET_FILE:${target}>"
          COMMENT "windeployqt ${target}" VERBATIM)
      endif()
    endfunction()
  endif()
endif()

add_subdirectory(core)
add_subdirectory(cli)
if(DLSSVID_BUILD_APP)
  add_subdirectory(app)
endif()

if(DLSSVID_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
