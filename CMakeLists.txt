cmake_minimum_required(VERSION 3.28)

project(dlss-video
  VERSION 0.3.1
  DESCRIPTION "Offline DLSS video pipeline (SR / NR / FG) for Windows + NVIDIA RTX"
  LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "dlss-video targets Windows 10/11 x64 only (D3D12 + NVIDIA).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# No C++20 modules: skip the per-TU dependency scan (kept off; the stage-4 stale-object incident turned out to be the localized /showIncludes prefix, see below).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header dependency tracking with Ninja + MSVC relies on the /showIncludes prefix CMake detects at configure time and
# writes into rules.ninja as UTF-8. A localized compiler (Russian Visual Studio, no English language pack) prints that
# prefix in the console code page (866), the bytes never match, ninja records NO header dependencies for the object
# (`ninja -t deps` shows "#deps 0") and objects silently survive header changes (incident: stages 4 and 8; the
# stage-4 guess "module scan" below was wrong). Cure: build with the console at code page 65001 (scripts/*.cmd and the
# scratch helper do `chcp 65001`) or install the English language pack (then VSLANG=1033 from the presets makes the
# prefix ASCII). tests/check_ninja_deps.cmake (ctest build.ninja_header_deps) fails when any object lacks dependencies.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja" AND DEFINED CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  string(REGEX MATCH "[^\t -~]" DLSSVID_NONASCII_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
  if(DLSSVID_NONASCII_PREFIX)
    message(WARNING "MSVC /showIncludes prefix is localized: build with the console at code page 65001 (chcp 65001, as scripts/ci-build.cmd does) "
      "or install the English language pack for Visual Studio; otherwise ninja tracks no header dependencies (ctest build.ninja_header_deps).")
  endif()
endif()
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

# ---- DLSS SDK / NGX (optional, stage 5) ----
option(DLSSVID_WITH_DLSS "Build the DLSS Super Resolution backend (needs the DLSS SDK: DLSS_SDK_ROOT)" ON)
if(DLSSVID_WITH_DLSS)
  set(DLSS_SDK_ROOT "" CACHE PATH "NVIDIA/DLSS checkout: include/nvsdk_ngx.h, lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
  if(NOT DLSS_SDK_ROOT)
    foreach(cand "$ENV{DLSS_SDK_ROOT}" "${CMAKE_SOURCE_DIR}/third_party/DLSS")
      if(cand AND EXISTS "${cand}/include/nvsdk_ngx.h")
        set(DLSS_SDK_ROOT "${cand}" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
  set(DLSSVID_NGX_LIB "${DLSS_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
  if(DLSS_SDK_ROOT AND EXISTS "${DLSS_SDK_ROOT}/include/nvsdk_ngx.h" AND EXISTS "${DLSSVID_NGX_LIB}")
    message(STATUS "DLSS SDK: ${DLSS_SDK_ROOT}")
    target_compile_definitions(dlssvid_settings INTERFACE DLSSVID_WITH_DLSS=1)
  else()
    message(WARNING "DLSS SDK not found (set DLSS_SDK_ROOT to a clone of https://github.com/NVIDIA/DLSS): building without the DLSS SR backend")
    set(DLSSVID_WITH_DLSS OFF)
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

# ---- stage 8: install layout and the package (docs/release.md) ------------------------------------------------
# Everything the executables need already sits in build/<preset>/bin (vcpkg DLLs, windeployqt output, DXC);
# NVIDIA binaries are never packaged (bin/nvidia keeps only its README, docs/dll-setup.md says what to add).
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/ DESTINATION bin USE_SOURCE_PERMISSIONS
  PATTERN "*.pdb" EXCLUDE PATTERN "*.ilk" EXCLUDE PATTERN "*_tests.exe" EXCLUDE
  REGEX "nvidia/nvngx.*\\.dll$" EXCLUDE)
if(TARGET dlssvid_golden_tests)
  install(TARGETS dlssvid_golden_tests RUNTIME DESTINATION bin)
  install(DIRECTORY tests/golden/ DESTINATION tests/golden PATTERN "*.cpp" EXCLUDE)
endif()
install(FILES bin/nvidia/README.md DESTINATION bin/nvidia)
install(FILES models/registry.json DESTINATION models)
install(DIRECTORY models/export/ DESTINATION models/export PATTERN ".venv" EXCLUDE PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY depth_worker/ DESTINATION depth_worker PATTERN ".venv" EXCLUDE PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY sr_worker/ DESTINATION sr_worker PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY docs/ DESTINATION docs)
install(FILES README.md CHANGELOG.md LICENSE DESTINATION .)

set(CPACK_PACKAGE_NAME "dlss-video")
set(CPACK_PACKAGE_VENDOR "VideoDLSS")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "dlss-video-${PROJECT_VERSION}-win64")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

# ---- full package: TensorRT runtime + CUDA runtime + ONNX models, self-contained (scripts/package.cmd full) ----
option(DLSSVID_PACKAGE_FULL "Package the TensorRT runtime, cudart64_12.dll and the ONNX models (dlss-video-<ver>-win64-full.zip)" OFF)
set(DLSSVID_PACKAGE_TENSORRT_DIR "" CACHE PATH "TensorRT runtime DLLs to package (default: env DLSSVID_TENSORRT_DIR, else the export venv)")
set(DLSSVID_PACKAGE_MODELS_CACHE "" CACHE PATH "models/cache with the ONNX files to package (default: env DLSSVID_MODELS_DIR/cache, else models/cache)")
set(DLSSVID_PACKAGE_TRT_SMS "75;86;89;120" CACHE STRING "GPU architectures (sm) whose TensorRT builder resources are packaged; ptx is always included")
if(DLSSVID_PACKAGE_FULL)
  set(_trt "${DLSSVID_PACKAGE_TENSORRT_DIR}")
  if(NOT _trt AND DEFINED ENV{DLSSVID_TENSORRT_DIR})
    file(TO_CMAKE_PATH "$ENV{DLSSVID_TENSORRT_DIR}" _trt)
  endif()
  if(NOT _trt)
    set(_trt "${CMAKE_SOURCE_DIR}/models/export/.venv/Lib/site-packages/tensorrt_libs")
  endif()
  if(NOT EXISTS "${_trt}/nvinfer_10.dll")
    message(FATAL_ERROR "DLSSVID_PACKAGE_FULL: TensorRT runtime (nvinfer_10.dll) not found in ${_trt} — set DLSSVID_PACKAGE_TENSORRT_DIR")
  endif()
  set(_trt_files nvinfer_10.dll nvonnxparser_10.dll nvinfer_plugin_10.dll nvinfer_builder_resource_ptx_10.dll)
  foreach(sm IN LISTS DLSSVID_PACKAGE_TRT_SMS)
    list(APPEND _trt_files "nvinfer_builder_resource_sm${sm}_10.dll")
  endforeach()
  set(_trt_present "")
  foreach(f IN LISTS _trt_files)
    if(EXISTS "${_trt}/${f}")
      list(APPEND _trt_present "${_trt}/${f}")
    else()
      message(WARNING "full package: ${f} not found in ${_trt}")
    endif()
  endforeach()
  install(FILES ${_trt_present} DESTINATION bin/tensorrt)
  # the executables import cudart64_12.dll (CudaInterop); ship it so a clean machine needs only the driver
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda)
  if(EXISTS "${_cuda}/bin/cudart64_12.dll")
    install(FILES "${_cuda}/bin/cudart64_12.dll" DESTINATION bin)
  else()
    message(WARNING "full package: cudart64_12.dll not found under CUDA_PATH (${_cuda})")
  endif()
  # ONNX models; TensorRT engines are GPU/driver specific and are built on first use into models/cache
  set(_models "${DLSSVID_PACKAGE_MODELS_CACHE}")
  if(NOT _models AND DEFINED ENV{DLSSVID_MODELS_DIR})
    file(TO_CMAKE_PATH "$ENV{DLSSVID_MODELS_DIR}" _models)
    set(_models "${_models}/cache")
  endif()
  if(NOT _models)
    set(_models "${CMAKE_SOURCE_DIR}/models/cache")
  endif()
  file(GLOB _onnx "${_models}/*.onnx")
  if(NOT _onnx)
    message(FATAL_ERROR "DLSSVID_PACKAGE_FULL: no ONNX models in ${_models} — run the pipeline once or set DLSSVID_PACKAGE_MODELS_CACHE")
  endif()
  install(FILES ${_onnx} DESTINATION models/cache)
  list(LENGTH _onnx _n_onnx)
  list(LENGTH _trt_present _n_trt)
  set(CPACK_PACKAGE_FILE_NAME "dlss-video-${PROJECT_VERSION}-win64-full")
  message(STATUS "Full package: ${_n_trt} TensorRT DLLs from ${_trt}, ${_n_onnx} ONNX models from ${_models}")
endif()
set(CPACK_PACKAGE_INSTALL_DIRECTORY "dlss-video")
set(CPACK_GENERATOR "ZIP")
find_program(DLSSVID_MAKENSIS makensis)
if(DLSSVID_MAKENSIS)
  list(APPEND CPACK_GENERATOR "NSIS")
  set(CPACK_NSIS_DISPLAY_NAME "dlss-video ${PROJECT_VERSION}")
  set(CPACK_NSIS_MODIFY_PATH ON)
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY "bin")
  set(CPACK_PACKAGE_EXECUTABLES "dlssvid-gui;dlss-video")
endif()
include(CPack)
