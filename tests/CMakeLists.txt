find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# ---- Unit tests (no video files, WARP adapter unless a GPU is present) ----
add_executable(dlssvid_unit_tests
  unit/test_frame.cpp
  unit/test_d3d12_device.cpp
  unit/test_gpu_frame_cache.cpp
  unit/test_pipeline.cpp
  unit/test_cuda_interop.cpp
  unit/test_half.cpp
  unit/test_pass_image.cpp
  unit/test_manifest.cpp
  unit/test_formats.cpp
  unit/test_depth_convert.cpp
  unit/test_mv_convert.cpp
  unit/test_color_convert.cpp
)
target_link_libraries(dlssvid_unit_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_unit_tests PRIVATE DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp")
catch_discover_tests(dlssvid_unit_tests TEST_PREFIX "unit." DISCOVERY_MODE PRE_TEST)

# ---- Integration tests (synthetic clips written with FFmpeg, CLI invoked as a process) ----
add_executable(dlssvid_integration_tests
  integration/TestClips.h
  integration/TestClips.cpp
  integration/test_passthrough.cpp
  integration/test_cli.cpp
  integration/test_pass_sequence.cpp
  integration/test_cli_passes.cpp
)
target_link_libraries(dlssvid_integration_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_integration_tests PRIVATE
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
)
add_dependencies(dlssvid_integration_tests dlssvid)
catch_discover_tests(dlssvid_integration_tests TEST_PREFIX "integration." DISCOVERY_MODE PRE_TEST)
