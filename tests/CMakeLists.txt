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
  unit/test_depth_preprocess.cpp
  unit/test_depth_postprocess.cpp
  unit/test_worker.cpp
  unit/test_trt_engine.cpp
  unit/test_model_registry.cpp
  unit/test_warp.cpp
  unit/test_ofa.cpp
  unit/test_viewport_state.cpp
  unit/test_viewport_renderer.cpp
  unit/test_frame_store.cpp
  unit/test_project.cpp
)
target_link_libraries(dlssvid_unit_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_unit_tests PRIVATE
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
  DLSSVID_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
catch_discover_tests(dlssvid_unit_tests TEST_PREFIX "unit." DISCOVERY_MODE PRE_TEST)

# ---- Integration tests (synthetic clips written with FFmpeg, CLI invoked as a process) ----
add_executable(dlssvid_integration_tests
  integration/TestClips.h
  integration/TestClips.cpp
  integration/test_passthrough.cpp
  integration/test_cli.cpp
  integration/test_pass_sequence.cpp
  integration/test_cli_passes.cpp
  integration/test_depth_stage.cpp
  integration/test_cli_depth.cpp
  integration/test_flow_stage.cpp
  integration/test_gpu_decode.cpp
  integration/test_cli_flow.cpp
  integration/test_cli_render.cpp
  integration/test_viewport_sources.cpp
)
target_link_libraries(dlssvid_integration_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_integration_tests PRIVATE
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
  DLSSVID_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
add_dependencies(dlssvid_integration_tests dlssvid)
catch_discover_tests(dlssvid_integration_tests TEST_PREFIX "integration." DISCOVERY_MODE PRE_TEST)

# ---- Qt application tests (offscreen platform: the model and the task queue without windows) ----
if(DLSSVID_BUILD_APP)
  add_executable(dlssvid_app_tests app/test_app_model.cpp)
  target_link_libraries(dlssvid_app_tests PRIVATE dlssvid_app Catch2::Catch2WithMain Qt6::Test)
  target_compile_definitions(dlssvid_app_tests PRIVATE
    DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
    DLSSVID_QT_PLUGINS="${DLSSVID_QT_BIN}/../plugins")
  dlssvid_deploy_qt(dlssvid_app_tests)
  # windeployqt deploys only the windows platform plugin: copy the offscreen one next to the executable
  add_custom_command(TARGET dlssvid_app_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSSVID_QT_BIN}/../plugins/platforms/qoffscreen.dll"
            "$<TARGET_FILE_DIR:dlssvid_app_tests>/platforms/qoffscreen.dll"
    VERBATIM)
  # (a single ENVIRONMENT entry: catch_discover_tests splits multi-value properties)
  catch_discover_tests(dlssvid_app_tests TEST_PREFIX "app." DISCOVERY_MODE PRE_TEST
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
  # Regression: a platform plugin that cannot be found must fail fast with a message. Without the
  # check in the tests' QApplication setup Qt shows a modal message box (no console) and aborts,
  # which hung automated runs until someone clicked it (stage 4).
  add_test(NAME app.platform_plugin_missing_fails_fast COMMAND dlssvid_app_tests "TaskQueue*")
  set_tests_properties(app.platform_plugin_missing_fails_fast PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=nonexistent" TIMEOUT 60
    PASS_REGULAR_EXPRESSION "platform plugin qnonexistent.dll not found")
endif()
