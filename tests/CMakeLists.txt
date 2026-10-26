find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# The tests run from a folder OUTSIDE the checkout. Every NGX initialisation (DLSS / NR / FG) spawns NVIDIA's OTA helper
# nvngx_update.exe, which inherits the working directory of the test process and outlives it by up to a minute; with
# the default working directory (build/release/tests) those helpers kept the folder locked and `git clean` of the next
# CI job failed with "Permission denied" (main pipelines 5881/5886/5892, 2026-09-22). The tests use absolute paths only.
file(TO_CMAKE_PATH "$ENV{TEMP}" DLSSVID_TEST_CWD_BASE)
if(NOT DLSSVID_TEST_CWD_BASE)
  set(DLSSVID_TEST_CWD_BASE "${CMAKE_BINARY_DIR}/..")
endif()
set(DLSSVID_TEST_CWD "${DLSSVID_TEST_CWD_BASE}/dlssvid-test-cwd")
file(MAKE_DIRECTORY "${DLSSVID_TEST_CWD}")

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
  unit/test_depth_models.cpp
  unit/test_worker.cpp
  unit/test_trt_engine.cpp
  unit/test_trt_upscaler.cpp
  unit/test_model_registry.cpp
  unit/test_warp.cpp
  unit/test_scene_cut.cpp
  unit/test_scenes_pass.cpp
  unit/test_ofa.cpp
  unit/test_viewport_state.cpp
  unit/test_viewport_renderer.cpp
  unit/test_frame_store.cpp
  unit/test_project.cpp
  unit/test_pass_fingerprint.cpp
  unit/test_pass_versions.cpp
  unit/test_param_schema.cpp
  unit/test_estimates.cpp
  unit/test_settings_note.cpp
  unit/test_encode_defaults.cpp
  unit/test_output_format.cpp
  unit/test_paths.cpp
  unit/test_upscale.cpp
  unit/test_tonemap.cpp
  unit/test_nr.cpp
  unit/test_fg.cpp
  unit/test_fg_process.cpp
  unit/test_repo_hygiene.cpp
)
target_link_libraries(dlssvid_unit_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_unit_tests PRIVATE
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
  DLSSVID_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
catch_discover_tests(dlssvid_unit_tests TEST_PREFIX "unit." DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY "${DLSSVID_TEST_CWD}")

# ---- Integration tests (synthetic clips written with FFmpeg, CLI invoked as a process) ----
add_executable(dlssvid_integration_tests
  integration/TestClips.h
  integration/TestClips.cpp
  integration/test_passthrough.cpp
  integration/test_formats.cpp
  integration/test_precision.cpp
  integration/test_cli.cpp
  integration/test_pass_sequence.cpp
  integration/test_cli_passes.cpp
  integration/test_depth_stage.cpp
  integration/test_vda_models.cpp
  integration/test_viewport_proxy.cpp
  integration/test_cli_depth.cpp
  integration/test_flow_stage.cpp
  integration/test_gpu_decode.cpp
  integration/test_cli_flow.cpp
  integration/test_cli_render.cpp
  integration/test_viewport_sources.cpp
  integration/test_upscale_stage.cpp
  integration/test_cli_upscale.cpp
  integration/test_nr_stage.cpp
  integration/test_cli_nr.cpp
  integration/test_fg_stage.cpp
  integration/test_cli_fg.cpp
  integration/test_process.cpp
  integration/test_cli_process.cpp
  integration/test_process_versions.cpp
  integration/test_cli_pass_versions.cpp
  integration/test_scenes_stage.cpp
)
target_link_libraries(dlssvid_integration_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_integration_tests PRIVATE
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
  DLSSVID_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
add_dependencies(dlssvid_integration_tests dlssvid)
catch_discover_tests(dlssvid_integration_tests TEST_PREFIX "integration." DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY "${DLSSVID_TEST_CWD}")

# ---- Qt application tests (offscreen platform: the model and the task queue without windows) ----
if(DLSSVID_BUILD_APP)
  add_executable(dlssvid_app_tests app/test_app_model.cpp integration/TestClips.h integration/TestClips.cpp)
  target_include_directories(dlssvid_app_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/integration)
  target_link_libraries(dlssvid_app_tests PRIVATE dlssvid_app Catch2::Catch2WithMain Qt6::Test)
  target_compile_definitions(dlssvid_app_tests PRIVATE
    DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
    DLSSVID_QT_PLUGINS="${DLSSVID_QT_BIN}/../plugins"
    DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
    DLSSVID_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_dependencies(dlssvid_app_tests dlssvid)
  if(TARGET dlssvid-gui)  # both deploy Qt into bin/ in POST_BUILD: one after the other, or two windeployqt race (CI 2026-09-25)
    add_dependencies(dlssvid_app_tests dlssvid-gui)
  endif()
  dlssvid_deploy_qt(dlssvid_app_tests)
  # windeployqt deploys only the windows platform plugin: copy the offscreen one next to the executable
  add_custom_command(TARGET dlssvid_app_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSSVID_QT_BIN}/../plugins/platforms/qoffscreen.dll"
            "$<TARGET_FILE_DIR:dlssvid_app_tests>/platforms/qoffscreen.dll"
    VERBATIM)
  # (a single ENVIRONMENT entry: catch_discover_tests splits multi-value properties)
  catch_discover_tests(dlssvid_app_tests TEST_PREFIX "app." DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY "${DLSSVID_TEST_CWD}"
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
  # Regression: a platform plugin that cannot be found must fail fast with a message. Without the
  # check in the tests' QApplication setup Qt shows a modal message box (no console) and aborts,
  # which hung automated runs until someone clicked it (stage 4).
  add_test(NAME app.platform_plugin_missing_fails_fast COMMAND dlssvid_app_tests "TaskQueue*" WORKING_DIRECTORY "${DLSSVID_TEST_CWD}")
  set_tests_properties(app.platform_plugin_missing_fails_fast PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=nonexistent" TIMEOUT 60
    PASS_REGULAR_EXPRESSION "platform plugin qnonexistent.dll not found")
endif()

# ---- Golden tests (stage 8): the whole pipeline through the CLI against stored reference frames; runs against
# the build tree or, with DLSSVID_CLI / DLSSVID_GOLDEN_DIR, against an installed copy (docs/release.md) ----
add_executable(dlssvid_golden_tests golden/test_golden.cpp integration/TestClips.h integration/TestClips.cpp)
target_include_directories(dlssvid_golden_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/integration)
target_link_libraries(dlssvid_golden_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_golden_tests PRIVATE
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp"
  DLSSVID_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_dependencies(dlssvid_golden_tests dlssvid)
catch_discover_tests(dlssvid_golden_tests TEST_PREFIX "golden." DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY "${DLSSVID_TEST_CWD}" PROPERTIES LABELS golden)

# ---- Performance (TASK-0040): scrubbing at >= 30 fps on 1080p, the p90 of random jumps into frames not resident; an
# NVIDIA GPU (skipped otherwise), serially - parallel tests would time each other (`ctest -LE perf` leaves it out) ----
add_executable(dlssvid_perf_tests perf/test_scrub_perf.cpp integration/TestClips.h integration/TestClips.cpp)
target_include_directories(dlssvid_perf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/integration)
target_link_libraries(dlssvid_perf_tests PRIVATE dlssvid_core Catch2::Catch2WithMain)
target_compile_definitions(dlssvid_perf_tests PRIVATE
  DLSSVID_CLI_PATH="$<TARGET_FILE:dlssvid>"
  DLSSVID_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp")
add_dependencies(dlssvid_perf_tests dlssvid)
catch_discover_tests(dlssvid_perf_tests TEST_PREFIX "perf." DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY "${DLSSVID_TEST_CWD}"
  PROPERTIES RUN_SERIAL TRUE LABELS perf TIMEOUT 900)

# ---- build hygiene: ninja must have recorded header dependencies for every C++ object (stale-object incident) ----
if(CMAKE_GENERATOR MATCHES "Ninja")
  add_test(NAME build.ninja_header_deps
    COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DNINJA=${CMAKE_MAKE_PROGRAM} -P ${CMAKE_SOURCE_DIR}/tests/check_ninja_deps.cmake)
endif()

# ---- release workflow: scripts/attach_release.cmd against a fake gh (incident TASK-0037: the attach step of release.yml
# died when the Release of the tag did not exist yet) ----
add_test(NAME scripts.attach_release
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DWORK_DIR=${CMAKE_BINARY_DIR}/test-tmp/attach_release
          -P ${CMAKE_SOURCE_DIR}/tests/check_attach_release.cmake)
set_tests_properties(scripts.attach_release PROPERTIES LABELS "scripts" TIMEOUT 60)

# ---- VDA export (TASK-0039): patched attention and external data, with the models/export venv (CPU; skipped without it) ----
add_test(NAME models.vda_export
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DSCRIPT=${CMAKE_SOURCE_DIR}/models/export/test_vda_export.py
          -P ${CMAKE_SOURCE_DIR}/tests/run_venv_python.cmake)
set_tests_properties(models.vda_export PROPERTIES LABELS "python;export" TIMEOUT 900 SKIP_REGULAR_EXPRESSION "venv not found")

# ---- SDK layout: the machine's SDK checkouts match the pinned revisions of scripts/setup_sdk.py (skipped without Python or SDK env) ----
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME sdk.layout_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/setup_sdk.py --check --from-env)
  set_tests_properties(sdk.layout_check PROPERTIES SKIP_RETURN_CODE 77 LABELS "sdk" TIMEOUT 120)
  # the benchmark script's guards against stale passes (review of TASK-0038); standard library only
  add_test(NAME scripts.bench_clips COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/tests/data/test_bench_clips.py)
  set_tests_properties(scripts.bench_clips PROPERTIES LABELS "scripts" TIMEOUT 60)
  # scripts/nr_ab.py (TASK-0050): its helpers without dlssvid, then a whole A/B on the stub backend and WARP over a generated Y4M clip - the
  # measurements of PR-2 run the same code on NGX. No NGX and no GPU, so it is safe while CI shares the GPU with development.
  add_test(NAME scripts.nr_ab COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/tests/scripts/test_nr_ab.py -k Helpers)
  set_tests_properties(scripts.nr_ab PROPERTIES LABELS "scripts" TIMEOUT 60)
  add_test(NAME scripts.nr_ab_stub
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_SOURCE_DIR}/tests/scripts/test_nr_ab.py -k StubRun --cli $<TARGET_FILE:dlssvid> --work ${CMAKE_BINARY_DIR}/test-tmp/nr_ab
           WORKING_DIRECTORY "${DLSSVID_TEST_CWD}")
  set_tests_properties(scripts.nr_ab_stub PROPERTIES LABELS "scripts" TIMEOUT 180)
endif()
