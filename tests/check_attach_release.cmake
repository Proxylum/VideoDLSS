# Regression test for incident TASK-0037: the release workflow failed to attach the package when the Release of the
# tag did not exist yet. `gh release view` prints "release not found" to stderr and exits 1; under Windows PowerShell
# 5.1 with $ErrorActionPreference = 'stop' (GitHub's `shell: powershell`) that stderr was a terminating error, so the
# step died before `gh release create`. The logic now lives in scripts/attach_release.cmd (plain cmd); this test runs
# it against a fake `gh` that answers exactly like the real one and checks which gh commands it issues.
#   cmake -DSOURCE_DIR=<repo> -DWORK_DIR=<scratch dir> -P tests/check_attach_release.cmake
if(NOT SOURCE_DIR OR NOT WORK_DIR)
  message(FATAL_ERROR "SOURCE_DIR and WORK_DIR are required")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/fakebin" "${WORK_DIR}/pkg")
# the package of the tag plus an older one left in the folder: the script must pick the tag's version, not "the first"
file(WRITE "${WORK_DIR}/pkg/dlss-video-9.9.9-win64.zip" "zip")
file(WRITE "${WORK_DIR}/pkg/dlss-video-0.0.1-win64.zip" "old zip")

file(TO_NATIVE_PATH "${WORK_DIR}/fakebin" FAKEBIN)
file(TO_NATIVE_PATH "${WORK_DIR}/pkg" PKGDIR)
file(TO_NATIVE_PATH "${SOURCE_DIR}/scripts/attach_release.cmd" SCRIPT)

# Fake gh: logs its arguments; `release view` answers like gh does for a missing Release (stderr + exit 1) unless
# FAKE_GH_VIEW_RC=0; `release upload` / `release create` exit with FAKE_GH_WRITE_RC.
file(CONFIGURE OUTPUT "${WORK_DIR}/fakebin/gh.cmd" NEWLINE_STYLE CRLF @ONLY CONTENT [=[@echo off
>>"%FAKE_GH_LOG%" echo gh %*
if not "%~1 %~2"=="release view" exit /b %FAKE_GH_WRITE_RC%
if "%FAKE_GH_VIEW_RC%"=="0" exit /b 0
>&2 echo release not found
exit /b 1
]=])

# run_case(<name> <tag> <repository> <view rc> <write rc>) -> case_rc, case_log
function(run_case name tag repository view_rc write_rc)
  set(LOG "${WORK_DIR}/${name}.log")
  file(REMOVE "${LOG}")
  file(TO_NATIVE_PATH "${LOG}" LOG_NATIVE)
  set(TAG "${tag}")
  set(REPOSITORY "${repository}")
  set(VIEW_RC "${view_rc}")
  set(WRITE_RC "${write_rc}")
  file(CONFIGURE OUTPUT "${WORK_DIR}/${name}.cmd" NEWLINE_STYLE CRLF @ONLY CONTENT [=[@echo off
set "PATH=@FAKEBIN@;%PATH%"
set "GITHUB_REPOSITORY=@REPOSITORY@"
set "FAKE_GH_LOG=@LOG_NATIVE@"
set "FAKE_GH_VIEW_RC=@VIEW_RC@"
set "FAKE_GH_WRITE_RC=@WRITE_RC@"
call "@SCRIPT@" @TAG@ "@PKGDIR@"
exit /b %errorlevel%
]=])
  file(TO_NATIVE_PATH "${WORK_DIR}/${name}.cmd" driver)
  execute_process(COMMAND cmd /d /c "${driver}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  set(text "")
  if(EXISTS "${LOG}")
    file(READ "${LOG}" text)
  endif()
  message(STATUS "[${name}] exit ${rc}\n${out}${err}gh calls:\n${text}")
  set(case_rc "${rc}" PARENT_SCOPE)
  set(case_log "${text}" PARENT_SCOPE)
endfunction()

function(fail what)
  message(SEND_ERROR "FAILED: ${what}")
  set_property(GLOBAL APPEND PROPERTY attach_release_failures "${what}")
endfunction()

function(expect_rc expected what)
  if(NOT case_rc EQUAL expected)
    fail("${what}: exit code ${case_rc}, expected ${expected}")
  endif()
endfunction()

function(expect_contains var needle what)
  string(FIND "${${var}}" "${needle}" pos)
  if(pos EQUAL -1)
    fail("${what}: expected to find '${needle}'")
  endif()
endfunction()

function(expect_lacks var needle what)
  string(FIND "${${var}}" "${needle}" pos)
  if(NOT pos EQUAL -1)
    fail("${what}: did not expect '${needle}'")
  endif()
endfunction()

function(expect_empty var what)
  if(NOT "${${var}}" STREQUAL "")
    fail("${what}: expected nothing, got '${${var}}'")
  endif()
endfunction()

set(ZIP "${PKGDIR}\\dlss-video-9.9.9-win64.zip")

# 1. The incident: no Release yet - view fails with "release not found" on stderr, the script creates the Release.
run_case(absent v9.9.9 owner/repo 1 0)
expect_rc(0 "absent")
expect_contains(case_log "gh release view \"v9.9.9\" --repo \"owner/repo\"" "absent")
expect_contains(case_log "gh release create \"v9.9.9\" \"${ZIP}\" --repo \"owner/repo\" --title \"dlss-video 9.9.9\" --generate-notes --verify-tag" "absent")
expect_lacks(case_log "release upload" "absent")
expect_lacks(case_log "0.0.1" "absent")

# 2. The Release exists (made by hand first): upload with --clobber, no create.
run_case(exists v9.9.9 owner/repo 0 0)
expect_rc(0 "exists")
expect_contains(case_log "gh release upload \"v9.9.9\" \"${ZIP}\" --repo \"owner/repo\" --clobber" "exists")
expect_lacks(case_log "release create" "exists")

# 3-4. gh fails while creating or uploading: the failure reaches the workflow.
run_case(create_fails v9.9.9 owner/repo 1 1)
expect_rc(1 "create_fails")
run_case(upload_fails v9.9.9 owner/repo 0 1)
expect_rc(1 "upload_fails")

# 5-7. Refusals before any gh call: no package for the tag's version, a tag without "v", no repository.
run_case(no_package v9.9.8 owner/repo 1 0)
expect_rc(2 "no_package")
expect_empty(case_log "no_package: gh calls")
run_case(bad_tag 9.9.9 owner/repo 1 0)
expect_rc(2 "bad_tag")
expect_empty(case_log "bad_tag: gh calls")
run_case(no_repository v9.9.9 "" 1 0)
expect_rc(2 "no_repository")
expect_empty(case_log "no_repository: gh calls")

# 8. The workflow calls the script through cmd and has no inline gh release calls left.
file(READ "${SOURCE_DIR}/.github/workflows/release.yml" workflow)
expect_contains(workflow "scripts\\attach_release.cmd" "release.yml")
expect_contains(workflow "shell: cmd" "release.yml")
string(REGEX MATCH "\n[^#\n]*gh release (view|upload|create)" inline "${workflow}")
expect_empty(inline "release.yml: gh release outside scripts/attach_release.cmd")

get_property(failed GLOBAL PROPERTY attach_release_failures)
list(LENGTH failed count)
if(count GREATER 0)
  message(FATAL_ERROR "${count} check(s) failed")
endif()
message(STATUS "attach_release: all checks passed")
