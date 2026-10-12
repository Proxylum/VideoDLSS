@echo off
rem Attach the release package to the GitHub Release of a version tag; create the Release when it does not exist yet.
rem   attach_release.cmd <tag> [<package dir>]      e.g. attach_release.cmd v0.4.0 build\release
rem The package is <package dir>\dlss-video-<version>-win64.zip with <version> = the tag without its leading "v", so a
rem tag that does not match the CMake version fails here instead of shipping an older ZIP left in the folder.
rem Needs gh on PATH, GH_TOKEN and GITHUB_REPOSITORY (owner/name) in the environment. Used by .github/workflows/release.yml
rem (shell: cmd). Plain cmd on purpose: under Windows PowerShell 5.1 with $ErrorActionPreference = 'stop' (how GitHub runs
rem `shell: powershell`) the "release not found" gh prints to stderr becomes a terminating error (incident TASK-0037).
rem Exit codes: 0 attached, 1 gh failed, 2 bad arguments or environment (gh is not called).
setlocal
set "TAG=%~1"
set "PKGDIR=%~2"
if "%PKGDIR%"=="" set "PKGDIR=build\release"
if "%TAG%"=="" goto usage
if not "%TAG:~0,1%"=="v" (
  echo attach_release: the tag must look like vX.Y.Z, got "%TAG%"
  exit /b 2
)
set "VERSION=%TAG:~1%"
set "ZIP=%PKGDIR%\dlss-video-%VERSION%-win64.zip"
if not exist "%ZIP%" (
  echo attach_release: package not found: %ZIP%
  exit /b 2
)
if "%GITHUB_REPOSITORY%"=="" (
  echo attach_release: GITHUB_REPOSITORY is not set
  exit /b 2
)

rem `call`: gh is gh.exe on the runner, but the test puts a gh.cmd first on PATH, and a batch file run without `call`
rem never returns to this script.
call gh release view "%TAG%" --repo "%GITHUB_REPOSITORY%" >nul 2>&1
if errorlevel 1 goto create

echo attach_release: release %TAG% exists - uploading %ZIP%
call gh release upload "%TAG%" "%ZIP%" --repo "%GITHUB_REPOSITORY%" --clobber
if errorlevel 1 goto failed
exit /b 0

:create
echo attach_release: release %TAG% not found - creating it with %ZIP%
call gh release create "%TAG%" "%ZIP%" --repo "%GITHUB_REPOSITORY%" --title "dlss-video %VERSION%" --generate-notes --verify-tag
if errorlevel 1 goto failed
exit /b 0

:failed
echo attach_release: gh failed
exit /b 1

:usage
echo usage: attach_release.cmd ^<tag^> [^<package dir^>]
exit /b 2
