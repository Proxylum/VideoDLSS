@echo off
rem CI / developer build: MSVC environment -> configure -> build -> tests (unit, integration, app, golden).
rem Used by .github/workflows/ci.yml on the self-hosted Windows runner (docs/ci.md) and runnable by hand from the repo root.
rem Environment expected: VCPKG_ROOT, CUDA_PATH_V12_4, TENSORRT_ROOT, NV_OPTICAL_FLOW_SDK_ROOT, QT_ROOT, DLSS_SDK_ROOT
rem (see docs/dll-setup.md); DLSSVID_PRESET selects the CMake preset (default: release).
setlocal
if "%DLSSVID_PRESET%"=="" set DLSSVID_PRESET=release
set VSLANG=1033
chcp 65001 >nul
cd /d "%~dp0.."

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" ( echo vswhere.exe not found: install Visual Studio 2022+ with the C++ toolset & exit /b 1 )
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if "%VSPATH%"=="" ( echo Visual Studio with the C++ toolset not found & exit /b 1 )
for %%v in (VCPKG_ROOT TENSORRT_ROOT NV_OPTICAL_FLOW_SDK_ROOT QT_ROOT DLSS_SDK_ROOT) do if not defined %%v echo warning: %%v is not set (docs/dll-setup.md)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

echo === configure (%DLSSVID_PRESET%)
cmake --preset %DLSSVID_PRESET% || exit /b 1
echo === build
cmake --build --preset %DLSSVID_PRESET% -- -k 0 || exit /b 1
echo === tests
ctest --preset %DLSSVID_PRESET% --output-on-failure --output-junit test-report.xml || exit /b 1
echo === done
endlocal
