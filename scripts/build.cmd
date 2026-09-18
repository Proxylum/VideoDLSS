@echo off
setlocal
rem Usage: scripts\build.cmd [debug|release] [--no-tests]
set PRESET=%1
if "%PRESET%"=="" set PRESET=release
set RUN_TESTS=1
if "%2"=="--no-tests" set RUN_TESTS=0

rem vcvars64.bat replaces VCPKG_ROOT with the VS-bundled vcpkg; keep the user's choice.
set USER_VCPKG_ROOT=%VCPKG_ROOT%

if not defined VSCMD_VER (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
  if "%VSPATH%"=="" (echo Visual Studio 2022 with C++ tools not found & exit /b 1)
  call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
)

if not "%USER_VCPKG_ROOT%"=="" set VCPKG_ROOT=%USER_VCPKG_ROOT%
if "%VCPKG_ROOT%"=="" (
  if exist C:\vcpkg\vcpkg.exe (set VCPKG_ROOT=C:\vcpkg) else (
    echo VCPKG_ROOT is not set & exit /b 1
  )
)

pushd "%~dp0.."
cmake --preset %PRESET% || (popd & exit /b 1)
cmake --build --preset %PRESET% || (popd & exit /b 1)
if "%RUN_TESTS%"=="1" ctest --preset %PRESET% || (popd & exit /b 1)
popd
endlocal
