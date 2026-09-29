@echo off
rem Build the release package (CPack ZIP, plus an NSIS installer when makensis is installed): docs/release.md.
rem   package.cmd        lean package (executables, docs, golden tests) — what CI builds
rem   package.cmd full   + TensorRT runtime, cudart64_12.dll and the ONNX models: self-contained, several GB
setlocal
if "%DLSSVID_PRESET%"=="" set DLSSVID_PRESET=release
set VSLANG=1033
chcp 65001 >nul
cd /d "%~dp0.."

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" ( echo vswhere.exe not found: install Visual Studio 2022+ with the C++ toolset & exit /b 1 )
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if "%VSPATH%"=="" ( echo Visual Studio with the C++ toolset not found & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set PKG_FULL=OFF
if /i "%~1"=="full" set PKG_FULL=ON
call "%~dp0configure.cmd" -DDLSSVID_PACKAGE_FULL=%PKG_FULL% || exit /b 1
cmake --build --preset %DLSSVID_PRESET% || exit /b 1
cpack --preset %DLSSVID_PRESET% || exit /b 1
echo === package written to build\%DLSSVID_PRESET%\
dir /b build\%DLSSVID_PRESET%\dlss-video-*
endlocal
