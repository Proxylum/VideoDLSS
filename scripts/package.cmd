@echo off
rem Build the release package (CPack ZIP, plus an NSIS installer when makensis is installed): docs/release.md.
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

cmake --preset %DLSSVID_PRESET% || exit /b 1
cmake --build --preset %DLSSVID_PRESET% || exit /b 1
cpack --preset %DLSSVID_PRESET% || exit /b 1
echo === package written to build\%DLSSVID_PRESET%\
dir /b build\%DLSSVID_PRESET%\dlss-video-*
endlocal
