@echo off
rem CMake configure with retries: `cmake --preset %DLSSVID_PRESET% <extra args>` up to 3 times, 30 s apart.
rem The configure step runs `vcpkg install`, which fetches the vcpkg registry from GitHub; on the self-hosted runner that
rem fetch fails now and then ("Fetching registry information ... failed") and a plain retry succeeds. Used by
rem ci-build.cmd and package.cmd; runnable by hand from the repo root (MSVC environment already set up).
setlocal
if "%DLSSVID_PRESET%"=="" set DLSSVID_PRESET=release
set TRIES=0
:configure
set /a TRIES+=1
cmake --preset %DLSSVID_PRESET% %*
if not errorlevel 1 goto ok
if %TRIES% geq 3 (
  echo configure failed 3 times
  exit /b 1
)
echo configure failed (attempt %TRIES% of 3^) - retrying in 30 s
rem `timeout` needs a console (it fails at once under a CI runner with redirected input); ping is the portable sleep
ping -n 31 127.0.0.1 >nul
goto configure
:ok
endlocal
exit /b 0
