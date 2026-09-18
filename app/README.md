# app — Qt 6 shell (stage 4)

`dlssvid-gui`: Qt 6.8 Widgets around the viewport. Everything here is UI only — the state model
(`core/viewport/ViewportState`), the frame cache (`core/viewport/FrameStore`), the D3D12 composite
(`core/viewport/ViewportRenderer` + `core/viewport/shaders/*.hlsl`) and the project file
(`core/viewport/Project`) live in `core/` and are shared with the CLI (`dlssvid render`,
`dlssvid project`).

| File | Role |
|---|---|
| `AppModel` | project + viewport state + FrameStore + playback timer; the panels only talk to it |
| `ViewportWindow` | `QWindow` with its own DXGI swapchain on the pipeline's device: zoom/pan, wipe drag, probe, cell click |
| `TimelineWidget` | scrubber, frame number, play/pause, ±1 |
| `ProjectPanel` | source, passes with status, stages with enable/params and «Запустить» (CLI via `TaskQueue`); the `nr` stage adds «Пропатчить DLL…» — picks your `nvngx_dlssnr.dll` (and the patcher / CUDA 13.3 folders when `DLSSNR_PATCHER_ROOT` / `CUDA_PATH_V13_3` are unset) and queues `dlssvid nr-patch` (stage 6) |
| `InspectorPanel` | layer stack and display settings, wipe, grid cell sources, pixel probe |
| `TaskQueue` | sequential `dlssvid <stage>` processes with progress parsed from `N/M frames` |
| `LogPanel` | spdlog sink |
| `MainWindow` | docks, menus, hotkeys (1..9 sources, F fit, Ctrl+0 1:1, W wipe, Space play, ,/. step), PNG screenshot with cell labels |

Build: `-DDLSSVID_BUILD_APP=ON` (default) with `QT_ROOT` pointing at `.../Qt/6.8.x/msvc2022_64`;
without Qt the GUI is skipped with a warning. `windeployqt` runs post-build, so the executable
starts from the build tree. Tests: `tests/app/` (offscreen platform).
