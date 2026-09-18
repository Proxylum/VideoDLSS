# dlss-video

Offline DLSS video pipeline for Windows + NVIDIA RTX: a video file goes through
`decode → depth / motion vectors → upscale (RTX VSR / DLSS SR) → tonemap → DLSS 5 Neural
Rendering → DLSS Frame Generation → encode`, every intermediate **pass** (depth, motion
vectors, masks, colour stages) can be exported, imported and inspected in a viewport.

Current stage: **4 — viewport** (see [docs/plans/04-viewport.md](docs/plans/04-viewport.md); earlier:
[00-skeleton](docs/plans/00-skeleton.md), [01-passes](docs/plans/01-passes.md), [02-depth](docs/plans/02-depth.md),
[03-motion-vectors](docs/plans/03-motion-vectors.md)). `dlssvid-gui` is a Qt 6.8 shell around a D3D12 viewport
that shows the source video and every pass folder in single / overlay / 2x2 modes with colour maps, motion-vector
and mask displays, a wipe, a pixel probe and a timeline; the viewport state lives in a project file
(`*.dlssvid.json`) and everything the GUI shows can be rendered from the CLI (`dlssvid render`).
`dlssvid flow` estimates `mv_raw` (forward flow) with the NVIDIA Optical Flow Accelerator fed straight from NVDEC
frames, or with SEA-RAFT through TensorRT, converts to `mv_dlss` (backward, occlusions resolved with depth) and
reports the warp-PSNR test; `dlssvid depth` (stage 2) gives `depth_raw` / `depth_dlss` with Depth Anything 3 or Video
Depth Anything through TensorRT or the PyTorch worker, now with motion-compensated TAE (`--mv-dir`). Stage 1 gave
passes as file sequences + `manifest.json` and the raw ↔ DLSS conventions ([docs/conventions.md](docs/conventions.md)).

## Requirements

| Component | Version | Notes |
|---|---|---|
| Windows | 10/11 x64 | |
| Visual Studio 2022 or newer | MSVC 14.4x+, Windows SDK 10.0.22621+ | Desktop C++ workload (includes Ninja). Use the same VS instance that vcpkg picks (the newest one) — mixing toolsets breaks linking |
| CMake | ≥ 3.28 | |
| vcpkg | any 2026 checkout | `VCPKG_ROOT` must be set; manifest mode, baseline pinned in `vcpkg.json` |
| CUDA Toolkit | 12.x | `CUDA_PATH_V12_4` (or pass `-DCUDAToolkit_ROOT`); optional — without it the build has no CUDA interop |
| NVIDIA driver | ≥ 616.56 for NR (stage 6); any recent driver for stage 0 | |
| Python | 3.12 | `models/export/.venv` with PyTorch cu126 + `tensorrt-cu12` (see [models/export/README.md](models/export/README.md)); used by `depth_worker` and by the ONNX export the TensorRT backend triggers on first use |
| TensorRT | 10.16 headers (`TENSORRT_ROOT`, e.g. a checkout of NVIDIA/TensorRT tag v10.16) | DLLs come from the `tensorrt-cu12` pip package in the venv and are loaded at runtime — no import libraries |
| Optical Flow SDK | headers (`NV_OPTICAL_FLOW_SDK_ROOT`, a checkout of NVIDIA/NVIDIAOpticalFlowSDK) | `nvofapi64.dll` ships with the driver (API 5.0 on 591.86); the public headers are API 2.0 and stay compatible |
| Qt | 6.8 (msvc2022_64), `QT_ROOT` = `.../Qt/6.8.x/msvc2022_64` | GUI only (`DLSSVID_BUILD_APP`, default ON; skipped with a warning when Qt is not found). `windeployqt` copies the runtime next to the executables |
| DXC | vcpkg `directx-dxc` (automatic) | the viewport shaders are compiled at build time into headers (SM 6.0, runs on WARP) |

NVIDIA SDKs (DLSS/NGX, Streamline, RTX Video, Optical Flow) and Qt are needed from stage 4
onwards — see [docs/dll-setup.md](docs/dll-setup.md). No NVIDIA binaries or model weights are
committed to this repository.

## Build

```bat
:: Developer Command Prompt for VS 2022 (or call vcvars64.bat), with VCPKG_ROOT set
cmake --preset release
cmake --build --preset release
ctest --preset release
```

`scripts\build.cmd [debug|release]` does the three steps and sets up the MSVC environment itself.
The first configure builds FFmpeg and friends through vcpkg (10–30 minutes); results are cached
in `%LOCALAPPDATA%\vcpkg\archives`.

## Use

```bat
build\release\bin\dlssvid info input.mp4
build\release\bin\dlssvid process --passthrough -i input.mp4 -o output.mp4 --codec hevc_nvenc
build\release\bin\dlssvid process --passthrough -i input.mp4 -o output.mkv --codec ffv1   :: lossless
```

`--hwaccel cuda` decodes with NVDEC; `--warp` runs on the D3D12 software adapter (tests, CI without a GPU).

```bat
dlssvid export  -i input.mp4 -o passes\color --range 0-299            :: color_source as EXR half + manifest.json
dlssvid export  -i input.mp4 -o passes\comfy --preset comfyui           :: PNG16 (colour) / NPZ (depth, mv)
dlssvid export  --from-dir passes\depth_raw -o passes\depth_npz --format npz
dlssvid export  -i input.mp4 -o passes
uke --preset nuke --depth-dir passes\depth_raw --mv-dir passes\mv_raw
dlssvid import  -i passes\depth_raw --expect-size 1920x1080 --expect-frames 300
dlssvid convert -i passes\depth_raw -o passes\depth_dlss --to depth_dlss --near 0.1 --far 1000
dlssvid convert -i passes\mv_raw -o passes\mv_dlss --to mv_dlss --target 3840x2160 --depth-dir passes\depth_raw
dlssvid depth   -i input.mp4 -o passes --backend da3                     :: DA3METRIC-LARGE via TensorRT -> depth_raw + depth_dlss
dlssvid depth   -i input.mp4 -o passes --backend vda --model metric-vda-small
dlssvid depth   -i input.mp4 -o passes --backend worker:da3              :: PyTorch reference path (depth_worker)
dlssvid flow    -i input.mp4 -o passes --backend ofa                     :: NVDEC -> OFA, mv_raw + mv_dlss, warp PSNR
dlssvid flow    -i input.mp4 -o passes --backend searaft --depth-dir passes\depth_raw --target 3840x2160
dlssvid depth   -i input.mp4 -o passes --backend da3 --mv-dir passes\mv_dlss   :: TAE with motion compensation
dlssvid models  list
```

Viewport (stage 4): the GUI and the CLI share one renderer, frame cache and project file.

```bat
build\release\bin\dlssvid-gui input.mp4                                    :: opens/creates input.dlssvid.json next to the video
build\release\bin\dlssvid-gui project.dlssvid.json
dlssvid project init -i input.mp4 --passes passes -o input.dlssvid.json       :: project file + discovered passes
dlssvid project show --project input.dlssvid.json
dlssvid render --project input.dlssvid.json --frame 42 -o frame.png           :: the saved viewport state
dlssvid render --project p.json --frame 42 --source depth_raw --display turbo -o depth.png
dlssvid render --project p.json --frame 42 --mode grid --sources source,depth_raw,mv_raw,depth_dlss --size 1920x1080 -o grid.png
dlssvid render --project p.json --frame 42 --layers source,depth_raw:viridis:0.5:multiply --wipe v:0.5 -o overlay.png
dlssvid render --project p.json --frame 42 --source mv_raw --display mv_arrows --zoom 2 --center-x 960 --center-y 540 -o arrows.png
dlssvid render -i input.mp4 --passes passes --bench 60                         :: scrubbing benchmark (cold / warm / random jumps)
dlssvid render --project p.json --frame 10 --source depth_raw --save-state -o x.png   :: write the state back into the project
```

GUI keys: `1`…`9` source, `Ctrl+1/2/3` single / overlay / grid, wheel = zoom to cursor (25–800 %), middle drag = pan,
`F` fit, `Ctrl+0` 1:1, `W` wipe (left drag moves it), click a 2x2 cell = expand, `Space` play, `,`/`.` step,
`Ctrl+Shift+S` PNG screenshot with cell labels, `Ctrl+S` save project. Stages are started from the project panel
(`dlssvid depth|flow` as a task with progress) and their passes appear in the viewport when finished.

## Layout

```
core/       gpu/ (D3D12, CUDA interop, frame cache) · io/ (decode, encode) · pipeline/ (IStage, Pipeline)
            passes/ (PassImage, Manifest, PassSequence, formats/: EXR, PNG, TIFF, NPZ, raw)
            convert/ (depth raw ↔ reverse-Z, mv forward ↔ backward, YUV → RGB)
            ml/ (TrtLoader: runtime-loaded TensorRT, TrtEngine, ModelRegistry) · util/ (Subprocess, Sha256, Half)
            stages/passthrough · stages/depth (IDepthEstimator, DA3/VDA via TensorRT, worker client, pre/post-processing, DepthStage)
            stages/flow (IFlowEstimator, OfaFlowEstimator via nvofapi, TrtFlowEstimator for SEA-RAFT, FlowStage) · convert/Warp (warp-PSNR, warped TAE)
            viewport/ (ViewportState, ViewportRenderer + shaders/, FrameStore, Project)
cli/        dlssvid (+ ViewportCommands: project, render)
app/        dlssvid-gui — Qt 6.8 Widgets shell (AppModel, ViewportWindow, panels, TaskQueue)
tests/      unit/ (Catch2, WARP-capable) · integration/ (synthetic clips, CLI as a process) · app/ (Qt offscreen) · golden/ (stage 8)
docs/       architecture.md · conventions.md · dll-setup.md · plans/
models/     registry.json (models, URLs, hashes, licences) · export/ (ONNX export scripts)
depth_worker/ worker.py — PyTorch reference backends (da3, vda), icdepth placeholder, stub for tests
models/     registry.json · export/ (fetch.py, export_da3.py, export_vda.py, loaders) · cache/ (weights, ONNX, engines; git-ignored)
fg_worker/  placeholder until stage 7
bin/nvidia/ user-supplied NVIDIA DLLs (git-ignored)
```

Tests: `ctest --preset release` (or run `dlssvid_unit_tests` / `dlssvid_integration_tests` directly;
Catch2 tags: `[gpu]`, `[cuda]`, `[integration]`, `[cli]`, `[nvdec]`, `[passes]`, `[formats]`, `[convert]`). The NPZ ↔ numpy test needs `python` with numpy on PATH and skips otherwise. GPU-specific tests skip themselves when no NVIDIA GPU is present.
