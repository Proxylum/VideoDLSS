# dlss-video

Offline DLSS video pipeline for Windows + NVIDIA RTX: a video file goes through
`decode → depth / motion vectors → upscale (RTX VSR / DLSS SR) → tonemap → DLSS 5 Neural
Rendering → DLSS Frame Generation → encode`, every intermediate **pass** (depth, motion
vectors, masks, colour stages) can be exported, imported and inspected in a viewport.

Current stage: **2 — depth** (see [docs/plans/02-depth.md](docs/plans/02-depth.md); earlier: [00-skeleton](docs/plans/00-skeleton.md), [01-passes](docs/plans/01-passes.md)).
`dlssvid depth` estimates `depth_raw` / `depth_dlss` with Depth Anything 3 (metric / mono) or Video Depth Anything
through TensorRT FP16, or through the PyTorch reference worker; temporal scale/shift stabilisation, edge-aware
upsampling, TAE metric. Stage 1 gave passes as file sequences + `manifest.json` (EXR / PNG16 / TIFF / NPZ / raw),
export presets and the raw ↔ DLSS conventions ([docs/conventions.md](docs/conventions.md)).

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
dlssvid models  list
```

## Layout

```
core/       gpu/ (D3D12, CUDA interop, frame cache) · io/ (decode, encode) · pipeline/ (IStage, Pipeline)
            passes/ (PassImage, Manifest, PassSequence, formats/: EXR, PNG, TIFF, NPZ, raw)
            convert/ (depth raw ↔ reverse-Z, mv forward ↔ backward, YUV → RGB)
            ml/ (TrtLoader: runtime-loaded TensorRT, TrtEngine, ModelRegistry) · util/ (Subprocess, Sha256, Half)
            stages/passthrough · stages/depth (IDepthEstimator, DA3/VDA via TensorRT, worker client, pre/post-processing, DepthStage)
cli/        dlssvid
tests/      unit/ (Catch2, WARP-capable) · integration/ (synthetic clips, CLI as a process) · golden/ (stage 8)
docs/       architecture.md · conventions.md · dll-setup.md · plans/
models/     registry.json (models, URLs, hashes, licences) · export/ (ONNX export scripts)
depth_worker/ worker.py — PyTorch reference backends (da3, vda), icdepth placeholder, stub for tests
models/     registry.json · export/ (fetch.py, export_da3.py, export_vda.py, loaders) · cache/ (weights, ONNX, engines; git-ignored)
fg_worker/  app/   — placeholders until their stages
bin/nvidia/ user-supplied NVIDIA DLLs (git-ignored)
```

Tests: `ctest --preset release` (or run `dlssvid_unit_tests` / `dlssvid_integration_tests` directly;
Catch2 tags: `[gpu]`, `[cuda]`, `[integration]`, `[cli]`, `[nvdec]`, `[passes]`, `[formats]`, `[convert]`). The NPZ ↔ numpy test needs `python` with numpy on PATH and skips otherwise. GPU-specific tests skip themselves when no NVIDIA GPU is present.
