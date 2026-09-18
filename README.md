# dlss-video

Offline DLSS video pipeline for Windows + NVIDIA RTX: a video file goes through
`decode → depth / motion vectors → upscale (RTX VSR / DLSS SR) → tonemap → DLSS 5 Neural
Rendering → DLSS Frame Generation → encode`, every intermediate **pass** (depth, motion
vectors, masks, colour stages) can be exported, imported and inspected in a viewport.

Current stage: **0 — skeleton** (see [docs/plans/00-skeleton.md](docs/plans/00-skeleton.md)).
Build system, D3D12 device, CUDA interop, FFmpeg decode → GPU → encode, CLI `dlssvid process --passthrough`.

## Requirements

| Component | Version | Notes |
|---|---|---|
| Windows | 10/11 x64 | |
| Visual Studio 2022 or newer | MSVC 14.4x+, Windows SDK 10.0.22621+ | Desktop C++ workload (includes Ninja). Use the same VS instance that vcpkg picks (the newest one) — mixing toolsets breaks linking |
| CMake | ≥ 3.28 | |
| vcpkg | any 2026 checkout | `VCPKG_ROOT` must be set; manifest mode, baseline pinned in `vcpkg.json` |
| CUDA Toolkit | 12.x | `CUDA_PATH_V12_4` (or pass `-DCUDAToolkit_ROOT`); optional — without it the build has no CUDA interop |
| NVIDIA driver | ≥ 616.56 for NR (stage 6); any recent driver for stage 0 | |
| Python | 3.12 | model export (stage 2), `depth_worker` |

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

## Layout

```
core/       gpu/ (D3D12, CUDA interop, frame cache) · io/ (decode, encode) · pipeline/ (IStage, Pipeline)
            stages/ (passthrough; depth/flow/upscale/tonemap/nr/fg arrive in stages 2–7) · convert/ (stage 1)
cli/        dlssvid
tests/      unit/ (Catch2, WARP-capable) · integration/ (synthetic clips, CLI as a process) · golden/ (stage 8)
docs/       architecture.md · conventions.md · dll-setup.md · plans/
models/     registry.json (models, URLs, hashes, licences) · export/ (ONNX export scripts)
fg_worker/  depth_worker/  app/   — placeholders until their stages
bin/nvidia/ user-supplied NVIDIA DLLs (git-ignored)
```

Tests: `ctest --preset release` (or run `dlssvid_unit_tests` / `dlssvid_integration_tests` directly;
Catch2 tags: `[gpu]`, `[cuda]`, `[integration]`, `[cli]`, `[nvdec]`). GPU-specific tests skip themselves when no NVIDIA GPU is present.
