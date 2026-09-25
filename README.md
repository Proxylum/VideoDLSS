# dlss-video

Offline DLSS video pipeline for Windows + NVIDIA RTX: a video file goes through
`decode → depth / motion vectors → upscale (DLSS SR; Real-ESRGAN via TensorRT; RealBasicVSR via a PyTorch worker; NIS / bicubic) → tonemap → DLSS 5 Neural
Rendering → DLSS Frame Generation → encode`, every intermediate **pass** (depth, motion
vectors, masks, colour stages) can be exported, imported and inspected in a viewport.

Version **0.3.0** ([CHANGELOG.md](CHANGELOG.md)): stages 0–9 are done — stage 9 is the operator UI
([docs/plans/09-ui-ux.md](docs/plans/09-ui-ux.md): start page, project screen, processing page, «До | После», pass fingerprints
and versions); release procedure in [docs/release.md](docs/release.md). Stage 8 — release (see [docs/plans/08-release.md](docs/plans/08-release.md); earlier:
[00-skeleton](docs/plans/00-skeleton.md), [01-passes](docs/plans/01-passes.md), [02-depth](docs/plans/02-depth.md),
[03-motion-vectors](docs/plans/03-motion-vectors.md), [04-viewport](docs/plans/04-viewport.md),
[05-upscale](docs/plans/05-upscale.md), [06-nr](docs/plans/06-nr.md), [07-fg](docs/plans/07-fg.md)). `dlssvid process` runs the
whole pipeline over the pass cache (depth → flow → upscale → nr → fg, complete passes are reused) and encodes the result
with the source audio; `dlssvid bench` prints ms/frame per stage; golden tests (`dlssvid_golden_tests`) check the pipeline
against stored reference frames, also on an installed copy; the package is built with CPack ([docs/release.md](docs/release.md)),
CI runs on a self-hosted RTX runner ([docs/ci.md](docs/ci.md)). Stage 7: `dlssvid fg` doubles (x2..x4) the frame rate into
`color_fg`: DLSS Frame Generation through the NGX API of the DLSS SDK (no swapchain, no Streamline — the current frame,
`depth_dlss` and `mv_dlss` go in as textures, the interpolated frame comes out), RIFE 4.9 through TensorRT as the
baseline, and a naive blend; the generated frames are scored against ground truth (a half-rate clip) in
[docs/benchmarks.md](docs/benchmarks.md). Stage 6: `dlssvid nr` runs DLSS 5 Neural Rendering (NGX Feature 18 through the
user-supplied `nvngx_dlssnr.dll`, patched for RTX 20/30/40 with `dlssvid nr-patch` / the GUI button) over `color_sr`
with depth / motion-vector guides and masks into `color_nr`, after a tonemap step; `dlssvid nr --check` prints the
GPU / driver / DLL / CreateFeature(18) diagnostics ([docs/dll-setup.md](docs/dll-setup.md)); verified on an RTX 4070 Ti SUPER
(driver 616.92, patched DLL): ~16 ms per 1440p frame on the GPU ([docs/benchmarks.md](docs/benchmarks.md)). Stage 5: `dlssvid upscale`
produces the `color_sr` pass through one `IUpscaler` interface: DLSS Super Resolution over NGX (with the jitter
emulation of ТЗ §3) as the default, open-source models through TensorRT (`trt`: Real-ESRGAN x2plus by default, the compact
general-x4v3 and x4plus; one ONNX per model with dynamic size, one engine per tile, tiles with 16 px of context — TASK-0022),
RealBasicVSR through the PyTorch worker `sr_worker/` (`worker`: temporal propagation over windows of frames, the temporally
consistent option — TASK-0023),
NVIDIA Image Scaling (always available, WARP-capable, the fallback) and a bicubic baseline (the RTX VSR stub was removed on
2026-09-24: the RTX Video SDK needs an NVIDIA developer account). `dlssvid compare` measures PSNR/SSIM for the A/B of
[docs/benchmarks.md](docs/benchmarks.md). Stage 4: `dlssvid-gui` is a Qt 6.8 shell around a D3D12 viewport
that shows the source video and every pass folder in single / overlay / 2x2 modes with colour maps, motion-vector
and mask displays, a wipe, a pixel probe and a timeline; the viewport state lives in a project file
(`*.dlssvid.json`) and everything the GUI shows can be rendered from the CLI (`dlssvid render`).
`dlssvid flow` estimates `mv_raw` (forward flow) with the NVIDIA Optical Flow Accelerator fed straight from NVDEC
frames, or with SEA-RAFT through TensorRT, converts to `mv_dlss` (backward, occlusions resolved with depth) and
reports the warp-PSNR test; `dlssvid depth` (stage 2) gives `depth_raw` / `depth_dlss` with Depth Anything 3 or Video
Depth Anything through TensorRT or the PyTorch worker, now with motion-compensated TAE (`--mv-dir`). Stage 1 gave
passes as file sequences + `manifest.json` and the raw ↔ DLSS conventions ([docs/conventions.md](docs/conventions.md)).

## Contents

- [Quick start (binary package)](#quick-start-binary-package)
- [Requirements](#requirements)
- [Install](#install)
- [NVIDIA runtimes](#nvidia-runtimes)
- [Models](#models)
- [Build from source](#build-from-source)
- [Use](#use)
- [Layout](#layout)
- [Tests](#tests)
- [Licences and third-party components](#licences-and-third-party-components)

## Quick start (binary package)

1. Download `dlss-video-<version>-win64.zip` from the releases page (or build it yourself: `scripts\package.cmd`,
   [Build from source](#build-from-source)) and unzip it into a folder you can write to — not `Program Files`: models
   and TensorRT engines are cached next to the executables (`models\cache\`, or wherever `DLSSVID_MODELS_DIR` points).
2. Install the [NVIDIA runtimes](#nvidia-runtimes) you need: the CUDA 12 runtime and TensorRT 10.16 for the neural
   stages (depth, motion vectors, Real-ESRGAN, RIFE), the DLSS DLLs in `bin\nvidia\` for DLSS Super Resolution,
   Frame Generation and Neural Rendering.
3. Create the Python environment ([Install → Python environment](#python-environment)): it exports the models to
   ONNX on first use and runs the PyTorch workers (RealBasicVSR, the depth reference path).
4. [Models](#models) download themselves on first use (`dlssvid models list` shows what is already there).
5. Check the machine and go:

```bat
bin\dlssvid.exe info clip.mp4                         :: streams, GPU, NVENC encoders, TensorRT
bin\dlssvid.exe nr --check                            :: Neural Rendering: driver >= 616.56, DLL, CreateFeature(18)
bin\dlssvid.exe fg --check                            :: Frame Generation: DLL, MultiFrameCountMax
bin\dlssvid-gui.exe                                   :: drop a video on the start page, «Обработать», «До | После»
bin\dlssvid.exe process -i clip.mp4 -o result.mp4     :: depth -> flow -> upscale x2 -> nr -> fg x2 -> encode (+ audio)
```

Without an NVIDIA GPU the CLI and the viewport still work on the D3D12 software adapter (`--warp`) with the
non-NVIDIA backends: NIS / bicubic upscale, the blend frame generator, the stub depth / NR backends.

## Requirements

### To run

| Component | Needed for | Notes |
|---|---|---|
| Windows 10/11 x64 | everything | the pipeline is D3D12 + CUDA, Windows only |
| NVIDIA GPU | DLSS SR and NIS: RTX 20 and newer · DLSS Frame Generation: RTX 40 and newer (x3/x4 on RTX 50) · Neural Rendering: RTX 50 with the official DLL, RTX 20/30/40 with a patched copy · TensorRT models (depth, flow, Real-ESRGAN, RIFE): RTX 20 and newer | engines are built for your GPU on first use; other GPUs and the WARP software adapter run the non-NVIDIA backends only |
| NVIDIA driver | ≥ 616.56 for Neural Rendering; any recent driver for the rest | `nvofapi64.dll` (Optical Flow) and NVDEC / NVENC come with the driver |
| CUDA 12 runtime (`cudart64_12.dll`) | NVDEC frames on the GPU, TensorRT | from the CUDA Toolkit 12.x installer (on `PATH`) or bundled in the full package |
| TensorRT 10.16 runtime (`nvinfer_10.dll`, `nvonnxparser_10.dll`, `nvinfer_plugin_10.dll`, builder resources) | depth (DA3, VDA), SEA-RAFT flow, Real-ESRGAN, RIFE | the pip package `tensorrt-cu12==10.16.1.11` in the Python environment is found by itself; or `DLSSVID_TENSORRT_DIR`; or `bin\tensorrt\` of the full package |
| Python 3.12 + the venv of `models\export\requirements.txt` (PyTorch cu126) | ONNX export of every model on first use, `depth_worker` (DA3 / VDA in PyTorch), `sr_worker` (RealBasicVSR), `nr-patch` | found at `models\export\.venv\Scripts\python.exe` or `DLSSVID_PYTHON` |
| Disk | models and engines: ~5 GB of ONNX plus TensorRT engines per GPU; passes: EXR sequences, GBs per minute of 4K | `DLSSVID_MODELS_DIR` moves the cache; the passes folder is chosen per project |

### To build from source

| Component | Version | Notes |
|---|---|---|
| Windows | 10/11 x64 | |
| Visual Studio 2022 or newer | MSVC 14.4x+, Windows SDK 10.0.22621+ | Desktop C++ workload (includes Ninja). Use the same VS instance that vcpkg picks (the newest one) — mixing toolsets breaks linking |
| CMake | ≥ 3.28 | |
| vcpkg | any 2026 checkout | `VCPKG_ROOT` must be set; manifest mode, baseline pinned in `vcpkg.json` |
| CUDA Toolkit | 12.x | `CUDA_PATH_V12_4` (or pass `-DCUDAToolkit_ROOT`); optional — without it the build has no CUDA interop and no TensorRT backends |
| TensorRT | 10.16 headers (`TENSORRT_ROOT`, e.g. a checkout of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) tag v10.16) | DLLs come from the `tensorrt-cu12` pip package in the venv and are loaded at runtime — no import libraries |
| Optical Flow SDK | headers (`NV_OPTICAL_FLOW_SDK_ROOT`, a checkout of [NVIDIA/NVIDIAOpticalFlowSDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK)) | `nvofapi64.dll` ships with the driver (API 5.0); the public headers are API 2.0 and stay compatible |
| DLSS SDK | clone of [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) (`DLSS_SDK_ROOT`): NGX headers, `nvsdk_ngx_d.lib`, `nvngx_dlss.dll` | DLSS SR, NR and FG (`DLSSVID_WITH_DLSS`, default ON when found). The build copies `nvngx_dlss.dll` / `nvngx_dlssg.dll` into `build\<preset>\bin\nvidia\` for development |
| Qt | 6.8 (msvc2022_64), `QT_ROOT` = `.../Qt/6.8.x/msvc2022_64` | GUI only (`DLSSVID_BUILD_APP`, default ON; skipped with a warning when Qt is not found). `windeployqt` copies the runtime next to the executables |
| DXC | vcpkg `directx-dxc` (automatic) | the viewport shaders are compiled at build time into headers (SM 6.0, runs on WARP) |
| Python | 3.12 | the same venv as above (`models/export/README.md`); also used by the tests that need numpy |

No NVIDIA binaries or model weights are committed to this repository; the environment variables are listed in
[docs/dll-setup.md](docs/dll-setup.md).

## Install

### Binary package

`dlss-video-<version>-win64.zip` (what CI builds, [docs/release.md](docs/release.md)) unpacks to:

```
dlss-video-<version>-win64/
  bin/                  dlssvid.exe, dlssvid-gui.exe, dlssvid_golden_tests.exe, the vcpkg DLLs (FFmpeg, OpenEXR, PNG, TIFF, spdlog),
                        Qt 6 (Qt6*.dll, platforms/, imageformats/, ...), dxcompiler.dll, dxil.dll, nvngx.dll_dlssvid.dll (the NR forwarder)
  bin/nvidia/           README.md — your NVIDIA DLLs go here (nvngx_dlss.dll, nvngx_dlssg.dll, nvngx_dlssnr.dll)
  models/registry.json  the model registry; models/export/ — the ONNX export scripts and requirements.txt (make the venv here)
  depth_worker/         PyTorch depth backends (DA3, VDA) behind a JSON-lines protocol
  sr_worker/            PyTorch video super-resolution (RealBasicVSR)
  tests/golden/         data of the golden tests (expected.json, ref/) — run them on the installed copy
  docs/, README.md, CHANGELOG.md, LICENSE
```

Not inside, on purpose: NVIDIA binaries (`nvngx_*.dll`, TensorRT, cudart), model weights and ONNX files, TensorRT
engines — see the next two sections. `scripts\package.cmd full` builds a self-contained package (TensorRT runtime,
`cudart64_12.dll`, the ONNX models: several GB) for machines without CUDA / TensorRT / Python; it is not published.

1. Unzip into a writable folder (models and engines are cached in `models\cache\` next to `registry.json`; set
   `DLSSVID_MODELS_DIR` to keep them elsewhere). Paths longer than 260 characters are fine (`\\?\`).
2. [NVIDIA runtimes](#nvidia-runtimes): the CUDA 12 runtime and TensorRT on `PATH` / in the venv, the DLSS DLLs in
   `bin\nvidia\`.
3. The [Python environment](#python-environment).
4. `bin\dlssvid.exe info clip.mp4` (streams, GPU, encoders, TensorRT), `bin\dlssvid.exe nr --check`, `bin\dlssvid.exe fg --check`.
5. Optional: the golden tests against the installed copy ([docs/release.md](docs/release.md)):

```bat
set DLSSVID_CLI=C:\dlss-video\bin\dlssvid.exe
set DLSSVID_GOLDEN_DIR=C:\dlss-video\tests\golden
C:\dlss-video\bin\dlssvid_golden_tests.exe "[golden]~[gpu]"   :: deterministic part (WARP); drop the filter for the GPU backends
```

### Python environment

One virtual environment serves the ONNX export, `depth_worker`, `sr_worker` and `nr-patch`. Make it inside
`models\export\` (the executables look there first; `DLSSVID_PYTHON` overrides):

```bat
cd models\export
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` pins `tensorrt-cu12==10.16.1.11`, so the TensorRT runtime DLLs land in the venv and the executables
find them there. Two depth models need their upstream code next to the venv: DA3 —
`pip install --no-deps -e <clone of ByteDance-Seed/Depth-Anything-3>`, VDA — a plain checkout of
`DepthAnything/Video-Depth-Anything` in `VDA_REPO` ([models/export/README.md](models/export/README.md)).

## NVIDIA runtimes

The application never redistributes NVIDIA binaries; every file below comes from NVIDIA under its own licence
(the DLSS SDK EULA for the SDK DLLs, the CUDA and TensorRT EULAs for the runtimes). Details and troubleshooting:
[docs/dll-setup.md](docs/dll-setup.md).

| Component | Where to get it | Where it goes | Used by |
|---|---|---|---|
| `cudart64_12.dll` (CUDA 12 runtime) | [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) — the installer puts it on `PATH` | `PATH`, or `bin\` (the full package bundles it) | NVDEC frames staying on the GPU, TensorRT |
| TensorRT 10.16: `nvinfer_10.dll`, `nvonnxparser_10.dll`, `nvinfer_plugin_10.dll`, `nvinfer_builder_resource_*.dll` | `pip install tensorrt-cu12==10.16.1.11` (part of `requirements.txt`) or the [TensorRT](https://developer.nvidia.com/tensorrt) zip | the venv (found automatically), or `DLSSVID_TENSORRT_DIR`, or `bin\tensorrt\` | depth (`depth --backend da3|vda`), SEA-RAFT (`flow --backend searaft`), Real-ESRGAN (`upscale --backend trt`), RIFE (`fg --backend rife`) |
| `nvngx_dlss.dll` | [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) → `lib/Windows_x86_64/rel/` | `bin\nvidia\` | DLSS Super Resolution (`upscale --backend dlss`, the default) |
| `nvngx_dlssg.dll` | the same DLSS SDK folder | `bin\nvidia\` | DLSS Frame Generation (`fg`, RTX 40 and newer) |
| `nvngx_dlssnr.dll` | your own copy from a driver or game with DLSS 5 — RTX 50: as is; RTX 20/30/40: patched on your machine with `dlssvid nr-patch` ([dlssnr-patcher](https://github.com/dev-camo/dlssnr-patcher) + the `ptxas` / `fatbinary` / `cuobjdump` tools of CUDA Toolkit 13.3) | `bin\nvidia\` (the patch leaves a sidecar `nvngx_dlssnr.dll.patch.json`) | Neural Rendering (`nr`; the GUI has a «Пропатчить DLL…» button) |
| `nvofapi64.dll` | ships with the driver | nothing to do | Optical Flow Accelerator (`flow --backend ofa`) |
| NVDEC / NVENC | ship with the driver | nothing to do | `--hwaccel cuda` decoding, `h264_nvenc` / `hevc_nvenc` encoding |

`DLSSVID_NVIDIA_DLL_DIR` or `--dll-dir` points at another folder of DLSS DLLs; `bin\dlssvid.exe nr --check` and
`fg --check` print which file was found, its SHA-256 and whether the feature can be created on this GPU / driver.

## Models

Weights are not in the repository. [models/registry.json](models/registry.json) lists every model with its stage,
source, licence, download location and parameters; the files land in `models\cache\` (git-ignored;
`DLSSVID_MODELS_DIR` moves the whole folder).

```bat
bin\dlssvid.exe models list                            :: what the registry knows and what is already cached
bin\dlssvid.exe models fetch realesrgan-x2plus         :: models with a plain download URL (Real-ESRGAN): download + sha256 check
models\export\.venv\Scripts\python models\export\fetch.py da3metric-large sea-raft-spring-m   :: HuggingFace models (needs the venv)
models\export\.venv\Scripts\python models\export\fetch.py --all
```

Nothing has to be fetched in advance: the first run of a stage downloads its model, exports the ONNX for the input
geometry (`models\export\export_*.py`, through the venv) and builds the TensorRT engine for your GPU — DA3 ≈ 2 min,
VDA ≈ 4 min, SEA-RAFT ≈ 2.5 min, Real-ESRGAN ≈ 1 min per tile size; the engines are cached next to the ONNX and rebuilt
only for another GPU or TensorRT version. RealBasicVSR is downloaded by `sr_worker` on first use.

| Model (registry id) | Stage, backend | Source | Licence |
|---|---|---|---|
| Depth Anything 3 metric / mono large (`da3metric-large`, `da3mono-large`) | depth, `da3` | HuggingFace `depth-anything/DA3METRIC-LARGE`, `DA3MONO-LARGE` | Apache-2.0 |
| Metric Video Depth Anything small (`metric-vda-small`, `vda-small`) | depth, `vda` | HuggingFace `depth-anything/Metric-Video-Depth-Anything-Small` | Apache-2.0 |
| Metric Video Depth Anything large (`metric-vda-large`) | depth, `vda` (optional) | HuggingFace | CC-BY-NC-4.0 — research only |
| SEA-RAFT Spring-M / S (`sea-raft-spring-m`, `-s`) | flow, `searaft` | HuggingFace mirror of the official weights | BSD-3-Clause code; confirm the weights' terms before commercial use |
| RIFE 4.9 / 4.8 / 4.7 (`rife49`, …) | frame generation, `rife` (the baseline) | ONNX export by yuvraj108c (ComfyUI-Rife-Tensorrt) | MIT |
| Real-ESRGAN x2plus, x4plus, general-x4v3 (`realesrgan-x2plus`, …) | upscale, `trt` | GitHub releases of xinntao/Real-ESRGAN, sha256-pinned | BSD-3-Clause |
| RealBasicVSR (`realbasicvsr`) | upscale, `worker` (temporally consistent) | HuggingFace mirror `akhaliq/RealBasicVSR_x4` of the official checkpoint, sha256-pinned | Apache-2.0 |

DLSS Super Resolution, Frame Generation and Neural Rendering need no model files: their networks live inside the
NVIDIA DLLs above.

## Build from source

```bat
:: Developer Command Prompt for VS 2022 (or call vcvars64.bat), with VCPKG_ROOT set
cmake --preset release
cmake --build --preset release
ctest --preset release
```

`scripts\build.cmd [debug|release]` does the three steps and sets up the MSVC environment itself.
The first configure builds FFmpeg and friends through vcpkg (10–30 minutes); results are cached
in `%LOCALAPPDATA%\vcpkg\archives`. The executables land in `build\release\bin\` with the Qt runtime deployed
next to them; put the DLSS DLLs into `build\release\bin\nvidia\` (the build copies `nvngx_dlss.dll` and
`nvngx_dlssg.dll` there when `DLSS_SDK_ROOT` is set). The package: `scripts\package.cmd` → `build\release\dlss-video-<version>-win64.zip`
(and an NSIS installer when `makensis` is on `PATH`); `scripts\package.cmd full` adds the TensorRT runtime, `cudart64_12.dll`
and the ONNX models from `models\cache\`. Release steps: [docs/release.md](docs/release.md); CI: [docs/ci.md](docs/ci.md).

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
dlssvid render --project input.dlssvid.json --frame 42 -o frame.png           :: the saved viewport state (--frame counts at the base rate)
dlssvid render --project p.json --time 1.75 -o t.png                           :: by time: sources with other rates (a 48 fps FG result) show their own frame
dlssvid render --project p.json --frame 42 --source depth_raw --display turbo -o depth.png
dlssvid render --project p.json --frame 42 --mode grid --sources source,depth_raw,mv_raw,depth_dlss --size 1920x1080 -o grid.png
dlssvid render --project p.json --frame 42 --layers source,depth_raw:viridis:0.5:multiply --wipe v:0.5 -o overlay.png
dlssvid render --project p.json --frame 42 --source mv_raw --display mv_arrows --zoom 2 --center-x 960 --center-y 540 -o arrows.png
dlssvid render -i input.mp4 --passes passes --bench 60                         :: scrubbing benchmark (cold / warm / random jumps)
dlssvid render --project p.json --frame 10 --source depth_raw --save-state -o x.png   :: write the state back into the project
```

Upscale (stage 5): `color_sr` at x1.5 / x2 / x3 (output capped at 3840x2160), depth/MV guides from pass folders or
from the same pipeline, jitter emulation for DLSS.

```bat
dlssvid upscale -i input.mp4 -o passes --backend nis --scale 2                         :: NVIDIA Image Scaling (any GPU, WARP)
dlssvid upscale -i input.mp4 -o passes --backend dlss --scale 2 --depth-dir passes\depth_dlss --mv-dir passes\mv_dlss --preset K
dlssvid upscale -i input.mp4 -o passes --scale 2                                       :: DLSS SR (default); --backend trt [--model realesrgan-x2plus | realesr-general-x4v3 | realesrgan-x4plus] | worker [--window 15 --overlap 3] | nis | bicubic
dlssvid upscale -i input.mp4 -o passes --backend nis --artifact-reduction-only         :: no scaling (NVSharpen / VSR artifact reduction)
dlssvid upscale -i input.mp4 -o passes --backend dlss --video sr.mp4 --codec hevc_nvenc  :: plus a preview video (no audio)
dlssvid compare --ref reference.mp4 --test passes\color_sr --json ab.json              :: PSNR Y/RGB + SSIM per frame
```

Neural Rendering (stage 6): `color_nr` from `color_sr` (or the video) with `depth_dlss` / `mv_dlss` guides and
`mask_ui` / `mask_ignore` / `mask_face` / `mask_skin` pass folders found under the pass root; the model runs through
`bin\nvidia\nvngx_dlssnr.dll` (RTX 50: official; RTX 20/30/40: your copy patched with `nr-patch`).

```bat
dlssvid nr --check                                                                    :: GPU, driver (>= 616.56), DLL + SHA-256, CreateFeature(18)
dlssvid nr-patch --input C:\dlls\nvngx_dlssnr.dll --patcher <SDK>\dlssnr-patcher --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
dlssvid nr -i input.mp4 -o passes                                                     :: color_sr + guides + masks under passes\ are picked up
dlssvid nr -i input.mp4 -o passes --intensity 1.2 --style cinematic --passes 2 --model-scale 0.75 --temporal 0.4
dlssvid nr -i input.mp4 -o passes --no-guides --video nr.mp4                          :: A/B without depth / MV, plus a preview video
dlssvid nr -i input.mp4 -o passes --backend stub --warp                               :: deterministic stand-in (tests, any GPU)
```

Frame Generation (stage 7): `color_fg` with `multiplier` x frames (real frame i at index i * multiplier), fps x multiplier;
colour from `color_nr` / `color_sr` / the video, guides from `depth_dlss` / `mv_dlss` under the pass root.

```bat
dlssvid fg --check                                                                    :: FrameGeneration.Available, MultiFrameCountMax, CreateFeature
dlssvid fg -i input.mp4 -o passes                                                     :: DLSS Frame Generation x2 (RTX 40+), nvngx_dlssg.dll in bin\nvidia
dlssvid fg -i input.mp4 -o passes --multiplier 3 --video fg.mp4                       :: x3 where Multi Frame Generation is available, plus a preview at 3x fps
dlssvid fg -i input.mp4 -o passes --backend rife --model rife49                       :: RIFE 4.9 through TensorRT (ONNX from the model registry)
dlssvid fg -i input.mp4 -o passes --backend blend --warp                              :: naive blend baseline (any GPU)
dlssvid compare --ref full_rate.mp4 --test passes\color_fg --start 1 --step 2         :: generated frames only, against the dropped originals
```

The whole pipeline (stage 8): stages run one after another over the pass cache, a complete pass is reused, the last
colour pass is encoded with the audio copied from the source. Stage parameters are the project's JSON keys. A pass is
reused only when its fingerprint matches (stage 9: source hash, canonical parameters, the fingerprints of the passes it
read, the tool — app version, backend, model, NVIDIA DLL hash); a changed parameter recomputes that stage and the ones
below it, the replaced folder is kept as a version in `<pass>.v/` and comes back without recomputing when the
parameters return. `process --plan` shows the decisions first; `passes list|use|gc` manage the versions
(`docs/architecture.md`, «Отпечатки и версии пассов»).

```bat
dlssvid process -i input.mp4 -o result.mp4                                            :: depth -> flow -> upscale x2 -> nr -> fg x2 -> encode (+ audio)
dlssvid process --project clip.dlssvid.json                                           :: the stages and parameters the GUI saved (button «Обработать → result»)
dlssvid process -i input.mp4 -o result.mp4 --stages upscale,fg --scale 1.5 --multiplier 2 --param fg.backend=rife
dlssvid process -i input.mp4 -o result.mp4 --no-skip-existing --disable-unavailable   :: recompute everything; skip nr / fg without a DLL
dlssvid process --project clip.dlssvid.json --plan [--json plan.json]                 :: what would run or be reused and why (fingerprints, versions) — nothing is processed
dlssvid process -i input.mp4 -o result.mp4 --param nr.intensity=1.4 --keep-versions 3 :: only nr, fg and the encode rerun; the old color_nr / color_fg stay as versions (default keep: 2)
dlssvid process --project clip.dlssvid.json --force nr,fg                            :: recompute these stages in place even when their passes match; parameters are validated first
dlssvid passes list --passes result_passes [--pass color_nr] [--json list.json]       :: every pass: the current version and the previous ones (fingerprint, parameters, size)
dlssvid passes use color_nr 20260922-140200 --passes result_passes                   :: switch a pass (or a whole stage: nr) to a previous version; the current one is kept
dlssvid passes gc --project clip.dlssvid.json [--keep 2] [--dry-run]                  :: drop old versions beyond the newest N per pass (versions other passes list as inputs stay)
dlssvid bench -i input.mp4 --frames 30 --json bench.json                              :: ms/frame per stage (ТЗ §9)
dlssvid process -i input.mp4 -o out.mp4 --passthrough --codec ffv1                    :: stage 0: decode -> GPU -> encode
```

GUI keys: `1`…`9` source, `Ctrl+1/2/3` single / overlay / grid, wheel = zoom to cursor (25–800 %), middle drag = pan,
`F` fit, `Ctrl+0` 1:1, `W` wipe (left drag moves it), click a 2x2 cell = expand, `Space` play, `,`/`.` step,
`L` loop playback, `Ctrl+Shift+S` PNG screenshot with cell labels, `Ctrl+S` save project. The timeline runs in seconds (stage 9): a 24 fps
source and a 48 fps FG result stay in step, `,`/`.` step by the base layer's frame, the rate switch next to the timecode
picks what the slider counts, and a cell whose frame is missing or still loading shows a plate and says so in its label.
While a frame is still decoding, the cell keeps the source's last loaded frame (its label says «загрузка…»), and playback
shows every frame: a decode slower than real time plays slower instead of skipping frames or flashing plates.
Comparison is one action (stage 9): the bar above the viewport switches «До | После» (a wipe between the source and the
result, `W`), «Только после» and «Сетка 2×2»; one chip per source with its human name («Исходник», «Глубина», «Апскейл»,
«Улучшение», «Генерация», «Результат»; technical names in tooltips) picks the «after» side, chips with a history open a
menu of previous versions (`<pass>@<id>`, also valid in `render --layers`); presets «Апскейл ↔ Улучшение» and
«Исходник ↔ Глубина» live in the «Вид» menu; «Инженерный режим» (`Ctrl+E`) reveals the layer stack and the inspector.
The project panel (stage 9) shows the source (frame, sound, passes, hash and whether the passes match it), «Что получится»
(the result's size and rate, the time and disk of what will run, one button «Обработать · N стадий», `Ctrl+Enter`), one
card per stage with a form built from the stage schema (backend list, scale / multiplier toggles, intensity slider; the
advanced keys in the JSON editor of the engineer mode) and what the plan does with it («переиспользуется», «пересчёт:
Интенсивность 1 → 1.4», «будет посчитано», a warning when NR / FG would run without depth and vectors), the encoder
card (stored in the project; the bitrate is «Авто» — chosen from the result's size, rate and codec, see
`core/io/EncodeDefaults` — or a preset in Mbit/s or a custom value, with the rough file size) and the pass versions on disk
(«Сравнить» / «Вернуть» / «Удалить» / «Очистить старые»). The look follows the mockups of `docs/ux-guidelines.md`
(`app/Theme`: dark palette, control sizes and paddings, the accent for the one primary action).
Without a project the window shows the start page (stage 9, laid out as the mockup «1 · Стартовый экран»): drop a video on it,
«Открыть видео…» / «Открыть проект…», the GPU / driver / DLL readiness card and, on the right, the recent files as cards
with a preview frame, their state («результат готов», «глубина и NR посчитаны») and «1920×800 24 fps → 3840×1600 48 fps ·
вчера»; the window remembers its size, the dock
layout and the dialog folders (QSettings), opens maximised the first time, brings a closed panel back from «Вид», marks
unsaved changes with «*» in the title and saves the project next to the video on close. Stages are started from the project panel
(`dlssvid depth|flow|upscale|nr` as a task with progress) and their passes appear in the viewport when finished; the
`nr` stage has a «Пропатчить DLL…» button that runs `dlssvid nr-patch` over your `nvngx_dlssnr.dll`.
«Обработать» (stage 9) opens the processing page: elapsed time and the time left (the plan's estimates until each
stage has measured its own rate), one row per stage (reused ones are ticked at once), «Отменить» (the child process is
killed; finished passes keep their manifests and are reused next time), «Свернуть в фон» (a system notification tells
when the run ends, clicking it brings the window back), the tail of the log and «Показать полный лог»; a finished run
switches the viewport to «До | После» by itself, a cancelled or failed one explains itself and leads back to the project.
The result line above the viewport (size, rate, frames, sound, when it was written, how long the run took) offers
«Сохранить как…» (a copy of the result video), «Открыть папку», «Экспорт пассов…» (one `dlssvid export … --format exr`
task per pass into a chosen folder) and «Другое видео» (= «Файл → Закрыть проект», `Ctrl+W`, back to the start page);
keys `1`…`9` follow the chips of the compare bar. The passes folder can be changed from the project panel («Пассы ·
изменить…», the project remembers it); cell labels carry the source size («Исходник · 1920×800  #123 · 00:05.12»).

## Layout

```
core/       gpu/ (D3D12, CUDA interop, frame cache) · io/ (decode, encode) · pipeline/ (IStage, Pipeline)
            passes/ (PassImage, Manifest, PassSequence, formats/: EXR, PNG, TIFF, NPZ, raw)
            convert/ (depth raw ↔ reverse-Z, mv forward ↔ backward, YUV → RGB)
            ml/ (TrtLoader: runtime-loaded TensorRT, TrtEngine, ModelRegistry) · util/ (Subprocess, Sha256, Half)
            stages/passthrough · stages/depth (IDepthEstimator, DA3/VDA via TensorRT, worker client, pre/post-processing, DepthStage)
            stages/flow (IFlowEstimator, OfaFlowEstimator via nvofapi, TrtFlowEstimator for SEA-RAFT, FlowStage) · convert/Warp (warp-PSNR, warped TAE)
            viewport/ (ViewportState, ViewportRenderer + shaders/, FrameStore, Project)
            stages/upscale (IUpscaler, DlssUpscaler — NGX, NisUpscaler, BicubicUpscaler / Resampler, TrtUpscaler — ONNX models through TensorRT, Tiling, WorkerUpscaler — RealBasicVSR through sr_worker/, Jitter, UpscaleStage)
            gpu/ComputeKernel (compute PSO + descriptor rings) · passes/ImageMetrics (PSNR, SSIM) · gpu/Ngx (NGX core, shared by DLSS SR and NR)
            pipeline/ProcessRunner (the whole pipeline over the pass cache + encode with audio; `process`, `bench`; `PlanProcess` = `process --plan`)
            pipeline/PassFingerprint (canonical JSON, tool identity, the fingerprint a pass is reused by) · pipeline/PassVersions (`<pass>.v/` history: retire, use, list, gc; `passes`)
            stages/tonemap (Tonemapper: passthrough / ACES / Reinhard, sRGB / linear / PQ / HLG in)
            stages/nr (INrBackend, NgxNrBackend — NGX Feature 18 + forwarder/ nvngx.dll_dlssvid.dll, StubNrBackend, NrCompose — resolve / masks / temporal, NrStage, NrPatch)
            stages/fg (IFrameGenerator, DlssgFrameGenerator — NGX Frame Generation, RifeFrameGenerator — TensorRT, BlendFrameGenerator, FgStage)
cli/        dlssvid (+ ViewportCommands: project, render; UpscaleCommands: upscale, compare; NrCommands: nr, nr-patch; FgCommands: fg; ProcessCommands: process, bench)
app/        dlssvid-gui — Qt 6.8 Widgets shell (AppModel, ViewportWindow, panels, TaskQueue)
tests/      unit/ (Catch2, WARP-capable) · integration/ (synthetic clips, CLI as a process) · app/ (Qt offscreen) · golden/ (the pipeline vs stored references, also for an installed copy) · check_ninja_deps.cmake (build hygiene)
docs/       architecture.md · conventions.md · dll-setup.md · benchmarks.md · release.md · ci.md · plans/
scripts/    ci-build.cmd (configure, build, tests), package.cmd (CPack ZIP / NSIS; `package.cmd full` bundles the TensorRT runtime, cudart and the ONNX models)
third_party/nis/  NVIDIA Image Scaling 1.0.3 (MIT, vendored headers)
models/     registry.json (models, URLs, hashes, licences) · export/ (ONNX export scripts)
depth_worker/ worker.py — PyTorch reference backends (da3, vda), icdepth placeholder, stub for tests
models/     registry.json · export/ (fetch.py, export_da3.py, export_vda.py, loaders) · cache/ (weights, ONNX, engines; git-ignored)
tests/data/ fetch.py (the 5 acceptance clips from open sources, sha256-checked; cache/ and clips/ are git-ignored) · bench_clips.py (measurements on them) · README.md (licences)
fg_worker/  README only: why there is no worker executable (DLSS-G runs through NGX inside `dlssvid fg`)
bin/nvidia/ user-supplied NVIDIA DLLs (git-ignored)
```

## Tests

`ctest --preset release` (or run `dlssvid_unit_tests` / `dlssvid_integration_tests` directly;
Catch2 tags: `[gpu]`, `[cuda]`, `[integration]`, `[cli]`, `[nvdec]`, `[passes]`, `[formats]`, `[convert]`). The NPZ ↔ numpy test needs `python` with numpy on PATH and skips otherwise. GPU-specific tests skip themselves when no NVIDIA GPU is present.

## Licences and third-party components

- This project is released under the MIT License ([LICENSE](LICENSE)); the package ships the file next to
  README and CHANGELOG.
- The models are downloaded from their authors' releases and mirrors under their own licences (the table in
  [Models](#models); `models/registry.json` is the source of truth) — check them before commercial use.
- NVIDIA components (DLSS SDK DLLs, `nvngx_dlssnr.dll`, TensorRT, CUDA, the driver's NVDEC / NVENC / Optical Flow)
  are not part of this repository or its packages and are used under NVIDIA's licences.
- Vendored: [NVIDIA Image Scaling 1.0.3](third_party/nis/) (MIT). Build-time dependencies come from vcpkg
  (FFmpeg, OpenEXR, libpng, libtiff, spdlog, nlohmann-json, CLI11, Catch2, DirectX Shader Compiler) and Qt 6 (LGPL 3,
  dynamically linked, redeployed as is).
