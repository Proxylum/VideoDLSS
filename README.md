# dlss-video

Offline DLSS video pipeline for Windows + NVIDIA RTX: a video file goes through
`decode → depth / motion vectors → upscale (DLSS SR; NIS / bicubic; RTX VSR optional) → tonemap → DLSS 5 Neural
Rendering → DLSS Frame Generation → encode`, every intermediate **pass** (depth, motion
vectors, masks, colour stages) can be exported, imported and inspected in a viewport.

Current stage: **8 — release** (see [docs/plans/08-release.md](docs/plans/08-release.md); earlier:
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
emulation of ТЗ §3) as the default, NVIDIA Image Scaling (always available, WARP-capable, the fallback), a bicubic
baseline, and an optional RTX VSR integration point (needs the RTX Video SDK; dropped as the primary backend on
2026-09-20 — no NVIDIA developer account). `dlssvid compare` measures PSNR/SSIM for the A/B of
[docs/benchmarks.md](docs/benchmarks.md). Stage 4: `dlssvid-gui` is a Qt 6.8 shell around a D3D12 viewport
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
| DLSS SDK | clone of [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) (`DLSS_SDK_ROOT`): NGX headers, `nvsdk_ngx_d.lib`, `nvngx_dlss.dll` | stage 5 `--backend dlss`; optional (`DLSSVID_WITH_DLSS`). `nvngx_dlss.dll` goes to `bin/nvidia/` next to the executables (the build copies it from the SDK for development) |
| RTX Video SDK | 1.1 (`RTX_VIDEO_SDK_ROOT`, NVIDIA developer account) | optional `--backend rtxvsr` only; not integrated (no account) — the stage falls back to NIS with an instruction |

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
dlssvid upscale -i input.mp4 -o passes --scale 2                                       :: DLSS SR (default); --backend nis | bicubic | rtxvsr (optional SDK)
dlssvid upscale -i input.mp4 -o passes --backend nis --artifact-reduction-only         :: no scaling (NVSharpen / VSR artifact reduction)
dlssvid upscale -i input.mp4 -o passes --backend dlss --video sr.mp4 --codec hevc_nvenc  :: plus a preview video (no audio)
dlssvid compare --ref reference.mp4 --test passes\color_sr --json ab.json              :: PSNR Y/RGB + SSIM per frame
```

Neural Rendering (stage 6): `color_nr` from `color_sr` (or the video) with `depth_dlss` / `mv_dlss` guides and
`mask_ui` / `mask_ignore` / `mask_face` / `mask_skin` pass folders found under the pass root; the model runs through
`bin\nvidia\nvngx_dlssnr.dll` (RTX 50: official; RTX 20/30/40: your copy patched with `nr-patch`).

```bat
dlssvid nr --check                                                                    :: GPU, driver (>= 616.56), DLL + SHA-256, CreateFeature(18)
dlssvid nr-patch --input C:\dlls\nvngx_dlssnr.dll --patcher D:\SDK\dlssnr-patcher --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
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
`Ctrl+Shift+S` PNG screenshot with cell labels, `Ctrl+S` save project. The timeline runs in seconds (stage 9): a 24 fps
source and a 48 fps FG result stay in step, `,`/`.` step by the base layer's frame, the rate switch next to the timecode
picks what the slider counts, and a cell whose frame is missing or still loading shows a plate and says so in its label.
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
card (stored in the project) and the pass versions on disk («Сравнить» / «Вернуть» / «Удалить» / «Очистить старые»).
Without a project the window shows the start page (stage 9): drop a video on it, «Открыть видео…» / «Открыть проект…»,
the GPU / driver / DLL readiness line and the recent files with their state; the window remembers its size, the dock
layout and the dialog folders (QSettings), opens maximised the first time, brings a closed panel back from «Вид», marks
unsaved changes with «*» in the title and saves the project next to the video on close. Stages are started from the project panel
(`dlssvid depth|flow|upscale|nr` as a task with progress) and their passes appear in the viewport when finished; the
`nr` stage has a «Пропатчить DLL…» button that runs `dlssvid nr-patch` over your `nvngx_dlssnr.dll`.

## Layout

```
core/       gpu/ (D3D12, CUDA interop, frame cache) · io/ (decode, encode) · pipeline/ (IStage, Pipeline)
            passes/ (PassImage, Manifest, PassSequence, formats/: EXR, PNG, TIFF, NPZ, raw)
            convert/ (depth raw ↔ reverse-Z, mv forward ↔ backward, YUV → RGB)
            ml/ (TrtLoader: runtime-loaded TensorRT, TrtEngine, ModelRegistry) · util/ (Subprocess, Sha256, Half)
            stages/passthrough · stages/depth (IDepthEstimator, DA3/VDA via TensorRT, worker client, pre/post-processing, DepthStage)
            stages/flow (IFlowEstimator, OfaFlowEstimator via nvofapi, TrtFlowEstimator for SEA-RAFT, FlowStage) · convert/Warp (warp-PSNR, warped TAE)
            viewport/ (ViewportState, ViewportRenderer + shaders/, FrameStore, Project)
            stages/upscale (IUpscaler, DlssUpscaler — NGX, NisUpscaler, BicubicUpscaler / Resampler, RtxVsrUpscaler stub, Jitter, UpscaleStage)
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

Tests: `ctest --preset release` (or run `dlssvid_unit_tests` / `dlssvid_integration_tests` directly;
Catch2 tags: `[gpu]`, `[cuda]`, `[integration]`, `[cli]`, `[nvdec]`, `[passes]`, `[formats]`, `[convert]`). The NPZ ↔ numpy test needs `python` with numpy on PATH and skips otherwise. GPU-specific tests skip themselves when no NVIDIA GPU is present.
