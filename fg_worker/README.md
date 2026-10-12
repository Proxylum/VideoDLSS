# fg_worker (stage 7) — resolved: no separate worker executable

ТЗ §4 planned a separate `fg_worker` process because NVIDIA Streamline runs DLSS Frame Generation only by
intercepting the swapchain `Present`, which can take the host down with it. The DLSS SDK 310.9 exposes Frame
Generation directly through NGX (`NVSDK_NGX_Feature_FrameGeneration`, `nvsdk_ngx_helpers_dlssg_d3d.h`, "DLSS-FG
Programming Guide"): the current frame, depth and motion vectors go in as textures and the interpolated frame comes
out as a texture — no swapchain, no hook, no Streamline binaries (which the GitHub checkout does not even ship).

So the stage lives in `core/stages/fg/` (`DlssgFrameGenerator`, `RifeFrameGenerator`, `BlendFrameGenerator`,
`FgStage`) and runs inside `dlssvid fg`, which is the worker process. A single stage from the project panel runs as
`dlssvid fg` through `TaskQueue`; the whole pipeline (`dlssvid process`, «Обработать», `bench`) starts FG as a child
`dlssvid fg --as-stage` (`core/pipeline/FgProcess.h`, audit fg-2): a crash of DLSS-G ends that process only, the
stage is reported `failed` and the result is encoded from color_nr (tests: `--crash-after`, `--crash-on-exit`,
`--stall` in `tests/integration/test_cli_process.cpp`, `tests/app`). See `docs/plans/07-fg.md` and
`docs/architecture.md`.
