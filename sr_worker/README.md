# sr_worker (stage 5, TASK-0023)

PyTorch video super-resolution behind the JSON-lines protocol of `depth_worker`: **RealBasicVSR** (Chan et al.,
CVPR 2022, Apache-2.0) — an image-cleaning module in front of BasicVSR, whose bidirectional propagation aligns
64-channel features across a window of frames with SPyNet flow, so the result is temporally consistent where
per-frame models (Real-ESRGAN) flicker. The network is `models/export/realbasicvsr_loader.py` in plain PyTorch (no
mmcv / mmagic); the official checkpoint `RealBasicVSR_x4.pth` comes from the HuggingFace mirror
`akhaliq/RealBasicVSR_x4` (sha256 in `models/registry.json`) and loads strictly (EMA generator, as at test time).

`core/stages/upscale/WorkerUpscaler` starts `worker.py` with the interpreter from `models/export/.venv` (or
`DLSSVID_PYTHON`), sends one JSON object per line on stdin, reads replies on stdout and exchanges frames as `.npz`
files in a scratch folder (protocol in the docstring of `worker.py`). One request is one window of frames
(`window`, default 15, overlap 3 — the stage blends the overlap); the worker resizes the model's x4 output to the
target size itself (antialiased bicubic), so `--scale 2` and the 4K cap need no resampling in the stage.

Backends: `realbasicvsr` (default, GPU; ~0.2–0.4 s per 960×400 frame on an RTX 4070 Ti in fp32), `stub` (nearest x4
then the target size — the tests, no GPU).

Environment: `models/export/requirements.txt` (the shared venv). CLI: `dlssvid upscale --backend worker [--window 15
--overlap 3]`; GUI: method «RealBasicVSR (PyTorch)» of the upscale card.
