# depth_worker (stage 2)

The only Python in the runtime (ТЗ §2): PyTorch reference path for Depth Anything 3 and Video
Depth Anything, and the home of **ICDepth** once its code is public (as of 2026-09-18 the paper
[arXiv 2607.01677](https://arxiv.org/pdf/2607.01677) has no released code or weights — backend
`icdepth` reports that and the "high quality" mode stays unavailable).

`core/stages/depth/WorkerDepthEstimator` starts `worker.py` with the interpreter from
`models/export/.venv` (or `DLSSVID_PYTHON`), talks JSON lines over stdin/stdout and exchanges
frames as `.npz` files in a scratch folder (see the docstring in `worker.py` for the protocol).

Backends: `da3` (default `da3metric-large`), `vda` (default `metric-vda-small`), `icdepth`
(unavailable), `stub` (synthetic, used by tests without a GPU).

Environment: `models/export/requirements.txt` (shared venv). Weights are downloaded from
HuggingFace into `models/cache/` on first use (`models/registry.json` lists ids and licences).
