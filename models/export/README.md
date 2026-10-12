# models/export — weights, ONNX export, PyTorch reference

One virtual environment serves both the export scripts and `depth_worker`:

```bat
cd models\export
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps -e <SDK>\models\Depth-Anything-3     :: DA3 code (Apache-2.0)
set VDA_REPO=<SDK>\models\Video-Depth-Anything                            :: VDA code (Apache-2.0), plain checkout
```

Scripts:

| Script | What |
|---|---|
| `fetch.py <id>…` / `--all` | download weights from HuggingFace into `models/cache/` (ids and licences: `models/registry.json`) |
| `export_da3.py --model da3metric-large --source-size 1920x1080` | ONNX `(1,3,H,W) -> (1,H,W)` for the pipeline's input geometry; verifies ONNX Runtime vs torch |
| `export_vda.py --model metric-vda-small --source-size 1920x1080 [--window 16] [--external-data auto\|on\|off]` | ONNX `(1,T,3,H,W) -> (1,T,H,W)`; the window defaults to the registry's (32), a multiple of 4 above 4. The motion module's attention is exported without the empty `baddbmm` input (`vda_loader.patch_attention`: the same scores, about 1 GB of zeros less at 518×924, T = 32, in the file and in the engine); the DINOv2 encoder is traced through `scaled_dot_product_attention` (the same MatMul / Softmax in the graph; the original attention materialises q·kᵀ of all frames on the CPU — 9 GB per layer for Base at T = 32, the export ran out of memory). A graph above 1.9 GB goes to `<out>.onnx` + `<out>.onnx.data` — delete both together. `<out>.lock` serialises exports of one file (the cache is shared with the CI runner); the onnxruntime check is skipped by default when the encoder's attention would take more than 6 GB per layer in onnxruntime (Base and Large at 32 frames; `--verify on` forces it; the TensorRT tests check the engine) |
| `da3_loader.py`, `vda_loader.py` | model construction used by the exporters and by `depth_worker` |
| `export_realesrgan.py --model realesrgan-x2plus` | Real-ESRGAN ONNX `(1,3,H,W) -> (1,3,sH,sW)` with dynamic H/W for the `trt` upscale backend; downloads the checkpoint from the GitHub release (sha256 in the registry), verifies ONNX Runtime vs torch on two geometries |
| `realesrgan_loader.py` | RRDBNet / SRVGGNetCompact in plain PyTorch (no basicsr), loaded with `torch.load(weights_only=True)` |
| `realbasicvsr_loader.py` | RealBasicVSR (cleaning + BasicVSR + SPyNet) in plain PyTorch (no mmcv / mmagic) with MMEditing's attribute names; the official checkpoint loads strictly (EMA generator); used by `sr_worker/worker.py` |

The C++ TensorRT backend calls the exporter itself when the ONNX for the current geometry is
missing (`core/stages/depth/TrtDepthEstimator.cpp`), then builds and caches the engine next to it
(`*.engine`). Both are git-ignored.

`fetch.py` also downloads registry entries that have a plain `url` (Real-ESRGAN weights). HuggingFace files with one
pinned file (`hf_file`) are fetched at the registry's `hf_revision` and checked against its `sha256` (VDA Base and
Large; a mismatch removes the file); a sidecar `<file>.sha256` skips re-hashing an unchanged checkpoint.

Licences: `realesrgan-x2plus`, `realesrgan-x4plus`, `realesr-general-x4v3` — BSD-3-Clause (code and weights); `realbasicvsr` — Apache-2.0
(weights from the HuggingFace mirror `akhaliq/RealBasicVSR_x4` of the official release, sha256-pinned);
`da3metric-large`, `da3mono-large`, `metric-vda-small`, `vda-small` — Apache-2.0;
`metric-vda-base`, `metric-vda-large` — CC-BY-NC-4.0 (non-commercial only: the plan and the GUI warn, the full package
does not ship their ONNX); the any-view `DA3-*` models are CC-BY-NC-4.0 and not used.
