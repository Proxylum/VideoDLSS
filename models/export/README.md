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
| `export_vda.py --model metric-vda-small --window 32 --source-size 1920x1080` | ONNX `(1,T,3,H,W) -> (1,T,H,W)` |
| `da3_loader.py`, `vda_loader.py` | model construction used by the exporters and by `depth_worker` |
| `export_realesrgan.py --model realesrgan-x2plus` | Real-ESRGAN ONNX `(1,3,H,W) -> (1,3,sH,sW)` with dynamic H/W for the `trt` upscale backend; downloads the checkpoint from the GitHub release (sha256 in the registry), verifies ONNX Runtime vs torch on two geometries |
| `realesrgan_loader.py` | RRDBNet / SRVGGNetCompact in plain PyTorch (no basicsr), loaded with `torch.load(weights_only=True)` |
| `realbasicvsr_loader.py` | RealBasicVSR (cleaning + BasicVSR + SPyNet) in plain PyTorch (no mmcv / mmagic) with MMEditing's attribute names; the official checkpoint loads strictly (EMA generator); used by `sr_worker/worker.py` |

The C++ TensorRT backend calls the exporter itself when the ONNX for the current geometry is
missing (`core/stages/depth/TrtDepthEstimator.cpp`), then builds and caches the engine next to it
(`*.engine`). Both are git-ignored.

`fetch.py` also downloads registry entries that have a plain `url` (Real-ESRGAN weights).

Licences: `realesrgan-x2plus`, `realesrgan-x4plus`, `realesr-general-x4v3` — BSD-3-Clause (code and weights); `realbasicvsr` — Apache-2.0
(weights from the HuggingFace mirror `akhaliq/RealBasicVSR_x4` of the official release, sha256-pinned);
`da3metric-large`, `da3mono-large`, `metric-vda-small`, `vda-small` — Apache-2.0;
`metric-vda-large` — CC-BY-NC-4.0 (research only); the any-view `DA3-*` models are CC-BY-NC-4.0 and not used.
