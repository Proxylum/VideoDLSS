# models/export — weights, ONNX export, PyTorch reference

One virtual environment serves both the export scripts and `depth_worker`:

```bat
cd models\export
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps -e D:\SDK\models\Depth-Anything-3     :: DA3 code (Apache-2.0)
set VDA_REPO=D:\SDK\models\Video-Depth-Anything                            :: VDA code (Apache-2.0), plain checkout
```

Scripts:

| Script | What |
|---|---|
| `fetch.py <id>…` / `--all` | download weights from HuggingFace into `models/cache/` (ids and licences: `models/registry.json`) |
| `export_da3.py --model da3metric-large --source-size 1920x1080` | ONNX `(1,3,H,W) -> (1,H,W)` for the pipeline's input geometry; verifies ONNX Runtime vs torch |
| `export_vda.py --model metric-vda-small --window 32 --source-size 1920x1080` | ONNX `(1,T,3,H,W) -> (1,T,H,W)` |
| `da3_loader.py`, `vda_loader.py` | model construction used by the exporters and by `depth_worker` |

The C++ TensorRT backend calls the exporter itself when the ONNX for the current geometry is
missing (`core/stages/depth/TrtDepthEstimator.cpp`), then builds and caches the engine next to it
(`*.engine`). Both are git-ignored.

Licences: `da3metric-large`, `da3mono-large`, `metric-vda-small`, `vda-small` — Apache-2.0;
`metric-vda-large` — CC-BY-NC-4.0 (research only); the any-view `DA3-*` models are CC-BY-NC-4.0 and not used.
