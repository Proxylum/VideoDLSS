# ONNX export (stage 2)

Python scripts that export Depth Anything 3, Video Depth Anything and SEA-RAFT from PyTorch to ONNX,
plus pytest checks that the exported graph matches the PyTorch reference within tolerance.
TensorRT engines are built from these ONNX files on first run and cached next to them.

Environment (created in stage 2): `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`.
