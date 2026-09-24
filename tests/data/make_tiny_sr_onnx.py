#!/usr/bin/env python
"""Writes tests/data/tiny_sr.onnx: a nearest-neighbour x2 "super-resolution" model with dynamic height / width —
input "image" float32 (1, 3, H, W) -> output "upscaled" float32 (1, 3, 2H, 2W). The TensorRT upscaler tests run it
as the model `tiny-sr`: the result must equal a plain nearest x2 of the frame, which proves the tiling (padding,
overlap, border tiles) reassembles the frame exactly. Needs the `onnx` package (models/export/.venv)."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent

scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], np.array([1, 1, 2, 2], dtype=np.float32))
resize = helper.make_node("Resize", inputs=["image", "", "scales"], outputs=["upscaled"], mode="nearest",
                          nearest_mode="floor", coordinate_transformation_mode="asymmetric")
graph = helper.make_graph([resize], "tiny_sr",
                          [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])],
                          [helper.make_tensor_value_info("upscaled", TensorProto.FLOAT, [1, 3, "out_height", "out_width"])],
                          initializer=[scales])
model = helper.make_model(graph, producer_name="dlssvid-tests", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
out = HERE / "tiny_sr.onnx"
onnx.save(model, str(out))
print(f"saved {out} ({out.stat().st_size} bytes)")
