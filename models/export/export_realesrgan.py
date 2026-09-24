#!/usr/bin/env python
"""Export a Real-ESRGAN model to ONNX with dynamic height / width and verify it.

  python export_realesrgan.py --model realesrgan-x2plus

Input "image": float32 (1, 3, H, W) RGB in [0, 1]; output "upscaled": float32 (1, 3, scale*H, scale*W).
The C++ backend (core/stages/upscale/TrtUpscaler) builds a TensorRT engine for its tile size from this file, so one
ONNX serves every video geometry. The checkpoint is downloaded on first use (models/registry.json: url, sha256).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import MODELS_DIR, sha256_file
from realesrgan_loader import load_realesrgan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="realesrgan-x2plus")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tile", type=int, default=64, help="tile size of the verification run")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tolerance", type=float, default=2e-3, help="max abs error ORT vs torch (fp32, [0, 1] values)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.cache) / f"{args.model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    net, scale, entry = load_realesrgan(args.model, args.cache)
    print(f"loaded {args.model} (x{scale}, {entry['license']}) in {time.time() - t0:.1f}s")

    torch.manual_seed(0)
    tile = max(16, args.tile // 4 * 4)
    image = torch.rand(1, 3, tile, tile)
    with torch.no_grad():
        ref = net(image).float().cpu().numpy()
    print(f"torch output {ref.shape}, range [{ref.min():.3f}, {ref.max():.3f}]")

    tmp = out.with_suffix(".tmp.onnx")
    torch.onnx.export(net, (image,), str(tmp), input_names=["image"], output_names=["upscaled"], opset_version=args.opset,
                      do_constant_folding=True, dynamo=False, dynamic_axes={"image": {2: "height", 3: "width"}, "upscaled": {2: "out_height", 3: "out_width"}})
    import onnx

    m = onnx.load(str(tmp))
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    tmp.unlink(missing_ok=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {sha256_file(out)[:16]}")

    if not args.no_verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(["upscaled"], {"image": image.numpy()})[0]
        err = float(np.abs(got - ref).max())
        print(f"onnxruntime vs torch: max abs err {err:.3e}")
        if err > args.tolerance:
            print("verification FAILED", file=sys.stderr)
            return 2
        # a second geometry proves the dynamic axes
        other = torch.rand(1, 3, tile // 2 + 8, tile + 12)
        got2 = sess.run(["upscaled"], {"image": other.numpy()})[0]
        if got2.shape != (1, 3, other.shape[2] * scale, other.shape[3] * scale):
            print(f"verification FAILED: dynamic shape gave {got2.shape}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
