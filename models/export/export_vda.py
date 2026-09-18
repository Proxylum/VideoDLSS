#!/usr/bin/env python
"""Export Video Depth Anything to ONNX for a fixed window and input geometry.

  python export_vda.py --model metric-vda-small --window 32 --width 924 --height 518

Input  "image": float32 (1, T, 3, H, W)  Output "depth": float32 (1, T, H, W)
The temporal model is heavy at T=32, 518x924 — the export needs a few minutes and several GB
of RAM; TensorRT builds it FP16.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import MODELS_DIR, model_input_size, sha256_file
from vda_loader import load_vda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="metric-vda-small")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--source-size", default="")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tolerance", type=float, default=2e-2)
    args = ap.parse_args()

    if args.source_size:
        w, h = map(int, args.source_size.lower().split("x"))
        args.width, args.height = model_input_size(w, h, args.input_size)
    if not args.width or not args.height:
        ap.error("--width/--height or --source-size required")
    out = Path(args.out) if args.out else Path(args.cache) / f"{args.model}_{args.window}x{args.height}x{args.width}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, metric, output, window, overlap = load_vda(args.model, args.cache)
    print(f"loaded {args.model} (metric={metric}, output={output}, window={window}) in {time.time() - t0:.1f}s")
    T = args.window
    dummy = torch.randn(1, T, 3, args.height, args.width)
    with torch.no_grad():
        ref = model(dummy).float().cpu().numpy()
    print(f"torch output shape {ref.shape}")

    tmp = out.with_suffix(".tmp.onnx")
    torch.onnx.export(model, (dummy,), str(tmp), input_names=["image"], output_names=["depth"], opset_version=args.opset,
                      do_constant_folding=True, dynamo=False)
    import onnx

    m = onnx.load(str(tmp))
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    tmp.unlink(missing_ok=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {sha256_file(out)[:16]}")

    if not args.no_verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(["depth"], {"image": dummy.numpy()})[0]
        rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6)
        print(f"onnxruntime vs torch: max rel err {rel:.2e}")
        if rel > args.tolerance:
            print("verification FAILED", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
