#!/usr/bin/env python
"""Export Depth Anything 3 (metric/mono) to ONNX for a fixed input geometry and verify it.

  python export_da3.py --model da3metric-large --width 924 --height 518 --out cache/da3metric-large_1x518x924.onnx

Input  "image": float32 (1, 3, H, W) ImageNet-normalised RGB
Output "depth": float32 (1, H, W)  (metres for metric models, relative depth for mono)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import MODELS_DIR, model_input_size, sha256_file
from da3_loader import load_da3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="da3metric-large")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--source-size", default="", help="WxH of the video instead of --width/--height (uses the pipeline's rule)")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--window", type=int, default=1, help="ignored for DA3 (single frame)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tolerance", type=float, default=2e-2, help="max relative error ORT vs torch (fp32) on a random frame")
    args = ap.parse_args()

    if args.source_size:
        w, h = map(int, args.source_size.lower().split("x"))
        args.width, args.height = model_input_size(w, h, args.input_size)
    if not args.width or not args.height:
        ap.error("--width/--height or --source-size required")
    if args.width % 14 or args.height % 14:
        ap.error("width and height must be multiples of 14")
    out = Path(args.out) if args.out else Path(args.cache) / f"{args.model}_1x{args.height}x{args.width}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, metric, output = load_da3(args.model, args.cache)
    model = model.eval()
    print(f"loaded {args.model} (metric={metric}, output={output}) in {time.time() - t0:.1f}s")

    dummy = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        outs = model(dummy)
    has_sky = isinstance(outs, tuple)
    ref = (outs[0] if has_sky else outs).float().cpu().numpy()
    output_names = ["depth", "sky"] if has_sky else ["depth"]
    print(f"torch output shape {ref.shape}, range [{ref.min():.3f}, {ref.max():.3f}], outputs {output_names}")

    tmp = out.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model, (dummy,), str(tmp), input_names=["image"], output_names=output_names, opset_version=args.opset,
        do_constant_folding=True, dynamo=False,
    )
    import onnx

    m = onnx.load(str(tmp))
    onnx.checker.check_model(m)
    # merge external data if any into one file
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
