#!/usr/bin/env python
"""Export SEA-RAFT to ONNX for a fixed input geometry (multiples of 8) and verify it.

  python export_searaft.py --model sea-raft-spring-m --width 1280 --height 720

Inputs "image1", "image2": float32 (1, 3, H, W) RGB in [0, 255]; output "flow": float32 (1, 2, H, W).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import MODELS_DIR, sha256_file
from searaft_loader import load_searaft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sea-raft-spring-m")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tolerance", type=float, default=5e-2, help="max abs error in pixels ORT vs torch on a synthetic shift")
    args = ap.parse_args()
    if args.width % 8 or args.height % 8:
        ap.error("width and height must be multiples of 8")
    out = Path(args.out) if args.out else Path(args.cache) / f"{args.model}_2x{args.height}x{args.width}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, iters, cfg = load_searaft(args.model, args.cache)
    print(f"loaded {args.model} (iters={iters}) in {time.time() - t0:.1f}s")

    # synthetic pair: smooth noise shifted by (3, -2) px
    torch.manual_seed(0)
    base = torch.rand(1, 3, args.height // 8, args.width // 8)
    base = torch.nn.functional.interpolate(base, size=(args.height, args.width), mode="bicubic", align_corners=False) * 255
    image1 = base
    image2 = torch.roll(base, shifts=(-2, 3), dims=(2, 3))
    with torch.no_grad():
        ref = model(image1, image2).float().cpu().numpy()
    inner = ref[0, :, 8:-8, 8:-8]
    print(f"torch flow shape {ref.shape}, mean ({inner[0].mean():.2f}, {inner[1].mean():.2f}) px, expected (3, -2)")

    tmp = out.with_suffix(".tmp.onnx")
    torch.onnx.export(model, (image1, image2), str(tmp), input_names=["image1", "image2"], output_names=["flow"],
                      opset_version=args.opset, do_constant_folding=True, dynamo=False)
    import onnx

    m = onnx.load(str(tmp))
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    tmp.unlink(missing_ok=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {sha256_file(out)[:16]}")

    if not args.no_verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(["flow"], {"image1": image1.numpy(), "image2": image2.numpy()})[0]
        err = np.abs(got - ref)[0, :, 8:-8, 8:-8].max()
        print(f"onnxruntime vs torch: max abs err {err:.3e} px")
        if err > args.tolerance:
            print("verification FAILED", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
