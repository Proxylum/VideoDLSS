#!/usr/bin/env python
"""Export Video Depth Anything to ONNX for a fixed window and input geometry.

  python export_vda.py --model metric-vda-small --width 924 --height 518          # window from the registry
  python export_vda.py --model metric-vda-large --window 16 --source-size 1920x1080

Input  "image": float32 (1, T, 3, H, W)  Output "depth": float32 (1, T, H, W)
The motion module's attention is exported without the empty baddbmm input (vda_loader.patch_attention): the graph
carries the weights only (Large ~1.5 GB, Base ~0.46 GB) instead of ~1 GB of zeros more. The DINOv2 encoder is traced
through scaled_dot_product_attention (exported as the same MatMul / Softmax / MatMul): the original materialises q k^T of
all T frames at once on the CPU - 9 GB per layer for Base at T = 32 on 518x924, 12 GB for Large. A graph above 1.9 GB is written
with external data (<out>.data next to <out>; delete both together). The export holds <out>.lock: the model cache is
shared with the CI runner, and two exports of one geometry must not write the same files.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import MODELS_DIR, model_input_size, sha256_file
from vda_loader import check_window, load_vda, patch_attention

EXTERNAL_THRESHOLD = int(1.9e9)  # protobuf's limit is 2 GiB


def _process_alive(pid):
    """Windows: the process exists and has not exited (a lock of a crashed export is taken over at once)."""
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    except Exception:
        return True  # unknown: trust the age


class ExportLock:
    """<out>.lock created exclusively, with the exporting process id; a lock of a live process younger than `stale`
    seconds is waited for (another process exports the same file), a lock of a dead process or an older one is taken
    over."""

    def __init__(self, out: Path, stale=3 * 3600, wait=4 * 3600):
        self.path = out.with_name(out.name + ".lock")
        self.stale, self.wait = stale, wait
        self.waited = False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    owner = self.path.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if age > self.stale or (owner.isdigit() and not _process_alive(owner)):
                    print(f"removing a stale lock {self.path} ({age / 60:.0f} min, process {owner})")
                    self.path.unlink(missing_ok=True)
                    continue
                if time.time() - t0 > self.wait:
                    raise TimeoutError(f"{self.path}: another export holds the lock")
                if not self.waited:
                    print(f"waiting for {self.path}: another process exports the same graph")
                self.waited = True
                time.sleep(5)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def save(m, out: Path, external: str):
    """Writes the model to `out` (and `out`.data with external data): into temporary names first, the .onnx last - its
    presence means a complete export. Returns True when external data was used."""
    import onnx

    size = m.ByteSize() if external == "auto" else 0
    use_external = external == "on" or (external == "auto" and size > EXTERNAL_THRESHOLD)
    data = out.with_name(out.name + ".data")
    tmp_out = out.with_name(out.name + ".part")
    for p in (out, data, tmp_out):
        p.unlink(missing_ok=True)
    if use_external:
        # `location` is relative to the graph file and already the final name; the graph is renamed last
        onnx.save_model(m, str(tmp_out), save_as_external_data=True, all_tensors_to_one_file=True, location=data.name, size_threshold=1024)
    else:
        onnx.save(m, str(tmp_out))
    os.replace(tmp_out, out)
    return use_external


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="metric-vda-small")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--source-size", default="")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--window", type=int, default=0, help="frames per window (0 = the registry's)")
    ap.add_argument("--registry", default="", help="registry.json (default: models/registry.json)")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--external-data", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--verify", choices=["auto", "on", "off"], default="auto",
                    help="onnxruntime against torch on the CPU; auto: off when the encoder's attention takes > 6 GB per layer in onnxruntime")
    ap.add_argument("--no-verify", action="store_true", help="same as --verify off")
    ap.add_argument("--tolerance", type=float, default=2e-2)
    args = ap.parse_args()

    if args.source_size:
        w, h = map(int, args.source_size.lower().split("x"))
        args.width, args.height = model_input_size(w, h, args.input_size)
    if not args.width or not args.height:
        ap.error("--width/--height or --source-size required")

    t0 = time.time()
    patch_attention(runtime=True)  # SDPA in the encoder too: the trace runs eagerly on the CPU
    model, metric, output, window, overlap = load_vda(args.model, args.cache, registry=args.registry or None)
    # no autograd state: with requires_grad the trace keeps every activation for a backward pass - 67 GB of commit for
    # Large at 8 frames on 518x924 (11 GB without), and 32 frames crashed the exporter (TASK-0039)
    model.requires_grad_(False)
    T = check_window(args.window or window)
    out = Path(args.out) if args.out else Path(args.cache) / f"{args.model}_{T}x{args.height}x{args.width}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"loaded {args.model} (metric={metric}, output={output}, window={T}, registry window {window}) in {time.time() - t0:.1f}s")

    with ExportLock(out) as lock:
        if lock.waited and out.exists():
            print(f"{out} was exported by the process that held the lock")
            return 0
        dummy = torch.randn(1, T, 3, args.height, args.width)
        # onnxruntime materialises the encoder's attention (frames x heads x tokens^2 floats): checked when it is small
        heads = {384: 6, 768: 12, 1024: 16}.get(model.net.pretrained.embed_dim, 16)  # ViT-S / B / L
        tokens = (args.height // 14) * (args.width // 14) + 1
        attention_gb = T * heads * tokens * tokens * 4 / 1e9
        verify = {"on": True, "off": False}.get("off" if args.no_verify else args.verify, attention_gb < 6.0)
        if not verify:
            print(f"onnxruntime check skipped (attention {attention_gb:.1f} GB per layer at window {T}); the TensorRT engine is checked by its tests")
        ref = None
        if verify:
            with torch.no_grad():
                ref = model(dummy).float().cpu().numpy()
            print(f"torch output shape {ref.shape}")

        import onnx

        # one scratch folder per output, under the lock: what a crashed export left there is removed first
        tmpdir = out.with_name(out.name + ".export_tmp")
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True)
        try:
            traced = tmpdir / "model.onnx"
            # the legacy exporter spreads tensors over files next to the graph above 2 GB: onnx.load gathers them
            torch.onnx.export(model, (dummy,), str(traced), input_names=["image"], output_names=["depth"], opset_version=args.opset,
                              do_constant_folding=True, dynamo=False)
            m = onnx.load(str(traced))
            external = save(m, out, args.external_data)
            del m
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        onnx.checker.check_model(str(out))  # by path: works with external data
        data = out.with_name(out.name + ".data")
        sizes = f"{out.stat().st_size / 1e6:.1f} MB" + (f" + {data.name} {data.stat().st_size / 1e6:.1f} MB" if external else "")
        print(f"saved {out} ({sizes}) sha256 {sha256_file(out)[:16]}")

        if verify:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            got = sess.run(["depth"], {"image": dummy.numpy()})[0]
            rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6)
            print(f"onnxruntime vs torch: max rel err {rel:.2e}")
            if rel > args.tolerance:
                print("verification FAILED", file=sys.stderr)
                for p in (out, data):
                    p.unlink(missing_ok=True)
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
