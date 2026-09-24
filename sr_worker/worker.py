#!/usr/bin/env python
"""sr_worker — PyTorch video super-resolution behind the JSON-lines protocol of depth_worker (stage 5, TASK-0023).

The C++ core (WorkerUpscaler) starts this script, sends one JSON object per line on stdin and reads one JSON object per
line from stdout. Frames are exchanged as .npz files:
input  rgb_<i>.npz  key "rgb"  (H, W, 3)   float16, [0, 1], R G B (sRGB-encoded, as the viewport shows them)
output sr_<i>.npz   key "rgb"  (H', W', 3) float16, [0, 1] at the requested target size

Requests
  {"cmd": "init", "backend": "realbasicvsr|stub", "model": "<registry id or ''>", "fp16": false,
   "models_dir": "...", "extra": {"window": 15, "overlap": 3}}
  {"cmd": "infer", "inputs": ["rgb_0.npz", ...], "out_dir": "...", "width": W, "height": H}
  {"cmd": "quit"}
Replies
  {"ok": true, ...} | {"ok": false, "error": "..."}   and free-form {"log": "..."} lines.

The window of frames is one propagation sequence of the model (BasicVSR is bidirectional: every frame sees the whole
window); the stage overlaps consecutive windows and blends the overlap. The model's native x4 output is resized to the
requested size with an antialiased bicubic (torch, antialias=True), so the stage never has to resample on its own.
"""
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "models" / "export"))


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log(msg):
    reply({"log": str(msg)})


class StubBackend:
    """Nearest x4 then the target size: deterministic, no GPU (tests)."""

    model = "stub"
    scale = 4

    def __init__(self, req):
        extra = req.get("extra", {}) or {}
        self.window = int(extra.get("window", 4))
        self.overlap = int(extra.get("overlap", 1))

    def infer(self, frames, width, height):
        import torch
        import torch.nn.functional as F

        out = []
        for f in frames:
            x = torch.from_numpy(f.astype(np.float32)).permute(2, 0, 1)[None]
            up = F.interpolate(x, scale_factor=4, mode="nearest")
            if (up.shape[3], up.shape[2]) != (width, height):
                up = F.interpolate(up, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
            out.append(up[0].permute(1, 2, 0).clamp(0, 1).numpy())
        return out


class RealBasicVsrBackend:
    """RealBasicVSR (Apache-2.0): cleaning + BasicVSR propagation over the window, x4, resized to the target."""

    scale = 4

    def __init__(self, req):
        import torch
        from realbasicvsr_loader import load_realbasicvsr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = bool(req.get("fp16", False)) and self.device == "cuda"
        models_dir = Path(req.get("models_dir") or (PROJECT / "models"))
        cache = models_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.model = req.get("model") or "realbasicvsr"
        net, self.scale, entry = load_realbasicvsr(self.model, cache, log=log)
        self.net = net.to(self.device)
        self.license = entry.get("license", "")
        extra = req.get("extra", {}) or {}
        params = entry.get("params", {})
        self.window = int(extra.get("window", params.get("window", 15)))
        self.overlap = int(extra.get("overlap", params.get("overlap", 3)))
        log(f"RealBasicVSR ready on {self.device} ({'fp16' if self.fp16 else 'fp32'}), window {self.window} overlap {self.overlap}")

    def infer(self, frames, width, height):
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.stack(frames).astype(np.float32)).permute(0, 3, 1, 2).to(self.device)[None]  # (1, T, 3, H, W)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.fp16):
            up = self.net(x, to_cpu=False)[0].float()  # (T, 3, 4H, 4W)
        if (up.shape[3], up.shape[2]) != (width, height):
            up = F.interpolate(up, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        up = up.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return [np.ascontiguousarray(f) for f in up]


BACKENDS = {"stub": StubBackend, "realbasicvsr": RealBasicVsrBackend}


def load_npz_rgb(path):
    with np.load(path) as z:
        key = "rgb" if "rgb" in z.files else z.files[0]
        a = z[key]
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"{path}: expected (H, W, 3) array, got {a.shape}")
    return a[..., :3].astype(np.float32)


def main():
    backend = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "init":
                name = req.get("backend", "realbasicvsr")
                if name not in BACKENDS:
                    raise ValueError(f"unknown backend '{name}' ({' | '.join(BACKENDS)})")
                backend = BACKENDS[name](req)
                reply({"ok": True, "backend": name, "model": backend.model, "scale": backend.scale, "window": backend.window, "overlap": backend.overlap,
                       "license": getattr(backend, "license", "")})
            elif cmd == "infer":
                if backend is None:
                    raise RuntimeError("init first")
                frames = [load_npz_rgb(p) for p in req["inputs"]]
                out_dir = Path(req["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                h0, w0 = frames[0].shape[:2]
                width = int(req.get("width") or w0 * backend.scale)
                height = int(req.get("height") or h0 * backend.scale)
                results = backend.infer(frames, width, height)
                outputs = []
                for i, r in enumerate(results):
                    p = out_dir / f"sr_{i}.npz"
                    np.savez(p, rgb=r.astype(np.float16))
                    outputs.append(str(p))
                reply({"ok": True, "outputs": outputs})
            elif cmd == "quit":
                reply({"ok": True})
                return 0
            elif cmd == "ping":
                reply({"ok": True})
            else:
                raise ValueError(f"unknown cmd '{cmd}'")
        except Exception as e:  # noqa: BLE001
            reply({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
