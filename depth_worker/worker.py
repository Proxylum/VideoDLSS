#!/usr/bin/env python
"""depth_worker — PyTorch depth backends behind a JSON-lines protocol (stage 2).

The C++ core (WorkerDepthEstimator) starts this script, sends one JSON object per line on
stdin and reads one JSON object per line from stdout. Frames are exchanged as .npz files:
input  rgb_<i>.npz   key "rgb"   (H, W, 3) float16, [0, 1], R G B
output depth_<i>.npz key "depth" (H, W)    float32, larger = farther (metres when metric)

Requests
  {"cmd": "init", "backend": "da3|vda|icdepth|stub", "model": "<registry id or ''>",
   "input_size": 518, "max_res": 1080, "fp16": true, "models_dir": "...", "extra": {}}
  {"cmd": "infer", "inputs": ["rgb_0.npz", ...], "out_dir": "..."}
  {"cmd": "quit"}
Replies
  {"ok": true, ...} | {"ok": false, "error": "..."}   and free-form {"log": "..."} lines.

This is the only Python in the runtime (ТЗ §2): the reference PyTorch path for DA3 / VDA and
the home of ICDepth when its code is released.
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


# ---- backends --------------------------------------------------------------------------


class StubBackend:
    """Deterministic synthetic depth (matches core/stages/depth/StubDepthEstimator)."""

    metric = True
    window = 1
    overlap = 0
    model = "stub"

    def __init__(self, req):
        self.window = int(req.get("extra", {}).get("window", 1))
        self.overlap = int(req.get("extra", {}).get("overlap", 0))

    def infer(self, frames):
        out = []
        for f in frames:
            luma = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
            out.append((1.0 + 4.0 * luma).astype(np.float32))
        return out


class TorchBackend:
    """Shared plumbing for the PyTorch backends (resize to input_size, normalise, upsample back)."""

    def __init__(self, req):
        import torch  # noqa: F401

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = bool(req.get("fp16", True)) and self.device == "cuda"
        self.input_size = int(req.get("input_size", 518))
        self.max_res = int(req.get("max_res", 1080))
        self.models_dir = Path(req.get("models_dir") or (PROJECT / "models"))
        self.cache = self.models_dir / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def model_input_size(w, h, input_size, multiple=14, max_aspect=1.78):
        ratio = max(w, h) / min(w, h)
        short = input_size * max_aspect / ratio if ratio > max_aspect else input_size
        short = max(multiple, round(short / multiple) * multiple)
        long_ = max(short, round(max(w, h) * short / min(w, h) / multiple) * multiple)
        return (long_, short) if w >= h else (short, long_)

    def preprocess(self, frames):
        """frames: list of (H, W, 3) float arrays -> tensor (N, 3, h, w) normalised, plus (h, w)."""
        import torch
        import torch.nn.functional as F

        h0, w0 = frames[0].shape[:2]
        w, h = self.model_input_size(w0, h0, self.input_size)
        x = torch.from_numpy(np.stack(frames).astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std, (h, w), (h0, w0)

    def upsample(self, depth, size):
        import torch.nn.functional as F

        d = F.interpolate(depth[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
        return [np.ascontiguousarray(t.float().cpu().numpy()) for t in d]


class Da3Backend(TorchBackend):
    """Depth Anything 3 metric / mono (DINOv2 + DPT), per frame. Weights from HuggingFace."""

    window = 1
    overlap = 0

    def __init__(self, req):
        super().__init__(req)
        from da3_loader import load_da3  # models/export/da3_loader.py

        self.model_id = req.get("model") or "da3metric-large"
        self.model, self.metric, self.output = load_da3(self.model_id, self.cache, log=log, raw=False)
        self.model = self.model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()
        self.model_name = self.model_id

    def infer(self, frames):
        import torch

        x, _, (h0, w0) = self.preprocess(frames)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            depth = self.model(x).float()  # (N, h, w)
        if self.output == "disparity":
            depth = 1.0 / depth.clamp(min=1e-6)
        return self.upsample(depth, (h0, w0))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m


class VdaBackend(TorchBackend):
    """Video Depth Anything (temporal window of 32 frames)."""

    def __init__(self, req):
        super().__init__(req)
        from vda_loader import load_vda  # models/export/vda_loader.py

        self.model_id = req.get("model") or "metric-vda-small"
        self.model, self.metric, self.output, self.window, self.overlap = load_vda(self.model_id, self.cache, log=log)
        self.model = self.model.to(self.device).eval()
        self.model_name = self.model_id

    def infer(self, frames):
        import torch

        n = len(frames)
        x, _, (h0, w0) = self.preprocess(frames)
        # pad the window by repeating the last frame; the model expects exactly `window` frames
        if x.shape[0] < self.window:
            x = torch.cat([x, x[-1:].repeat(self.window - x.shape[0], 1, 1, 1)], dim=0)
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.fp16):
            depth = self.model(x[None])[0].float()  # (T, h, w)
        depth = depth[:n]
        if self.output == "disparity":
            depth = 1.0 / depth.clamp(min=1e-6)
        return self.upsample(depth, (h0, w0))


class IcDepthBackend:
    def __init__(self, req):
        raise RuntimeError(
            "ICDepth (arXiv 2607.01677) has no public code or weights as of 2026-09-18; "
            "the 'high quality' mode is unavailable. Use backend da3 or vda."
        )


BACKENDS = {"stub": StubBackend, "da3": Da3Backend, "vda": VdaBackend, "icdepth": IcDepthBackend}


# ---- protocol --------------------------------------------------------------------------


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
                name = req.get("backend", "da3")
                if name not in BACKENDS:
                    raise ValueError(f"unknown backend '{name}' (known: {', '.join(BACKENDS)})")
                backend = BACKENDS[name](req)
                reply({"ok": True, "backend": name, "model": getattr(backend, "model_name", getattr(backend, "model", name)) if isinstance(getattr(backend, "model_name", None), str) else name,
                       "metric": bool(backend.metric), "window": int(backend.window), "overlap": int(backend.overlap),
                       "device": getattr(backend, "device", "cpu"), "fp16": bool(getattr(backend, "fp16", False))})
            elif cmd == "infer":
                if backend is None:
                    raise RuntimeError("infer before init")
                frames = [load_npz_rgb(p) for p in req["inputs"]]
                if not frames:
                    raise ValueError("no inputs")
                depths = backend.infer(frames)
                out_dir = Path(req["out_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                outputs = []
                for i, d in enumerate(depths):
                    p = out_dir / f"depth_{i}.npz"
                    np.savez(p, depth=np.ascontiguousarray(d, dtype=np.float32))
                    outputs.append(str(p))
                reply({"ok": True, "outputs": outputs})
            elif cmd == "quit":
                reply({"ok": True})
                break
            elif cmd == "ping":
                reply({"ok": True, "pong": True})
            else:
                raise ValueError(f"unknown cmd '{cmd}'")
        except Exception as e:  # noqa: BLE001
            reply({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
