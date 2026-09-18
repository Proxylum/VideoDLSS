"""Video Depth Anything as a window -> depth module (T = 32 frames)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from common import hf_download, registry_entry

VDA_REPO = Path(__file__).resolve().parents[3] / "SDK" / "models" / "Video-Depth-Anything"  # D:/SDK/models/... when checked out next to DEV
VDA_CANDIDATES = [Path(r"D:\SDK\models\Video-Depth-Anything"), VDA_REPO, Path(__file__).resolve().parent / "third_party" / "Video-Depth-Anything"]

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _import_vda():
    import os

    for cand in [os.environ.get("VDA_REPO", ""), *map(str, VDA_CANDIDATES)]:
        if cand and Path(cand, "video_depth_anything", "video_depth.py").exists():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

            return VideoDepthAnything
    raise ImportError("Video-Depth-Anything checkout not found: set VDA_REPO to the repository path")


class VdaDepth(nn.Module):
    """x: (1, T, 3, H, W) normalised RGB -> depth/disparity (1, T, H, W)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def load_vda(model_id, cache_dir, log=print):
    """Returns (module, metric, output, window, overlap)."""
    VideoDepthAnything = _import_vda()
    entry = registry_entry(model_id)
    params = entry["params"]
    enc = params.get("encoder", "vits")
    ckpt = hf_download(entry["hf"], entry["hf_file"], cache_dir, log)
    metric = bool(params.get("metric", False))
    net = VideoDepthAnything(**MODEL_CONFIGS[enc], metric=metric)
    net.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    return VdaDepth(net).eval(), metric, params.get("output", "depth"), int(params.get("window", 32)), int(params.get("overlap", 8))
