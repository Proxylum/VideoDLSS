"""Video Depth Anything as a window -> depth module (T frames, 32 by default)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import REGISTRY, hf_download, registry_entry, verify_sha256

# VDA_REPO, else the workspace layout (<workspace>/SDK from scripts/setup_sdk.py), else build/sdk, else a checkout next to this file
VDA_CANDIDATES = [Path(__file__).resolve().parents[4] / "SDK" / "models" / "Video-Depth-Anything",
                  Path(__file__).resolve().parents[2] / "build" / "sdk" / "models" / "Video-Depth-Anything",
                  Path(__file__).resolve().parent / "third_party" / "Video-Depth-Anything"]

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


def _bmm_attention(self, query, key, value, attention_mask=None):
    """CrossAttention._attention of the VDA motion module without `baddbmm(torch.empty(B, T, T), q, k^T, beta=0)`: the
    same scores (beta = 0 ignores the empty input), but the traced graph no longer carries that B x T x T tensor as a
    constant - about 1 GB of zeros at 518x924 with T = 32, kept by the ONNX file and allocated by the TensorRT engine
    (TASK-0039). Mathematically identical; the cached ONNX of Small keeps its name and is not exported again."""
    if self.upcast_attention:
        query = query.float()
        key = key.float()
    scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if attention_mask is not None:
        scores = scores + attention_mask
    if self.upcast_softmax:
        scores = scores.float()
    probs = scores.softmax(dim=-1).to(value.dtype)
    return self.reshape_batch_dim_to_heads(torch.bmm(probs, value))


def _sdpa_forward(self, x, attn_bias=None):
    """DINOv2 Attention.forward through F.scaled_dot_product_attention (flash / memory-efficient kernels): without
    xformers the original materialises q k^T and a fp32 softmax - 12 GB per layer for ViT-L at T = 32 on 518x924. The
    default scale 1/sqrt(head_dim) equals the module's `scale`; dropout is 0 in eval. PyTorch path only (the worker)."""
    if attn_bias is not None:
        raise NotImplementedError("attn_bias needs xformers")
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


def patch_attention(runtime=False):
    """Export: the bmm attention of the motion module. Runtime (worker): also SDPA in the DINOv2 encoder."""
    _import_vda()
    from video_depth_anything.motion_module import attention as motion_attention

    motion_attention.CrossAttention._attention = _bmm_attention
    if runtime:
        from video_depth_anything.dinov2_layers import attention as dino_attention

        dino_attention.Attention.forward = _sdpa_forward
        dino_attention.MemEffAttention.forward = _sdpa_forward


def check_window(window, max_window=32):
    """The window the C++ side resolved (ResolveDepthModel): 1..max_window and a multiple of 4 above 4 - the DPT head
    runs in micro-batches of 4 and takes all frames at once otherwise (a full-resolution spike)."""
    if not 1 <= window <= max_window or (window > 4 and window % 4):
        raise ValueError(f"VDA window {window}: needs 1..{max_window}, a multiple of 4 above 4")
    return window


class VdaDepth(nn.Module):
    """x: (1, T, 3, H, W) normalised RGB -> depth/disparity (1, T, H, W)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def load_vda(model_id, cache_dir, log=print, registry=None):
    """Returns (module, metric, output, window, overlap): the registry's window and overlap. `registry`: the registry.json
    the caller resolved the model with (models_dir of the worker), default models/registry.json. The checkpoint is the
    registry's HF revision, checked against its sha256 when pinned."""
    VideoDepthAnything = _import_vda()
    entry = registry_entry(model_id, Path(registry) if registry else REGISTRY)
    params = entry.get("params", {})
    enc = params.get("encoder", "vits")
    if params.get("family") != "vda" or enc not in MODEL_CONFIGS:
        raise ValueError(f"'{model_id}' is not a VDA model (family {params.get('family')!r}, encoder {enc!r})")
    ckpt = hf_download(entry["hf"], entry["hf_file"], cache_dir, log, revision=entry.get("hf_revision") or None)
    verify_sha256(ckpt, entry.get("sha256", ""), log)
    metric = bool(params.get("metric", False))
    net = VideoDepthAnything(**MODEL_CONFIGS[enc], metric=metric)
    net.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    return VdaDepth(net).eval(), metric, params.get("output", "depth"), int(params.get("window", 32)), int(params.get("overlap", 8))
