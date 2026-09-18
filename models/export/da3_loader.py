"""Depth Anything 3 (metric / mono, Apache-2.0) as a plain image -> depth module.

The any-view DA3 models (DA3-Large etc.) are CC-BY-NC and need camera tokens; the metric and
mono series are DINOv2-L + DPT and predict depth directly, which is what the pipeline wants.
The network is built from the DA3 config registry directly (depth_anything_3.api pulls in
trimesh / plyfile / moviepy for its export helpers, which we do not need).

Two forward modes:
  raw=True  (ONNX export): backbone -> DPT head only; outputs (depth, sky) with no data-dependent
            post-processing, so the traced graph is exact. The sky rule (sky < 0.3 is non-sky,
            sky pixels get the 99th percentile of non-sky depth) is applied by the consumer
            (core/stages/depth/TrtDepthEstimator.cpp, worker.py) exactly as DA3 does it.
  raw=False (PyTorch reference): the full DepthAnything3Net forward including the sky rule.
"""
import json

import torch
import torch.nn as nn

from common import hf_download, registry_entry

SKY_THRESHOLD = 0.3
SKY_QUANTILE = 0.99


def apply_sky_rule(depth, sky, threshold=SKY_THRESHOLD, quantile=SKY_QUANTILE):
    """depth, sky: (N, H, W) tensors. Mirrors DepthAnything3Net._process_mono_sky_estimation."""
    non_sky = sky < threshold
    out = depth.clone()
    for i in range(depth.shape[0]):
        ns = non_sky[i]
        if ns.sum() <= 10 or (~ns).sum() <= 10:
            continue
        vals = depth[i][ns]
        if vals.numel() > 100000:
            idx = torch.randint(0, vals.numel(), (100000,), device=vals.device)
            vals = vals[idx]
        out[i][~ns] = torch.quantile(vals.float(), quantile).to(depth.dtype)
    return out


class Da3Depth(nn.Module):
    """x: (N, 3, H, W) normalised RGB -> depth (N, H, W) [, sky (N, H, W)]."""

    def __init__(self, net, raw=True):
        super().__init__()
        self.net = net
        self.raw = raw

    def forward(self, x):
        if not self.raw:
            out = self.net(x[None])
            depth = out["depth"]
            if depth.dim() == 5:
                depth = depth[:, :, 0]
            return depth[0]
        feats, _ = self.net.backbone(x[None], cam_token=None, export_feat_layers=[], ref_view_strategy="first")
        H, W = x.shape[-2], x.shape[-1]
        out = self.net.head(feats, H, W, patch_start_idx=0)
        depth = out["depth"]
        if depth.dim() == 5:
            depth = depth[:, :, 0]
        depth = depth[0]
        if "sky" in out:
            sky = out["sky"]
            if sky.dim() == 5:
                sky = sky[:, :, 0]
            return depth, sky[0]
        return depth


def build_da3(model_name):
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    config = load_config(MODEL_REGISTRY[model_name])
    return create_object(config)


def load_da3(model_id, cache_dir, log=print, raw=True):
    """Returns (module, metric: bool, output: 'depth'|'disparity')."""
    entry = registry_entry(model_id)
    repo = entry["hf"]
    cfg = hf_download(repo, "config.json", cache_dir, log)
    weights = hf_download(repo, "model.safetensors", cache_dir, log)
    with open(cfg, encoding="utf-8") as f:
        model_name = json.load(f).get("model_name", model_id)
    net = build_da3(model_name)
    from safetensors.torch import load_file

    state = load_file(str(weights))
    # PyTorchModelHubMixin saved the wrapper: keys are "model.<...>"
    if all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        log(f"warning: {len(missing)} missing keys when loading {model_id} (first: {missing[:3]})")
    if unexpected:
        log(f"warning: {len(unexpected)} unexpected keys when loading {model_id} (first: {unexpected[:3]})")
    return Da3Depth(net, raw=raw).eval(), bool(entry["params"].get("metric", False)), entry["params"].get("output", "depth")
