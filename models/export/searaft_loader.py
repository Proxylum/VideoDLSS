"""SEA-RAFT (princeton-vl, BSD-3) as a plain (image1, image2) -> forward flow module."""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from common import hf_download, registry_entry

# SEARAFT_REPO, else the workspace layout (<workspace>/SDK from scripts/setup_sdk.py), else a checkout next to this file
SEARAFT_CANDIDATES = [Path(__file__).resolve().parents[4] / "SDK" / "models" / "SEA-RAFT",
                      Path(__file__).resolve().parents[2] / "build" / "sdk" / "models" / "SEA-RAFT",
                      Path(__file__).resolve().parent / "third_party" / "SEA-RAFT"]


def _import_searaft():
    for cand in [os.environ.get("SEARAFT_REPO", ""), *map(str, SEARAFT_CANDIDATES)]:
        if cand and Path(cand, "core", "raft.py").exists():
            core = str(Path(cand) / "core")
            if core not in sys.path:
                sys.path.insert(0, core)
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from raft import RAFT  # noqa: E402

            return RAFT, Path(cand)
    raise ImportError("SEA-RAFT checkout not found: set SEARAFT_REPO to the repository path")


class Args:
    def __init__(self, d):
        self.__dict__.update(d)


class SeaRaftFlow(nn.Module):
    """image1, image2: (1, 3, H, W) RGB float in [0, 255] -> flow (1, 2, H, W) forward, pixels."""

    def __init__(self, net, iters):
        super().__init__()
        self.net = net
        self.iters = iters

    def forward(self, image1, image2):
        out = self.net(image1, image2, iters=self.iters, test_mode=True)
        return out["flow"][-1]


def load_searaft(model_id, cache_dir, log=print):
    """Returns (module, iters, config dict)."""
    RAFT, repo = _import_searaft()
    entry = registry_entry(model_id)
    cfg_path = repo / "config" / "eval" / entry["params"].get("config", "spring-M.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["iters"] = int(entry["params"].get("iters", cfg.get("iters", 4)))
    ckpt = hf_download(entry["hf"], entry["hf_file"], cache_dir, log)
    args = Args(cfg)
    net = RAFT(args)
    state = torch.load(str(ckpt), map_location="cpu")
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        log(f"warning: {len(missing)} missing keys loading {model_id} (first: {missing[:3]})")
    return SeaRaftFlow(net, cfg["iters"]).eval(), cfg["iters"], cfg
