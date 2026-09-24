"""Real-ESRGAN networks in plain PyTorch (no basicsr): RRDBNet (RealESRGAN_x2plus / x4plus) and SRVGGNetCompact
(realesr-general-x4v3), built from the `params` of models/registry.json and loaded from the official checkpoints
(github.com/xinntao/Real-ESRGAN, BSD-3-Clause). Checkpoints are read with torch.load(weights_only=True): tensors only,
no pickled code. Input and output are RGB in [0, 1], NCHW.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import download_url, registry_entry


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN / Real-ESRGAN generator. scale 2 pixel-unshuffles the input by 2 (scale 1: by 4), the body works at that
    resolution and two nearest-x2 + conv steps bring it to the output."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """The compact VGG-style network of realesr-general-x4v3 (and the anime video model): convs + PReLU, pixel shuffle,
    plus the nearest-upsampled input."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def build_network(params):
    arch = params.get("arch", "rrdb")
    scale = int(params.get("scale", 4))
    if arch == "rrdb":
        return RRDBNet(3, 3, scale, int(params.get("num_feat", 64)), int(params.get("num_block", 23)), int(params.get("num_grow_ch", 32)))
    if arch == "srvgg":
        return SRVGGNetCompact(3, 3, int(params.get("num_feat", 64)), int(params.get("num_conv", 32)), scale)
    raise ValueError(f"unknown Real-ESRGAN arch '{arch}' (rrdb | srvgg)")


def checkpoint_path(entry, cache_dir):
    name = entry.get("file") or entry["url"].rsplit("/", 1)[1]
    return Path(cache_dir) / name


def load_realesrgan(model_id, cache_dir, log=print):
    """Downloads the checkpoint listed in the registry (sha256-checked when pinned), builds the network from `params`
    and loads the weights strictly. Returns (network in eval mode, scale, entry)."""
    entry = registry_entry(model_id)
    if entry.get("params", {}).get("family") != "realesrgan":
        raise ValueError(f"'{model_id}' is not a Real-ESRGAN model (family {entry.get('params', {}).get('family')})")
    path = download_url(entry["url"], checkpoint_path(entry, cache_dir), entry.get("sha256", ""), log=log)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    net = build_network(entry["params"])
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, int(entry["params"]["scale"]), entry
