"""RealBasicVSR (Chan et al., CVPR 2022) in plain PyTorch, no mmcv / mmagic: the image-cleaning module, BasicVSR
(SPyNet flow + bidirectional propagation with flow warping, x4 pixel-shuffle upsampling) and SPyNet, with the attribute
names of MMEditing so the official checkpoint (`RealBasicVSR_x4.pth`, Apache-2.0) loads strictly after its
`generator.` prefix is stripped. Checkpoints are read with torch.load(weights_only=True); the EMA generator is used, as MMEditing does at test time. Input and output are RGB in
[0, 1], (N, T, 3, H, W) -> (N, T, 3, 4H, 4W).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import hf_download, registry_entry, sha256_file


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp `x` (n, c, h, w) by `flow` (n, h, w, 2) in pixels (MMEditing's flow_warp)."""
    n, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=x.device, dtype=x.dtype), torch.arange(0, w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    grid_flow = grid + flow
    gx = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gx, gy), dim=3), mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)


class ConvModule(nn.Module):
    """MMCV's ConvModule as used by SPyNet: conv (+ ReLU); the weight lives at `.conv`."""

    def __init__(self, in_ch, out_ch, k, padding, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, 1, padding)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        return self.act(x) if self.act is not None else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(ConvModule(8, 32, 7, 3), ConvModule(32, 64, 7, 3), ConvModule(64, 32, 7, 3), ConvModule(32, 16, 7, 3), ConvModule(16, 2, 7, 3, act=False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = flow if level == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + self.basic_module[level](torch.cat([ref[level], warped, flow_up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow = torch.stack([flow[:, 0] * (float(w) / float(w_up)), flow[:, 1] * (float(h) / float(h_up))], dim=1)
        return flow


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]))

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, upsample_kernel, padding=(upsample_kernel - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class BasicVSRNet(nn.Module):
    """BasicVSR (Chan et al., CVPR 2021): bidirectional propagation of 64-channel features aligned by SPyNet flow."""

    def __init__(self, mid_channels=64, num_blocks=30):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs, to_cpu=False):
        n, t, c, h, w = lrs.size()
        flows_forward, flows_backward = self.compute_flow(lrs)
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat_prop = flow_warp(feat_prop, flows_backward[:, i].permute(0, 2, 3, 1))
            feat_prop = self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(t):
            lr_curr = lrs[:, i]
            if i > 0:
                feat_prop = flow_warp(feat_prop, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            feat_prop = self.forward_resblocks(torch.cat([lr_curr, feat_prop], dim=1))
            out = self.lrelu(self.fusion(torch.cat([outputs[i], feat_prop], dim=1)))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out) + F.interpolate(lr_curr, scale_factor=4, mode="bilinear", align_corners=False)
            outputs[i] = out.cpu() if to_cpu else out
        return torch.stack(outputs, dim=1)


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR: a residual image-cleaning module (dynamic refinement, at most 3 passes) in front of BasicVSR."""

    def __init__(self, mid_channels=64, num_propagation_blocks=20, num_cleaning_blocks=20, dynamic_refine_thres=255):
        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.0
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks)
        self.image_cleaning = nn.Sequential(ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks), nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True))

    def clean(self, lqs):
        n, t, c, h, w = lqs.size()
        for _ in range(3):
            flat = lqs.view(-1, c, h, w)
            residues = self.image_cleaning(flat)
            lqs = (flat + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break
        return lqs

    def forward(self, lqs, to_cpu=False):
        return self.basicvsr(self.clean(lqs), to_cpu=to_cpu)


def build_network(params):
    return RealBasicVSRNet(int(params.get("mid_channels", 64)), int(params.get("num_propagation_blocks", 20)), int(params.get("num_cleaning_blocks", 20)),
                           int(params.get("dynamic_refine_thres", 255)))


def load_realbasicvsr(model_id, cache_dir, log=print):
    """Downloads the checkpoint of the registry entry (HuggingFace mirror `akhaliq/RealBasicVSR_x4` of the official release,
    sha256-checked when pinned), builds the network from `params` and loads the generator weights strictly."""
    entry = registry_entry(model_id)
    if entry.get("params", {}).get("family") != "realbasicvsr":
        raise ValueError(f"'{model_id}' is not a RealBasicVSR model")
    path = hf_download(entry["hf"], entry["hf_file"], cache_dir, log=log)
    got = sha256_file(path)
    if entry.get("sha256") and got != entry["sha256"]:
        Path(path).unlink()
        raise RuntimeError(f"{Path(path).name}: sha256 {got} does not match the registry ({entry['sha256']}); the file was removed")
    if not entry.get("sha256"):
        log(f"{Path(path).name}: sha256 {got} (not pinned in the registry yet)")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    # MMEditing tests RealBasicVSR with the EMA generator (is_use_ema in the config): prefer it, then the plain one
    generator = {k[len("generator_ema."):]: v for k, v in state.items() if k.startswith("generator_ema.")}
    if not generator:
        generator = {k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")}
    if not generator:  # a bare generator checkpoint
        generator = {k: v for k, v in state.items() if not k.startswith("discriminator.")}
    net = build_network(entry["params"])
    net.load_state_dict(generator, strict=True)
    net.eval()
    return net, 4, entry
