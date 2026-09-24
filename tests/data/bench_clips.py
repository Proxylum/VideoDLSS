#!/usr/bin/env python3
"""Benchmarks of the real acceptance clips (TASK-0010): runs dlssvid on tests/data/clips/*.mp4 and prints markdown tables.

    python tests/data/bench_clips.py [--sections flow,depth,upscale] [--only face] [--frames N] [--out DIR]

Sections (docs/benchmarks.md):
  flow     — OFA vs SEA-RAFT: warp-PSNR (mean / min), mean |v|, ms/frame            (stage 3)
  depth    — DA3 vs VDA: metric scale (mean / median depth of depth_raw), TAE with motion compensation
             (--mv-dir from the OFA run), ms/frame                                    (stage 2)
  upscale  — A/B ×2 against the clip itself: NIS / bicubic / DLSS (jitter emulation) / DLSS + guides /
             Real-ESRGAN x2plus and general-x4v3 through TensorRT (trt)
             (depth + MV estimated on the half-resolution input), PSNR Y / RGB, SSIM Y (stage 5 methodology)
  fg       — clips at >= 48 fps only: every other frame dropped, x2 generation (dlssg / rife / blend, with guides),
             generated frames compared with the dropped ones (stage 7 methodology)
Results: <out>/results.json and the tables on stdout. Needs an NVIDIA GPU (OFA, TensorRT, DLSS) and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLIPS = HERE / "clips"


def run(cmd: list[str], log: Path) -> str:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + f"\n\n{out}\n[exit {p.returncode}, {time.time() - t0:.1f} s]\n", encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... failed (exit {p.returncode}): see {log}\n{out[-800:]}")
    return out


def grab(pattern: str, text: str, group: int = 1, default=None):
    m = re.search(pattern, text)
    return float(m.group(group)) if m else default


def depth_stats(pass_dir: Path) -> dict:
    """Mean / median of the depth_raw NPZ frames (metres for metric models)."""
    files = sorted(pass_dir.glob("*.npz"))
    if not files:
        return {}
    means, medians = [], []
    for f in files[:: max(1, len(files) // 24)]:  # every ~10th frame is enough for a scale estimate
        z = np.load(f)
        a = np.asarray(z[z.files[0]], dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            means.append(a.mean())
            medians.append(np.median(a))
    return dict(mean_m=float(np.mean(means)), median_m=float(np.mean(medians)), median_min_m=float(np.min(medians)), median_max_m=float(np.max(medians)),
                frames_sampled=len(means))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables are UTF-8 markdown whatever the console code page is
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default=str(ROOT / "build/release/bin/dlssvid.exe"))
    ap.add_argument("--clips", default=str(CLIPS))
    ap.add_argument("--out", default=str(ROOT / "build/release/clipbench"))
    ap.add_argument("--sections", default="flow,depth,upscale,fg")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--frames", type=int, default=-1, help="frames per clip (default: all)")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--force", action="store_true", help="re-run measurements already present in results.json")
    args = ap.parse_args()

    cli = args.cli
    out_root = Path(args.out)
    sections = set(args.sections.split(","))
    clips = sorted(Path(args.clips).glob("*.mp4"))
    if args.only:
        clips = [c for c in clips if c.stem in args.only]
    if not clips:
        print("no clips: run tests/data/fetch.py first", file=sys.stderr)
        return 1
    frames = ["--frames", str(args.frames)] if args.frames > 0 else []
    results_path = out_root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    def save():
        out_root.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for clip in clips:
        name = clip.stem
        d = out_root / name
        r = results.setdefault(name, {})
        print(f"== {name}", flush=True)

        def done(section: str, key: str) -> bool:
            if args.force or not r.get(section, {}).get(key):
                return False
            print(f"  {section} {key}: already in results.json (use --force to redo)", flush=True)
            return True

        if "flow" in sections:
            for backend in ("ofa", "searaft"):
                if done("flow", backend):
                    continue
                text = run([cli, "flow", "--backend", backend, "-i", str(clip), "-o", str(d / f"flow_{backend}"), *frames], d / f"flow_{backend}.log")
                r.setdefault("flow", {})[backend] = dict(
                    warp_psnr_db=grab(r"warp PSNR:\s+([\d.]+) dB", text), warp_psnr_min_db=grab(r"warp PSNR:\s+[\d.]+ dB \(min ([\d.]+)\)", text),
                    mean_v_px=grab(r"mean \|v\|:\s+([\d.]+) px", text), ms_per_frame=grab(r"ms/frame:\s+([\d.]+)", text), frames=grab(r"frames:\s+(\d+)", text))
                print(f"  flow {backend}: {r['flow'][backend]}", flush=True)
                save()

        if "depth" in sections:
            mv_dir = d / "flow_ofa" / "mv_dlss"
            mv = ["--mv-dir", str(mv_dir)] if mv_dir.exists() else []
            for backend in ("da3", "vda"):
                if done("depth", backend):
                    continue
                text = run([cli, "depth", "--backend", backend, "--format", "npz", "-i", str(clip), "-o", str(d / f"depth_{backend}"), *mv, *frames], d / f"depth_{backend}.log")
                r.setdefault("depth", {})[backend] = dict(
                    tae=grab(r"mean TAE:\s+([\d.]+)", text), tae_mode="warped" if mv else "static", ms_per_frame=grab(r"ms/frame:\s+([\d.]+)", text),
                    frames=grab(r"frames:\s+(\d+)", text), **depth_stats(d / f"depth_{backend}" / "depth_raw"))
                print(f"  depth {backend}: {r['depth'][backend]}", flush=True)
                save()
            a, b = r["depth"].get("da3", {}), r["depth"].get("vda", {})
            if a.get("median_m") and b.get("median_m"):
                r["depth"]["scale_ratio_da3_over_vda"] = a["median_m"] / b["median_m"]

        if "fg" in sections and json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "json", str(clip)],
                                                           check=True, capture_output=True, text=True).stdout)["streams"][0]["r_frame_rate"] in ("60/1", "50/1", "48/1"):
            # ground truth: drop every other frame, generate x2, compare the generated frames with the dropped ones (stage 7 methodology)
            half_rate = d / "in_halfrate.mp4"
            if not half_rate.exists():
                run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip), "-an", "-vf", "select=not(mod(n\\,2)),setpts=N/(30*TB)", "-r", "30",
                     "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                     *(["-frames:v", str(args.frames // 2)] if args.frames > 0 else []), str(half_rate)], d / "in_halfrate.log")
            fg_guides = d / "halfrate_passes"
            if not (fg_guides / "mv_dlss").exists():
                run([cli, "flow", "--backend", "ofa", "-i", str(half_rate), "-o", str(fg_guides)], d / "halfrate_flow.log")
            if not (fg_guides / "depth_dlss").exists():
                run([cli, "depth", "--backend", "da3", "-i", str(half_rate), "-o", str(fg_guides), "--mv-dir", str(fg_guides / "mv_dlss")], d / "halfrate_depth.log")
            for backend in ("dlssg", "rife", "blend"):
                if done("fg", backend):
                    continue
                fg_dir = d / f"fg_{backend}"
                text = run([cli, "fg", "--backend", backend, "--multiplier", "2", "-i", str(half_rate), "-o", str(fg_dir), "--no-auto",
                            "--depth-dir", str(fg_guides / "depth_dlss"), "--mv-dir", str(fg_guides / "mv_dlss")], d / f"fg_{backend}.log")
                cmp_text = run([cli, "compare", "--ref", str(clip), "--test", str(fg_dir / "color_fg"), "--start", "1", "--step", "2", "-q", *frames], d / f"cmp_fg_{backend}.log")
                r.setdefault("fg", {})[backend] = dict(
                    psnr_y_db=grab(r"PSNR Y:\s+([\d.]+) dB", cmp_text), psnr_y_min_db=grab(r"PSNR Y:\s+[\d.]+ dB \(min ([\d.]+)\)", cmp_text),
                    psnr_rgb_db=grab(r"PSNR RGB:\s+([\d.]+) dB", cmp_text), ssim_y=grab(r"SSIM Y:\s+([\d.]+)", cmp_text), ms_per_frame=grab(r"ms/frame:\s+([\d.]+)", text))
                print(f"  fg {backend}: {r['fg'][backend]}", flush=True)
                save()
                shutil.rmtree(fg_dir, ignore_errors=True)

        if "upscale" in sections:
            probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(clip)],
                                              check=True, capture_output=True, text=True).stdout)["streams"][0]
            w, h = probe["width"] // 2 // 2 * 2, probe["height"] // 2 // 2 * 2
            half = d / "in_half.mp4"
            if not half.exists():
                run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip), "-an", *(["-frames:v", str(args.frames)] if args.frames > 0 else []),
                     "-vf", f"scale={w}:{h}:flags=lanczos", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                     "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(half)], d / "in_half.log")
            guides = d / "half_passes"
            if not (guides / "mv_dlss").exists():
                run([cli, "flow", "--backend", "ofa", "-i", str(half), "-o", str(guides)], d / "half_flow.log")
            if not (guides / "depth_dlss").exists():
                run([cli, "depth", "--backend", "da3", "-i", str(half), "-o", str(guides), "--mv-dir", str(guides / "mv_dlss")], d / "half_depth.log")
            variants = {
                "nis": ["--backend", "nis"],
                "bicubic": ["--backend", "bicubic"],
                "dlss": ["--backend", "dlss", "--no-fallback"],
                "dlss+guides": ["--backend", "dlss", "--no-fallback", "--depth-dir", str(guides / "depth_dlss"), "--mv-dir", str(guides / "mv_dlss")],
                # open-source models through TensorRT (TASK-0022): the ONNX is exported on first use, engines are cached
                "trt-x2plus": ["--backend", "trt", "--model", "realesrgan-x2plus", "--no-fallback"],
                "trt-general-x4v3": ["--backend", "trt", "--model", "realesr-general-x4v3", "--no-fallback"],
            }
            for key, extra in variants.items():
                if done("upscale", key):
                    continue
                up = d / f"up_{key.replace('+', '_')}"
                text = run([cli, "upscale", *extra, "--scale", "2", "-i", str(half), "-o", str(up)], d / f"up_{key}.log")
                ms = grab(r"ms/frame:\s+([\d.]+)", text)
                cmp_text = run([cli, "compare", "--ref", str(clip), "--test", str(up / "color_sr"), "-q", *frames], d / f"cmp_{key}.log")
                r.setdefault("upscale", {})[key] = dict(
                    psnr_y_db=grab(r"PSNR Y:\s+([\d.]+) dB", cmp_text), psnr_y_min_db=grab(r"PSNR Y:\s+[\d.]+ dB \(min ([\d.]+)\)", cmp_text),
                    psnr_rgb_db=grab(r"PSNR RGB:\s+([\d.]+) dB", cmp_text), ssim_y=grab(r"SSIM Y:\s+([\d.]+)", cmp_text), ms_per_frame=ms, input=f"{w}x{h}")
                print(f"  upscale {key}: {r['upscale'][key]}", flush=True)
                save()
            shutil.rmtree(d / "up_bicubic", ignore_errors=True)  # keep the disk usage in check
            shutil.rmtree(d / "up_nis", ignore_errors=True)

    # ---- tables
    def f(v, spec=".2f"):
        return format(v, spec) if isinstance(v, (int, float)) else "—"

    if "flow" in sections:
        print("\n| Клип | Бэкенд | Кадров | mean |v|, px | warp-PSNR, дБ (min) | мс/кадр |\n|---|---|---|---|---|---|")
        for name, r in results.items():
            for backend, m in r.get("flow", {}).items():
                print(f"| {name} | {backend} | {f(m.get('frames'), '.0f')} | {f(m.get('mean_v_px'))} | {f(m.get('warp_psnr_db'))} ({f(m.get('warp_psnr_min_db'))}) | {f(m.get('ms_per_frame'), '.1f')} |")
    if "depth" in sections:
        print("\n| Клип | Бэкенд | Кадров | median depth, м | mean depth, м | TAE (warp) | мс/кадр |\n|---|---|---|---|---|---|---|")
        for name, r in results.items():
            for backend, m in r.get("depth", {}).items():
                if not isinstance(m, dict):
                    continue
                print(f"| {name} | {backend} | {f(m.get('frames'), '.0f')} | {f(m.get('median_m'))} | {f(m.get('mean_m'))} | {f(m.get('tae'), '.4f')} | {f(m.get('ms_per_frame'), '.1f')} |")
            if "scale_ratio_da3_over_vda" in r.get("depth", {}):
                print(f"| {name} | DA3 / VDA (median) | | ×{r['depth']['scale_ratio_da3_over_vda']:.2f} | | | |")
    if "fg" in sections:
        print("\n| Клип | Бэкенд FG | PSNR Y, дБ (min) | PSNR RGB, дБ | SSIM Y | мс/кадр |\n|---|---|---|---|---|---|")
        for name, r in results.items():
            for key, m in r.get("fg", {}).items():
                print(f"| {name} | {key} | {f(m.get('psnr_y_db'))} ({f(m.get('psnr_y_min_db'))}) | {f(m.get('psnr_rgb_db'))} | {f(m.get('ssim_y'), '.4f')} | {f(m.get('ms_per_frame'), '.1f')} |")
    if "upscale" in sections:
        print("\n| Клип | Вход | Бэкенд | PSNR Y, дБ (min) | PSNR RGB, дБ | SSIM Y | мс/кадр |\n|---|---|---|---|---|---|---|")
        for name, r in results.items():
            for key, m in r.get("upscale", {}).items():
                print(f"| {name} | {m.get('input')} | {key} | {f(m.get('psnr_y_db'))} ({f(m.get('psnr_y_min_db'))}) | {f(m.get('psnr_rgb_db'))} | {f(m.get('ssim_y'), '.4f')} | {f(m.get('ms_per_frame'), '.1f')} |")
    print(f"\nresults: {results_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"bench_clips.py: {e}", file=sys.stderr)
        sys.exit(1)
