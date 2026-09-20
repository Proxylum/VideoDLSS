#!/usr/bin/env python3
"""The five acceptance clips (ТЗ §9, TASK-0010): 10 s each, cut from open (CC BY 3.0) sources.

    python tests/data/fetch.py            # download the sources (sha256-checked), cut the clips, write clips/manifest.json
    python tests/data/fetch.py --check    # verify what is on disk (no network), exit 1 when a clip is missing / short
    python tests/data/fetch.py --sources-only [--print-hashes]
    python tests/data/fetch.py --only face --force

Sources land in tests/data/cache/ (git-ignored; ~940 MB: Tears of Steel 1080p and Big Buck Bunny 1080p60 come as ZIPs that are extracted and deleted),
clips in tests/data/clips/ (git-ignored). Cutting needs the ffmpeg / ffprobe CLI on PATH (or --ffmpeg DIR,
env FFMPEG_BIN). Licences and credits: tests/data/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CLIPS_DIR = HERE / "clips"
MANIFEST = CLIPS_DIR / "manifest.json"

# sha256 of the file as stored in cache/ (for the ZIP source: the extracted member); `zip_sha256` covers the archive.
SOURCES = {
    "tos_1080p": dict(
        url="https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip",
        file="tears_of_steel_1080p.mov",
        zip_member="tears_of_steel_1080p.mov",
        zip_size=583774281,
        zip_sha256="d87a41de040d3814dbde143e9ab85ef122caf22265f660b0bebf476cd8b357a5",
        size=583774083,
        sha256="99486359be7e3681168a0fe94e1cbb0284c48b57b2a3ce7df4fa75e185987a15",
        title="Tears of Steel (2012, live action)",
        license="CC BY 3.0",
        credit="(CC) Blender Foundation | mango.blender.org",
    ),
    "bbb_1080p_60fps": dict(
        url="https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_60fps_normal.mp4.zip",
        file="bbb_sunflower_1080p_60fps_normal.mp4",
        zip_member="bbb_sunflower_1080p_60fps_normal.mp4",
        zip_size=355019001,
        zip_sha256="68c456673409f8df09b80d0afe29ecb38ef110551fa8a93c83e54a96ebdaec78",
        size=355856562,
        sha256="3aa2d8946ec7cf3b50b7148fb3338ca421e6ec34987b20a4044845d674794e3e",
        title="Big Buck Bunny (2008; 2013 re-render, 1080p 60 fps)",
        license="CC BY 3.0",
        credit="(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org",
    ),
    "bbb_720_10s_1mb": dict(
        url="https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
        file="Big_Buck_Bunny_720_10s_1MB.mp4",
        size=969201,
        sha256="18b99ec25f32f6bd2223aa54e4b5632533328bf5cc81c283eba7604c42649f75",
        title="Big Buck Bunny — 720p, 10 s at 1 Mbit/s (test-videos.co.uk encode)",
        license="CC BY 3.0 (Big Buck Bunny); re-encoded by test-videos.co.uk",
        credit="(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org; encode: test-videos.co.uk",
    ),
}

# start = seconds into the source; crop = ffmpeg crop expression for letterboxed sources (None = as is).
# Chosen from contact sheets (docs: tests/data/README.md): continuous shots without cuts wherever the films allow it.
CLIPS = [
    dict(name="static", source="tos_1080p", start=103.0, duration=10.0, crop=None,
         description="static camera, live action: the round window of the lab while an airship passes outside (ТЗ «статичная сцена»)"),
    dict(name="pan", source="tos_1080p", start=274.5, duration=10.0, crop=None,
         description="lateral tracking shot along the canal: Thom walking, trees and railings with parallax, a red car passes (ТЗ «быстрая панорама»)"),
    dict(name="face", source="tos_1080p", start=239.0, duration=10.0, crop=None,
         description="three real faces in a dialogue, medium close-up, static camera, continuous take (ТЗ «лицо крупно»)"),
    dict(name="foliage_hair", source="bbb_1080p_60fps", start=41.6, duration=10.0, crop=None,
         description="grass in the foreground, moss and leaves, Big Buck Bunny's fur as he climbs out of the burrow; 60 fps = ground truth for frame generation (ТЗ «листва/волосы»)"),
    dict(name="lowbitrate_720", source="bbb_720_10s_1mb", copy=True,
         description="720p at 1 Mbit/s with visible compression artefacts, as published (ТЗ «низкобитрейтное 720p»)"),
]

X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, expected_size: int | None, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dlssvid-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                if expected_size and total and total != expected_size:
                    raise RuntimeError(f"{url}: Content-Length {total} != expected {expected_size}")
                done = 0
                next_report = 0.0
                t0 = time.time()
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    done += len(b)
                    if total and done / total >= next_report:
                        print(f"    {dest.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB  {done / max(time.time() - t0, 1e-3) / 1e6:.1f} MB/s", flush=True)
                        next_report += 0.1
            if expected_size and part.stat().st_size != expected_size:
                raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected_size}")
            part.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 — retry on any network error
            last_err = e
            print(f"    attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed: {url}: {last_err}")


def ensure_source(key: str, print_hashes: bool, force: bool) -> Path:
    src = SOURCES[key]
    dest = CACHE / src["file"]
    if dest.exists() and not force:
        if src.get("sha256"):
            actual = sha256_of(dest)
            if actual == src["sha256"]:
                print(f"  {key}: cached, sha256 ok")
                return dest
            print(f"  {key}: cached file has sha256 {actual}, expected {src['sha256']} — re-downloading")
        else:
            print(f"  {key}: cached (no reference sha256 yet: {sha256_of(dest)})")
            return dest
    if "zip_member" in src:
        zpath = CACHE / (src["file"] + ".zip")
        print(f"  {key}: downloading {src['url']} ({(src.get('zip_size') or 0) / 1e6:.0f} MB)")
        download(src["url"], zpath, src.get("zip_size"))
        actual = sha256_of(zpath)
        if src.get("zip_sha256") and actual != src["zip_sha256"]:
            raise RuntimeError(f"{key}: zip sha256 {actual} != {src['zip_sha256']}")
        if print_hashes or not src.get("zip_sha256"):
            print(f"    zip sha256 {actual}")
        print(f"    extracting {src['zip_member']}")
        with zipfile.ZipFile(zpath) as z:
            with z.open(src["zip_member"]) as m, open(dest.with_suffix(dest.suffix + ".part"), "wb") as out:
                shutil.copyfileobj(m, out, 1 << 20)
        dest.with_suffix(dest.suffix + ".part").replace(dest)
        zpath.unlink()
    else:
        print(f"  {key}: downloading {src['url']} ({(src.get('size') or 0) / 1e6:.1f} MB)")
        download(src["url"], dest, src.get("size"))
    actual = sha256_of(dest)
    if src.get("sha256") and actual != src["sha256"]:
        raise RuntimeError(f"{key}: sha256 {actual} != {src['sha256']}")
    if print_hashes or not src.get("sha256"):
        print(f"    sha256 {actual}  size {dest.stat().st_size}")
    return dest


def ffmpeg_bin(name: str, ffdir: str | None) -> str:
    for d in [ffdir, os.environ.get("FFMPEG_BIN")]:
        if d:
            p = Path(d) / (name + (".exe" if os.name == "nt" else ""))
            if p.exists():
                return str(p)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found on PATH: install ffmpeg or pass --ffmpeg DIR / FFMPEG_BIN")
    return found


def probe(ffprobe: str, path: Path) -> dict:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt,color_space:format=duration,bit_rate",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    audio = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
                           check=True, capture_output=True, text=True).stdout.strip()
    return dict(width=s["width"], height=s["height"], fps=round(int(num) / int(den), 3), frames=int(s.get("nb_read_frames") or 0),
                codec=s["codec_name"], pix_fmt=s.get("pix_fmt"), color_space=s.get("color_space"),
                duration=round(float(j["format"]["duration"]), 3), bitrate_kbps=round(int(j["format"].get("bit_rate") or 0) / 1000),
                audio=audio or None)


def cut_clip(clip: dict, src_path: Path, ffmpeg: str, force: bool) -> Path:
    out = CLIPS_DIR / f"{clip['name']}.mp4"
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        print(f"  {clip['name']}: exists ({out.name})")
        return out
    if clip.get("copy"):
        shutil.copyfile(src_path, out)
        print(f"  {clip['name']}: copied {src_path.name} as is")
        return out
    if clip["start"] is None:
        raise RuntimeError(f"{clip['name']}: no start time defined")
    vf = ["-vf", clip["crop"]] if clip.get("crop") else []
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{clip['start']:.3f}", "-i", str(src_path), "-t", f"{clip['duration']:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?", *vf, *X264, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    print(f"  {clip['name']}: {src_path.name} @ {clip['start']:.1f} s + {clip['duration']:.0f} s -> {out.name}")
    subprocess.run(cmd, check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the clips on disk without downloading; exit 1 on problems")
    ap.add_argument("--sources-only", action="store_true", help="download / verify the sources, do not cut clips")
    ap.add_argument("--print-hashes", action="store_true", help="print sha256 of every source (to fill SOURCES)")
    ap.add_argument("--only", action="append", default=[], help="clip name(s) to (re)build")
    ap.add_argument("--force", action="store_true", help="re-download sources / re-cut clips")
    ap.add_argument("--ffmpeg", help="directory with ffmpeg / ffprobe")
    args = ap.parse_args()

    clips = [c for c in CLIPS if not args.only or c["name"] in args.only]
    if args.only and len(clips) != len(args.only):
        print("unknown clip name; known:", ", ".join(c["name"] for c in CLIPS), file=sys.stderr)
        return 2

    if args.check:
        ffprobe = ffmpeg_bin("ffprobe", args.ffmpeg)
        problems = 0
        man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        for c in clips:
            p = CLIPS_DIR / f"{c['name']}.mp4"
            if not p.exists():
                print(f"  {c['name']}: MISSING ({p})")
                problems += 1
                continue
            info = probe(ffprobe, p)
            expected = man.get("clips", {}).get(c["name"], {}).get("probe", {}).get("frames")
            ok = info["frames"] > 0 and (expected is None or info["frames"] == expected) and info["duration"] >= 9.5
            print(f"  {c['name']}: {info['width']}x{info['height']} {info['fps']} fps {info['frames']} frames {info['duration']} s {'ok' if ok else 'BAD'}")
            problems += 0 if ok else 1
        return 1 if problems else 0

    needed = {c["source"] for c in clips}
    print("sources:")
    paths = {k: ensure_source(k, args.print_hashes, args.force) for k in SOURCES if k in needed}
    if args.sources_only:
        return 0

    ffmpeg = ffmpeg_bin("ffmpeg", args.ffmpeg)
    ffprobe = ffmpeg_bin("ffprobe", args.ffmpeg)
    print("clips:")
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"clips": {}}
    man.setdefault("clips", {})
    man["ffmpeg"] = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    for c in clips:
        src = SOURCES[c["source"]]
        out = cut_clip(c, paths[c["source"]], ffmpeg, args.force)
        info = probe(ffprobe, out)
        man["clips"][c["name"]] = dict(
            file=out.name, description=c["description"],
            source=dict(key=c["source"], title=src["title"], url=src["url"], file=src["file"], sha256=src.get("sha256"), license=src["license"], credit=src["credit"]),
            start=c.get("start"), duration=c.get("duration"), crop=c.get("crop"), copied=bool(c.get("copy")), probe=info)
        print(f"    {info['width']}x{info['height']} {info['fps']} fps, {info['frames']} frames, {info['duration']} s, {info['codec']} {info['bitrate_kbps']} kbps"
              f"{', audio ' + info['audio'] if info['audio'] else ''}")
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"fetch.py: {e}", file=sys.stderr)
        sys.exit(1)
