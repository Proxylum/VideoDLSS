"""Shared helpers for models/export and depth_worker: registry lookup, HuggingFace download, hashing."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent
REGISTRY = MODELS_DIR / "registry.json"


def load_registry(path=REGISTRY):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def registry_entry(model_id, path=REGISTRY):
    for m in load_registry(path)["models"]:
        if m["id"] == model_id:
            return m
    known = ", ".join(m["id"] for m in load_registry(path)["models"])
    raise KeyError(f"unknown model '{model_id}' (registry has: {known})")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_download(repo, filename, cache_dir, log=print):
    """Download one file from a HuggingFace repo into cache_dir (flat layout). Returns the path."""
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{repo.replace('/', '__')}__{filename}"
    if dest.exists():
        return dest
    log(f"downloading {repo}/{filename} ...")
    got = hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache_dir / "hf"), token=os.environ.get("HF_TOKEN"))
    # copy out of the hub cache into a flat, predictable file
    import shutil

    shutil.copyfile(got, dest)
    return dest


def download_url(url, dest, sha256="", log=print):
    """Download `url` to `dest` (atomic: .part then rename) unless it exists; verify the sha256 when one is pinned,
    otherwise print the hash so it can be pinned in models/registry.json. Returns the path."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log(f"downloading {url} ...")
        part = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "dlssvid-models-export"})
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    log(f"  {done * 100 // total}% of {total / 1e6:.1f} MB", ) if done % (16 << 20) < (1 << 20) else None
        part.replace(dest)
    got = sha256_file(dest)
    if sha256 and got != sha256:
        dest.unlink()
        raise RuntimeError(f"{dest.name}: sha256 {got} does not match the registry ({sha256}); the file was removed")
    if not sha256:
        log(f"{dest.name}: sha256 {got} (not pinned in the registry yet)")
    return dest


def model_input_size(w, h, input_size=518, multiple=14, max_aspect=1.78):
    """Same rule as core/stages/depth/DepthPreprocess.cpp::ComputeModelInputSize."""
    ratio = max(w, h) / min(w, h)
    short = input_size * max_aspect / ratio if ratio > max_aspect else input_size
    short = max(multiple, round(short / multiple) * multiple)
    long_ = max(short, round(max(w, h) * short / min(w, h) / multiple) * multiple)
    return (int(long_), int(short)) if w >= h else (int(short), int(long_))
