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


def model_input_size(w, h, input_size=518, multiple=14, max_aspect=1.78):
    """Same rule as core/stages/depth/DepthPreprocess.cpp::ComputeModelInputSize."""
    ratio = max(w, h) / min(w, h)
    short = input_size * max_aspect / ratio if ratio > max_aspect else input_size
    short = max(multiple, round(short / multiple) * multiple)
    long_ = max(short, round(max(w, h) * short / min(w, h) / multiple) * multiple)
    return (int(long_), int(short)) if w >= h else (int(short), int(long_))
