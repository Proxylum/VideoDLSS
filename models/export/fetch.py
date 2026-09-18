#!/usr/bin/env python
"""Download model weights listed in models/registry.json into models/cache/.

  python fetch.py da3metric-large metric-vda-small
  python fetch.py --all
"""
import argparse
import sys

from common import MODELS_DIR, hf_download, load_registry, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cache", default=str(MODELS_DIR / "cache"))
    args = ap.parse_args()
    reg = load_registry()
    ids = [m["id"] for m in reg["models"] if m.get("hf")] if args.all else args.ids
    if not ids:
        ap.error("give model ids or --all")
    rc = 0
    for mid in ids:
        entry = next((m for m in reg["models"] if m["id"] == mid), None)
        if not entry or not entry.get("hf"):
            print(f"{mid}: no HuggingFace source", file=sys.stderr)
            rc = 1
            continue
        files = [entry["hf_file"]] if entry.get("hf_file") else ["config.json", "model.safetensors"]
        for f in files:
            p = hf_download(entry["hf"], f, args.cache)
            print(f"{mid}: {p} sha256 {sha256_file(p)[:16]} license {entry.get('license', '?')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
