#!/usr/bin/env python3
"""Third-party SDKs, Qt and the model-code checkouts the build and the ONNX export need, pinned to the revisions this project was
developed against (docs/dll-setup.md). Nothing here is committed: the folder is outside the repository or git-ignored.

    python scripts/setup_sdk.py                       # clone the required checkouts into <root>, write <root>/env.cmd
    python scripts/setup_sdk.py --qt                  # also Qt 6.8.3 msvc2022_64 through aqtinstall (~2 GB)
    python scripts/setup_sdk.py --reference           # also the reference sources (Streamline, NIS, OptiScaler, ComfyUI-DLSS5-NR)
    python scripts/setup_sdk.py --check [--from-env]  # verify folders and pinned revisions; exit 1 on a mismatch, 77 when nothing to check
    python scripts/setup_sdk.py --env                 # print the environment variables (cmd syntax) for <root>

<root> is --root, else SDK_ROOT from the environment, else <workspace>/SDK when the project lives in <workspace>/Project/dlss-video
(the agent workspace layout), else build/sdk. Needs git on PATH; --qt needs pip (aqtinstall is installed into the running Python).
The DLSS DLLs themselves (nvngx_dlss.dll, nvngx_dlssg.dll) are copied from the DLSS SDK by the build; nvngx_dlssnr.dll and the
TensorRT / CUDA runtimes are not part of this script (README: NVIDIA runtimes).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QT_VERSION = "6.8.3"
QT_ARCH = "win64_msvc2022_64"
QT_SUBDIR = Path("Qt") / QT_VERSION / "msvc2022_64"

# dir (under <root>) -> what it is. `ref` is the tag or branch the commit came from; `commit` is what gets checked out.
REQUIRED = {
    "DLSS": dict(url="https://github.com/NVIDIA/DLSS.git", ref="v310.9.1", commit="374959484e79a640feaba44c93ac8cfb0a03f5b5",
                 env="DLSS_SDK_ROOT", note="DLSS SDK 310.9.1: NGX headers, import library, nvngx_dlss*.dll (stage 5 / 7)"),
    "OpticalFlowSDK": dict(url="https://github.com/NVIDIA/NVIDIAOpticalFlowSDK.git", ref="master", commit="edb50da3cf849840d680249aa6dbef248ebce2ca",
                           env="NV_OPTICAL_FLOW_SDK_ROOT", note="Optical Flow SDK headers (stage 3)"),
    "TensorRT": dict(url="https://github.com/NVIDIA/TensorRT.git", ref="v10.16", commit="52399f555c2f80cb690a4a558b604e1a5f227e7c",
                     env="TENSORRT_ROOT", note="TensorRT 10.16 headers; the DLLs come from pip tensorrt-cu12 in models/export/.venv"),
    "dlssnr-patcher": dict(url="https://github.com/dev-camo/dlssnr-patcher.git", ref="main", commit="98dac112990e5cea30ab03d74625fc848bce5c92",
                           env="DLSSNR_PATCHER_ROOT", note="external tool (GPLv2, not vendored) for nvngx_dlssnr.dll on RTX 20/30/40 (stage 6)"),
    "models/Depth-Anything-3": dict(url="https://github.com/ByteDance-Seed/Depth-Anything-3.git", ref="main", commit="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
                                    env=None, note="DA3 code for the ONNX export; installed editable into models/export/.venv"),
    "models/Video-Depth-Anything": dict(url="https://github.com/DepthAnything/Video-Depth-Anything.git", ref="main", commit="4f5ae23172ba60fd7bc11ef671cca678842c7072",
                                        env="VDA_REPO", note="VDA code for the ONNX export (not a pip package)"),
    "models/SEA-RAFT": dict(url="https://github.com/princeton-vl/SEA-RAFT.git", ref="main", commit="9137517ba24e628442aec097d3afe71d03503b75",
                            env="SEARAFT_REPO", note="SEA-RAFT code for the ONNX export"),
}
REFERENCE = {
    "Streamline": dict(url="https://github.com/NVIDIAGameWorks/Streamline.git", ref="v2.9.0", commit="b998246a3d499c08765c5681b229c9e6b4513348",
                       env="STREAMLINE_ROOT", note="not used by the build (FG goes through the NGX API); kept for reference"),
    "NVIDIAImageScaling": dict(url="https://github.com/NVIDIAGameWorks/NVIDIAImageScaling.git", ref="v1.0.3", commit="35e13ba316c98eeecf16f37eae70ce88019911f6",
                               env=None, note="NIS 1.0.3 — already vendored in third_party/nis"),
    "ComfyUI-DLSS5-NR": dict(url="https://github.com/lisitskyaa/ComfyUI-DLSS5-NR.git", ref="v0.3.1", commit="41dcdfa593cb61b6a98c65bb8ed27606260bb598",
                             env=None, note="reference for the NGX Feature 18 call sequence (docs/plans/06-nr.md)"),
    "OptiScaler_DLSSNR": dict(url="https://github.com/Dagherbou/OptiScaler_DLSSNR.git", ref="dlss-neural-rendering", commit="973761621353b99bee3dc7d4bb27b117fef2644f",
                              env=None, note="reference for the NR parameter names and the forwarder (docs/plans/06-nr.md)"),
}


def default_root() -> Path:
    if os.environ.get("SDK_ROOT"):
        return Path(os.environ["SDK_ROOT"])
    if ROOT.parent.name == "Project":
        return ROOT.parent.parent / "SDK"
    return ROOT / "build" / "sdk"


def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(r.stdout + r.stderr).strip()[-400:]}")
    return r.stdout.strip()


def head_of(d: Path) -> str | None:
    if not (d / ".git").exists():
        return None
    try:
        return git("rev-parse", "HEAD", cwd=d)
    except RuntimeError:
        return None


def clone(name: str, item: dict, root: Path) -> None:
    dest = root / name
    head = head_of(dest)
    if head:
        print(f"  {name}: present at {head[:12]}" + ("" if head == item["commit"] else f" — pinned {item['commit'][:12]} ({item['ref']}); "
              f"update by hand: git -C \"{dest}\" fetch origin {item['ref']} && git -C \"{dest}\" checkout {item['commit'][:12]}"))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: cloning {item['url']} @ {item['ref']} ({item['commit'][:12]})")
    git("init", "-q", str(dest))
    git("remote", "add", "origin", item["url"], cwd=dest)
    try:
        git("fetch", "--depth", "1", "origin", item["commit"], cwd=dest)  # GitHub serves reachable commits by SHA
    except RuntimeError:
        git("fetch", "--depth", "50", "origin", item["ref"], cwd=dest)
    git("checkout", "-q", "--detach", item["commit"], cwd=dest)
    got = head_of(dest)
    if got != item["commit"]:
        raise RuntimeError(f"{name}: checked out {got}, expected {item['commit']}")


def install_qt(root: Path) -> None:
    qt = root / QT_SUBDIR
    if (qt / "bin" / "Qt6Core.dll").exists():
        print(f"  Qt {QT_VERSION}: present at {qt}")
        return
    print(f"  Qt {QT_VERSION} {QT_ARCH}: installing with aqtinstall into {root / 'Qt'} (~2 GB)")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "aqtinstall>=3.3,<4"], check=True)
    subprocess.run([sys.executable, "-m", "aqt", "install-qt", "windows", "desktop", QT_VERSION, QT_ARCH, "--outputdir", str(root / "Qt")], check=True)
    if not (qt / "bin" / "Qt6Core.dll").exists():
        raise RuntimeError(f"Qt not found at {qt} after aqtinstall")


def install_da3_editable(root: Path) -> None:
    venv_py = ROOT / "models" / "export" / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        print("  DA3: models/export/.venv not created yet — after `pip install -r models/export/requirements.txt` run:"
              f"\n      {venv_py} -m pip install --no-deps -e \"{root / 'models' / 'Depth-Anything-3'}\"")
        return
    r = subprocess.run([str(venv_py), "-c", "import depth_anything_3"], capture_output=True)
    if r.returncode == 0:
        print("  DA3: importable from the export venv")
        return
    subprocess.run([str(venv_py), "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(root / "models" / "Depth-Anything-3")], check=True)
    print("  DA3: installed editable into models/export/.venv")


def env_lines(root: Path) -> list[str]:
    lines = []
    for name, item in {**REQUIRED, **REFERENCE}.items():
        if item["env"] and (root / name).exists():
            lines.append(f"set {item['env']}={root / name}")
    if (root / QT_SUBDIR).exists():
        lines.append(f"set QT_ROOT={root / QT_SUBDIR}")
    return lines


def write_env(root: Path) -> None:
    lines = env_lines(root)
    (root / "env.cmd").write_text("@echo off\r\n:: generated by scripts/setup_sdk.py — the SDK roots for the build (CMakePresets.json reads them)\r\n"
                                  + "\r\n".join(lines) + "\r\n", encoding="utf-8")
    (root / "env.ps1").write_text("# generated by scripts/setup_sdk.py\n" + "\n".join(l.replace("set ", "$env:", 1).replace("=", " = '", 1) + "'" for l in lines) + "\n",
                                  encoding="utf-8")
    print(f"  env: {root / 'env.cmd'} ({len(lines)} variables), {root / 'env.ps1'}")


def check(root: Path | None, from_env: bool) -> int:
    """0 — every checkout that could be located is at its pinned revision; 1 — mismatch or a required one missing; 77 — nothing to check."""
    problems, checked = 0, 0
    for name, item in {**REQUIRED, **REFERENCE}.items():
        required = name in REQUIRED
        d = None
        if from_env and item["env"] and os.environ.get(item["env"]):
            d = Path(os.environ[item["env"]])
        elif root is not None:
            d = root / name
        if d is None:
            continue
        head = head_of(d)
        if head is None:
            if required and not from_env:
                print(f"  MISSING {name}: {d}")
                problems += 1
            elif required:
                print(f"  {name}: {d} is not a git checkout — cannot verify the revision")
            continue
        checked += 1
        ok = head == item["commit"]
        print(f"  {'ok     ' if ok else 'DIFFERS'} {name}: {head[:12]} ({'pinned ' + item['commit'][:12] if not ok else item['ref']})")
        problems += 0 if ok else 1
    qt = Path(os.environ["QT_ROOT"]) if from_env and os.environ.get("QT_ROOT") else (root / QT_SUBDIR if root else None)
    if qt is not None:
        qt_ok = (qt / "bin" / "Qt6Core.dll").exists()
        print(f"  {'ok     ' if qt_ok else 'MISSING'} Qt {QT_VERSION}: {qt}")
        checked += 1
        if not qt_ok and not from_env:
            problems += 1
    if checked == 0:
        print("  nothing to check (no SDK environment variables and no root)")
        return 77
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="SDK folder (default: SDK_ROOT, <workspace>/SDK or build/sdk)")
    ap.add_argument("--qt", action="store_true", help=f"install Qt {QT_VERSION} {QT_ARCH} with aqtinstall")
    ap.add_argument("--reference", action="store_true", help="also clone the reference sources")
    ap.add_argument("--check", action="store_true", help="verify the layout against the pinned revisions")
    ap.add_argument("--from-env", action="store_true", help="--check: locate the checkouts through the environment variables instead of --root")
    ap.add_argument("--env", action="store_true", help="print the environment variables for the build and exit")
    args = ap.parse_args()
    root = args.root or (None if args.from_env and not os.environ.get("SDK_ROOT") else default_root())
    if args.check:
        print(f"SDK layout check ({'environment' if args.from_env else root}):")
        return check(root, args.from_env)
    if args.env:
        print("\n".join(env_lines(root)))
        return 0
    print(f"SDK root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name, item in REQUIRED.items():
        clone(name, item, root)
    if args.reference:
        for name, item in REFERENCE.items():
            clone(name, item, root)
    if args.qt:
        install_qt(root)
    install_da3_editable(root)
    write_env(root)
    print("done — set the variables from env.cmd (or CMakePresets picks them up from the environment) and configure the build")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"setup_sdk.py: {e}", file=sys.stderr)
        sys.exit(1)
