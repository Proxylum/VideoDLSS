#!/usr/bin/env python3
"""A/B of the Neural Rendering settings (experiments E1-E11 of docs/benchmarks.md «A/B настроек»; TASK-0050).

Every variant is one `dlssvid nr --start F --frames N --warmup K --params-file <variant>/params.json [A/B flags]` into a pass root of its
own (<out>/<run>/<experiment>/<variant>/ - never the project's passes root), then `dlssvid compare` against the baseline variant, against
what the variant read (color_sr, else the video) and against the next frame of the same output. Tables: stdout (this invocation) and
<out>/nr_ab.md (every run of results.json); numbers: <out>/results.json. A stored experiment is reused only when its signature is the
same - the range, the variants' settings and flags, the content of the inputs, the dlssvid binary, for NGX the DLL, GPU and driver; --only
adds variants to it, --force redoes.

    python scripts/nr_ab.py --list
    python scripts/nr_ab.py --clip face.mp4 --passes <passes> --experiment E1 --start 48
    python scripts/nr_ab.py --project face.dlssvid.json --experiment E5,E6
    python scripts/nr_ab.py --clip clip.mp4 --prepare --experiment E7 --dry-run          # the commands only
    python scripts/nr_ab.py --clip clip.mp4 --passes <passes> --variants my_ab.json      # own variants

NGX runs use the GPU the CI runner shares: run them only while CI is idle. `--backend stub --warp` runs anywhere on given passes, or with
`--prepare --prepare-backends stub,stub,bicubic` (ctest scripts.nr_ab_stub); the default --prepare backends (da3,ofa,dlss) need the NVIDIA
GPU. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SAME_DB = 45.0        # spec §2b / §2i: two outputs above it are one picture (E2 identity, E5 / E7 / E11 «no effect»)
LINEAR_TOL = 0.05     # PR-7, E1: the edit at strength x <= 1 is x times the edit at 1 (our lerp, no clipping inside [input, model])
GAIN_DB = 1.0         # spec §2e: an inter-frame PSNR gain this large makes an MV variant the default (E4)
IDENTICAL_DB = 100.0  # `dlssvid compare` counts an identical frame as 100 dB in its means and writes null for an infinite minimum
# intensity 1: since PR-7 the strength is our blend after the model - a base intensity of --params / --project-params would scale every
# difference between variants (0: every variant one bypass run); E1 / E2 set it in their variants
STILL = {"guides": False, "temporal": 0.0, "intensity": 1.0}  # the model alone: Reset on every frame (still mode), no history of our filter either
MODEL = {"temporal": 0.0, "intensity": 1.0}                   # model history on, our flicker filter off: the breathing measured is the model's
# E1 / E2 measure the edit against the input, the blend starts from the tone-mapped frame: an identity tone stage
TONE = {"tonemap": "passthrough", "input_transfer": "srgb", "exposure": 0.0}
METRICS = ("base", "edit", "temporal", "time", "pairs")
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
RUN_KEYS = ("backend", "format")  # nr schema keys nr_ab passes as options (--backend / --format): a base file's would be overridden anyway
STYLES = ("default", "natural", "cinematic")  # DLSSNR.Style 0..2 by name, as NormalizeStageParams folds them
# options of `dlssvid nr` (name -> takes a value) a variant's flags / --nr-arg may carry. A setting there would win over every params.json,
# an A/B switch in --nr-arg over every variant's own: both make the variants one run
AB_FLAGS = {"--mv-scale": True, "--create-with-mv": False, "--no-depth-guide": False}
NR_ARG_OK = {"--skip-driver-check": False, "--dll-dir": True, "--params-block": True}
# options of `dlssvid nr` that are not stages.nr.params keys (cli/NrCommands.cpp): ResolveNrParams refuses them in --params-file, after the
# variants before ran - known_keys of --help minus these is what a params.json may hold
NOT_SETTINGS = {"input", "output", "color_dir", "depth_dir", "mv_dir", "masks_dir", "no_auto", "auto", "video", "codec", "frames", "start", "warmup",
                "params_file", "warp", "check", "check_size", "check_repeat", "check_features", "json", "help", "depth_guide"} | {o[2:].replace("-", "_") for o in (*AB_FLAGS, *NR_ARG_OK)}
INPUT_KEYS = ("color", "depth", "mv")  # a variant's own folders: what nr_command passes as --color-dir / --depth-dir / --mv-dir
EXPECT_KEYS = ("depth_used", "mv_used", "work_size", "passes", "mv_scale", "frames_written", "warmup_frames", "masks")  # stats read_nr keeps
NR_DLLS = ("nvngx_dlssnr.dll", "nvngx.dll_dlssvid.dll")  # kNrDllName, kNrForwarderName (core/stages/nr/INrBackend.h)
# a path in a dlssvid message (printed unquoted): the report keeps its last part ([hygiene]). A drive or UNC path may have spaces in its
# folders (each ends with a separator); only the last part stops at a space, a POSIX path at the first one (prose like «unit / auto» stays)
LOG_PREFIX = re.compile(r"^\[[0-9:.]+\] \[[a-z]+\] ")  # «[14:46:39.828] [error] » of a dlssvid log line: the reason starts after it
TIMEOUT: Optional[float] = None  # seconds per dlssvid call (--timeout): a hung NGX call must not hold the GPU the CI runner shares
# E8: the preview's phase 2 re-creates the feature on a change of its settings (spec §2f); proposed thresholds (TASK-0053), the spec
# gives none: every re-creation ok, the last ten no slower than 1.5x the first ten, the process's video memory back within 64 MiB
# after each release and no more than 128 MiB above the first init at the end
PHASE2_SLOWDOWN, PHASE2_RELEASE_MIB, PHASE2_DRIFT_MIB = 1.5, 64.0, 128.0
ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)(?:[^\\/\r\n'\"()]*[\\/])*[^\s'\"()]+|/[^\s'\"()]+")


@dataclass
class Variant:
    name: str                                     # folder name: [A-Za-z0-9_.-]+
    params: dict = field(default_factory=dict)    # nr schema keys on top of the base and the experiment's fixed ones
    flags: list = field(default_factory=list)     # A/B options of the standalone `dlssvid nr` (AB_FLAGS), not schema keys (spec §0.1)
    inputs: dict = field(default_factory=dict)    # "color" / "depth" / "mv": another pass folder; "@mv_target" = --mv-target-dir (E3 C)
    expect: dict = field(default_factory=dict)    # stats `nr --json` must report ({"depth_used": False}): a flag that did nothing fails the run
    optional: bool = False                        # a failure is recorded and the experiment goes on (passes 3-4 before PR-5); never the baseline


@dataclass
class Experiment:
    id: str
    title: str                                    # Russian: the table heading
    variants: list
    baseline: str = ""
    fixed: dict = field(default_factory=dict)     # every variant: STILL / MODEL
    metrics: tuple = ("base", "time")
    groups: list = field(default_factory=list)    # pairwise PSNR inside each group (E7; the "pairs" metric)
    frames: int = 12                              # spec §4: 12 frames of a real clip; breathing (E3 / E4) on 30
    needs: tuple = ()                             # "depth", "mv": inputs the experiment cannot run without - every run must report them used
    requires: tuple = ()                          # options `dlssvid nr --help` must list (E9: --scene-cut, PR-6)
    kind: str = "run"                             # run | check (`nr --check`: E8 its runs and re-creations, E6f features at once)
    check: str = ""                               # check kind: "runs" (E8) | "features" (E6f)
    unsupported: str = ""                         # why no build of PR-1d can run it (E8b)


def _v(name, **kw) -> Variant:
    return Variant(name, **kw)


EXPERIMENTS = [
    # PR-7: nr.intensity is our blend after the model (the model always gets DLSSNR.Intensity 1) - E1 and E2 check the blend, not NGX;
    # NGX's own reading of DLSSNR.Intensity (it saturated at 1, 0 was black: PR-2) needs a hidden A/B flag for a new DLL, not added yet
    Experiment("E1", "Сила правки: линейность смешивания (PR-7, §2b)",
               [_v(f"i{x:g}", params={"intensity": x}) for x in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)], "i1", {**STILL, **TONE}, ("base", "edit", "time")),
    Experiment("E2", "Сила 0: обход модели (PR-7, §2b)", [_v("i0", params={"intensity": 0}), _v("i1", params={"intensity": 1})], "i1", {**STILL, **TONE},
               ("edit", "base", "time")),
    Experiment("E3", "Единицы векторов движения (§2e)",
               [_v("unit", flags=["--mv-scale", "unit"]), _v("auto", flags=["--mv-scale", "auto"]),
                _v("target", flags=["--mv-scale", "unit"], inputs={"mv": "@mv_target"}, optional=True)],
               "unit", MODEL, ("temporal", "base", "time"), frames=30, needs=("mv",)),
    Experiment("E4", "Фича с нулевыми векторами при создании (§2e)", [_v("null"), _v("create_mv", flags=["--create-with-mv"])], "null", MODEL,
               ("temporal", "base", "time"), frames=30, needs=("mv",)),
    Experiment("E5", "Глубина для модели (§2e)",
               [_v("depth", expect={"depth_used": True}), _v("no_depth", flags=["--no-depth-guide"], expect={"depth_used": False})], "depth", MODEL,
               ("base", "temporal", "time"), needs=("depth", "mv")),
    Experiment("E6", "Проходы: время и видеопамять (§2c)",
               [_v(f"p{p}", params={"passes": p}, optional=p > 2) for p in (1, 2, 3, 4)], "p1", MODEL, ("time", "base")),
    # the schema's three styles (PR-2: 3..6 gave cinematic bit for bit, the preset changed nothing - both left the schema, TASK-0052);
    # a new DLL is re-checked here for these three; 3..6 or a preset again need a code change (ParseNrStyle, a preset flag)
    Experiment("E7", "Стили (§2i)", [_v(n, params={"style": n}) for n in ("default", "natural", "cinematic")],
               "natural", STILL, ("pairs", "edit", "time"), groups=[["default", "natural", "cinematic"]]),
    Experiment("E6f", "Несколько фич одновременно: VRAM на фичу (§2c)", [], kind="check", check="features", metrics=("time",),
               requires=("--check-features",)),
    Experiment("E8", "Цена создания фичи (§2f)", [], kind="check", check="runs", metrics=("time",)),
    Experiment("E8b", "Смена настроек без пересоздания фичи (§2f)", [],
               unsupported="нужен режим смены tone/structure посреди прогона без пересоздания фичи — не входит в PR-1d"),
    Experiment("E9", "Сброс на склейках (§2d)",
               [_v("cut0", params={"scene_cut": 0}), _v("cut024", params={"scene_cut": 0.24}), _v("cut024_t04", params={"scene_cut": 0.24, "temporal": 0.4})],
               "cut0", {"intensity": 1.0}, ("temporal", "base", "time"), needs=("mv",), requires=("--scene-cut",)),
    Experiment("E10", "Локальный тон и структура (§2a)",
               [_v("base"), _v("tone0", params={"local_tone": 0}), _v("structure0", params={"local_structure": 0})], "base", STILL, ("base", "edit", "time")),
    # the control with the auto mask (am_*): «no effect without it» means something only if the key acts with it on this clip
    Experiment("E11", "Структура кожи без автомаски (§2a)",
               [_v("follow", params={"skin_structure": -1}), _v("s0", params={"skin_structure": 0}), _v("s1", params={"skin_structure": 1}),
                _v("s2", params={"skin_structure": 2})]
               + [_v(f"am_{n}", params={"auto_mask": True, "skin_structure": s}) for n, s in (("follow", -1), ("s0", 0), ("s2", 2))],
               "follow", {**STILL, "auto_mask": False}, ("base", "edit", "time", "pairs"), groups=[["am_follow", "am_s0"], ["am_follow", "am_s2"]]),
]


# ---- pure helpers (tests/scripts/test_nr_ab.py) ----------------------------------------------------------------------------

def normalize_params(p: dict) -> dict:
    """Schema keys in snake_case and values as dlssvid nr reads them (NormalizeStageParams): kebab aliases and no_guides are folded in, so a base
    file with `local-tone` and a variant with `local_tone` do not both reach --params-file (the later one would win by accident); a style given as
    a number or as the index of a named one becomes the schema's string, so the check that it reached the run compares like with like."""
    out = {}
    for k, v in p.items():
        k = k.replace("-", "_")
        if k == "no_guides":
            k, v = "guides", not v
        if k == "style":
            if isinstance(v, int) and not isinstance(v, bool):
                v = str(v)
            if v in ("0", "1", "2"):
                v = STYLES[int(v)]
        out[k] = v
    return out


def merge_params(base: dict, fixed: dict, variant: dict) -> dict:
    return {**normalize_params(base), **normalize_params(fixed), **normalize_params(variant)}  # base < experiment < variant


def run_params(base: dict, fixed: dict, variant: dict) -> dict:
    """A variant's params.json: the merged settings without the keys nr_ab gives as options (RUN_KEYS)."""
    return {k: v for k, v in merge_params(base, fixed, variant).items() if k not in RUN_KEYS}


def load_base_params(path: Optional[Path]) -> dict:
    """A JSON object of nr keys, or a dlssvid-preset (spec §2h, PR-9) - its stages.nr.params."""
    if not path:
        return {}
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(j, dict) and j.get("format") == "dlssvid-preset":
        j = j.get("stages", {}).get("nr", {}).get("params", {})
    if not isinstance(j, dict):
        raise RuntimeError(f"{path}: not a JSON object of nr settings")
    return normalize_params(j)


def known_keys(help_text: str) -> set:
    """Option names of `dlssvid nr --help` in snake_case (a default-on flag also as the key without no_). A merged key outside the set would be
    refused by dlssvid nr; nr_ab says so before anything runs."""
    keys = set()
    for opt in re.findall(r"--([a-z0-9][a-z0-9_-]*)", help_text):
        k = opt.replace("-", "_")
        keys.add(k)
        if k.startswith("no_"):
            keys.add(k[3:])
    return keys


def setting_keys(help_text: str) -> set:
    """The nr settings `dlssvid nr --help` lists: every option but the run's own (NOT_SETTINGS) - those in a params.json fail the run - and
    the negated spellings of a flag (no_auto_mask): no nr key starts with no_, normalize_params folds no_guides into guides first."""
    return {k for k in known_keys(help_text) - NOT_SETTINGS if not k.startswith("no_")}


def check_options(tokens: list, allowed: dict, what: str) -> list:
    """`tokens` hold only options of `allowed` (name -> takes a value; `--a=b` or `--a b`, either spelling of the name); the tokens back. An
    empty `--a=` is refused: CLI11 would take the next token - a variant's first switch - as its value."""
    i, toks = 0, list(tokens)
    while i < len(toks):
        name, eq, val = toks[i].partition("=")
        key = name.replace("_", "-")
        if key not in allowed:
            raise RuntimeError(f"{what}: '{toks[i]}' is not one of {', '.join(allowed)} - a setting goes into params (--params / a variant's "
                               "params), never beside the variants")
        if allowed[key] and (eq and not val or not eq and i + 1 >= len(toks)):
            raise RuntimeError(f"{what}: {name} needs a value")
        if allowed[key] and not eq:
            i += 1
        if not allowed[key] and eq:
            raise RuntimeError(f"{what}: {name} takes no value")
        i += 1
    return toks


def expected_ab(flags: list) -> dict:
    """`nr --json` "ab" (NrAbJson) the variant's A/B switches must give: one that did not reach the run would leave the variants one run."""
    ab, toks = {}, list(flags)
    for i, t in enumerate(toks):
        name, eq, val = t.partition("=")
        key = name.replace("_", "-")
        if key == "--mv-scale":
            scale = val if eq else (toks[i + 1] if i + 1 < len(toks) else "")
            if scale != "unit":
                ab["mv_scale"] = scale
        elif key == "--create-with-mv":
            ab["create_with_mv"] = True
        elif key == "--no-depth-guide":
            ab["depth_guide"] = False
    return ab


def names_only(text: str) -> str:
    """A dlssvid message with its absolute paths cut to their last part: the reasons go into nr_ab.md and from there into docs ([hygiene])."""
    return ABS_PATH.sub(lambda m: PureWindowsPath(m.group(0)).name or m.group(0), LOG_PREFIX.sub("", text))


def option_value(tokens: list, name: str) -> Optional[str]:
    for i, t in enumerate(tokens):
        k, eq, v = t.partition("=")
        if k.replace("_", "-") == name:
            return v if eq else (tokens[i + 1] if i + 1 < len(tokens) else None)
    return None


def lost_params(params: dict, got: dict) -> list:
    """Keys of a variant's params.json the run reports with another value (`nr --json` params). A key the run leaves out (another backend's, or
    one at its default that the schema omits) is not a loss."""
    def same(x, y):
        num = lambda z: isinstance(z, (int, float)) and not isinstance(z, bool)
        return abs(float(x) - float(y)) <= 1e-6 if num(x) and num(y) else x == y

    return [f"{k}={v!r} (прогон: {got[k]!r})" for k, v in params.items() if k in got and not same(v, got[k])]


def pass_range(pass_dir: Optional[Path]):
    """(first, last, frame_count) of a pass folder's manifest; None without one."""
    if not pass_dir:
        return None
    try:
        m = json.loads((Path(pass_dir) / "manifest.json").read_text(encoding="utf-8"))
        r = m["frame_range"]
        return int(r["first"]), int(r["last"]), int(m.get("frame_count", r["last"] - r["first"] + 1))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def covers(pass_dir: Optional[Path], first: int, last: int) -> bool:
    """The pass has every frame first..last (bench_clips.mv_covers: a shorter pass would silently change the run)."""
    r = pass_range(pass_dir)
    return bool(r) and r[0] <= first and r[1] >= last


# ---- identities: what a stored result was measured with ----------------------------------------------------------------------

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Optional[Path]):
    """A clip: name, size and modification time (hashing gigabytes per invocation is not worth it)."""
    if not path:
        return None
    try:
        st = Path(path).stat()
        return [Path(path).name, st.st_size, st.st_mtime_ns]
    except OSError:
        return [Path(path).name]


def pass_identity(pass_dir: Optional[Path]):
    """A pass folder by its manifest's content, not its path: `dlssvid process` (--prepare, another backend, more frames) rewrites a folder in
    place, and the manifest's range, fingerprint and creation time change with it."""
    if not pass_dir:
        return None
    try:
        return hashlib.sha256((Path(pass_dir) / "manifest.json").read_bytes()).hexdigest()
    except OSError:
        return ["нет", Path(pass_dir).name]


def masks_identity(root: Optional[Path]):
    return sorted([d.name, pass_identity(d)] for d in Path(root).glob("mask_*") if (d / "manifest.json").exists()) if root else None


def nvidia_dll_dirs(cli: Path, nr_arg: list) -> list:
    """NvidiaDllSearchPaths (core/stages/upscale/IUpscaler.cpp): --dll-dir, DLSSVID_NVIDIA_DLL_DIR, <exe>/nvidia, <exe>."""
    dirs = [Path(d) for d in (option_value(nr_arg, "--dll-dir"), os.environ.get("DLSSVID_NVIDIA_DLL_DIR")) if d]
    return dirs + [Path(cli).parent / "nvidia", Path(cli).parent]


def nvidia_smi() -> Optional[str]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_identity(cli: Path, backend: str, nr_arg: list) -> dict:
    """Known before a run: the dlssvid binary (`--version` is the release, the same across unreleased builds - the NR code and the schema are in
    the exe), for NGX the DLL and forwarder the CLI would load and the GPU / driver nvidia-smi reports."""
    ident = {"dlssvid": file_sha(cli)}
    if backend == "ngx":
        for name in NR_DLLS:
            f = next((d / name for d in nvidia_dll_dirs(cli, nr_arg) if (d / name).is_file()), None)
            ident[name] = file_sha(f) if f else None
        ident["gpu"] = nvidia_smi()
    return ident


@dataclass
class Inputs:
    clip: Path                          # dlssvid nr decodes it even when colour comes from color_sr (the decoder drives the run)
    color: Optional[Path] = None        # color_sr; None: NR reads the video
    depth: Optional[Path] = None
    mv: Optional[Path] = None
    masks: Optional[Path] = None        # a passes root with mask_* passes (--masks)
    mv_target: Optional[Path] = None

    def describe(self) -> str:
        return ", ".join(n for n, d in (("color_sr", self.color), ("depth_dlss", self.depth), ("mv_dlss", self.mv)) if d) or "видео"


def frame_index(color: Optional[Path], frame: int) -> int:
    """Index of `frame` for `dlssvid compare` in what NR read: a pass is read from its first frame, the video (no color_sr) from 0."""
    r = pass_range(color) if color else None
    return frame - (r[0] if r else 0)


def own_inputs(inp: Inputs, v: Variant) -> dict:
    """The folders a variant reads instead of the run's (None: "@mv_target" without --mv-target-dir)."""
    return {k: (inp.mv_target if d == "@mv_target" else Path(d)) for k, d in v.inputs.items()}


def variant_dirs(inp: Inputs, v: Variant) -> dict:
    return {"color": inp.color, "depth": inp.depth, "mv": inp.mv, **own_inputs(inp, v)}


def missing_input(inp: Inputs, v: Variant, first: int, last: int, dry: bool) -> str:
    """Why a variant cannot run: a folder it reads instead of the run's is not given or does not cover first..last ('' = it can). Without the
    check E3 C ran with no vectors at all (--no-auto) and its numbers entered the verdict."""
    for k, p in own_inputs(inp, v).items():
        if p is None:
            return "нет --mv-target-dir (mv_dlss размера модели из `dlssvid flow --target`)"
        if not dry and not covers(p, first, last):
            return f"{k}: {p.name} не покрывает кадры {first}..{last}"
    return ""


def nr_command(cli, inp: Inputs, v: Variant, root: Path, params_file: Path, start: int, frames: int, warmup: int, backend: str, fmt: str,
               warp: bool, extra: list) -> list:
    dirs = variant_dirs(inp, v)
    cmd = [str(cli), "nr", "--backend", backend, "-i", str(inp.clip), "-o", str(root), "--no-auto",  # nothing under -o is picked up
           "--start", str(start), "--frames", str(frames), "--warmup", str(min(warmup, start)),  # no history before frame 0
           "--params-file", str(params_file), "--format", fmt, "--json", str(root / "nr.json")]
    for opt, d in (("--color-dir", dirs["color"]), ("--depth-dir", dirs["depth"]), ("--mv-dir", dirs["mv"]), ("--masks-dir", inp.masks)):
        if d:
            cmd += [opt, str(d)]
    return cmd + (["--warp"] if warp else []) + list(extra) + list(v.flags)


def compare_command(cli, ref, test, out_json, *, start: int = 0, offset: int = 0, frames: int = -1) -> list:
    cmd = [str(cli), "compare", "--ref", str(ref), "--test", str(test), "-q", "--json", str(out_json)]
    return cmd + (["--start", str(start)] if start else []) + (["--offset", str(offset)] if offset else []) + (["--frames", str(frames)] if frames > 0 else [])


def read_compare(path: Path, expected: Optional[int] = None) -> dict:
    """The summary of `dlssvid compare --json`; with `expected`, a different frame count is an error (a pair missing or one too many)."""
    s = json.loads(Path(path).read_text(encoding="utf-8"))["summary"]
    if expected is not None and s["frames"] != expected:
        raise RuntimeError(f"{Path(path).name}: compare read {s['frames']} frames, expected {expected}")

    def db(v):
        return IDENTICAL_DB if v is None or v >= IDENTICAL_DB else float(v)  # null = infinite minimum (identical frames)

    return {"frames": s["frames"], "psnr_y": db(s["psnr_y_mean"]), "psnr_y_min": db(s.get("psnr_y_min")), "psnr_rgb": db(s["psnr_rgb_mean"]),
            "ssim_y": s["ssim_y_mean"], "mae_rgb": s.get("mae_rgb_mean")}


def read_nr(path: Path) -> dict:
    """stats + the diagnostics of the feature (TASK-0048) + the run's settings; absent -> None, shown as «—»."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    st, dg = j.get("stats", {}), j.get("diagnostics", {}) or {}
    out = {k: st.get(k) for k in ("ms_per_frame_gpu", "ms_per_frame_wall", "depth_used", "mv_used", "work_size", "passes", "mv_scale",
                                  "frames_written", "warmup_frames", "masks", "bypass")}
    out.update({k: dg.get(k) for k in ("init_ms", "create_ms", "gpu", "driver", "dll_sha256")})
    vram = dg.get("vram_feature_bytes")
    out["vram_mib"] = round(vram / 1048576.0, 1) if isinstance(vram, (int, float)) else None
    out["range"] = j.get("range")
    out["params"] = j.get("params")
    out["ab"] = j.get("ab")
    return out


def unmet(expect: dict, nr: dict) -> list:
    return [f"{k}={nr.get(k)!r}, expected {v!r}" for k, v in expect.items() if nr.get(k) != v]


def run_expect(exp: Experiment, v: Variant) -> dict:
    """What a run must report: the guides the experiment needs were bound (E3 C without its vectors, E4 without any), then the variant's own."""
    return {**{f"{n}_used": True for n in exp.needs if n in ("depth", "mv")}, **v.expect}


def fmt_db(v) -> str:
    if v is None:
        return "—"
    return "∞" if v >= IDENTICAL_DB else f"{v:.2f}"


def fmt_num(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def describe(v: Variant) -> str:
    """The settings column: names only - a variants file may give absolute input folders, the table goes into docs ([hygiene])."""
    parts = [f"{k}={json.dumps(val, ensure_ascii=False) if isinstance(val, str) else val}" for k, val in v.params.items()]
    parts += [" ".join(v.flags)] if v.flags else []
    parts += [f"{k}: {'--mv-target-dir' if d == '@mv_target' else Path(d).name}" for k, d in v.inputs.items()]
    return ", ".join(parts) or "—"


def load_variants(path: Path) -> Experiment:
    """A variants file (docs/benchmarks.md «A/B настроек»): its own experiment, validated before anything runs."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    eid = j.get("id") or Path(path).stem
    if not NAME.match(eid):
        raise RuntimeError(f"{path}: id '{eid}' must match {NAME.pattern}")

    def own(params: dict, where: str) -> dict:
        p = normalize_params(params)
        bad = [k for k in p if k in RUN_KEYS]
        if bad:  # nr_ab gives them as options: an A/B over them would run one setting twice
            raise RuntimeError(f"{path}: {where} sets {', '.join(bad)} - give it to nr_ab as --backend / --format")
        return p

    vs, names, folders = [], set(), {}
    for x in j.get("variants", []):
        name = x.get("name", "")
        if not NAME.match(name) or name.endswith("."):  # Win32 drops a trailing dot: "x." is the folder "x"
            raise RuntimeError(f"{path}: variant name '{name}' must match {NAME.pattern} and not end with '.'")
        if name.casefold() in folders:  # every variant runs in <out>/<run>/<id>/<name>: case-blind on Windows, like the experiment ids
            raise RuntimeError(f"{path}: variant '{name}' and '{folders[name.casefold()]}' are one folder on Windows")
        folders[name.casefold()] = name
        names.add(name)
        flags = check_options(x.get("flags", []), AB_FLAGS, f"{path}: variant '{name}' flags")
        if expected_ab(flags).get("mv_scale") not in (None, "auto"):
            raise RuntimeError(f"{path}: variant '{name}': --mv-scale is unit | auto")
        inputs, expect = dict(x.get("inputs", {})), dict(x.get("expect", {}))
        bad = [k for k in inputs if k not in INPUT_KEYS] + [k for k in expect if k not in EXPECT_KEYS]
        if bad:  # an unknown input would run on the run's folder under another name; an unknown stat could never be met
            raise RuntimeError(f"{path}: variant '{name}': {', '.join(bad)} - inputs are {', '.join(INPUT_KEYS)}, expect checks {', '.join(EXPECT_KEYS)}")
        vs.append(Variant(name, own(x.get("params", {}), f"variant '{name}'"), flags, inputs, expect, bool(x.get("optional", False))))
    if not vs:
        raise RuntimeError(f"{path}: no variants")
    baseline = j.get("baseline") or vs[0].name
    if baseline not in names:
        raise RuntimeError(f"{path}: baseline '{baseline}' is not a variant")
    if next(v for v in vs if v.name == baseline).optional:
        raise RuntimeError(f"{path}: baseline '{baseline}' cannot be optional - every other variant is measured against it")
    metrics = tuple(j.get("metrics", ("base", "time")))
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise RuntimeError(f"{path}: unknown metric {bad} (known: {', '.join(METRICS)})")
    groups = [list(g) for g in j.get("groups", [])]
    for g in groups:
        for n in g:
            if n not in names:
                raise RuntimeError(f"{path}: group member '{n}' is not a variant")
    if bool(groups) != ("pairs" in metrics):
        raise RuntimeError(f"{path}: groups and the \"pairs\" metric go together (pairwise PSNR inside each group)")
    needs = tuple(j.get("needs", ()))
    if [n for n in needs if n not in ("depth", "mv")]:
        raise RuntimeError(f"{path}: needs may list depth and mv only")
    fixed = {**(STILL if j.get("still") else {}), **own(j.get("fixed", {}), "fixed")}
    return Experiment(eid, j.get("title", eid), vs, baseline, fixed, metrics, groups, int(j.get("frames", 12)), needs)


def selected(exp: Experiment, only: list) -> list:
    return [v for v in exp.variants if not only or v.name in only or v.name == exp.baseline]


def check_size(a, inp: Inputs) -> str:
    """E8 / E6f create the feature at --check-size when it is given, else at the size NR runs at: color_sr's, else 1280x720."""
    if a.check_size:
        return a.check_size
    if inp.color:
        try:
            m = json.loads((inp.color / "manifest.json").read_text(encoding="utf-8"))
            return f"{m['width']}x{m['height']}"
        except (OSError, ValueError, KeyError):
            pass
    return "1280x720"


def run_signature(a, inp: Inputs, exp: Experiment, base_params: dict, build: dict) -> dict:
    """Everything a stored experiment's numbers depend on: `common` for every variant (range, inputs by content, the build, the baseline, the
    metrics), `variants` per variant (its settings, flags, own inputs, expectations) - so --mv-target-dir changes E3 C only."""
    common = {"kind": exp.kind, "backend": a.backend, "warp": bool(a.warp), "nr_arg": list(a.nr_arg), "build": build}
    variants = {}
    if exp.kind == "check":
        common.update({"check": exp.check, "size": check_size(a, inp),
                       **({"runs": a.check_runs, "repeat": a.check_repeat} if exp.check == "runs" else {"features": a.check_features})})
    else:
        still = exp.fixed.get("guides") is False
        common.update({"start": a.start, "frames": a.frames or exp.frames, "warmup": 0 if still else min(a.warmup, a.start), "format": a.format,
                       "clip": file_identity(inp.clip), "inputs": {k: pass_identity(d) for k, d in (("color", inp.color), ("depth", inp.depth), ("mv", inp.mv))},
                       "masks": masks_identity(inp.masks), "baseline": exp.baseline, "metrics": sorted(exp.metrics), "groups": exp.groups})
        variants = {v.name: {"params": run_params(base_params, exp.fixed, v.params), "flags": list(v.flags), "expect": run_expect(exp, v),
                             "inputs": {k: pass_identity(p) for k, p in own_inputs(inp, v).items()}, "optional": v.optional} for v in exp.variants}
    return json.loads(json.dumps({"common": common, "variants": variants}, sort_keys=True))  # the shape a results.json round trip gives


def valid_stored(prev: Optional[dict], sig: dict, baseline: str) -> set:
    """Stored variants (measured or failed) still true under `sig`: the same common part, an unchanged baseline (every other variant is measured
    against it), an unchanged entry of their own. Empty for a result stored before signatures."""
    ps = (prev or {}).get("sig") or {}
    if ps.get("common") is None or ps.get("common") != sig["common"]:
        return set()
    pv, sv = ps.get("variants", {}), sig["variants"]
    if baseline and pv.get(baseline) != sv.get(baseline):
        return set()
    return {n for n in set(prev.get("variants", {})) | set(prev.get("failed", {})) if n in sv and pv.get(n) == sv[n]}


def reusable(prev: Optional[dict], sig: dict, want: list, force: bool, baseline: str = "") -> bool:
    """A stored experiment answers this invocation: every stored variant still true, every wanted one among them, no pair left unmeasured."""
    if force or not prev or prev.get("pairs_missing") or prev.get("measured") is False or (prev.get("sig") or {}).get("common") != sig["common"]:
        return False  # an unmeasured check (a timeout, a crash, no JSON) is measured again
    stored = set(prev.get("variants", {})) | set(prev.get("failed", {}))
    return set(want) <= stored and valid_stored(prev, sig, baseline) == stored


# ---- verdicts (pure; None = no data) ---------------------------------------------------------------------------------------

def _get(res, name, *path):
    x = res.get("variants", {}).get(name, {})
    for p in path:
        x = x.get(p) if isinstance(x, dict) else None
    return x


def static_input(res):
    """E4 measures the breathing on identical frames (spec §2e, «дыхание» below in docs/benchmarks.md): the input's own inter-frame PSNR - the
    minimum, every pair, not the mean that a few identical pairs at 100 dB would lift - says whether the clip is one."""
    t = (res.get("input_temporal") or {}).get("psnr_y_min")
    return t, t is not None and t > SAME_DB


def _not_static(t) -> str:
    return f"вход не статичен (межкадровый PSNR входа, min {fmt_db(t)} дБ): «дыхание» меряется на одинаковых кадрах (§2e) — вывода нет"


def verdict_e1(res):
    """PR-7: out = in + x * (model - in), the model run at 1. Up to 1 the edit (MAE to the input) is x times the edit at 1 - the lerp stays
    inside [input, model], nothing clips; above 1 saturate clips the extrapolation, and the ratio's shortfall from x is the clipped share.
    The GPU time at x > 0 is listed, not judged: every run does the same model work plus one lerp, and one run's mean over a few frames
    spreads by x1.4-x4.8 between identical runs (PR-2 E1 / E7) - no threshold separates a lerp from that noise."""
    e1 = _get(res, "i1", "edit", "mae_rgb")
    low = {x: _get(res, f"i{x:g}", "edit", "mae_rgb") for x in (0.25, 0.5, 0.75)}
    worst = None if not e1 or None in low.values() else max(abs(m - x * e1) / e1 for x, m in low.items())
    linear = None if worst is None else worst <= LINEAR_TOL
    ratios = {f"{x:g}": (None if not e1 or _get(res, f"i{x:g}", "edit", "mae_rgb") is None else _get(res, f"i{x:g}", "edit", "mae_rgb") / e1)
              for x in (1.25, 1.5, 2)}
    edits = [_get(res, f"i{x:g}", "edit", "mae_rgb") for x in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)]
    known = [m for m in edits if m is not None]
    monotone = None if len(known) < 2 else all(b > a for a, b in zip(known, known[1:]))
    ms = [t for t in (_get(res, f"i{x:g}", "nr", "ms_per_frame_gpu") for x in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)) if isinstance(t, (int, float)) and t > 0]
    text = []
    if linear is not None:
        text.append(f"смешивание линейно на [0, 1]: худшее отклонение правки от x·правка(1) — {worst * 100:.1f}% (допуск {LINEAR_TOL * 100:g}%)" if linear
                    else f"правка на [0, 1] не пропорциональна силе: отклонение {worst * 100:.1f}% > {LINEAR_TOL * 100:g}% — регрессия NrCompose (PR-7)")
    over = [(x, r) for x, r in ratios.items() if r is not None]
    if over:
        text.append("выше 1 (экстраполяция): " + ", ".join(f"правка({x})/правка(1) = {r:.2f} при {x}, обрезано {max(0.0, 1 - r / float(x)) * 100:.0f}%"
                                                         for x, r in over))
    if monotone is False:
        text.append("правка растёт с силой не монотонно — регрессия NrCompose (PR-7)")
    if len(ms) >= 2:
        text.append(f"время GPU при x > 0: {fmt_num(min(ms))}–{fmt_num(max(ms))} мс/кадр — одна и та же работа модели, разброс — шум прогонов")
    return {"linear": linear, "worst": worst, "ratios": ratios, "monotone": monotone, "text": text}


def verdict_e2(res):
    """PR-7: at 0 the model is not called (no feature, no DLL, no guides) and the output is the input: identity, bypass and the GPU time saved."""
    db = _get(res, "i0", "edit", "psnr_y")
    identity = None if db is None else db >= SAME_DB
    bypass = _get(res, "i0", "nr", "bypass")
    t0, t1 = _get(res, "i0", "nr", "ms_per_frame_gpu"), _get(res, "i1", "nr", "ms_per_frame_gpu")
    text = []
    if identity is not None:
        text.append(f"тождество: PSNR ко входу {fmt_db(db)} дБ" if identity
                    else f"не тождество (PSNR ко входу {fmt_db(db)} дБ) — регрессия обхода модели при 0 (PR-7)")
    if bypass is not True and identity is not None:
        text.append("прогон при 0 не сообщил обход модели (stats.bypass) — сборка до PR-7?")
    if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
        text.append(f"GPU {fmt_num(t0)} мс/кадр при 0 против {fmt_num(t1)} при 1")
    return {"identity": identity, "bypass": bypass, "ms_gpu": [t0, t1], "text": text}


def verdict_e3(res):
    """No decision yet: on identical frames (the breathing method) the vectors are zero and unit / auto / C hand the model the same 0 x scale,
    on moving content the unaligned inter-frame PSNR measures the motion. The units need a PSNR compensated by mv on a pan (§2e, «Пока нет»)."""
    a, b, c = (_get(res, n, "temporal", "psnr_y") for n in ("unit", "auto", "target"))
    sa, sb = _get(res, "unit", "nr", "mv_scale"), _get(res, "auto", "nr", "mv_scale")
    text = []
    if sa is not None and sb is not None and sa == sb:
        text.append("масштаб векторов у unit и auto одинаковый (размер mv = размер модели): E3 на этом входе ничего не меряет")
    if c is None:
        text.append("нет варианта C (--mv-target-dir: mv_dlss размера модели из `dlssvid flow --target`)")
    if a is not None and b is not None:
        text.append("вывода нет: на одинаковых кадрах векторы нулевые и unit / auto / C дают модели одно и то же, на движении межкадровый PSNR "
                    "без компенсации меряет движение — нужен PSNR с компенсацией по mv (§2e); пока только таблица")
    return {"auto_better": None, "text": text}


def verdict_e4(res):
    a, b = _get(res, "null", "temporal", "psnr_y"), _get(res, "create_mv", "temporal", "psnr_y")
    if None in (a, b):
        return {"create_better": None, "text": []}
    t, static = static_input(res)
    if not static:
        return {"create_better": None, "text": [_not_static(t)]}
    better = b - a >= GAIN_DB
    return {"create_better": better,
            "text": [f"межкадровый PSNR {fmt_db(a)} → {fmt_db(b)} дБ: " + ("создание с нулевыми векторами по умолчанию" if better else "разницы на порог нет")]}


def verdict_e5(res):
    db = _get(res, "no_depth", "vs_base", "psnr_y")
    same = None if db is None else db > SAME_DB
    text = [] if same is None else [f"без глубины к базе {fmt_db(db)} дБ: " + ("разницы нет — глубину оставляем (дешевле не станет)" if same else "глубина меняет картинку")]
    return {"no_difference": same, "text": text}


def verdict_e6(res):
    p1, p2 = _get(res, "p1", "nr", "ms_per_frame_gpu"), _get(res, "p2", "nr", "ms_per_frame_gpu")
    step = None if None in (p1, p2) else p2 - p1
    return {"ms_per_pass": step, "text": [] if step is None else [f"GPU на добавленный проход: {step:.1f} мс/кадр (константа оценок — 6.4 мс на 1440p)"]}


def verdict_e7(res):
    """Over the pairs the groups call for: one without a number (a variant not run under --only, a pair without a color_nr) leaves it partial."""
    pairs = res.get("pairs", [])
    want = {tuple(sorted(c)) for g in res.get("groups", []) for c in itertools.combinations(g, 2)}
    unmeasured = len(want - {tuple(sorted((p["a"], p["b"]))) for p in pairs}) if want else len(res.get("pairs_missing") or [])
    same = [f"{p['a']} ≈ {p['b']} ({fmt_db(p['psnr_y'])} дБ)" for p in pairs if p.get("psnr_y") is not None and p["psnr_y"] > SAME_DB]
    text = ["совпадают: " + "; ".join(same)] if same else (["все пары различаются"] if pairs and not unmeasured else [])
    if unmeasured:
        text.append(f"пар без замера: {unmeasured} — вывод неполный")
    return {"same_pairs": len(same), "missing_pairs": unmeasured, "text": text}


def phase2(rep: Optional[dict]):
    """The preview's phase 2 condition from N re-creations in one process (None: not measured)."""
    if not rep:
        return None, []
    if rep.get("error"):
        return None, [f"пересоздания не измерены: {rep['error']} — условие фазы 2 превью этим не проверено"]
    if rep.get("stop") in ("budget", "growth"):
        return False, [f"пересоздания остановлены ({'бюджет VRAM' if rep['stop'] == 'budget' else 'память растёт больше чем на фичу'}) после "
                       f"{rep.get('ok')} из {rep.get('requested')} — условие фазы 2 превью не выполнено"]
    creates = [x for x in rep.get("create_ms", []) if x is not None]
    released = [x for x in rep.get("vram_after_release_mib", []) if x is not None]
    made = [x for x in rep.get("vram_after_create_mib", []) if x is not None]

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    first, last = med(creates[:10]), med(creates[-10:])
    spread = max((abs(x - released[0]) for x in released), default=None) if released else None
    drift = made[-1] - rep["usage_after_init_mib"] if made and rep.get("usage_after_init_mib") is not None else None
    checks = [rep.get("ok") == rep.get("requested") and rep.get("requested", 0) > 0,
              first is not None and last is not None and last <= PHASE2_SLOWDOWN * first,
              spread is not None and spread <= PHASE2_RELEASE_MIB, drift is not None and drift <= PHASE2_DRIFT_MIB]
    text = [f"пересозданий в одном процессе: {rep.get('ok')} из {rep.get('requested')}; create, медиана первых 10 / последних 10: "
            f"{fmt_num(first, 0)} / {fmt_num(last, 0)} мс; после освобождения VRAM в пределах {fmt_num(spread)} МиБ; дрейф к концу "
            f"{fmt_num(drift)} МиБ"]
    met = all(checks)
    text.append("условие фазы 2 превью выполнено" if met else
                f"условие фазы 2 превью не выполнено (пороги: все ok, замедление ≤ {PHASE2_SLOWDOWN:g}×, VRAM ≤ {PHASE2_RELEASE_MIB:g} МиБ, "
                f"дрейф ≤ {PHASE2_DRIFT_MIB:g} МиБ)")
    return met, text


def verdict_e8(res):
    runs = [r for r in res.get("checks", []) if r.get("ok")]

    def med(k):
        xs = sorted(r[k] for r in runs if r.get(k) is not None)
        return xs[len(xs) // 2] if xs else None

    text = [f"медианы: init {fmt_num(med('init_ms'), 0)} мс, create {fmt_num(med('create_ms'), 0)} мс, фича {fmt_num(med('vram_mib'))} МиБ, процесс {fmt_num(med('run_s'), 2)} с"]
    failed = [i for i, r in enumerate(res.get("checks", []), 1) if not r.get("ok")]  # numbered as the table's «Прогон»
    text.append("все проверки ok" if not failed else f"сбой в проверках {', '.join(map(str, failed))}")
    met, more = phase2(res.get("repeat"))
    text += more or ["пересоздания в одном процессе не измерены (--check-repeat 0): условие фазы 2 превью этим не проверено"]
    return {"init_ms": med("init_ms"), "create_ms": med("create_ms"), "phase2": met, "text": text}


def verdict_e6f(res):
    feats = res.get("features", [])
    if not res.get("wrote_json", True):
        return {"alive": None, "mib_per_feature": None,
                "text": [f"не измерено на {res.get('size')}: dlssvid не записал JSON (тайм-аут, сбой или опция не поддерживается) — см. features.log"]}
    if not feats or not feats[0].get("ok"):
        return {"alive": 0, "mib_per_feature": None, "text": [f"первая фича на {res.get('size')} не создалась — вывода нет (см. features.log)"]}
    alive = res.get("alive", 0)
    sizes = sorted(f["vram_mib"] for f in feats if f.get("ok") and f.get("vram_mib") is not None)
    per = sizes[len(sizes) // 2] if sizes else None
    stop = {"budget": "остановлено у 80 % бюджета DXGI", "failure": "следующая фича не создалась — наблюдаемый предел"}.get(res.get("stop") or "",
                                                                                                                          "все запрошенные созданы")
    return {"alive": alive, "mib_per_feature": per,
            "text": [f"одновременно {alive} фич на {res.get('size')}: VRAM на фичу (медиана) {fmt_num(per)} МиБ, процесс {fmt_num(res.get('usage_mib'))} "
                     f"из {fmt_num(res.get('budget_mib'))} МиБ бюджета; {stop}"]}


def verdict_e11(res):
    """Without the auto mask s0 / s1 / s2 against follow; the control with the mask (am_s0, am_s2 against am_follow) says whether the key acts
    on this clip at all - without an effect there «no effect without the mask» proves nothing."""
    dbs = [_get(res, n, "vs_base", "psnr_y") for n in ("s0", "s1", "s2")]
    none = None if None in dbs else all(x > SAME_DB for x in dbs)
    ctrl = [p["psnr_y"] for p in res.get("pairs", []) if p.get("a") == "am_follow" and p.get("psnr_y") is not None]
    acts = None if not ctrl else any(x <= SAME_DB for x in ctrl)
    if none is None:
        return {"no_effect": None, "control_acts": acts, "text": []}
    if none and acts is False:
        return {"no_effect": None, "control_acts": False,
                "text": [f"с автомаской эффекта тоже нет (контроль {', '.join(fmt_db(x) for x in ctrl)} дБ) — кожа не найдена или эффект не виден: вывода нет"]}
    if none:
        return {"no_effect": True, "control_acts": acts,
                "text": ["без автомаски эффекта нет" + (", с автомаской есть" if acts else " (контроля нет)") + " — условие доступности остаётся"]}
    return {"no_effect": False, "control_acts": acts, "text": ["структура кожи действует и без автомаски — условие снять"]}


VERDICTS = {"E1": verdict_e1, "E2": verdict_e2, "E3": verdict_e3, "E4": verdict_e4, "E5": verdict_e5, "E6": verdict_e6, "E6f": verdict_e6f, "E7": verdict_e7,
            "E8": verdict_e8, "E11": verdict_e11}


# ---- running ----------------------------------------------------------------------------------------------------------------

def find_cli(given: Optional[str]) -> Path:
    for c in (given, os.environ.get("DLSSVID_CLI"), ROOT / "build" / "release" / "bin" / "dlssvid.exe"):
        if c and Path(c).exists():
            return Path(c)
    raise RuntimeError("dlssvid not found: give --cli or DLSSVID_CLI, or build the release preset")


def default_out() -> Path:
    workspace = ROOT.parents[1] / "Output" / "results"
    return workspace / "nr-ab" if workspace.is_dir() else ROOT / "build" / "release" / "nr-ab"


def run(cmd: list, log: Optional[Path], dry: bool) -> float:
    if dry:
        print(subprocess.list2cmdline([str(c) for c in cmd]))
        return 0.0
    args = [str(c) for c in cmd]
    t0 = time.perf_counter()
    # the output into a file, not pipes: a grandchild of dlssvid (the ONNX export of --prepare, a worker) holding inherited handles
    # cannot block the wait; on a timeout the whole tree is killed while dlssvid still lives (taskkill /T finds it by the parent)
    with tempfile.TemporaryFile() as buf:
        p = subprocess.Popen(args, stdout=buf, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, timeout=60)
            p.kill()
            p.wait()
            rc = None
        buf.seek(0)
        out = buf.read().decode("utf-8", "replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(subprocess.list2cmdline(args) + "\n\n" + out + ("" if rc is not None else f"\n[nr_ab: killed after {TIMEOUT:g} s]\n"), encoding="utf-8")
    if rc is None:
        raise RuntimeError(f"{Path(args[0]).name} {args[1]} timed out after {TIMEOUT:g} s (killed with its children)")
    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        raise RuntimeError(f"{Path(args[0]).name} {args[1]} failed ({rc}):\n{tail}")
    return time.perf_counter() - t0


def clear(folder: Path):
    """Frames of an earlier, longer run must not enter the metrics: a file left behind (open in a viewer, an AV scan) stops the experiment."""
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        raise RuntimeError(f"{folder}: could not remove the frames of an earlier run (a file is open?)")


def measure(cli, ref, test, out_json: Path, expected: int, dry: bool, **kw) -> Optional[dict]:
    """One `dlssvid compare` bounded to the frames the run wrote, its count checked; the JSON of an earlier run is removed first, so a compare
    that fails before writing cannot report it."""
    if not dry:
        out_json.unlink(missing_ok=True)
    run(compare_command(cli, ref, test, out_json, frames=expected, **kw), None, dry)
    return None if dry else read_compare(out_json, expected)


def prepare_root(out: Path, a, clip: Path) -> Path:
    """One folder per clip and backends: other --prepare-backends make other passes, never over the stored experiments' inputs; two clips of
    one name (another folder) get their own by a hash of the full path."""
    tag = hashlib.sha256(str(Path(clip).resolve()).lower().encode("utf-8")).hexdigest()[:8]
    backends = "-".join(re.sub(r"[^A-Za-z0-9._-]", "_", x) for x in a.prepare_backends.split(",") if x)  # worker:vda -> worker_vda (no ':' on Windows)
    return out / "prepare" / f"{Path(clip).stem}-{tag}-{backends}"


def resolve_inputs(a, out: Path, cli: Path, dry: bool) -> Inputs:
    clip, passes = (Path(a.clip) if a.clip else None), (Path(a.passes) if a.passes else None)
    if a.project:
        p = Path(a.project)
        proj = json.loads(p.read_text(encoding="utf-8"))
        src = proj.get("source") or proj.get("source_video") or ""
        clip = clip or (p.parent / src if src else None)
        root = proj.get("passes") or proj.get("passes_root") or ""
        passes = passes or (p.parent / root if root else None)
    if not clip:
        raise RuntimeError("give --clip or --project")
    if a.masks and (a.prepare or not passes or not any(passes.glob("mask_*/manifest.json"))):
        # the header would say «маски» over runs that had none: --masks-dir is the passes root, --prepare makes no masks
        raise RuntimeError("--masks needs mask_* passes under --passes / the project's passes root (--prepare makes none)")
    given = [o for o, v in (("--passes", a.passes), ("--color-dir", a.color_dir), ("--depth-dir", a.depth_dir), ("--mv-dir", a.mv_dir)) if v]
    if a.prepare and given:  # --prepare reads the passes it makes: a folder given beside it would be dropped without a word
        raise RuntimeError(f"--prepare makes color_sr / depth_dlss / mv_dlss itself: drop {', '.join(given)} or --prepare")
    if a.prepare:
        passes = prepare_root(out, a, clip) / "passes"
    pick = lambda name: (passes / name) if passes and (passes / name / "manifest.json").exists() else None
    inp = Inputs(clip, Path(a.color_dir) if a.color_dir else pick("color_sr"), Path(a.depth_dir) if a.depth_dir else pick("depth_dlss"),
                 Path(a.mv_dir) if a.mv_dir else pick("mv_dlss"), (passes if a.masks else None), Path(a.mv_target_dir) if a.mv_target_dir else None)
    if a.prepare:  # the planned folders, also in a dry run
        inp.color, inp.depth, inp.mv = passes / "color_sr", passes / "depth_dlss", passes / "mv_dlss"
    return inp


def prepare(a, cli: Path, inp: Inputs, out: Path, last: int, dry: bool) -> bool:
    """color_sr / depth_dlss / mv_dlss for frames 0..last with `dlssvid process` (it always encodes: the video is deleted after) - only when they
    do not cover the range or were made from other content of the clip (re-encoded in place: its stamp differs). Rewriting them changes the
    inputs of every stored experiment, which their signatures see. True: the passes are (dry: would be) rewritten."""
    work = prepare_root(out, a, inp.clip)
    stamp = work / "clip.json"
    try:
        same = json.loads(stamp.read_text(encoding="utf-8")) == file_identity(inp.clip)
    except (OSError, ValueError):
        same = False
    if same and all(covers(d, 0, last) for d in (inp.color, inp.depth, inp.mv)):
        return False
    d, f, u = (a.prepare_backends.split(",") + ["", "", ""])[:3]
    cmd = [cli, "process", "-i", inp.clip, "-o", work / "prepare.mkv", "--passes", work / "passes", "--stages", "depth,flow,upscale", "--depth-backend", d,
           "--flow-backend", f, "--upscale-backend", u, "--frames", str(last + 1), "--codec", "ffv1"] + (["--warp"] if a.warp else [])
    if not dry:
        stamp.unlink(missing_ok=True)  # written again only after a whole process
    run(cmd, work / "prepare.log", dry)
    if not dry:
        (work / "prepare.mkv").unlink(missing_ok=True)
        stamp.write_text(json.dumps(file_identity(inp.clip)), encoding="utf-8")
    return True


def _ident(entry) -> tuple:
    return tuple(((entry or {}).get("nr") or {}).get(k) for k in ("gpu", "driver", "dll_sha256"))


def _same_build(entry, now: tuple, baseline) -> bool:
    """A stored variant kept by an --only merge ran on the baseline's GPU / driver / DLL. A bypass run (intensity 0, PR-7) calls no model
    and reports none of them - nothing to compare, for it or against a bypass baseline; the build itself is in the signature."""
    bypass = lambda e: ((e or {}).get("nr") or {}).get("bypass") is True
    return bypass(entry) or bypass(baseline) or _ident(entry) == now


def run_experiment(a, cli: Path, inp: Inputs, exp: Experiment, base_params: dict, out: Path, dry: bool, prev: Optional[dict] = None,
                   sig: Optional[dict] = None) -> dict:
    """`prev` / `sig`: the stored result and this run's signature - the stored variants still true under it (valid_stored) and not run now stay
    (--only adds to a result), if the baseline ran on the same GPU / driver / DLL."""
    edir = out / exp.id
    if exp.kind == "check":
        return run_check(a, cli, inp, exp, edir, dry)
    frames = a.frames or exp.frames
    still = exp.fixed.get("guides") is False
    start = a.start
    warmup = 0 if still else min(a.warmup, start)  # still mode resets every frame: a warm-up only burns GPU
    last = start + frames - 1
    variants = selected(exp, a.only)
    ran = {v.name for v in variants}
    res = {"title": exp.title, "start": start, "frames": frames, "warmup": warmup, "still": still, "fixed": exp.fixed, "baseline": exp.baseline,
           "metrics": list(exp.metrics), "variants": {}, "failed": {}}
    todo = []
    for v in variants:  # every input before the first run: a later variant's missing folder would stop the experiment after the GPU ran the others
        dirs = variant_dirs(inp, v)
        why = missing_input(inp, v, start - warmup, last, dry)
        if not why and not dry:
            short = [n for n in exp.needs if not covers(dirs[n], start - warmup, last)] + (["color"] if dirs["color"] and not covers(dirs["color"], start - warmup, last) else [])
            why = f"{', '.join(short)}: нет кадров {start - warmup}..{last} (--passes / --<вход>-dir или --prepare)" if short else ""
        if why and (not v.optional or v.name == exp.baseline):
            raise RuntimeError(f"{exp.id}/{v.name}: {why}")
        if why:
            res["failed"][v.name] = why
        else:
            todo.append(v)
    for v in todo:
        vdir = edir / v.name
        params = run_params(base_params, exp.fixed, v.params)
        if dry:
            print(f"# {exp.id}/{v.name} params.json: {json.dumps(params, sort_keys=True)}")
        else:
            clear(vdir / "color_nr")
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "nr.json").unlink(missing_ok=True)
            (vdir / "params.json").write_text(json.dumps(params, sort_keys=True, indent=1), encoding="utf-8")
        try:
            run_s = run(nr_command(cli, inp, v, vdir, vdir / "params.json", start, frames, warmup, a.backend, a.format, a.warp, a.nr_arg), vdir / "nr.log", dry)
            if dry:
                res["variants"][v.name] = {}
                continue
            r = pass_range(vdir / "color_nr")
            if not r or r[0] != start or r[1] != last or r[2] != frames:
                raise RuntimeError(f"{exp.id}/{v.name}: dlssvid nr wrote frames {r and r[0]}..{r and r[1]}, expected {start}..{last}")
            nr = read_nr(vdir / "nr.json")
            bad = unmet(run_expect(exp, v), nr)
            if bad:
                raise RuntimeError(f"{exp.id}/{v.name}: {'; '.join(bad)} - the flag or the input did nothing")
            lost = lost_params(params, nr.get("params") or {})
            if lost:
                raise RuntimeError(f"{exp.id}/{v.name}: {'; '.join(lost)} - the setting did not reach the run")
            if (nr.get("ab") or {}) != expected_ab(v.flags):
                raise RuntimeError(f"{exp.id}/{v.name}: the run's A/B switches {nr.get('ab')}, expected {expected_ab(v.flags)}")
            res["variants"][v.name] = {"nr": nr, "run_s": round(run_s, 2), "params": params, "flags": v.flags, "inputs": v.inputs, "range": [r[0], r[1], r[2]]}
        except RuntimeError as e:
            if not v.optional or v.name == exp.baseline:
                raise  # every other metric is measured against the baseline
            res["failed"][v.name] = names_only(str(e).splitlines()[-1])
    base = edir / exp.baseline / "color_nr"
    for v in todo:
        if v.name not in res["variants"]:
            continue
        vdir, entry, own = edir / v.name, res["variants"][v.name], variant_dirs(inp, v)["color"]
        if "base" in exp.metrics and v.name != exp.baseline:
            entry["vs_base"] = measure(cli, base, vdir / "color_nr", vdir / "cmp_base.json", frames, dry)
        if "edit" in exp.metrics:  # against what this variant read: its own color_sr, the run's, else the video
            entry["edit"] = measure(cli, vdir / "color_nr", own or inp.clip, vdir / "cmp_input.json", frames, dry, offset=frame_index(own, start))
        if "temporal" in exp.metrics and frames > 1:
            entry["temporal"] = measure(cli, vdir / "color_nr", vdir / "color_nr", vdir / "cmp_next.json", frames - 1, dry, offset=1)
    if "temporal" in exp.metrics and frames > 1:  # the «вход» row: what the baseline read (a variants file may give it its own color_sr)
        own = variant_dirs(inp, next(v for v in exp.variants if v.name == exp.baseline))["color"]
        res["input_temporal"] = measure(cli, own or inp.clip, own or inp.clip, edir / "cmp_input_next.json", frames - 1, dry,
                                        start=frame_index(own, start), offset=1)
    valid = valid_stored(prev, sig, exp.baseline) if prev and sig else set()
    if valid:  # --only over a stored result: the variants not run now stay while true under this run's signature and of the same build
        base = (prev if dry else res)["variants"].get(exp.baseline)  # a dry run has no new baseline: it shows what the real one keeps
        now = _ident(base)
        merged = {**{n: e for n, e in prev.get("variants", {}).items() if n in valid and n not in ran and _same_build(e, now, base)}, **res["variants"]}
        res["variants"] = {v.name: merged[v.name] for v in exp.variants if v.name in merged}
        res["failed"] = {**{n: w for n, w in prev.get("failed", {}).items() if n in valid and n not in ran}, **res["failed"]}
    if "pairs" in exp.metrics:
        res["groups"] = [list(g) for g in exp.groups]  # verdict_e7 counts the pairs they call for
        res["pairs"], res["pairs_missing"] = [], []
        # stored pairs survive an --only merge only: not when one of theirs is named again (the baseline re-runs with the same entry,
        # valid_stored); a run without --only (--force included) ran every variant again and measures every pair
        again = set(a.only) if a.only else ran
        old = {(p["a"], p["b"]): p for p in (prev or {}).get("pairs", []) if p["a"] in valid and p["b"] in valid and not {p["a"], p["b"]} & again}

        def have(n):  # a color_nr of the frames the entry describes: one left by a later, interrupted run of other settings is not
            if n in ran:
                return dry or pass_range(edir / n / "color_nr") == (start, last, frames)
            if pass_range(edir / n / "color_nr") != (start, last, frames):
                return False
            try:  # nr.json is removed before every run and written on success only: equal to the stored one = the same run
                return read_nr(edir / n / "nr.json") == (res["variants"][n].get("nr") or {})
            except (OSError, ValueError):
                return False

        for g in exp.groups:
            for x, y in itertools.combinations([n for n in g if n in res["variants"]], 2):
                if (x, y) in old:
                    res["pairs"].append(old[(x, y)])
                elif not (have(x) and have(y)):
                    res["pairs_missing"].append([x, y])  # a stored variant's color_nr is gone (--keep keeps it)
                else:
                    c = measure(cli, edir / x / "color_nr", edir / y / "color_nr", edir / f"pair_{x}__{y}.json", frames, dry)
                    if c:
                        res["pairs"].append({"a": x, "b": y, "psnr_y": c["psnr_y"], "ssim_y": c["ssim_y"]})
    if not dry:
        # an own experiment with pairs gets E7's verdict (which pairs match, whether all were measured)
        res["verdict"] = (VERDICTS.get(exp.id) or (verdict_e7 if "pairs" in exp.metrics else lambda _r: {"text": []}))(res)
        if not a.keep:
            for v in todo:
                shutil.rmtree(edir / v.name / "color_nr", ignore_errors=True)
    return res


def check_once(a, cli, size: str, f: Path, log: Path, dry: bool, extra: list) -> dict:
    """One `dlssvid nr --check`; its JSON ({} when it wrote none), "_ok" and "_run_s" beside."""
    if not dry:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.unlink(missing_ok=True)  # a --check that fails before writing its JSON must not show an earlier run's numbers
    cmd = [cli, "nr", "--check", "--backend", a.backend, "--check-size", size, "--json", f] + extra + (["--warp"] if a.warp else []) + list(a.nr_arg)
    try:
        run_s, ok = run(cmd, log, dry), True
    except RuntimeError:
        run_s, ok = None, False  # a failed --check still writes its JSON (exit 1)
    if dry:
        return {}
    j = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return {**j, "_ok": ok and bool(j.get("ok")), "_run_s": None if run_s is None else round(run_s, 2)}


def mib(v):
    return round(v / 1048576.0, 1) if isinstance(v, (int, float)) else None


def run_check(a, cli: Path, inp: Inputs, exp: Experiment, edir: Path, dry: bool) -> dict:
    size = check_size(a, inp)
    res = {"title": exp.title, "size": size}
    if exp.check == "features":
        j = check_once(a, cli, size, edir / "features.json", edir / "features.log", dry, ["--check-features", str(a.check_features)])
        if not dry:
            wrote = any(not k.startswith("_") for k in j)  # dlssvid writes its JSON only at the end: none after a timeout, a crash, an unknown option
            res.update({"wrote_json": wrote, "measured": bool(j.get("features")), "requested": a.check_features, "alive": j.get("features_alive", 0),
                        "stop": j.get("features_stop"),
                        "budget_mib": mib(j.get("vram_budget_bytes")), "usage_mib": mib((j.get("features") or [{}])[-1].get("vram_usage_bytes")),
                        "features": [{"index": x.get("index"), "ok": x.get("ok"), "init_ms": x.get("init_ms"), "create_ms": x.get("create_ms"),
                                      "vram_mib": mib(x.get("vram_feature_bytes")), "usage_mib": mib(x.get("vram_usage_bytes"))} for x in j.get("features") or []]})
            res["diag"] = {k: j.get(k) for k in ("gpu", "driver", "dll_sha256")}
            res["verdict"] = verdict_e6f(res)
        return res
    res["checks"] = []
    for i in range(a.check_runs):
        f = edir / f"check_{i}.json"
        if not dry:
            edir.mkdir(parents=True, exist_ok=True)
            f.unlink(missing_ok=True)  # a --check that fails before writing its JSON must not show an earlier run's numbers
        cmd = [cli, "nr", "--check", "--backend", a.backend, "--check-size", size, "--json", f] + (["--warp"] if a.warp else []) + list(a.nr_arg)
        try:
            run_s = run(cmd, edir / f"check_{i}.log", dry)
            ok = True
        except RuntimeError:
            run_s, ok = None, False  # a failed --check still writes its JSON (exit 1)
        if dry:
            continue
        j = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if j.get("gpu") and "diag" not in res:
            res["diag"] = {k: j.get(k) for k in ("gpu", "driver", "dll_sha256")}  # the report's build line (experiment_meta)
        vram = j.get("vram_feature_bytes")
        res["checks"].append({"ok": ok and bool(j.get("ok")), "json": bool(j), "init_ms": j.get("init_ms"), "create_ms": j.get("create_ms"), "device_ms": j.get("device_ms"),
                              "vram_mib": round(vram / 1048576.0, 1) if isinstance(vram, (int, float)) else None, "run_s": None if run_s is None else round(run_s, 2)})
    if a.check_repeat > 0:  # the re-creations in one process: the preview's phase 2 condition (spec §2f)
        j = check_once(a, cli, size, edir / "repeat.json", edir / "repeat.log", dry, ["--check-repeat", str(a.check_repeat)])
        if not dry and "recreations" not in j:  # no JSON: a CLI error, a timeout kill or a crash - not a measurement
            res["repeat"] = {"requested": a.check_repeat, "error": "dlssvid не записал JSON (ошибка CLI, тайм-аут или сбой) — см. repeat.log",
                             "run_s": j.get("_run_s")}
        elif not dry:
            recs = j.get("recreations") or []
            res["repeat"] = {"requested": a.check_repeat, "ok": sum(1 for r in recs if r.get("ok")), "usage_after_init_mib": mib(j.get("vram_usage_after_init_bytes")),
                             "create_ms": [r.get("create_ms") for r in recs], "vram_after_release_mib": [mib(r.get("vram_after_release_bytes")) for r in recs],
                             "vram_after_create_mib": [mib(r.get("vram_after_create_bytes")) for r in recs], "stop": j.get("recreations_stop") or "",
                             "failed": next((r.get("result") for r in recs if not r.get("ok")), None), "run_s": j.get("_run_s")}
    if not dry:
        # a failed check is an answer; one without its JSON (a timeout, a crash) is not
        res["measured"] = all(c.get("json") for c in res["checks"]) and not (res.get("repeat") or {}).get("error")
        res["verdict"] = verdict_e8(res)
    return res


def experiment_meta(res: dict, version: str, build: dict, inp: Inputs, exp: Experiment) -> dict:
    """The build and the inputs an experiment was measured with - stored with it: an experiment reused from results.json keeps its own. The
    inputs are the baseline's, and the variants with folders of their own are named."""
    # the baseline's run first, then the first that reported its GPU: i0 calls no model and has no diagnostics (PR-7)
    variants = res.get("variants", {})
    runs = [x["nr"] for x in ([variants[exp.baseline]] if exp.baseline in variants else []) + list(variants.values()) if x.get("nr")]
    first = next((n for n in runs if n.get("gpu")), runs[0] if runs else None) or res.get("diag") or {}
    meta = {"version": version, "exe_sha256": build.get("dlssvid"), "gpu": first.get("gpu"), "driver": first.get("driver"), "dll_sha256": first.get("dll_sha256")}
    if exp.kind != "check":  # E8 and E6f read no passes
        used = sorted({m for x in res.get("variants", {}).values() for m in ((x.get("nr") or {}).get("masks") or [])})
        d = variant_dirs(inp, next(v for v in exp.variants if v.name == exp.baseline))
        own = [v.name for v in exp.variants if v.inputs and v.name in res.get("variants", {})]
        inputs = Inputs(inp.clip, d["color"], d["depth"], d["mv"]).describe() + (f"; свои входы у {', '.join(own)}" if own else "")
        meta.update({"inputs": inputs, "masks": ", ".join(used)})
    return meta


# ---- report ----------------------------------------------------------------------------------------------------------------

BUILD = ("version", "exe_sha256", "gpu", "driver", "dll_sha256")


def build_line(meta: dict) -> str:
    sha, exe = (meta.get("dll_sha256") or "")[:8], (meta.get("exe_sha256") or "")[:8]
    s = f"dlssvid {meta.get('version') or '?'}" + (f" (exe {exe}…)" if exe else "") + (f", {meta['gpu']}" if meta.get("gpu") else "")
    s += f", драйвер {meta['driver']}" if meta.get("driver") else ""
    return s + (f", nvngx_dlssnr.dll sha256 {sha}…" if sha else "")


def inputs_line(meta: dict) -> str:
    return f"Вход NR: {meta.get('inputs') or '—'}; маски: {meta.get('masks') or 'выкл'}."


def render_experiment(exp_id: str, res: dict, own: tuple = ()) -> str:
    """`own`: lines of the experiment's build / inputs when the experiments of the run differ in them (render_run)."""
    lines = [f"### {exp_id} — {res.get('title', '')}", ""] + ([*own, ""] if own else [])
    if "features" in res and "checks" not in res:
        lines += [f"`dlssvid nr --check --check-features {res.get('requested')}` на {res['size']}.", "",
                  "| Фича | ok | init, мс | create, мс | фича, МиБ | процесс, МиБ |", "|---|---|---|---|---|---|"]
        for f in res["features"]:
            lines.append(f"| {f['index']} | {'да' if f['ok'] else 'нет'} | {fmt_num(f['init_ms'], 0)} | {fmt_num(f['create_ms'], 0)} | {fmt_num(f['vram_mib'])} | "
                         f"{fmt_num(f['usage_mib'])} |")
    elif "checks" in res:
        lines += [f"`dlssvid nr --check` на {res['size']}, {len(res['checks'])} прогона.", "", "| Прогон | ok | init, мс | create, мс | фича, МиБ | процесс, с |", "|---|---|---|---|---|---|"]
        for i, c in enumerate(res["checks"]):
            lines.append(f"| {i + 1} | {'да' if c['ok'] else 'нет'} | {fmt_num(c['init_ms'], 0)} | {fmt_num(c['create_ms'], 0)} | {fmt_num(c['vram_mib'])} | {fmt_num(c['run_s'], 2)} |")
        rep = res.get("repeat")
        if rep and rep.get("error"):
            lines += ["", f"`--check-repeat {rep['requested']}`: не измерено — {rep['error']}"]
        elif rep:
            cs = sorted(x for x in rep["create_ms"] if x is not None)
            lines += ["", f"`--check-repeat {rep['requested']}`: {rep['ok']} пересозданий ok" + (f", сбой: {rep['failed']}" if rep.get("failed") else "") +
                      (f"; create min / медиана / max {fmt_num(cs[0], 0)} / {fmt_num(cs[len(cs) // 2], 0)} / {fmt_num(cs[-1], 0)} мс" if cs else "")]
    else:
        fixed = res.get("fixed", {})  # every fixed value is in the heading: still mode names its own, as the run used them
        own = ["guides выкл" if fixed.get("guides", False) is False else "guides вкл"] +             [f"{word} {fixed[k]:g}" for k, word in (("temporal", "temporal"), ("intensity", "сила")) if isinstance(fixed.get(k), (int, float))]
        extra = ", ".join(f"{k} {v}" for k, v in fixed.items() if k not in STILL)
        mode = (f"still ({', '.join(own)})" + (f", {extra}" if extra else "")) if res.get("still") else \
            ", ".join(f"{k} {v}" for k, v in fixed.items()) or "параметры по умолчанию"
        lines += [f"Кадры {res['start']}…{res['start'] + res['frames'] - 1} ({res['frames']}), разогрев {res['warmup']}, {mode}; база — {res['baseline']}.", ""]
        metrics, variants = res.get("metrics", []), res.get("variants", {})
        timed = any((e.get("nr") or {}).get("init_ms") is not None for e in variants.values())
        head = ["Вариант", "Параметры"]
        if "base" in metrics:
            head += ["PSNR Y к базе, дБ (min)", "SSIM Y к базе"]
        if "edit" in metrics:
            head += ["Правка: PSNR Y ко входу, дБ", r"mean \|out − in\| RGB"]
        if "temporal" in metrics:
            head += ["Межкадровый PSNR Y, дБ (min)"]
        if "time" in metrics:
            head += ["мс/кадр GPU", "мс/кадр wall", "прогон, с"] + (["init / create, мс", "фича, МиБ"] if timed else [])
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        if "temporal" in metrics and res.get("input_temporal"):
            row = ["вход", "—"] + (["—", "—"] if "base" in metrics else []) + (["—", "—"] if "edit" in metrics else [])
            t = res["input_temporal"]
            row += [f"{fmt_db(t['psnr_y'])} ({fmt_db(t['psnr_y_min'])})"]
            row += (["—"] * (3 + (2 if timed else 0))) if "time" in metrics else []
            lines.append("| " + " | ".join(row) + " |")
        for name, e in variants.items():
            nr = e.get("nr") or {}
            fixed = res.get("fixed", {})  # the experiment's fixed values are in the heading; a variant's own override of one stays in its row
            mine = {k: v for k, v in (e.get("params") or {}).items() if k not in fixed or fixed[k] != v}
            row = [name, describe(Variant(name, mine, e.get("flags") or [], e.get("inputs") or {}))]
            if "base" in metrics:
                b = e.get("vs_base")
                row += ["база", "база"] if name == res["baseline"] else [f"{fmt_db(b['psnr_y'])} ({fmt_db(b['psnr_y_min'])})" if b else "—",
                                                                       f"{b['ssim_y']:.4f}" if b else "—"]
            if "edit" in metrics:
                d = e.get("edit")
                row += [fmt_db(d["psnr_y"]) if d else "—", fmt_num(d.get("mae_rgb") if d else None, 4)]
            if "temporal" in metrics:
                t = e.get("temporal")
                row += [f"{fmt_db(t['psnr_y'])} ({fmt_db(t['psnr_y_min'])})" if t else "—"]
            if "time" in metrics:
                row += [fmt_num(nr.get("ms_per_frame_gpu")), fmt_num(nr.get("ms_per_frame_wall")), fmt_num(e.get("run_s"), 2)]
                if timed:
                    row += [f"{fmt_num(nr.get('init_ms'), 0)} / {fmt_num(nr.get('create_ms'), 0)}", fmt_num(nr.get("vram_mib"))]
            lines.append("| " + " | ".join(row) + " |")
        for name, why in res.get("failed", {}).items():
            lines += ["", f"{name}: не выполнен — {names_only(why)}"]
        if res.get("pairs"):
            lines += ["", "| A | B | PSNR Y, дБ | SSIM Y | вывод |", "|---|---|---|---|---|"]
            for p in res["pairs"]:
                same = p["psnr_y"] is not None and p["psnr_y"] > SAME_DB
                lines.append(f"| {p['a']} | {p['b']} | {fmt_db(p['psnr_y'])} | {p['ssim_y']:.4f} | {'совпадает' if same else 'различается'} |")
        if res.get("pairs_missing"):
            lines += ["", "Пары без замера (color_nr прошлого прогона удалён; --keep его сохраняет): " + ", ".join(f"{x}–{y}" for x, y in res["pairs_missing"])]
    text = (res.get("verdict") or {}).get("text", [])
    if text:
        lines += [""] + [f"- {t}" for t in text]
    return "\n".join(lines) + "\n"


def render_run(run_key: str, exps: dict) -> str:
    """Names only, never absolute paths: the tables are pasted into docs/benchmarks.md, which the [hygiene] test checks. The build and the inputs
    go into the heading when every experiment shares them, else into each experiment (one reused from results.json may be of another build)."""
    metas = {eid: res.get("meta") or {} for eid, res in exps.items()}
    builds = {eid: {k: m.get(k) for k in BUILD} for eid, m in metas.items()}
    ins = {eid: (m.get("inputs"), m.get("masks")) for eid, m in metas.items() if "inputs" in m}
    one_build = len({json.dumps(b, sort_keys=True) for b in builds.values()}) == 1
    one_input = len(set(ins.values())) <= 1
    out = [f"## {run_key}" + (f" — {build_line(next(iter(builds.values())))}" if one_build and builds else ""), ""]
    if one_input and ins:
        out += [inputs_line(metas[next(iter(ins))]), ""]
    for eid, res in exps.items():
        own = ([] if one_build else [f"Сборка: {build_line(builds[eid])}."]) + ([inputs_line(metas[eid])] if not one_input and eid in ins else [])
        out.append(render_experiment(eid, res, tuple(own)))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli")
    ap.add_argument("--clip")
    ap.add_argument("--project")
    ap.add_argument("--project-params", action="store_true", help="mix in the project's nr parameters (schema v2 only)")
    ap.add_argument("--passes")
    ap.add_argument("--color-dir")
    ap.add_argument("--depth-dir")
    ap.add_argument("--mv-dir")
    ap.add_argument("--mv-target-dir", help="mv_dlss at the model's size (E3 variant C)")
    ap.add_argument("--masks", action="store_true", help="pass --masks-dir <passes root>, which must hold mask_* passes (off: protected pixels would "
                                                         "mask the differences)")
    ap.add_argument("--prepare", action="store_true", help="make color_sr / depth_dlss / mv_dlss with dlssvid process first (when they do not "
                                                           "cover the range; one folder per clip and backends)")
    ap.add_argument("--prepare-backends", default="da3,ofa,dlss", help="depth,flow,upscale backends of --prepare (any GPU: stub,stub,bicubic)")
    ap.add_argument("--experiment", action="append", default=[], help="E1..E11, comma separated or repeated; all")
    ap.add_argument("--variants", action="append", default=[], help="a variants file (docs/benchmarks.md)")
    ap.add_argument("--only", action="append", default=[], help="run only these variants (the baseline always runs); added to a stored "
                                                                "experiment of the same signature")
    ap.add_argument("--start", type=int, default=6, help="first written frame (default 6 = the default warm-up)")
    ap.add_argument("--frames", type=int, default=0, help="frames per variant (0: the experiment's)")
    ap.add_argument("--warmup", type=int, default=6, help="frames before --start for the history (spec §2f)")
    ap.add_argument("--params", help="base nr settings: a JSON object or a dlssvid-preset")
    ap.add_argument("--backend", default="ngx", choices=("ngx", "stub"))
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--format", default="exr")
    ap.add_argument("--nr-arg", action="append", default=[], help=f"a run option of every dlssvid nr: {', '.join(NR_ARG_OK)} "
                                                                  "(--nr-arg=--dll-dir=D); settings go into --params")
    ap.add_argument("--check-runs", type=int, default=3)
    ap.add_argument("--check-size", help="feature size of E8 / E6f (default: color_sr's, else 1280x720)")
    ap.add_argument("--check-repeat", type=int, default=100, help="E8: re-creations in one process (0: none)")
    # 1..4 as `nr --check-features`: a probe of up to 18 NGX features hung the whole machine (INCIDENT 2026-10-25)
    ap.add_argument("--check-features", type=int, default=4, choices=(1, 2, 3, 4), help="E6f: features alive at once, 1..4 (the passes need no more)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per dlssvid call: a hung one is killed (the GPU is shared)")
    ap.add_argument("--out")
    ap.add_argument("--run-name")
    ap.add_argument("--keep", action="store_true", help="keep the color_nr folders")
    ap.add_argument("--force", action="store_true", help="redo experiments stored in results.json")
    ap.add_argument("--dry-run", action="store_true", help="the commands only; reads results.json, never writes it")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    a.only = [x for s in a.only for x in s.split(",") if x]
    if not 0 <= a.check_repeat <= 1000:
        ap.error("--check-repeat: 0..1000, as dlssvid nr --check-repeat")
    global TIMEOUT
    TIMEOUT = a.timeout if a.timeout and a.timeout > 0 else None
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    catalogue = {e.id: e for e in EXPERIMENTS}
    if a.list:
        for e in EXPERIMENTS:
            note = f" — {e.unsupported}" if e.unsupported else (f" (нужно {', '.join(e.requires)})" if e.requires else "")
            print(f"{e.id:4} {e.title}: {', '.join(v.name for v in e.variants) or e.kind}{note}")
        return 0
    check_options(a.nr_arg, NR_ARG_OK, "--nr-arg")
    cli = find_cli(a.cli)
    version = subprocess.run([str(cli), "--version"], capture_output=True, text=True).stdout.strip()
    help_text = subprocess.run([str(cli), "nr", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    keys = setting_keys(help_text)
    ids = [x for s in a.experiment for x in s.split(",") if x]
    if "all" in ids:
        ids = [e.id for e in EXPERIMENTS if not e.unsupported and all(r in help_text for r in e.requires)]
    exps = []
    for i in ids:
        if i not in catalogue:
            raise RuntimeError(f"unknown experiment {i} (--list)")
        e = catalogue[i]
        if e.unsupported:
            raise RuntimeError(f"{i}: {e.unsupported}")
        missing = [r for r in e.requires if r not in help_text]
        if missing:
            raise RuntimeError(f"{i} needs {', '.join(missing)} in dlssvid nr - not in this build")
        exps.append(e)
    taken = {i.casefold(): "the catalogue" for i in catalogue}  # results.json, the output folder (case-blind on Windows) and the verdict are keyed by id
    for p in a.variants:
        e = load_variants(Path(p))
        if e.id.casefold() in taken:
            raise RuntimeError(f"{p}: id '{e.id}' is taken by {taken[e.id.casefold()]} - give the file its own \"id\"")
        taken[e.id.casefold()] = p
        exps.append(e)
    if not exps:
        raise RuntimeError("nothing to run: give --experiment or --variants (--list)")
    base_params = load_base_params(Path(a.params) if a.params else None)
    if a.project and a.project_params:
        proj = json.loads(Path(a.project).read_text(encoding="utf-8"))
        if proj.get("schema_version", 1) < 2:
            raise RuntimeError("--project-params needs a project of schema_version 2 (v1 exposure is a linear factor)")
        nr = next((s for s in proj.get("stages", []) if s.get("name") == "nr"), {})
        base_params = {**normalize_params(nr.get("params", {})), **base_params}
    for e in exps:  # every merged key before anything runs: one dlssvid nr would refuse it after the others ran
        for v in e.variants:
            bad = [k for k in run_params(base_params, e.fixed, v.params) if k not in keys]
            if bad:
                raise RuntimeError(f"{e.id}/{v.name}: {', '.join(bad)} - not an nr setting of this dlssvid ({version}); an A/B switch goes into "
                                   "the variant's flags")
    out_root = Path(a.out) if a.out else default_out()
    inp = resolve_inputs(a, out_root, cli, a.dry_run)
    run_key = a.run_name or f"{inp.clip.stem}-{a.backend}"
    out = out_root / run_key
    repass = False
    if a.prepare:
        last = a.start + max((a.frames or e.frames) for e in exps if e.kind == "run") - 1 if any(e.kind == "run" for e in exps) else a.start
        repass = prepare(a, cli, inp, out_root, last, a.dry_run)
    build = build_identity(cli, a.backend, a.nr_arg)
    results_path = out_root / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    run_res = results.setdefault(run_key, {})
    stored = run_res.setdefault("experiments", {})
    for e in exps:
        sig = run_signature(a, inp, e, base_params, build)  # after prepare: the identities of the passes the variants read
        prev = stored.get(e.id)
        if a.dry_run and repass and (e.kind == "run" or (inp.color and not a.check_size)):  # the real run rewrites the passes first: no stored
            # experiment on them keeps its signature; E8 / E6f without --check-size create the feature at the new color_sr's size - provisional below
            print(f"# {e.id}: --prepare перепишет пассы — будет переснят" + (" (размер фичи — по новому color_sr)" if e.kind == "check" else ""))
            prev = None
        if reusable(prev, sig, [v.name for v in selected(e, a.only)], a.force, e.baseline):
            prev["title"] = e.title
            print(f"# {e.id}: уже в {results_path.name} с той же подписью (настройки, входы, сборка) — пропущен; --force переснимет")
            continue
        res = run_experiment(a, cli, inp, e, base_params, out, a.dry_run, prev, sig)
        if a.dry_run:
            continue
        res["sig"], res["meta"] = sig, experiment_meta(res, version, build, inp, e)
        stored[e.id] = res
        out_root.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")  # after each experiment: resumable
    if a.dry_run:
        return 0
    # nr_ab.md: every run and experiment of results.json (it goes into git beside it); stdout: this invocation's
    md = "\n".join(render_run(k, r["experiments"]) for k, r in results.items() if r.get("experiments"))
    (out_root / "nr_ab.md").write_text(md, encoding="utf-8")
    print(render_run(run_key, {e.id: stored[e.id] for e in exps if e.id in stored}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as err:
        print(f"nr_ab.py: {err}", file=sys.stderr)
        sys.exit(1)
