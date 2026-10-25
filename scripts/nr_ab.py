#!/usr/bin/env python3
"""A/B of the Neural Rendering settings (experiments E1-E11 of docs/benchmarks.md «A/B настроек»; TASK-0050).

Every variant is one `dlssvid nr --start F --frames N --warmup K --params-file <variant>/params.json [A/B flags]` into a pass root of its
own (<out>/<run>/<experiment>/<variant>/ - never the project's passes root), then `dlssvid compare` against the baseline variant, against
what NR read (color_sr, else the video) and against the next frame of the same output. Tables: stdout and <out>/nr_ab.md; numbers:
<out>/results.json (resumable per experiment, --force redoes).

    python scripts/nr_ab.py --list
    python scripts/nr_ab.py --clip face.mp4 --passes <passes> --experiment E1 --start 48
    python scripts/nr_ab.py --project face.dlssvid.json --experiment E5,E6
    python scripts/nr_ab.py --clip clip.mp4 --prepare --experiment E7 --dry-run          # the commands only
    python scripts/nr_ab.py --clip clip.mp4 --passes <passes> --variants my_ab.json      # own variants

NGX runs use the GPU the CI runner shares: run them only while CI is idle. `--backend stub --warp` runs anywhere (ctest
scripts.nr_ab_stub). Standard library only."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SAME_DB = 45.0        # spec §2b / §2i: two outputs above it are one picture (E1 saturation, E2 identity, E5 / E7 / E11 «no effect»)
GAIN_DB = 1.0         # spec §2e: an inter-frame PSNR gain this large makes an MV variant the default (E3 / E4)
IDENTICAL_DB = 100.0  # `dlssvid compare` counts an identical frame as 100 dB in its means and writes null for an infinite minimum
STILL = {"guides": False, "temporal": 0.0}  # the model alone: Reset on every frame (still mode), no history of our filter either
MODEL = {"temporal": 0.0}                   # model history on, our flicker filter off: the breathing measured is the model's
METRICS = ("base", "edit", "temporal", "time", "pairs")
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class Variant:
    name: str                                     # folder name: [A-Za-z0-9_.-]+
    params: dict = field(default_factory=dict)    # nr schema keys on top of the base and the experiment's fixed ones
    flags: list = field(default_factory=list)     # A/B options of the standalone `dlssvid nr`, not schema keys (spec §0.1)
    inputs: dict = field(default_factory=dict)    # "color" / "depth" / "mv": another pass folder; "@mv_target" = --mv-target-dir (E3 C)
    expect: dict = field(default_factory=dict)    # stats `nr --json` must report ({"depth_used": False}): a flag that did nothing fails the run
    optional: bool = False                        # a failure is recorded and the experiment goes on (passes 3-4 before PR-5)


@dataclass
class Experiment:
    id: str
    title: str                                    # Russian: the table heading
    variants: list
    baseline: str = ""
    fixed: dict = field(default_factory=dict)     # every variant: STILL / MODEL
    metrics: tuple = ("base", "time")
    groups: list = field(default_factory=list)    # pairwise PSNR inside each group (E7)
    frames: int = 12                              # spec §4: 12 frames of a real clip; breathing (E3 / E4) on 30
    needs: tuple = ()                             # "depth", "mv": inputs the experiment cannot run without
    requires: tuple = ()                          # options `dlssvid nr --help` must list (E9: --scene-cut, PR-6)
    kind: str = "run"                             # run | check (E8: `nr --check`)
    unsupported: str = ""                         # why no build of PR-1d can run it (E8b)


def _v(name, **kw) -> Variant:
    return Variant(name, **kw)


EXPERIMENTS = [
    Experiment("E1", "Интенсивность: насыщение и монотонность правки (§2b)",
               [_v(f"i{x:g}", params={"intensity": x}) for x in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)], "i1", STILL, ("base", "edit", "time")),
    Experiment("E2", "Интенсивность 0 — тождество? (§2b)", [_v("i0", params={"intensity": 0}), _v("i1", params={"intensity": 1})], "i1", STILL,
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
    Experiment("E7", "Стили и пресеты (§2i)",
               [_v(n, params={"style": s}) for n, s in (("default", "default"), ("natural", "natural"), ("cinematic", "cinematic"),
                                                        ("style3", "3"), ("style4", "4"), ("style5", "5"), ("style6", "6"))]
               + [_v(f"preset{p}", params={"preset": p}) for p in (0, 1, 2)],
               "natural", STILL, ("pairs", "edit", "time"),
               groups=[["default", "natural", "cinematic", "style3", "style4", "style5", "style6"], ["preset0", "preset1", "preset2", "natural"]]),
    Experiment("E8", "Цена создания фичи (§2f)", [], kind="check", metrics=("time",)),
    Experiment("E8b", "Смена настроек без пересоздания фичи (§2f)", [],
               unsupported="нужен режим смены intensity/tone посреди прогона без пересоздания фичи — не входит в PR-1d"),
    Experiment("E9", "Сброс на склейках (§2d)",
               [_v("cut0", params={"scene_cut": 0}), _v("cut024", params={"scene_cut": 0.24}), _v("cut024_t04", params={"scene_cut": 0.24, "temporal": 0.4})],
               "cut0", {}, ("temporal", "base", "time"), needs=("mv",), requires=("--scene-cut",)),
    Experiment("E10", "Локальный тон и структура (§2a)",
               [_v("base"), _v("tone0", params={"local_tone": 0}), _v("structure0", params={"local_structure": 0})], "base", STILL, ("base", "edit", "time")),
    Experiment("E11", "Структура кожи без автомаски (§2a)",
               [_v("follow", params={"skin_structure": -1}), _v("s0", params={"skin_structure": 0}), _v("s1", params={"skin_structure": 1}),
                _v("s2", params={"skin_structure": 2})], "follow", {**STILL, "auto_mask": False}, ("base", "edit", "time")),
]


# ---- pure helpers (tests/scripts/test_nr_ab.py) ----------------------------------------------------------------------------

def normalize_params(p: dict) -> dict:
    """Schema keys in snake_case, as dlssvid nr validates them: kebab aliases and no_guides are folded in, so a base file with `local-tone` and a
    variant with `local_tone` do not both reach --params-file (the later one would win by accident)."""
    out = {}
    for k, v in p.items():
        k = k.replace("-", "_")
        if k == "no_guides":
            k, v = "guides", not v
        out[k] = v
    return out


def merge_params(base: dict, fixed: dict, variant: dict) -> dict:
    return {**normalize_params(base), **normalize_params(fixed), **normalize_params(variant)}  # base < experiment < variant


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


@dataclass
class Inputs:
    clip: Path                          # dlssvid nr decodes it even when colour comes from color_sr (the decoder drives the run)
    color: Optional[Path] = None        # color_sr; None: NR reads the video
    depth: Optional[Path] = None
    mv: Optional[Path] = None
    masks: Optional[Path] = None
    mv_target: Optional[Path] = None

    def reference(self) -> Path:        # what NR read: the edit is measured against it
        return self.color or self.clip

    def index(self, frame: int) -> int:  # compare reads a pass from its first frame, a video from 0
        r = pass_range(self.color) if self.color else None
        return frame - (r[0] if r else 0)


def nr_command(cli, inp: Inputs, v: Variant, root: Path, params_file: Path, start: int, frames: int, warmup: int, backend: str, fmt: str,
               warp: bool, extra: list) -> list:
    dirs = {"color": inp.color, "depth": inp.depth, "mv": inp.mv}
    for k, d in v.inputs.items():
        dirs[k] = inp.mv_target if d == "@mv_target" else Path(d)
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


def read_compare(path: Path) -> dict:
    s = json.loads(Path(path).read_text(encoding="utf-8"))["summary"]

    def db(v):
        return IDENTICAL_DB if v is None or v >= IDENTICAL_DB else float(v)  # null = infinite minimum (identical frames)

    return {"frames": s["frames"], "psnr_y": db(s["psnr_y_mean"]), "psnr_y_min": db(s.get("psnr_y_min")), "psnr_rgb": db(s["psnr_rgb_mean"]),
            "ssim_y": s["ssim_y_mean"], "mae_rgb": s.get("mae_rgb_mean")}


def read_nr(path: Path) -> dict:
    """stats + the diagnostics of the feature (TASK-0048); absent -> None, shown as «—»."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    st, dg = j.get("stats", {}), j.get("diagnostics", {}) or {}
    out = {k: st.get(k) for k in ("ms_per_frame_gpu", "ms_per_frame_wall", "depth_used", "mv_used", "work_size", "passes", "mv_scale",
                                  "frames_written", "warmup_frames")}
    out.update({k: dg.get(k) for k in ("init_ms", "create_ms", "gpu", "driver", "dll_sha256")})
    vram = dg.get("vram_feature_bytes")
    out["vram_mib"] = round(vram / 1048576.0, 1) if isinstance(vram, (int, float)) else None
    out["range"] = j.get("range")
    return out


def unmet(expect: dict, nr: dict) -> list:
    return [f"{k}={nr.get(k)!r}, expected {v!r}" for k, v in expect.items() if nr.get(k) != v]


def fmt_db(v) -> str:
    if v is None:
        return "—"
    return "∞" if v >= IDENTICAL_DB else f"{v:.2f}"


def fmt_num(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def describe(v: Variant) -> str:
    parts = [f"{k}={json.dumps(val, ensure_ascii=False) if isinstance(val, str) else val}" for k, val in v.params.items()]
    parts += [" ".join(v.flags)] if v.flags else []
    parts += [f"{k}: {d}" for k, d in v.inputs.items()]
    return ", ".join(parts) or "—"


def load_variants(path: Path) -> Experiment:
    """A variants file (docs/benchmarks.md «A/B настроек»): its own experiment, validated before anything runs."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    eid = j.get("id") or Path(path).stem
    if not NAME.match(eid):
        raise RuntimeError(f"{path}: id '{eid}' must match {NAME.pattern}")
    vs, names = [], set()
    for x in j.get("variants", []):
        name = x.get("name", "")
        if not NAME.match(name):
            raise RuntimeError(f"{path}: variant name '{name}' must match {NAME.pattern}")
        if name in names:
            raise RuntimeError(f"{path}: variant '{name}' twice")
        names.add(name)
        vs.append(Variant(name, normalize_params(x.get("params", {})), list(x.get("flags", [])), dict(x.get("inputs", {})), dict(x.get("expect", {})),
                          bool(x.get("optional", False))))
    if not vs:
        raise RuntimeError(f"{path}: no variants")
    baseline = j.get("baseline") or vs[0].name
    if baseline not in names:
        raise RuntimeError(f"{path}: baseline '{baseline}' is not a variant")
    metrics = tuple(j.get("metrics", ("base", "time")))
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise RuntimeError(f"{path}: unknown metric {bad} (known: {', '.join(METRICS)})")
    groups = [list(g) for g in j.get("groups", [])]
    for g in groups:
        for n in g:
            if n not in names:
                raise RuntimeError(f"{path}: group member '{n}' is not a variant")
    fixed = {**(STILL if j.get("still") else {}), **normalize_params(j.get("fixed", {}))}
    return Experiment(eid, j.get("title", eid), vs, baseline, fixed, metrics, groups, int(j.get("frames", 12)), tuple(j.get("needs", ())))


# ---- verdicts (pure; None = no data) ---------------------------------------------------------------------------------------

def _get(res, name, *path):
    x = res.get("variants", {}).get(name, {})
    for p in path:
        x = x.get(p) if isinstance(x, dict) else None
    return x


def verdict_e1(res):
    hi = [_get(res, "i1.5", "vs_base", "psnr_y"), _get(res, "i2", "vs_base", "psnr_y")]
    saturated = None if None in hi else all(x > SAME_DB for x in hi)
    edits = [_get(res, f"i{x}", "edit", "mae_rgb") for x in ("0", "0.25", "0.5", "0.75", "1")]
    monotone = None if None in edits else all(b > a for a, b in zip(edits, edits[1:]))
    text = []
    if saturated is not None:
        text.append(f"NGX насыщается на 1: PSNR(1.5, 1) {fmt_db(hi[0])}, PSNR(2, 1) {fmt_db(hi[1])} дБ > {SAME_DB:g} — вариант А или Б выбирает пользователь (§2b)"
                    if saturated else f"насыщения нет (PSNR(1.5, 1) {fmt_db(hi[0])}, PSNR(2, 1) {fmt_db(hi[1])} дБ) — intensity идёт в NGX как есть")
    if monotone is False:
        text.append("правка на [0, 1] растёт не монотонно — семантику не меняем, сначала разбираемся, как читается параметр (E8b)")
    return {"saturated": saturated, "monotone": monotone, "text": text}


def verdict_e2(res):
    db = _get(res, "i0", "edit", "psnr_y")
    identity = None if db is None else db >= SAME_DB
    text = [] if identity is None else [f"тождество (PSNR ко входу {fmt_db(db)} дБ): обход модели не нужен" if identity
                                        else f"не тождество (PSNR ко входу {fmt_db(db)} дБ) — обход при 0 (+nr.i0, PR-7)"]
    return {"identity": identity, "text": text}


def verdict_e3(res):
    a, b, c = (_get(res, n, "temporal", "psnr_y") for n in ("unit", "auto", "target"))
    sa, sb = _get(res, "unit", "nr", "mv_scale"), _get(res, "auto", "nr", "mv_scale")
    text = []
    if sa is not None and sb is not None and sa == sb:
        text.append("масштаб векторов у unit и auto одинаковый (размер mv = размер модели): E3 на этом входе ничего не меряет")
    if c is None:
        text.append("нет варианта C (--mv-target-dir: mv_dlss размера модели из `dlssvid flow --target`)")
    better = None
    if None not in (a, b, c):
        better = abs(b - c) < GAIN_DB and min(b, c) - a >= GAIN_DB
        text.append("B ≈ C и оба лучше A — единицы auto по умолчанию" if better else "B и C не лучше A на порог — остаётся unit")
    return {"auto_better": better, "text": text}


def verdict_e4(res):
    a, b = _get(res, "null", "temporal", "psnr_y"), _get(res, "create_mv", "temporal", "psnr_y")
    better = None if None in (a, b) else b - a >= GAIN_DB
    text = [] if better is None else [f"межкадровый PSNR {fmt_db(a)} → {fmt_db(b)} дБ: " + ("создание с нулевыми векторами по умолчанию" if better else "разницы на порог нет")]
    return {"create_better": better, "text": text}


def verdict_e5(res):
    db = _get(res, "no_depth", "vs_base", "psnr_y")
    same = None if db is None else db > SAME_DB
    text = [] if same is None else [f"без глубины к базе {fmt_db(db)} дБ: " + ("разницы нет — глубину оставляем (дешевле не станет)" if same else "глубина меняет картинку")]
    return {"no_difference": same, "text": text}


def verdict_e6(res):
    p1, p2 = _get(res, "p1", "nr", "ms_per_frame_gpu"), _get(res, "p2", "nr", "ms_per_frame_gpu")
    step = None if None in (p1, p2) else p2 - p1
    return {"ms_per_pass": step, "text": [] if step is None else [f"GPU на добавленный проход: {step:.1f} мс/кадр (константа оценок — 17.6 мс на 1440p)"]}


def verdict_e7(res):
    same = [f"{p['a']} ≈ {p['b']} ({fmt_db(p['psnr_y'])} дБ)" for p in res.get("pairs", []) if p.get("psnr_y") is not None and p["psnr_y"] > SAME_DB]
    return {"same_pairs": len(same), "text": ["совпадают: " + "; ".join(same)] if same else ["все пары различаются"]}


def verdict_e8(res):
    runs = [r for r in res.get("checks", []) if r.get("ok")]

    def med(k):
        xs = sorted(r[k] for r in runs if r.get(k) is not None)
        return xs[len(xs) // 2] if xs else None

    text = [f"медианы: init {fmt_num(med('init_ms'), 0)} мс, create {fmt_num(med('create_ms'), 0)} мс, фича {fmt_num(med('vram_mib'))} МиБ, процесс {fmt_num(med('run_s'), 2)} с"]
    failed = [i for i, r in enumerate(res.get("checks", [])) if not r.get("ok")]
    text.append("все проверки ok" if not failed else f"сбой в проверках {failed}")
    return {"init_ms": med("init_ms"), "create_ms": med("create_ms"), "text": text}


def verdict_e11(res):
    dbs = [_get(res, n, "vs_base", "psnr_y") for n in ("s0", "s1", "s2")]
    none = None if None in dbs else all(x > SAME_DB for x in dbs)
    text = [] if none is None else ["без автомаски эффекта нет — условие доступности остаётся" if none else "структура кожи действует и без автомаски — условие снять"]
    return {"no_effect": none, "text": text}


VERDICTS = {"E1": verdict_e1, "E2": verdict_e2, "E3": verdict_e3, "E4": verdict_e4, "E5": verdict_e5, "E6": verdict_e6, "E7": verdict_e7, "E8": verdict_e8,
            "E11": verdict_e11}


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
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(subprocess.list2cmdline([str(c) for c in cmd]) + "\n\n" + r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        raise RuntimeError(f"{Path(str(cmd[0])).name} {cmd[1]} failed ({r.returncode}):\n{tail}")
    return time.perf_counter() - t0


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
    if a.prepare:
        passes = out / "prepare" / clip.stem / "passes"
    pick = lambda name: (passes / name) if passes and (passes / name / "manifest.json").exists() else None
    inp = Inputs(clip, Path(a.color_dir) if a.color_dir else pick("color_sr"), Path(a.depth_dir) if a.depth_dir else pick("depth_dlss"),
                 Path(a.mv_dir) if a.mv_dir else pick("mv_dlss"), (passes if a.masks else None), Path(a.mv_target_dir) if a.mv_target_dir else None)
    if a.prepare:  # the planned folders, also in a dry run
        inp.color, inp.depth, inp.mv = passes / "color_sr", passes / "depth_dlss", passes / "mv_dlss"
    return inp


def prepare(a, cli: Path, inp: Inputs, out: Path, last: int, dry: bool):
    """color_sr / depth_dlss / mv_dlss for frames 0..last with `dlssvid process` (it always encodes: the video is deleted after)."""
    if not a.force and all(covers(d, 0, last) for d in (inp.color, inp.depth, inp.mv)):
        return
    work = out / "prepare" / inp.clip.stem
    d, f, u = (a.prepare_backends.split(",") + ["", "", ""])[:3]
    cmd = [cli, "process", "-i", inp.clip, "-o", work / "prepare.mkv", "--passes", work / "passes", "--stages", "depth,flow,upscale", "--depth-backend", d,
           "--flow-backend", f, "--upscale-backend", u, "--frames", str(last + 1), "--codec", "ffv1"] + (["--warp"] if a.warp else [])
    run(cmd, work / "prepare.log", dry)
    if not dry:
        (work / "prepare.mkv").unlink(missing_ok=True)


def run_experiment(a, cli: Path, inp: Inputs, exp: Experiment, base_params: dict, out: Path, dry: bool) -> dict:
    edir = out / exp.id
    if exp.kind == "check":
        return run_check(a, cli, inp, exp, edir, dry)
    frames = a.frames or exp.frames
    still = exp.fixed.get("guides") is False
    start = a.start
    warmup = 0 if still else min(a.warmup, start)  # still mode resets every frame: a warm-up only burns GPU
    last = start + frames - 1
    for need in exp.needs:
        d = {"depth": inp.depth, "mv": inp.mv}[need]
        if not dry and not covers(d, start - warmup, last):
            raise RuntimeError(f"{exp.id} needs {need} for frames {start - warmup}..{last}: give --passes / --{need}-dir or --prepare")
    if inp.color and not dry and not covers(inp.color, start - warmup, last):
        raise RuntimeError(f"{exp.id}: color_sr {inp.color.name} does not cover frames {start - warmup}..{last}")
    variants = [v for v in exp.variants if not a.only or v.name in a.only or v.name == exp.baseline]
    res = {"title": exp.title, "start": start, "frames": frames, "warmup": warmup, "still": still, "fixed": exp.fixed, "baseline": exp.baseline,
           "metrics": list(exp.metrics), "variants": {}, "failed": {}}
    for v in variants:
        vdir = edir / v.name
        params = merge_params(base_params, exp.fixed, v.params)
        if dry:
            print(f"# {exp.id}/{v.name} params.json: {json.dumps(params, sort_keys=True)}")
        else:
            shutil.rmtree(vdir / "color_nr", ignore_errors=True)  # frames of an earlier, longer run must not enter the metrics
            vdir.mkdir(parents=True, exist_ok=True)
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
            bad = unmet(v.expect, nr)
            if bad:
                raise RuntimeError(f"{exp.id}/{v.name}: {'; '.join(bad)} - the flag did nothing")
            res["variants"][v.name] = {"nr": nr, "run_s": round(run_s, 2), "params": params, "flags": v.flags, "range": [r[0], r[1], r[2]]}
        except RuntimeError as e:
            if not v.optional:
                raise
            res["failed"][v.name] = str(e).splitlines()[-1]
    ok = [v for v in variants if v.name in res["variants"]]
    base = edir / exp.baseline / "color_nr"
    for v in ok:
        vdir, entry = edir / v.name, res["variants"][v.name]
        if "base" in exp.metrics and v.name != exp.baseline:
            run(compare_command(cli, base, vdir / "color_nr", vdir / "cmp_base.json"), None, dry)
            entry["vs_base"] = None if dry else read_compare(vdir / "cmp_base.json")
        if "edit" in exp.metrics:
            run(compare_command(cli, vdir / "color_nr", inp.reference(), vdir / "cmp_input.json", offset=inp.index(start)), None, dry)
            entry["edit"] = None if dry else read_compare(vdir / "cmp_input.json")
        if "temporal" in exp.metrics and frames > 1:
            run(compare_command(cli, vdir / "color_nr", vdir / "color_nr", vdir / "cmp_next.json", offset=1), None, dry)
            entry["temporal"] = None if dry else read_compare(vdir / "cmp_next.json")
    if "temporal" in exp.metrics and frames > 1:
        ref = inp.reference()
        run(compare_command(cli, ref, ref, edir / "cmp_input_next.json", start=inp.index(start), offset=1, frames=frames - 1), None, dry)
        res["input_temporal"] = None if dry else read_compare(edir / "cmp_input_next.json")
    if "pairs" in exp.metrics:
        res["pairs"] = []
        for g in exp.groups:
            for x, y in itertools.combinations([n for n in g if n in res["variants"]], 2):
                f = edir / f"pair_{x}__{y}.json"
                run(compare_command(cli, edir / x / "color_nr", edir / y / "color_nr", f), None, dry)
                if not dry:
                    c = read_compare(f)
                    res["pairs"].append({"a": x, "b": y, "psnr_y": c["psnr_y"], "ssim_y": c["ssim_y"]})
    if not dry:
        res["verdict"] = VERDICTS.get(exp.id, lambda _r: {"text": []})(res)
        if not a.keep:
            for v in variants:
                shutil.rmtree(edir / v.name / "color_nr", ignore_errors=True)
    return res


def run_check(a, cli: Path, inp: Inputs, exp: Experiment, edir: Path, dry: bool) -> dict:
    r = pass_range(inp.color)
    size = a.check_size
    if inp.color and r:
        try:
            m = json.loads((inp.color / "manifest.json").read_text(encoding="utf-8"))
            size = f"{m['width']}x{m['height']}"
        except (OSError, ValueError, KeyError):
            pass
    res = {"title": exp.title, "size": size, "checks": []}
    for i in range(a.check_runs):
        f = edir / f"check_{i}.json"
        if not dry:
            edir.mkdir(parents=True, exist_ok=True)
        cmd = [cli, "nr", "--check", "--backend", a.backend, "--check-size", size, "--json", f] + (["--warp"] if a.warp else []) + list(a.nr_arg)
        try:
            run_s = run(cmd, edir / f"check_{i}.log", dry)
            ok = True
        except RuntimeError:
            run_s, ok = None, False  # a failed --check still writes its JSON (exit 1)
        if dry:
            continue
        j = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        vram = j.get("vram_feature_bytes")
        res["checks"].append({"ok": ok and bool(j.get("ok")), "init_ms": j.get("init_ms"), "create_ms": j.get("create_ms"), "device_ms": j.get("device_ms"),
                              "vram_mib": round(vram / 1048576.0, 1) if isinstance(vram, (int, float)) else None, "run_s": None if run_s is None else round(run_s, 2)})
    if not dry:
        res["verdict"] = verdict_e8(res)
    return res


# ---- report ----------------------------------------------------------------------------------------------------------------

def render_experiment(exp_id: str, res: dict) -> str:
    lines = [f"### {exp_id} — {res.get('title', '')}", ""]
    if "checks" in res:
        lines += [f"`dlssvid nr --check` на {res['size']}, {len(res['checks'])} прогона.", "", "| Прогон | ok | init, мс | create, мс | фича, МиБ | процесс, с |", "|---|---|---|---|---|---|"]
        for i, c in enumerate(res["checks"]):
            lines.append(f"| {i + 1} | {'да' if c['ok'] else 'нет'} | {fmt_num(c['init_ms'], 0)} | {fmt_num(c['create_ms'], 0)} | {fmt_num(c['vram_mib'])} | {fmt_num(c['run_s'], 2)} |")
    else:
        mode = "still (guides выкл, temporal 0)" if res.get("still") else ", ".join(f"{k} {v}" for k, v in res.get("fixed", {}).items()) or "параметры по умолчанию"
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
            own = {k: v for k, v in (e.get("params") or {}).items() if k not in res.get("fixed", {})}  # the experiment's fixed keys are in the heading
            row = [name, describe(Variant(name, own, e.get("flags") or []))]
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
            lines += ["", f"{name}: не выполнен — {why}"]
        if res.get("pairs"):
            lines += ["", "| A | B | PSNR Y, дБ | SSIM Y | вывод |", "|---|---|---|---|---|"]
            for p in res["pairs"]:
                same = p["psnr_y"] is not None and p["psnr_y"] > SAME_DB
                lines.append(f"| {p['a']} | {p['b']} | {fmt_db(p['psnr_y'])} | {p['ssim_y']:.4f} | {'совпадает' if same else 'различается'} |")
    text = (res.get("verdict") or {}).get("text", [])
    if text:
        lines += [""] + [f"- {t}" for t in text]
    return "\n".join(lines) + "\n"


def render_run(run_key: str, meta: dict, exps: dict) -> str:
    """Names only, never absolute paths: the tables are pasted into docs/benchmarks.md, which the [hygiene] test checks."""
    sha = (meta.get("dll_sha256") or "")[:8]
    head = f"## {run_key} — dlssvid {meta.get('version', '?')}" + (f", {meta['gpu']}" if meta.get("gpu") else "") + (f", драйвер {meta['driver']}" if meta.get("driver") else "")
    head += f", nvngx_dlssnr.dll sha256 {sha}…" if sha else ""
    out = [head, "", f"Вход NR: {meta.get('inputs', '—')}; маски {'вкл' if meta.get('masks') else 'выкл'}.", ""]
    for eid, res in exps.items():
        out.append(render_experiment(eid, res))
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
    ap.add_argument("--masks", action="store_true", help="pass --masks-dir <passes root> (off: protected pixels would mask the differences)")
    ap.add_argument("--prepare", action="store_true", help="make color_sr / depth_dlss / mv_dlss with dlssvid process first")
    ap.add_argument("--prepare-backends", default="da3,ofa,dlss", help="depth,flow,upscale backends of --prepare (any GPU: stub,stub,bicubic)")
    ap.add_argument("--experiment", action="append", default=[], help="E1..E11, comma separated or repeated; all")
    ap.add_argument("--variants", action="append", default=[], help="a variants file (docs/benchmarks.md)")
    ap.add_argument("--only", action="append", default=[], help="run only these variants (the baseline always runs)")
    ap.add_argument("--start", type=int, default=6, help="first written frame (default 6 = the default warm-up)")
    ap.add_argument("--frames", type=int, default=0, help="frames per variant (0: the experiment's)")
    ap.add_argument("--warmup", type=int, default=6, help="frames before --start for the history (spec §2f)")
    ap.add_argument("--params", help="base nr settings: a JSON object or a dlssvid-preset")
    ap.add_argument("--backend", default="ngx", choices=("ngx", "stub"))
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--format", default="exr")
    ap.add_argument("--nr-arg", action="append", default=[], help="an extra option of every dlssvid nr (--nr-arg=--skip-driver-check)")
    ap.add_argument("--check-runs", type=int, default=3)
    ap.add_argument("--check-size", default="1280x720")
    ap.add_argument("--out")
    ap.add_argument("--run-name")
    ap.add_argument("--keep", action="store_true", help="keep the color_nr folders")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    a.only = [x for s in a.only for x in s.split(",") if x]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    catalogue = {e.id: e for e in EXPERIMENTS}
    if a.list:
        for e in EXPERIMENTS:
            note = f" — {e.unsupported}" if e.unsupported else (f" (нужно {', '.join(e.requires)})" if e.requires else "")
            print(f"{e.id:4} {e.title}: {', '.join(v.name for v in e.variants) or e.kind}{note}")
        return 0
    cli = find_cli(a.cli)
    version = subprocess.run([str(cli), "--version"], capture_output=True, text=True).stdout.strip()
    help_text = subprocess.run([str(cli), "nr", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    keys = known_keys(help_text)
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
    exps += [load_variants(Path(p)) for p in a.variants]
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
            bad = [k for k in merge_params(base_params, e.fixed, v.params) if k not in keys]
            if bad:
                raise RuntimeError(f"{e.id}/{v.name}: {', '.join(bad)} - not an option of this dlssvid nr ({version})")
    out_root = Path(a.out) if a.out else default_out()
    inp = resolve_inputs(a, out_root, cli, a.dry_run)
    run_key = a.run_name or f"{inp.clip.stem}-{a.backend}"
    out = out_root / run_key
    if a.prepare:
        last = a.start + max((a.frames or e.frames) for e in exps if e.kind == "run") - 1 if any(e.kind == "run" for e in exps) else a.start
        prepare(a, cli, inp, out_root, last, a.dry_run)
    results_path = out_root / "results.json"
    results = {} if a.dry_run or not results_path.exists() else json.loads(results_path.read_text(encoding="utf-8"))
    run_res = results.setdefault(run_key, {"meta": {}, "experiments": {}})
    for e in exps:
        prev = run_res["experiments"].get(e.id)
        if prev and not a.force and prev.get("start") == a.start and prev.get("frames") == (a.frames or e.frames):
            continue
        res = run_experiment(a, cli, inp, e, base_params, out, a.dry_run)
        if a.dry_run:
            continue
        run_res["experiments"][e.id] = res
        first = next((x["nr"] for x in res.get("variants", {}).values() if x.get("nr")), {})
        run_res["meta"] = {"version": version, "gpu": first.get("gpu"), "driver": first.get("driver"), "dll_sha256": first.get("dll_sha256"),
                           "masks": a.masks, "inputs": ", ".join(n for n, d in (("color_sr", inp.color), ("depth_dlss", inp.depth), ("mv_dlss", inp.mv)) if d) or "видео"}
        out_root.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")  # after each experiment: resumable
    if a.dry_run:
        return 0
    md = render_run(run_key, run_res["meta"], {e.id: run_res["experiments"][e.id] for e in exps if e.id in run_res["experiments"]})
    (out_root / "nr_ab.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as err:
        print(f"nr_ab.py: {err}", file=sys.stderr)
        sys.exit(1)
