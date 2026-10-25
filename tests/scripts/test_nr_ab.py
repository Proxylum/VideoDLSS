#!/usr/bin/env python3
"""scripts/nr_ab.py (TASK-0050): the helpers that decide what an A/B reads (ctest scripts.nr_ab, no dlssvid), and a whole A/B on the stub
backend and WARP over a generated Y4M clip (ctest scripts.nr_ab_stub: --cli <dlssvid> --work <scratch>). Standard library only."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import nr_ab  # noqa: E402

CLI: Path | None = None
WORK: Path | None = None


def write_y4m(path: Path, width: int = 96, height: int = 64, frames: int = 8) -> Path:
    """A clip without an FFmpeg CLI: dlssvid's FFmpeg reads YUV4MPEG2. Texture for the stub's local-contrast edit, a moving block so that
    the frames differ (finite inter-frame PSNR)."""
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(frames):
            f.write(b"FRAME\n")
            y = bytearray(16 + (x * 5 + yy * 3 + t * 7) % 200 for yy in range(height) for x in range(width))
            for yy in range(16, 32):
                for x in range(8 + 6 * t, min(width, 24 + 6 * t)):
                    y[yy * width + x] = 235
            f.write(bytes(y))
            f.write(bytes(96 + (x * 2) % 64 for yy in range(height // 2) for x in range(width // 2)))
            f.write(bytes(128 + (yy * 2) % 32 for yy in range(height // 2) for x in range(width // 2)))
    return path


def manifest(folder: Path, first: int, last: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"frame_range": {"first": first, "last": last}, "frame_count": last - first + 1, "width": 8, "height": 8}),
                                          encoding="utf-8")
    return folder


class Helpers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normalize_and_merge_params(self):
        self.assertEqual(nr_ab.merge_params({"intensity": 1, "local-tone": 2}, {"temporal": 0.0}, {"intensity": 0.5, "local_tone": 0}),
                         {"intensity": 0.5, "local_tone": 0, "temporal": 0.0})
        self.assertEqual(nr_ab.normalize_params({"no_guides": True}), {"guides": False})
        self.assertEqual(nr_ab.merge_params({}, {"guides": False}, {"no-guides": False}), {"guides": True})  # the variant wins

    def test_base_params_from_preset(self):
        p = self.root / "preset.json"
        p.write_text(json.dumps({"format": "dlssvid-preset", "stages": {"nr": {"params": {"intensity": 1.5}}}}), encoding="utf-8")
        self.assertEqual(nr_ab.load_base_params(p), {"intensity": 1.5})
        p.write_text(json.dumps({"local-tone": 0.5}), encoding="utf-8")
        self.assertEqual(nr_ab.load_base_params(p), {"local_tone": 0.5})
        self.assertEqual(nr_ab.load_base_params(None), {})

    def test_known_keys_from_help(self):
        help_text = "  --local-tone,--local_tone FLOAT [1]\n  --no-guides,--no_guides,--guides\n  --params-file,--params_file TEXT\n"
        keys = nr_ab.known_keys(help_text)
        self.assertTrue({"local_tone", "guides", "params_file"} <= keys)
        self.assertNotIn("scene_cut", keys)

    def test_nr_command_range_and_flags(self):
        inp = nr_ab.Inputs(Path("clip.mp4"), color=Path("sr"), depth=Path("d"), mv=Path("m"), mv_target=Path("mt"))
        v = nr_ab.Variant("x", flags=["--no-depth-guide"], inputs={"mv": "@mv_target"})
        cmd = nr_ab.nr_command("dlssvid", inp, v, Path("out"), Path("p.json"), 4, 3, 6, "stub", "exr", True, ["--skip-driver-check"])
        joined = " ".join(cmd)
        self.assertIn("--start 4 --frames 3 --warmup 4", joined)  # the warm-up is clamped to the frames before the start
        self.assertIn("--no-auto", cmd)
        self.assertIn("--params-file p.json", joined)
        self.assertIn("--mv-dir mt", joined)  # the variant's mv folder replaces the run's
        self.assertEqual(cmd[-1], "--no-depth-guide")  # the variant's flags come last
        self.assertIn("--warp", cmd)
        cmd = nr_ab.nr_command("dlssvid", inp, nr_ab.Variant("y"), Path("out"), Path("p.json"), 4, 3, 2, "ngx", "exr", False, [])
        self.assertNotIn("--warp", cmd)
        self.assertIn("--warmup 2", " ".join(cmd))

    def test_compare_commands_index_the_input_from_the_output(self):
        sr = manifest(self.root / "color_sr", 0, 9)
        inp = nr_ab.Inputs(Path("clip.mp4"), color=sr)
        self.assertEqual(inp.index(4), 4)
        manifest(sr, 2, 9)
        self.assertEqual(inp.index(4), 2)  # compare reads a pass from its first frame
        self.assertEqual(nr_ab.Inputs(Path("clip.mp4")).index(4), 4)  # a video from frame 0
        edit = nr_ab.compare_command("dlssvid", "out/color_nr", sr, "e.json", offset=inp.index(4))
        self.assertIn("--offset 2", " ".join(edit))
        nxt = nr_ab.compare_command("dlssvid", "a", "a", "n.json", start=4, offset=1, frames=2)
        self.assertIn("--start 4 --offset 1 --frames 2", " ".join(nxt))
        for c in (edit, nxt):
            self.assertFalse(any(x.startswith("-") and x[1:].isdigit() for x in map(str, c)))  # no negative numbers in an argv

    def test_pass_range_and_covers(self):
        d = manifest(self.root / "p", 2, 9)
        self.assertEqual(nr_ab.pass_range(d), (2, 9, 8))
        self.assertTrue(nr_ab.covers(d, 2, 9))
        self.assertFalse(nr_ab.covers(d, 1, 9))
        self.assertIsNone(nr_ab.pass_range(self.root / "none"))
        (self.root / "bad").mkdir()
        (self.root / "bad" / "manifest.json").write_text("{", encoding="utf-8")
        self.assertIsNone(nr_ab.pass_range(self.root / "bad"))
        self.assertFalse(nr_ab.covers(None, 0, 0))

    def test_read_compare_caps_identical(self):
        f = self.root / "c.json"
        f.write_text(json.dumps({"summary": {"frames": 3, "psnr_y_mean": 100.0, "psnr_y_min": None, "psnr_rgb_mean": 100.0, "ssim_y_mean": 1.0}}),
                     encoding="utf-8")
        c = nr_ab.read_compare(f)
        self.assertEqual((c["psnr_y"], c["psnr_y_min"]), (100.0, 100.0))
        self.assertEqual(nr_ab.fmt_db(c["psnr_y"]), "∞")
        self.assertIsNone(c["mae_rgb"])
        self.assertEqual(nr_ab.fmt_num(c["mae_rgb"]), "—")
        self.assertEqual(nr_ab.fmt_db(38.123), "38.12")

    def test_variants_file_validation(self):
        def load(j):
            p = self.root / "v.json"
            p.write_text(json.dumps(j), encoding="utf-8")
            return nr_ab.load_variants(p)

        ok = load({"id": "s", "still": True, "variants": [{"name": "a"}, {"name": "b", "params": {"local-tone": 0}}], "groups": [["a", "b"]]})
        self.assertEqual(ok.baseline, "a")
        self.assertEqual(ok.fixed, {"guides": False, "temporal": 0.0})
        self.assertEqual(ok.variants[1].params, {"local_tone": 0})
        for bad in ({"variants": [{"name": "a"}, {"name": "a"}]}, {"variants": [{"name": "a/b"}]}, {"baseline": "z", "variants": [{"name": "a"}]},
                    {"metrics": ["speed"], "variants": [{"name": "a"}]}, {"groups": [["a", "q"]], "variants": [{"name": "a"}]}, {"variants": []}):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    load(bad)

    def test_catalogue(self):
        cat = {e.id: e for e in nr_ab.EXPERIMENTS}
        self.assertEqual(len(cat["E1"].variants), 8)
        self.assertEqual(cat["E1"].baseline, "i1")
        for i in ("E1", "E2", "E7", "E10", "E11"):
            self.assertEqual({k: cat[i].fixed[k] for k in ("guides", "temporal")}, {"guides": False, "temporal": 0.0}, i)
        for i in ("E3", "E4", "E5", "E6"):
            self.assertEqual(cat[i].fixed, {"temporal": 0.0}, i)
        self.assertEqual([len(g) for g in cat["E7"].groups], [7, 4])
        styles = {v.params.get("style") for v in cat["E7"].variants}
        self.assertTrue({"3", "4", "5", "6"} <= styles)
        self.assertEqual([v.optional for v in cat["E6"].variants], [False, False, True, True])
        self.assertTrue(cat["E8b"].unsupported)
        self.assertEqual(cat["E9"].requires, ("--scene-cut",))
        self.assertEqual((cat["E3"].frames, cat["E4"].frames), (30, 30))

    def test_verdicts(self):
        def v(**variants):
            return {"variants": variants}

        e1 = nr_ab.verdict_e1(v(**{"i1.5": {"vs_base": {"psnr_y": 50}}, "i2": {"vs_base": {"psnr_y": 47}},
                                   **{f"i{x}": {"edit": {"mae_rgb": m}} for x, m in zip(("0", "0.25", "0.5", "0.75", "1"), (0, .01, .02, .015, .03))}}))
        self.assertTrue(e1["saturated"])
        self.assertFalse(e1["monotone"])
        self.assertIsNone(nr_ab.verdict_e1(v(**{"i1.5": {"vs_base": {"psnr_y": 50}}}))["saturated"])
        self.assertFalse(nr_ab.verdict_e2(v(i0={"edit": {"psnr_y": 40}}))["identity"])
        self.assertTrue(nr_ab.verdict_e4(v(null={"temporal": {"psnr_y": 46.4}}, create_mv={"temporal": {"psnr_y": 47.6}}))["create_better"])
        self.assertTrue(nr_ab.verdict_e5(v(no_depth={"vs_base": {"psnr_y": 52}}))["no_difference"])
        self.assertEqual(nr_ab.verdict_e7({"pairs": [{"a": "a", "b": "b", "psnr_y": 50}, {"a": "a", "b": "c", "psnr_y": 30}]})["same_pairs"], 1)
        self.assertTrue(nr_ab.verdict_e11(v(**{n: {"vs_base": {"psnr_y": 46.5}} for n in ("s0", "s1", "s2")}))["no_effect"])
        e3 = nr_ab.verdict_e3(v(unit={"temporal": {"psnr_y": 40}, "nr": {"mv_scale": [1, 1]}}, auto={"temporal": {"psnr_y": 41}, "nr": {"mv_scale": [1, 1]}}))
        self.assertTrue(any("ничего не меряет" in t for t in e3["text"]))

    def test_render_table(self):
        res = {"title": "Проверка", "start": 4, "frames": 3, "warmup": 2, "still": False, "fixed": {"temporal": 0.0}, "baseline": "base",
               "metrics": ["base", "edit", "temporal", "time"],
               "variants": {"base": {"nr": {"ms_per_frame_gpu": 1.0, "ms_per_frame_wall": 2.0}, "run_s": 0.5, "params": {"temporal": 0.0},
                                     "edit": {"psnr_y": 30.0, "psnr_y_min": 29.0, "mae_rgb": 0.01}, "temporal": {"psnr_y": 40.0, "psnr_y_min": 39.0}},
                            "x": {"nr": {}, "run_s": 0.4, "params": {"intensity": 0.5}, "vs_base": {"psnr_y": 100.0, "psnr_y_min": 100.0, "ssim_y": 1.0},
                                  "edit": None, "temporal": None}},
               "failed": {"p3": "nr.passes must be within 1..2"}, "input_temporal": {"psnr_y": 41.0, "psnr_y_min": 40.0},
               "verdict": {"text": ["вывод"]}}
        md = nr_ab.render_experiment("smoke", res)
        self.assertIn("### smoke — Проверка", md)
        self.assertIn("| base |", md)
        self.assertIn("| x | intensity=0.5 |", md)
        self.assertIn("база", md)
        self.assertIn("∞", md)
        self.assertIn("—", md)
        self.assertIn("p3: не выполнен", md)
        self.assertIn("- вывод", md)
        self.assertNotIn(str(self.root), md)


class StubRun(unittest.TestCase):
    """A whole A/B on the stub backend and WARP: prepare, ranged runs, compares, the table (scripts.nr_ab_stub)."""

    @classmethod
    def setUpClass(cls) -> None:
        cli = CLI or (Path(os.environ["NR_AB_CLI"]) if os.environ.get("NR_AB_CLI") else None)  # set by __main__ (--cli) after the import
        if not cli or not cli.exists():
            raise unittest.SkipTest("no dlssvid: --cli or NR_AB_CLI")
        cls.cli = cli
        cls.work = WORK or Path(tempfile.mkdtemp())
        shutil.rmtree(cls.work, ignore_errors=True)
        cls.work.mkdir(parents=True)
        cls.clip = write_y4m(cls.work / "clip.y4m")
        cls.smoke = cls.work / "smoke.json"
        cls.smoke.write_text(json.dumps({"id": "smoke", "title": "Смоук", "baseline": "base", "metrics": ["base", "edit", "temporal", "time"],
                                         "variants": [{"name": "base", "expect": {"depth_used": True, "mv_used": True}},
                                                      {"name": "i05", "params": {"intensity": 0.5}},
                                                      {"name": "nodepth", "flags": ["--no-depth-guide"], "expect": {"depth_used": False, "mv_used": True}},
                                                      {"name": "mv", "flags": ["--mv-scale", "auto", "--create-with-mv"]}]}), encoding="utf-8")
        cls.out = cls.work / "out"
        cls.result = cls.run_ab(["--variants", str(cls.smoke), "--experiment", "E2", "--start", "4", "--frames", "3", "--warmup", "2", "--out", str(cls.out)])

    @classmethod
    def run_ab(cls, args: list) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-I", str(ROOT / "scripts" / "nr_ab.py"), "--cli", str(cls.cli), "--clip", str(cls.clip), "--prepare",
               "--prepare-backends", "stub,stub,bicubic", "--backend", "stub", "--warp"] + args
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_ab_on_stub_writes_the_range_and_the_table(self):
        r = self.result
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        res = json.loads((self.out / "results.json").read_text(encoding="utf-8"))["clip-stub"]["experiments"]
        smoke = res["smoke"]["variants"]
        for name, e in smoke.items():
            self.assertEqual(e["range"], [4, 6, 3], name)  # checked against the manifest by nr_ab
        self.assertLess(smoke["i05"]["vs_base"]["psnr_y"], 100.0)
        self.assertLess(smoke["i05"]["edit"]["mae_rgb"], smoke["base"]["edit"]["mae_rgb"])  # the stub's edit grows with intensity
        self.assertEqual(smoke["nodepth"]["vs_base"]["psnr_y"], 100.0)  # the stub ignores the guides
        self.assertEqual(smoke["mv"]["vs_base"]["psnr_y"], 100.0)
        self.assertIs(smoke["nodepth"]["nr"]["depth_used"], False)
        self.assertIs(smoke["base"]["nr"]["depth_used"], True)
        self.assertEqual(smoke["mv"]["nr"]["mv_scale"], [2.0, 2.0])  # mv 96x64 -> model 192x128 (bicubic x2)
        self.assertLess(smoke["base"]["temporal"]["psnr_y"], 100.0)
        self.assertIsNotNone(res["smoke"]["input_temporal"])
        self.assertIsInstance(smoke["base"]["nr"]["ms_per_frame_gpu"], (int, float))
        self.assertEqual(res["E2"]["warmup"], 0)  # still mode: no warm-up
        self.assertIs(res["E2"]["verdict"]["identity"], True)  # the stub at intensity 0 returns its input
        md = (self.out / "nr_ab.md").read_text(encoding="utf-8")
        for s in ("### smoke", "| i05 |", "∞", "E2"):
            self.assertIn(s, md)
        self.assertEqual(list(self.out.rglob("color_nr")), [])  # cleaned up without --keep
        prep = self.out / "prepare" / "clip" / "passes"
        self.assertFalse((self.out / "prepare" / "clip" / "prepare.mkv").exists())
        for name in ("color_sr", "depth_dlss", "mv_dlss"):
            self.assertTrue(nr_ab.covers(prep / name, 0, 6), name)

    def test_dry_run_prints_commands_only(self):
        out = self.work / "dry"
        r = self.run_ab(["--dry-run", "--experiment", "E7", "--start", "4", "--frames", "3", "--out", str(out)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(sum(" process " in x for x in lines), 1)
        nr = [x for x in lines if " nr " in x]
        self.assertEqual(len(nr), 10)
        self.assertTrue(all("--warmup 0" in x and "--params-file" in x for x in nr))
        self.assertEqual(sum(" compare " in x for x in lines), 37)  # 21 + 6 pairs and 10 edit compares
        self.assertFalse(out.exists() and any(out.rglob("*")))

    def test_unknown_key_is_refused_before_any_run(self):
        bad = self.work / "bad.json"
        bad.write_text(json.dumps({"id": "bad", "variants": [{"name": "a", "params": {"no_such_key": 1}}]}), encoding="utf-8")
        out = self.work / "bad_out"
        r = self.run_ab(["--variants", str(bad), "--out", str(out)])
        self.assertEqual(r.returncode, 1)
        self.assertIn("no_such_key", r.stderr)
        self.assertEqual(list(out.rglob("nr.log")) if out.exists() else [], [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cli")
    ap.add_argument("--work")
    known, rest = ap.parse_known_args()
    if known.cli:
        CLI = Path(known.cli)
    elif os.environ.get("DLSSVID_CLI"):
        CLI = Path(os.environ["DLSSVID_CLI"])
    elif (ROOT / "build" / "release" / "bin" / "dlssvid.exe").exists():
        CLI = ROOT / "build" / "release" / "bin" / "dlssvid.exe"
    WORK = Path(known.work) if known.work else None
    unittest.main(argv=[sys.argv[0], *rest])
