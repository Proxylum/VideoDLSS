#!/usr/bin/env python3
"""scripts/nr_ab.py (TASK-0050): the helpers that decide what an A/B reads and when a stored result answers again (ctest scripts.nr_ab, no
dlssvid), and a whole A/B on the stub backend and WARP over a generated Y4M clip (ctest scripts.nr_ab_stub: --cli <dlssvid> --work
<scratch>). Standard library only."""
from __future__ import annotations

import argparse
import contextlib
import io
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


def run_res(meta: dict) -> dict:
    return {"title": "t", "start": 0, "frames": 1, "warmup": 0, "still": True, "fixed": {}, "baseline": "a", "metrics": ["time"], "variants": {},
            "failed": {}, "meta": meta}


class Helpers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cat = {e.id: e for e in nr_ab.EXPERIMENTS}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def args(**kw) -> argparse.Namespace:
        a = argparse.Namespace(start=6, frames=0, warmup=6, only=[], backend="stub", format="exr", warp=True, nr_arg=[], keep=False, check_runs=3,
                               check_size="1280x720", check_repeat=0, check_features=4, force=False, prepare_backends="stub,stub,bicubic")
        vars(a).update(kw)
        return a

    def load(self, j: dict) -> nr_ab.Experiment:
        p = self.root / "v.json"
        p.write_text(json.dumps(j), encoding="utf-8")
        return nr_ab.load_variants(p)

    def dry(self, exp, inp, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            res = nr_ab.run_experiment(self.args(**kw), "dlssvid", inp, exp, {}, self.root / "out", True)
        return res, out.getvalue().splitlines()

    def test_normalize_and_merge_params(self):
        self.assertEqual(nr_ab.merge_params({"intensity": 1, "local-tone": 2}, {"temporal": 0.0}, {"intensity": 0.5, "local_tone": 0}),
                         {"intensity": 0.5, "local_tone": 0, "temporal": 0.0})
        self.assertEqual(nr_ab.normalize_params({"no_guides": True}), {"guides": False})
        self.assertEqual(nr_ab.merge_params({}, {"guides": False}, {"no-guides": False}), {"guides": True})  # the variant wins

    def test_style_is_normalized_as_dlssvid_reads_it(self):
        self.assertEqual(nr_ab.normalize_params({"style": 1}), {"style": "natural"})
        self.assertEqual(nr_ab.normalize_params({"style": "2"}), {"style": "cinematic"})
        self.assertEqual(nr_ab.normalize_params({"style": 4}), {"style": "4"})
        self.assertEqual(nr_ab.normalize_params({"style": "natural"}), {"style": "natural"})

    def test_run_params_leave_out_the_options_of_nr_ab(self):
        # --backend / --format of nr_ab win over a base file's: params.json must not claim another
        self.assertEqual(nr_ab.run_params({"backend": "ngx", "format": "png16", "intensity": 1.5}, {}, {}), {"intensity": 1.5})

    def test_lost_params_find_a_setting_that_did_not_reach_the_run(self):
        params = {"intensity": 0, "style": "3", "guides": False, "skin_structure": -1}
        self.assertEqual(nr_ab.lost_params(params, {"intensity": 0.0, "style": "3", "guides": False, "skin_structure": -1.0, "passes": 1}), [])
        self.assertEqual(nr_ab.lost_params(params, {"style": "3"}), [])  # left out by the run (another backend's, a default): not a loss
        lost = nr_ab.lost_params(params, {"intensity": 1.0, "style": "natural", "guides": False})
        self.assertEqual(len(lost), 2)
        self.assertTrue(lost[0].startswith("intensity=0"))

    def test_options_beside_the_variants_are_run_options_only(self):
        # a setting or an A/B switch in --nr-arg would make every variant one run; flags of a variant are the A/B switches
        self.assertEqual(nr_ab.check_options(["--skip-driver-check", "--dll-dir=D", "--params_block", "alloc"], nr_ab.NR_ARG_OK, "x"),
                         ["--skip-driver-check", "--dll-dir=D", "--params_block", "alloc"])
        for bad in (["--intensity=1"], ["--create-with-mv"], ["--local-structure", "0"], ["D"], ["--dll-dir"],
                    ["--dll-dir=", "--create-with-mv"], ["--params-block="], ["--skip-driver-check=1"]):  # CLI11 would take the next token
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                nr_ab.check_options(bad, nr_ab.NR_ARG_OK, "--nr-arg")
        nr_ab.check_options(["--mv-scale", "auto", "--create-with-mv"], nr_ab.AB_FLAGS, "flags")
        with self.assertRaises(RuntimeError):
            nr_ab.check_options(["--intensity", "2"], nr_ab.AB_FLAGS, "flags")
        self.assertEqual(nr_ab.option_value(["--x", "--dll-dir", "D"], "--dll-dir"), "D")
        self.assertEqual(nr_ab.option_value(["--dll_dir=E"], "--dll-dir"), "E")
        self.assertIsNone(nr_ab.option_value([], "--dll-dir"))
        self.assertEqual(nr_ab.expected_ab(["--mv-scale", "auto", "--create-with-mv"]), {"mv_scale": "auto", "create_with_mv": True})  # NrAbJson
        self.assertEqual(nr_ab.expected_ab(["--mv-scale=unit", "--no-depth-guide"]), {"depth_guide": False})
        self.assertEqual(nr_ab.expected_ab([]), {})
        help_text = ("--local-tone,--local_tone FLOAT\n--no-guides,--no_guides,!--guides\n--mv-scale,--mv_scale TEXT\n--start INT\n--no-auto\n"
                     "--dll-dir TEXT\n--auto-mask,--auto_mask,!--no-auto-mask,!--no_auto_mask\n--check-repeat,--check_repeat INT\n"
                     "--check-features,--check_features INT")
        keys = nr_ab.setting_keys(help_text)  # a run option in a params.json fails the run after the variants before it ran
        self.assertTrue({"local_tone", "guides", "auto_mask"} <= keys)
        self.assertFalse({"mv_scale", "start", "auto", "no_auto", "dll_dir", "no_auto_mask", "no_guides", "check_repeat", "check_features"} & keys)
        self.assertEqual(nr_ab.names_only(r"missing pass frame 12: D:\clips\mv_dlss\000012.exr"), "missing pass frame 12: 000012.exr")
        self.assertEqual(nr_ab.names_only("pass folder not found: /tmp/x/color_sr (no manifest)"), "pass folder not found: color_sr (no manifest)")
        self.assertEqual(nr_ab.names_only(r"missing pass frame 12: D:\My Clips\face\mv_dlss\000012.exr"), "missing pass frame 12: 000012.exr")
        self.assertEqual(nr_ab.names_only(r"nr: color D:\John Doe\clips\color_sr has frames 0..9, the run reads 4..12"),
                         "nr: color color_sr has frames 0..9, the run reads 4..12")  # spaces in the folders, unquoted
        self.assertEqual(nr_ab.names_only("unit / auto"), "unit / auto")
        for e in nr_ab.EXPERIMENTS:  # the catalogue keeps the same rules as a variants file
            for v in e.variants:
                nr_ab.check_options(v.flags, nr_ab.AB_FLAGS, f"{e.id}/{v.name}")
                self.assertFalse(set(v.params) & set(nr_ab.RUN_KEYS), v.name)

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
        self.assertEqual(nr_ab.frame_index(sr, 4), 4)
        manifest(sr, 2, 9)
        self.assertEqual(nr_ab.frame_index(sr, 4), 2)  # compare reads a pass from its first frame
        self.assertEqual(nr_ab.frame_index(None, 4), 4)  # a video from frame 0
        edit = nr_ab.compare_command("dlssvid", "out/color_nr", sr, "e.json", offset=nr_ab.frame_index(sr, 4))
        self.assertIn("--offset 2", " ".join(edit))
        nxt = nr_ab.compare_command("dlssvid", "a", "a", "n.json", start=4, offset=1, frames=2)
        self.assertIn("--start 4 --offset 1 --frames 2", " ".join(nxt))
        for c in (edit, nxt):
            self.assertFalse(any(x.startswith("-") and x[1:].isdigit() for x in map(str, c)))  # no negative numbers in an argv

    def test_edit_is_measured_against_what_the_variant_read(self):
        bic = manifest(self.root / "bic" / "color_sr", 2, 40)
        exp = nr_ab.Experiment("x", "x", [nr_ab.Variant("a"), nr_ab.Variant("b", inputs={"color": str(bic)})], "a", metrics=("edit",))
        res, lines = self.dry(exp, nr_ab.Inputs(Path("clip.mp4")))
        edits = [x for x in lines if " compare " in x and "cmp_input" in x]
        self.assertEqual(len(edits), 2)
        self.assertIn("clip.mp4", edits[0])  # a: the run's input, the video from frame 0
        self.assertIn("--offset 6", edits[0])
        self.assertIn(str(bic), edits[1])  # b: its own color_sr, indexed from its first frame
        self.assertIn("--offset 4", edits[1])
        self.assertIn("--frames 12", edits[1])  # every compare bounded to the frames written

    def test_the_input_row_is_what_the_baseline_read(self):
        bic = manifest(self.root / "bic" / "color_sr", 2, 40)
        exp = nr_ab.Experiment("x", "x", [nr_ab.Variant("a", inputs={"color": str(bic)}), nr_ab.Variant("b")], "a", metrics=("temporal",))
        inp = nr_ab.Inputs(Path("clip.mp4"))
        _res, lines = self.dry(exp, inp)
        row = [x for x in lines if "cmp_input_next" in x]
        self.assertEqual(len(row), 1)
        self.assertIn(str(bic), row[0])
        self.assertIn("--start 4 --offset 1 --frames 11", row[0])
        res = {"variants": {"a": {"nr": {}}, "b": {"nr": {}}}}
        meta = nr_ab.experiment_meta(res, "1.0", {"dlssvid": "e"}, inp, exp)
        self.assertEqual(meta["inputs"], "color_sr; свои входы у a")

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

    def test_read_compare_caps_identical_and_checks_the_count(self):
        f = self.root / "c.json"
        f.write_text(json.dumps({"summary": {"frames": 3, "psnr_y_mean": 100.0, "psnr_y_min": None, "psnr_rgb_mean": 100.0, "ssim_y_mean": 1.0}}),
                     encoding="utf-8")
        c = nr_ab.read_compare(f)
        self.assertEqual((c["psnr_y"], c["psnr_y_min"]), (100.0, 100.0))
        self.assertEqual(nr_ab.fmt_db(c["psnr_y"]), "∞")
        self.assertIsNone(c["mae_rgb"])
        self.assertEqual(nr_ab.fmt_num(c["mae_rgb"]), "—")
        self.assertEqual(nr_ab.fmt_db(38.123), "38.12")
        self.assertEqual(nr_ab.read_compare(f, 3)["frames"], 3)
        with self.assertRaises(RuntimeError):  # a pair missing or one too many: the metric would mix runs
            nr_ab.read_compare(f, 4)

    def test_no_output_of_an_earlier_run_survives(self):
        def fails(*_a, **_k):
            raise RuntimeError("compare failed")

        old = self.root / "cmp.json"
        old.write_text(json.dumps({"summary": {"frames": 3}}), encoding="utf-8")
        run, nr_ab.run = nr_ab.run, fails
        try:
            with self.assertRaises(RuntimeError):
                nr_ab.measure("dlssvid", "a", "b", old, 3, False)
            self.assertFalse(old.exists())  # removed before the command, so a failure cannot report it
            edir = self.root / "E8"
            edir.mkdir()
            (edir / "check_0.json").write_text(json.dumps({"ok": True, "init_ms": 1.0, "gpu": "G"}), encoding="utf-8")
            res = nr_ab.run_check(self.args(check_runs=1), "dlssvid", nr_ab.Inputs(Path("clip.mp4")), self.cat["E8"], edir, False)
            self.assertEqual(res["checks"], [{"ok": False, "json": False, "init_ms": None, "create_ms": None, "device_ms": None, "vram_mib": None, "run_s": None}])
            self.assertIs(res["measured"], False)  # no JSON: not an answer - reusable() measures it again
        finally:
            nr_ab.run = run
        if os.name == "nt":  # an open file cannot be deleted on Windows: the frames left behind stop the run
            folder = manifest(self.root / "color_nr", 0, 0)
            with open(folder / "frame.exr", "wb"):
                with self.assertRaises(RuntimeError):
                    nr_ab.clear(folder)
        nr_ab.clear(self.root / "color_nr")
        self.assertFalse((self.root / "color_nr").exists())

    def test_a_hung_call_is_killed(self):
        old, nr_ab.TIMEOUT = nr_ab.TIMEOUT, 1.0
        try:
            log = self.root / "hung.log"
            t0 = __import__("time").perf_counter()
            with self.assertRaises(RuntimeError) as e:
                nr_ab.run([sys.executable, "-c", "import time; time.sleep(30)", "x"], log, False)
            self.assertIn("timed out", str(e.exception))
            self.assertLess(__import__("time").perf_counter() - t0, 15)  # killed, not waited for
            self.assertIn("killed after", log.read_text(encoding="utf-8"))
        finally:
            nr_ab.TIMEOUT = old

    def test_check_runs_record_recreations_and_features(self):
        def fake(cmd, log, dry):  # what `nr --check` writes: the JSON beside the command
            out = Path(cmd[cmd.index("--json") + 1])
            if "--check-repeat" in cmd:
                n = int(cmd[cmd.index("--check-repeat") + 1])
                recs = [{"index": i, "ok": True, "result": "Success", "create_ms": 200.0 + i, "vram_after_release_bytes": 2 << 30,
                         "vram_after_create_bytes": (2 << 30) + (504 << 20)} for i in range(1, n + 1)]
                out.write_text(json.dumps({"ok": True, "gpu": "G", "vram_usage_after_init_bytes": (2 << 30) + (504 << 20), "recreations": recs}), encoding="utf-8")
            elif "--check-features" in cmd:
                feats = [{"index": i, "ok": i < 3, "create_ms": 210.0, "vram_feature_bytes": 940 << 20, "vram_usage_bytes": (2 << 30) + i * (940 << 20)}
                         for i in (1, 2, 3)]
                out.write_text(json.dumps({"ok": True, "gpu": "G", "vram_budget_bytes": 14 << 30, "features": feats, "features_alive": 2, "features_stop": "failure"}),
                               encoding="utf-8")
            else:
                out.write_text(json.dumps({"ok": True, "gpu": "G", "init_ms": 1400.0, "create_ms": 240.0, "vram_feature_bytes": 504 << 20}), encoding="utf-8")
            return 2.5

        run, nr_ab.run = nr_ab.run, fake
        try:
            inp = nr_ab.Inputs(Path("clip.mp4"))
            e8 = nr_ab.run_check(self.args(check_runs=1, check_repeat=12, check_size="2560x1440"), "dlssvid", inp, self.cat["E8"], self.root / "E8", False)
            self.assertEqual(e8["size"], "2560x1440")
            self.assertEqual((e8["repeat"]["requested"], e8["repeat"]["ok"]), (12, 12))
            self.assertIs(e8["verdict"]["phase2"], True)  # all ok, no slowdown, no drift
            self.assertIn("условие фазы 2 превью выполнено", e8["verdict"]["text"])
            md = nr_ab.render_experiment("E8", e8)
            self.assertIn("`--check-repeat 12`: 12 пересозданий ok", md)
            f = nr_ab.run_check(self.args(check_features=3, check_size="3840x2160"), "dlssvid", inp, self.cat["E6f"], self.root / "E6f", False)
            self.assertEqual((f["alive"], f["stop"]), (2, "failure"))
            self.assertEqual(f["verdict"]["mib_per_feature"], 940.0)
            self.assertIn("наблюдаемый предел", f["verdict"]["text"][0])
            self.assertIn("| 3 | нет |", nr_ab.render_experiment("E6f", f))
        finally:
            nr_ab.run = run
        def silent(cmd, log, dry):  # a timeout kill or a crash: no JSON at all
            raise RuntimeError("timed out")

        def first_fails(cmd, log, dry):
            Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps({"ok": False, "features": [], "features_alive": 0, "features_stop": ""}), encoding="utf-8")
            raise RuntimeError("failed (1)")

        for k, (fake_run, word) in enumerate(((silent, "не измерено"), (first_fails, "не создалась"))):
            nr_ab.run = fake_run
            try:
                f = nr_ab.run_check(self.args(check_features=4), "dlssvid", inp, self.cat["E6f"], self.root / f"E6f_{k}", False)
                e8 = nr_ab.run_check(self.args(check_runs=1, check_repeat=5), "dlssvid", inp, self.cat["E8"], self.root / f"E8_{k}", False)
            finally:
                nr_ab.run = run
            self.assertIn(word, f["verdict"]["text"][0])
            self.assertNotIn("все запрошенные созданы", f["verdict"]["text"][0])
            self.assertIs(f["measured"], False)
            self.assertFalse(nr_ab.reusable({**f, "sig": {"common": {}, "variants": {}}}, {"common": {}, "variants": {}}, [], False))
            self.assertIsNone(e8["verdict"]["phase2"])  # the re-creations were not measured: neither met nor failed
            self.assertIn("не измерено", nr_ab.render_experiment("E8", e8))
        sr = nr_ab.Inputs(Path("clip.mp4"), color=manifest(self.root / "color_sr", 0, 0))  # manifest() writes 8x8
        self.assertEqual(nr_ab.check_size(self.args(check_size="2560x1440"), sr), "2560x1440")  # an explicit --check-size beats color_sr
        self.assertEqual(nr_ab.check_size(self.args(check_size=None), sr), "8x8")
        self.assertEqual(nr_ab.check_size(self.args(check_size=None), nr_ab.Inputs(Path("clip.mp4"))), "1280x720")
        stopped = {"requested": 100, "ok": 17, "stop": "growth", "create_ms": [100.0] * 17, "vram_after_release_mib": [2048.0] * 17,
                   "vram_after_create_mib": [2552.0] * 17, "usage_after_init_mib": 2552.0}
        self.assertIs(nr_ab.phase2(stopped)[0], False)  # a release that frees nothing
        rep = {"requested": 20, "ok": 20, "usage_after_init_mib": 2552.0, "create_ms": [200.0] * 10 + [400.0] * 10,
               "vram_after_release_mib": [2048.0] * 20, "vram_after_create_mib": [2552.0] * 20}
        self.assertIs(nr_ab.phase2(rep)[0], False)  # twice slower at the end
        self.assertIs(nr_ab.phase2({**rep, "create_ms": [200.0] * 20, "vram_after_release_mib": [2048.0] * 19 + [2200.0]})[0], False)  # memory not freed
        self.assertIs(nr_ab.phase2({**rep, "create_ms": [200.0] * 20, "ok": 19})[0], False)
        self.assertEqual(nr_ab.phase2(None), (None, []))

    def test_variants_file_validation(self):
        ok = self.load({"id": "s", "still": True, "metrics": ["base", "time", "pairs"], "variants": [{"name": "a"}, {"name": "b", "params": {"local-tone": 0}}],
                        "groups": [["a", "b"]]})
        self.assertEqual(ok.baseline, "a")
        self.assertEqual(ok.fixed, {"guides": False, "temporal": 0.0, "intensity": 1.0})  # PR-7: the strength pinned, like the catalogue's
        self.assertEqual(ok.variants[1].params, {"local_tone": 0})
        for bad in ({"variants": [{"name": "a"}, {"name": "a"}]}, {"variants": [{"name": "a/b"}]}, {"baseline": "z", "variants": [{"name": "a"}]},
                    {"metrics": ["speed"], "variants": [{"name": "a"}]}, {"metrics": ["pairs"], "groups": [["a", "q"]], "variants": [{"name": "a"}]},
                    {"variants": []},
                    {"variants": [{"name": "a", "optional": True}, {"name": "b"}]},  # the baseline: every other variant is measured against it
                    {"needs": ["color"], "variants": [{"name": "a"}]},
                    {"groups": [["a", "b"]], "variants": [{"name": "a"}, {"name": "b"}]},  # groups without the pairs metric do nothing
                    {"metrics": ["pairs"], "variants": [{"name": "a"}]},
                    {"variants": [{"name": "a"}, {"name": "b", "params": {"format": "png16"}}]},  # nr_ab's own options: one run twice
                    {"fixed": {"backend": "stub"}, "variants": [{"name": "a"}]},
                    {"variants": [{"name": "a"}, {"name": "b", "flags": ["--intensity", "2"]}]},  # a setting as a flag
                    {"variants": [{"name": "a"}, {"name": "b", "flags": ["--mv-scale", "half"]}]},
                    {"variants": [{"name": "a"}, {"name": "b", "inputs": {"colour": "x"}}]},  # would run on the run's colour under another name
                    {"variants": [{"name": "A"}, {"name": "a"}]}, {"variants": [{"name": "a"}, {"name": "b."}]},  # one folder on Windows
                    {"variants": [{"name": "a", "expect": {"color_source": "color_sr"}}]}):  # a stat nr_ab does not read: never met
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    self.load(bad)

    def test_catalogue(self):
        cat = self.cat
        self.assertEqual(len(cat["E1"].variants), 8)
        self.assertEqual(cat["E1"].baseline, "i1")
        for i in ("E1", "E2", "E7", "E10", "E11"):
            self.assertEqual({k: cat[i].fixed[k] for k in ("guides", "temporal")}, {"guides": False, "temporal": 0.0}, i)
        for i in ("E3", "E4", "E5", "E6"):
            self.assertEqual(cat[i].fixed, {"temporal": 0.0, "intensity": 1.0}, i)
        self.assertEqual(cat["E7"].groups, [["default", "natural", "cinematic"]])  # PR-2: 3..6 = cinematic, no preset (TASK-0052)
        self.assertEqual({v.params["style"] for v in cat["E7"].variants}, {"default", "natural", "cinematic"})
        self.assertFalse(any("preset" in v.params for e in nr_ab.EXPERIMENTS for v in e.variants))
        base = {"style": "cinematic"}  # a --params / --project-params style does not leak into the variants
        for v in cat["E7"].variants:
            self.assertEqual(nr_ab.run_params(base, cat["E7"].fixed, v.params)["style"], v.name)
        self.assertEqual(nr_ab.run_expect(cat["E3"], cat["E3"].variants[2]), {"mv_used": True})  # the experiment needs vectors: every run binds them
        self.assertEqual(nr_ab.run_expect(cat["E5"], cat["E5"].variants[1]), {"depth_used": False, "mv_used": True})
        self.assertEqual(nr_ab.run_expect(cat["E1"], cat["E1"].variants[0]), {})
        self.assertEqual([v.optional for v in cat["E6"].variants], [False, False, True, True])
        self.assertTrue(cat["E8b"].unsupported)
        self.assertEqual((cat["E8"].check, cat["E6f"].check, cat["E6f"].kind), ("runs", "features", "check"))
        self.assertEqual([v.name for v in cat["E11"].variants][-3:], ["am_follow", "am_s0", "am_s2"])
        self.assertTrue(all(v.params["auto_mask"] for v in cat["E11"].variants[-3:]))  # the control overrides the fixed auto_mask false
        self.assertEqual(cat["E9"].requires, ("--scene-cut",))
        self.assertEqual((cat["E3"].frames, cat["E4"].frames), (30, 30))

    def test_verdicts(self):
        def v(**variants):
            return {"variants": variants}

        # PR-7: E1 checks our blend - up to 1 the edit is x times the edit at 1, above 1 the clipped share is shown
        edits = lambda ms: v(**{f"i{x}": {"edit": {"mae_rgb": m}, "nr": {"ms_per_frame_gpu": 16.0}}
                                for x, m in zip(("0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"), ms)})
        e1 = nr_ab.verdict_e1(edits((0, .0101, .0199, .0302, .04, .049, .057, .07)))
        self.assertIs(e1["linear"], True)
        self.assertLess(e1["worst"], 0.01)
        self.assertAlmostEqual(e1["ratios"]["2"], 1.75)
        self.assertIs(e1["monotone"], True)
        self.assertIn("обрезано 12%", " ".join(e1["text"]))  # 1 - 1.75 / 2
        bad = nr_ab.verdict_e1(edits((0, .01, .02, .015, .04, .05, .06, .08)))  # NGX's reading of the strength would give this
        self.assertIs(bad["linear"], False)
        self.assertIs(bad["monotone"], False)
        self.assertIn("регрессия", " ".join(bad["text"]))
        self.assertIsNone(nr_ab.verdict_e1(v(**{"i1.5": {"edit": {"mae_rgb": 0.05}}}))["linear"])
        slow = edits((0, .01, .02, .03, .04, .05, .06, .08))
        slow["variants"]["i2"]["nr"]["ms_per_frame_gpu"] = 30.0  # one run's noise (PR-2 E1: x1.6..x4.8 between the same work): listed, not judged
        self.assertIn("16.0–30.0 мс/кадр", " ".join(nr_ab.verdict_e1(slow)["text"]))
        self.assertNotIn("регрессия", " ".join(nr_ab.verdict_e1(slow)["text"]))
        # E2: at 0 the model is not called - identity, the bypass reported, the GPU time beside 1
        e2 = nr_ab.verdict_e2(v(i0={"edit": {"psnr_y": 100.0}, "nr": {"bypass": True, "ms_per_frame_gpu": 0.4}}, i1={"nr": {"ms_per_frame_gpu": 16.0}}))
        self.assertEqual((e2["identity"], e2["bypass"]), (True, True))
        self.assertEqual(e2["ms_gpu"], [0.4, 16.0])
        self.assertNotIn("до PR-7", " ".join(e2["text"]))
        old = nr_ab.verdict_e2(v(i0={"edit": {"psnr_y": 7.6}, "nr": {}}))  # PR-2's black frame: the build before PR-7
        self.assertIs(old["identity"], False)
        self.assertIn("регрессия", " ".join(old["text"]))
        self.assertIn("до PR-7", " ".join(old["text"]))
        static = {"input_temporal": {"psnr_y": 100.0, "psnr_y_min": 100.0}}
        e4 = v(null={"temporal": {"psnr_y": 46.4}}, create_mv={"temporal": {"psnr_y": 47.6}})
        self.assertTrue(nr_ab.verdict_e4({**e4, **static})["create_better"])
        for moving in ({"input_temporal": {"psnr_y": 25.0, "psnr_y_min": 24.0}},  # a pan measures the motion (§2e: identical frames)
                       {"input_temporal": {"psnr_y": 66.0, "psnr_y_min": 30.0}}):  # a few identical pairs at 100 dB lift the mean, not the minimum
            r = nr_ab.verdict_e4({**e4, **moving})
            self.assertIsNone(r["create_better"])
            self.assertTrue(any("не статичен" in t for t in r["text"]))
        self.assertTrue(nr_ab.verdict_e5(v(no_depth={"vs_base": {"psnr_y": 52}}))["no_difference"])
        self.assertEqual(nr_ab.verdict_e7({"pairs": [{"a": "a", "b": "b", "psnr_y": 50}, {"a": "a", "b": "c", "psnr_y": 30}]})["same_pairs"], 1)
        part = nr_ab.verdict_e7({"pairs": [{"a": "a", "b": "c", "psnr_y": 30}], "pairs_missing": [["a", "b"]]})
        self.assertNotIn("все пары различаются", part["text"])  # a subset cannot say that
        self.assertTrue(any("неполный" in t for t in part["text"]))
        sub = nr_ab.verdict_e7({"groups": [["a", "b", "c"]], "pairs": [{"a": "a", "b": "b", "psnr_y": 30}]})  # --only: b, c never paired
        self.assertNotIn("все пары различаются", sub["text"])
        self.assertEqual(sub["missing_pairs"], 2)
        full = nr_ab.verdict_e7({"groups": [["a", "b"]], "pairs": [{"a": "a", "b": "b", "psnr_y": 30}]})
        self.assertEqual(full["text"], ["все пары различаются"])
        flat = v(**{n: {"vs_base": {"psnr_y": 46.5}} for n in ("s0", "s1", "s2")})
        self.assertTrue(nr_ab.verdict_e11({**flat, "pairs": [{"a": "am_follow", "b": "am_s0", "psnr_y": 38.0}]})["no_effect"])  # acts with the mask
        blind = nr_ab.verdict_e11({**flat, "pairs": [{"a": "am_follow", "b": "am_s0", "psnr_y": 60.0}, {"a": "am_follow", "b": "am_s2", "psnr_y": 58.0}]})
        self.assertIsNone(blind["no_effect"])  # no effect with the mask either: the clip proves nothing
        self.assertIs(nr_ab.verdict_e11({**v(s0={"vs_base": {"psnr_y": 30}}, s1={"vs_base": {"psnr_y": 50}}, s2={"vs_base": {"psnr_y": 50}})})["no_effect"], False)
        self.assertTrue(nr_ab.names_only("[14:46:39.828] [error] nr: nr.passes must be one of 1 | 2 (got 3)").startswith("nr: nr.passes"))
        e3 = nr_ab.verdict_e3(v(unit={"temporal": {"psnr_y": 40}, "nr": {"mv_scale": [1, 1]}}, auto={"temporal": {"psnr_y": 41}, "nr": {"mv_scale": [1, 1]}}))
        self.assertTrue(any("ничего не меряет" in t for t in e3["text"]))
        e3 = v(unit={"temporal": {"psnr_y": 44.0}}, auto={"temporal": {"psnr_y": 46.0}}, target={"temporal": {"psnr_y": 46.5}})
        for inp in (static, {"input_temporal": {"psnr_y": 25.0, "psnr_y_min": 24.0}}):  # zero vectors on identical frames: units cannot matter
            r = nr_ab.verdict_e3({**e3, **inp})
            self.assertIsNone(r["auto_better"])
            self.assertTrue(any("компенсаци" in t for t in r["text"]))
        e8 = nr_ab.verdict_e8({"checks": [{"ok": False}, {"ok": True, "init_ms": 1500.0, "create_ms": 200.0, "vram_mib": 249.0, "run_s": 3.1}]})
        self.assertIn("сбой в проверках 1", e8["text"])  # numbered as the table's «Прогон»
        self.assertEqual(e8["init_ms"], 1500.0)

    def test_render_table(self):
        res = {"title": "Проверка", "start": 4, "frames": 3, "warmup": 2, "still": False, "fixed": {"temporal": 0.0}, "baseline": "base",
               "metrics": ["base", "edit", "temporal", "time"],
               "variants": {"base": {"nr": {"ms_per_frame_gpu": 1.0, "ms_per_frame_wall": 2.0}, "run_s": 0.5, "params": {"temporal": 0.0},
                                     "edit": {"psnr_y": 30.0, "psnr_y_min": 29.0, "mae_rgb": 0.01}, "temporal": {"psnr_y": 40.0, "psnr_y_min": 39.0}},
                            "x": {"nr": {}, "run_s": 0.4, "params": {"intensity": 0.5}, "vs_base": {"psnr_y": 100.0, "psnr_y_min": 100.0, "ssim_y": 1.0},
                                  "edit": None, "temporal": None}},
               "failed": {"p3": "nr.passes must be within 1..2", "t": f"missing pass frame 12: {self.root / 'mv_dlss' / '000012.exr'}"},
               "input_temporal": {"psnr_y": 41.0, "psnr_y_min": 40.0},
               "verdict": {"text": ["вывод"]}}
        md = nr_ab.render_experiment("smoke", res)
        self.assertIn("### smoke — Проверка", md)
        self.assertIn("| base |", md)
        self.assertIn("| x | intensity=0.5 |", md)
        self.assertIn("| base | — |", md)  # its params are the fixed ones, in the heading
        self.assertIn("t: не выполнен — missing pass frame 12: 000012.exr", md)
        self.assertIn("база", md)
        self.assertIn("∞", md)
        self.assertIn("—", md)
        self.assertIn("p3: не выполнен", md)
        self.assertIn("- вывод", md)
        self.assertNotIn(str(self.root), md)
        res["variants"]["x"]["inputs"] = {"mv": str(self.root / "mv_dlss")}  # a variants file's own folder: its name only ([hygiene])
        res["variants"]["x"]["params"] = {"intensity": 0.5, "temporal": 0.4}  # an override of a fixed key: the heading says 0, the row 0.4
        res["pairs_missing"] = [["base", "x"]]
        md = nr_ab.render_experiment("smoke", res)
        still = nr_ab.render_experiment("s", {**res, "still": True, "fixed": {**nr_ab.STILL, "auto_mask": False}})  # E11: every fixed value shown
        self.assertIn("still (guides выкл, temporal 0, сила 1), auto_mask False", still)
        self.assertIn("mv: mv_dlss", md)
        self.assertIn("temporal=0.4", md)
        self.assertIn("Пары без замера", md)
        self.assertNotIn(str(self.root), md)

    def test_render_run_keeps_each_experiment_build(self):
        m = {"version": "1.0", "gpu": "G", "driver": "616", "dll_sha256": "aaaaaaaa11", "inputs": "color_sr", "masks": ""}
        md = nr_ab.render_run("run", {"E1": run_res(m), "E2": run_res(m)})
        self.assertIn("## run — dlssvid 1.0, G, драйвер 616, nvngx_dlssnr.dll sha256 aaaaaaaa…", md)
        self.assertEqual(md.count("Вход NR: color_sr; маски: выкл."), 1)
        self.assertNotIn("Сборка:", md)
        md = nr_ab.render_run("run", {"E1": run_res(m), "E2": run_res({**m, "dll_sha256": "bbbbbbbb22", "exe_sha256": "cccccccc33", "masks": "mask_face"})})
        self.assertNotIn("sha256", md.splitlines()[0])  # E1 reused from results.json, E2 of another build: each its own line
        self.assertIn("Сборка: dlssvid 1.0, G, драйвер 616, nvngx_dlssnr.dll sha256 aaaaaaaa…", md)
        self.assertIn("Сборка: dlssvid 1.0 (exe cccccccc…), G, драйвер 616, nvngx_dlssnr.dll sha256 bbbbbbbb…", md)
        self.assertIn("маски: mask_face", md)

    def test_signature_follows_the_content_and_the_build(self):
        e1, e3 = self.cat["E1"], self.cat["E3"]
        sr, mv = manifest(self.root / "color_sr", 0, 40), manifest(self.root / "mv_dlss", 0, 40)
        inp = nr_ab.Inputs(Path("clip.mp4"), color=sr, mv=mv)
        build = {"dlssvid": "exe1"}
        sig = lambda a=None, i=inp, e=e1, base=None, b=build: nr_ab.run_signature(a or self.args(), i, e, base or {}, b)
        s = sig()
        self.assertEqual(s, sig())
        self.assertEqual(s, sig(self.args(warmup=2)))  # still mode runs without a warm-up
        self.assertNotEqual(sig(e=e3), sig(self.args(warmup=2), e=e3))
        with_target = nr_ab.Inputs(Path("clip.mp4"), color=sr, mv=mv, mv_target=manifest(self.root / "mvt", 0, 40))
        self.assertEqual(sig(i=with_target), s)  # only E3 C reads --mv-target-dir
        t0, t1 = sig(e=e3), sig(i=with_target, e=e3)
        self.assertEqual(t0["common"], t1["common"])
        self.assertEqual({n for n in t0["variants"] if t0["variants"][n] != t1["variants"][n]}, {"target"})
        changed = [sig(base={"local_tone": 0.5}), sig(b={"dlssvid": "exe2"}), sig(self.args(start=8)), sig(self.args(nr_arg=["--skip-driver-check"])),
                   sig(i=nr_ab.Inputs(Path("clip.mp4"), color=manifest(self.root / "sr2", 0, 41), mv=mv)),
                   sig(e=nr_ab.Experiment("E1", e1.title, e1.variants, "i0", e1.fixed, e1.metrics)),  # another baseline
                   sig(e=nr_ab.Experiment("E1", e1.title, e1.variants, e1.baseline, e1.fixed, ("base",)))]
        for c in changed:
            self.assertNotEqual(c, s)
        manifest(sr, 0, 39)  # the same folder rewritten (--prepare, another backend): another input
        self.assertNotEqual(sig(), s)

    def test_reuse_and_what_an_only_run_keeps(self):
        e1 = self.cat["E1"]
        inp = nr_ab.Inputs(Path("clip.mp4"), color=manifest(self.root / "color_sr", 0, 40))
        sig = nr_ab.run_signature(self.args(), inp, e1, {}, {"dlssvid": "x"})
        prev = {"sig": sig, "variants": {"i1": {}, "i2": {}}, "failed": {"i0": "x"}}
        self.assertTrue(nr_ab.reusable(prev, sig, ["i1", "i2", "i0"], False, "i1"))
        self.assertFalse(nr_ab.reusable(prev, sig, ["i1", "i0.5"], False, "i1"))  # a stored --only subset does not answer more
        self.assertFalse(nr_ab.reusable(prev, sig, ["i1"], True, "i1"))
        self.assertFalse(nr_ab.reusable({**prev, "pairs_missing": [["i1", "i2"]]}, sig, ["i1"], False, "i1"))
        self.assertFalse(nr_ab.reusable({"variants": {"i1": {}}}, sig, ["i1"], False, "i1"))  # stored before signatures: redone
        self.assertFalse(nr_ab.reusable(None, sig, [], False))
        other = json.loads(json.dumps(sig))
        other["variants"]["i2"]["params"]["intensity"] = 3  # a variant changed since: its stored row is stale
        self.assertFalse(nr_ab.reusable(prev, other, ["i1"], False, "i1"))
        self.assertEqual(nr_ab.valid_stored(prev, other, "i1"), {"i1", "i0"})
        other["variants"]["i1"]["params"]["intensity"] = 3  # the baseline changed: nothing stored is measured against it
        self.assertEqual(nr_ab.valid_stored(prev, other, "i1"), set())
        self.assertEqual(nr_ab.valid_stored(prev, nr_ab.run_signature(self.args(), inp, e1, {}, {"dlssvid": "y"}), "i1"), set())
        check = nr_ab.run_signature(self.args(), inp, self.cat["E8"], {}, {"dlssvid": "x"})
        self.assertTrue(nr_ab.reusable({"sig": check, "checks": []}, check, [], False))

    def test_an_only_merge_keeps_a_bypass_run(self):
        # PR-7 (review): i0 calls no model and reports no GPU / driver / DLL - the merge compared them with the baseline's and dropped it
        e2 = self.cat["E2"]
        inp = nr_ab.Inputs(Path("clip.mp4"), color=manifest(self.root / "color_sr", 0, 40))
        sig = nr_ab.run_signature(self.args(), inp, e2, {}, {"dlssvid": "x"})
        gpu = {"gpu": "G", "driver": "D", "dll_sha256": "S"}
        prev = {"sig": sig, "variants": {"i0": {"nr": {"bypass": True, "passes": 0}}, "i1": {"nr": gpu}}, "failed": {}}
        with contextlib.redirect_stdout(io.StringIO()):
            res = nr_ab.run_experiment(self.args(only=["i1"]), "dlssvid", inp, e2, {}, self.root / "out", True, prev, sig)
        self.assertEqual(set(res["variants"]), {"i0", "i1"})
        other = {**prev, "variants": {**prev["variants"], "i0": {"nr": {"gpu": "H", "driver": "D", "dll_sha256": "S"}}}}
        with contextlib.redirect_stdout(io.StringIO()):
            res = nr_ab.run_experiment(self.args(only=["i1"]), "dlssvid", inp, e2, {}, self.root / "out", True, other, sig)
        self.assertEqual(set(res["variants"]), {"i1"})  # a model run on another GPU still goes

    def test_the_model_experiments_pin_the_strength(self):
        # PR-7 (review): a base intensity is our blend after the model - it would scale every difference between variants
        for e in nr_ab.EXPERIMENTS:
            if e.kind == "run":
                for v in e.variants:
                    self.assertEqual(nr_ab.run_params({"intensity": 0.0}, e.fixed, v.params).get("intensity"), v.params.get("intensity", 1.0), (e.id, v.name))
        for i in ("E1", "E2"):
            self.assertEqual({k: self.cat[i].fixed[k] for k in nr_ab.TONE}, nr_ab.TONE)  # the edit is measured against the input

    def test_e3_c_without_its_vectors_is_not_run(self):
        inp = nr_ab.Inputs(Path("clip.mp4"), mv=Path("m"))
        res, lines = self.dry(self.cat["E3"], inp)
        self.assertNotIn("target", res["variants"])
        self.assertIn("--mv-target-dir", res["failed"]["target"])
        self.assertEqual(sum(" nr " in x for x in lines), 2)
        self.assertTrue(any("нет варианта C" in t for t in nr_ab.verdict_e3(res)["text"]))
        must = nr_ab.Experiment("m", "m", [nr_ab.Variant("a"), nr_ab.Variant("b", inputs={"mv": "@mv_target"})], "a")
        with self.assertRaises(RuntimeError):  # not optional: the experiment stops before any run
            self.dry(must, inp)

    def test_masks_need_mask_passes(self):
        passes = self.root / "passes"
        a = self.args(clip="clip.mp4", passes=str(passes), project=None, prepare=False, color_dir=None, depth_dir=None, mv_dir=None, masks=True,
                      mv_target_dir=None)
        with self.assertRaises(RuntimeError):  # the report would say masks over runs that had none
            nr_ab.resolve_inputs(a, self.root, "dlssvid", False)
        manifest(passes / "mask_face", 0, 9)
        self.assertEqual(nr_ab.resolve_inputs(a, self.root, "dlssvid", False).masks, passes)
        a.prepare = True
        with self.assertRaises(RuntimeError):
            nr_ab.resolve_inputs(a, self.root, "dlssvid", False)
        a.masks = False
        with self.assertRaises(RuntimeError):  # --prepare reads the passes it makes: --passes beside it would be dropped without a word
            nr_ab.resolve_inputs(a, self.root, "dlssvid", False)
        a.passes = None
        inp = nr_ab.resolve_inputs(a, self.root, "dlssvid", False)
        self.assertIsNone(inp.masks)
        self.assertEqual(inp.color, nr_ab.prepare_root(self.root, a, Path("clip.mp4")) / "passes" / "color_sr")
        self.assertTrue(nr_ab.prepare_root(self.root, a, Path("clip.mp4")).name.startswith("clip-"))
        self.assertTrue(nr_ab.prepare_root(self.root, a, Path("clip.mp4")).name.endswith("-stub-stub-bicubic"))
        name = nr_ab.prepare_root(self.root, self.args(prepare_backends="worker:vda,ofa,dlss"), Path("clip.mp4")).name
        self.assertTrue(name.endswith("-worker_vda-ofa-dlss"))  # ':' is not a Windows file name character

    def test_prepared_passes_belong_to_the_clip(self):
        a = self.args()
        clips = [self.root / d / "face.mp4" for d in ("a", "b")]
        for c in clips:
            c.parent.mkdir()
            c.write_bytes(b"x" * 10)
        work = [nr_ab.prepare_root(self.root, a, c) for c in clips]
        self.assertNotEqual(work[0], work[1])  # one name, two clips: two folders
        passes = work[0] / "passes"
        inp = nr_ab.Inputs(clips[0], *(manifest(passes / n, 0, 20) for n in ("color_sr", "depth_dlss", "mv_dlss")))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(nr_ab.prepare(a, "dlssvid", inp, self.root, 17, True))  # no stamp: made from what?
        self.assertIn(" process ", out.getvalue())
        (work[0] / "clip.json").write_text(json.dumps(nr_ab.file_identity(clips[0])), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(nr_ab.prepare(a, "dlssvid", inp, self.root, 17, True))
            self.assertTrue(nr_ab.prepare(a, "dlssvid", inp, self.root, 30, True))  # do not cover the range
        clips[0].write_bytes(b"y" * 11)  # the clip re-encoded in place
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(nr_ab.prepare(a, "dlssvid", inp, self.root, 17, True))


class StubRun(unittest.TestCase):
    """A whole A/B on the stub backend and WARP: prepare, ranged runs, compares, the table, reuse (scripts.nr_ab_stub)."""

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
        cls.result = cls.run_ab(["--variants", str(cls.smoke), "--experiment", "E2,E8,E6f", "--check-runs", "1", "--check-repeat", "5", "--check-features", "3",
                                 "--start", "4", "--frames", "3", "--warmup", "2", "--out", str(cls.out)])

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
        # PR-7: the strength is NrCompose's lerp after the model - half the edit at 0.5 (the flicker filter fades with it, the clip moves)
        self.assertAlmostEqual(smoke["i05"]["edit"]["mae_rgb"] / smoke["base"]["edit"]["mae_rgb"], 0.5, delta=0.1)
        self.assertEqual(smoke["nodepth"]["vs_base"]["psnr_y"], 100.0)  # the stub ignores the guides
        self.assertEqual(smoke["mv"]["vs_base"]["psnr_y"], 100.0)
        self.assertIs(smoke["nodepth"]["nr"]["depth_used"], False)
        self.assertIs(smoke["base"]["nr"]["depth_used"], True)
        self.assertEqual(smoke["mv"]["nr"]["mv_scale"], [2.0, 2.0])  # mv 96x64 -> model 192x128 (bicubic x2)
        self.assertLess(smoke["base"]["temporal"]["psnr_y"], 100.0)
        self.assertIsNotNone(res["smoke"]["input_temporal"])
        self.assertIsInstance(smoke["base"]["nr"]["ms_per_frame_gpu"], (int, float))
        self.assertEqual(smoke["i05"]["nr"]["params"]["intensity"], 0.5)  # what the run reported it ran with
        self.assertEqual(res["E2"]["warmup"], 0)  # still mode: no warm-up
        self.assertIs(res["E2"]["verdict"]["identity"], True)  # intensity 0: the model is not called, the output is the input (PR-7)
        self.assertIs(res["E2"]["verdict"]["bypass"], True)
        self.assertEqual(res["E2"]["variants"]["i0"]["nr"]["passes"], 0)
        self.assertTrue(res["E2"]["meta"]["exe_sha256"])
        # i0 calls no model and has no diagnostics: the build line takes the baseline's GPU (review of PR-7)
        self.assertTrue(res["E2"]["meta"]["gpu"])
        self.assertEqual(res["E2"]["meta"]["gpu"], res["E2"]["variants"]["i1"]["nr"]["gpu"])
        e8 = res["E8"]
        self.assertEqual([c["ok"] for c in e8["checks"]], [True])
        self.assertEqual(e8["size"], "192x128")  # the feature at the size NR runs at: color_sr's
        self.assertTrue(e8["meta"]["gpu"])  # the build line keeps the check's GPU (nothing else ran in E8)
        self.assertEqual((e8["repeat"]["requested"], e8["repeat"]["ok"]), (5, 5))  # nr --check --check-repeat on the stub
        self.assertEqual((res["E6f"]["alive"], res["E6f"]["stop"]), (3, ""))
        self.assertEqual(res["E6f"]["size"], "192x128")
        self.assertTrue(res["smoke"]["meta"]["exe_sha256"])
        md = (self.out / "nr_ab.md").read_text(encoding="utf-8")
        for s in ("### smoke", "| i05 |", "∞", "E2", "### E8", e8["meta"]["gpu"], "Вход NR: color_sr, depth_dlss, mv_dlss; маски: выкл."):
            self.assertIn(s, md)
        self.assertEqual(list(self.out.rglob("color_nr")), [])  # cleaned up without --keep
        work = nr_ab.prepare_root(self.out, argparse.Namespace(prepare_backends="stub,stub,bicubic"), self.clip)
        self.assertEqual(json.loads((work / "clip.json").read_text(encoding="utf-8")), nr_ab.file_identity(self.clip))
        self.assertFalse((work / "prepare.mkv").exists())
        for name in ("color_sr", "depth_dlss", "mv_dlss"):
            self.assertTrue(nr_ab.covers(work / "passes" / name, 0, 6), name)

    def test_resume_reuses_only_the_same_run_and_only_adds(self):
        out = self.work / "resume"
        args = ["--variants", str(self.smoke), "--start", "4", "--frames", "3", "--warmup", "2", "--out", str(out)]
        stored = lambda: json.loads((out / "results.json").read_text(encoding="utf-8"))["clip-stub"]["experiments"]["smoke"]
        log = lambda n: (out / "clip-stub" / "smoke" / n / "nr.log").stat().st_mtime_ns
        r = self.run_ab(args + ["--only", "i05"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(set(stored()["variants"]), {"base", "i05"})
        r = self.run_ab(args)  # the stored subset does not answer the whole experiment
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("пропущен", r.stdout)
        full = stored()
        self.assertEqual(set(full["variants"]), {"base", "i05", "nodepth", "mv"})
        r = self.run_ab(args)  # the same run: reused
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("пропущен", r.stdout)
        self.assertEqual(stored(), full)
        r = self.run_ab(args + ["--dry-run"])  # a dry run tells what a real one would skip, and writes nothing
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("пропущен", r.stdout)
        self.assertFalse([x for x in r.stdout.splitlines() if " nr " in x or " compare " in x])
        before = {n: log(n) for n in ("i05", "nodepth")}
        r = self.run_ab(args + ["--only", "nodepth", "--force"])  # redo one variant: added to the stored run, not replacing it
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        now = stored()
        self.assertEqual(list(now["variants"]), ["base", "i05", "nodepth", "mv"])
        self.assertEqual(now["variants"]["i05"], full["variants"]["i05"])
        self.assertEqual(log("i05"), before["i05"])
        self.assertGreater(log("nodepth"), before["nodepth"])
        r = self.run_ab(args + ["--warmup", "1"])  # another warm-up: another signature, the whole experiment again
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("пропущен", r.stdout)
        self.assertGreater(log("i05"), before["i05"])
        self.assertEqual(stored()["warmup"], 1)

    def test_pairs_are_measured_again_unless_kept_by_an_only_merge(self):
        out, ab = self.work / "pairs", self.work / "pairs_ab.json"
        ab.write_text(json.dumps({"id": "pairs_ab", "baseline": "base", "metrics": ["base", "pairs"], "groups": [["base", "i05", "nodepth"]],
                                  "variants": [{"name": "base"}, {"name": "i05", "params": {"intensity": 0.5}}, {"name": "nodepth", "flags": ["--no-depth-guide"]}]}),
                      encoding="utf-8")
        args = ["--variants", str(ab), "--start", "4", "--frames", "3", "--warmup", "2", "--out", str(out)]
        path = out / "results.json"
        stored = lambda: json.loads(path.read_text(encoding="utf-8"))["clip-stub"]["experiments"]["pairs_ab"]
        pairs = lambda: {(p["a"], p["b"]): p["psnr_y"] for p in stored()["pairs"]}

        def tamper(pair, value):
            j = json.loads(path.read_text(encoding="utf-8"))
            for p in j["clip-stub"]["experiments"]["pairs_ab"]["pairs"]:
                if (p["a"], p["b"]) == pair:
                    p["psnr_y"] = value
            path.write_text(json.dumps(j), encoding="utf-8")

        r = self.run_ab(args + ["--keep"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        first = pairs()
        self.assertEqual(set(first), {("base", "i05"), ("base", "nodepth"), ("i05", "nodepth")})
        tamper(("base", "i05"), 1.0)
        r = self.run_ab(args + ["--keep", "--force"])  # every variant ran again: every pair is measured again
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(pairs()[("base", "i05")], first[("base", "i05")])
        tamper(("base", "nodepth"), 2.0)
        r = self.run_ab(args + ["--only", "i05", "--force"])  # an --only merge: the pair without i05 stays, i05's are measured (nodepth kept)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        now = pairs()
        self.assertEqual(now[("base", "nodepth")], 2.0)
        self.assertEqual(now[("i05", "nodepth")], first[("i05", "nodepth")])
        self.assertEqual(stored()["pairs_missing"], [])
        r = self.run_ab(args + ["--only", "nodepth", "--force"])  # i05's color_nr is gone (no --keep above): its pair with nodepth has no number
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(stored()["pairs_missing"], [["i05", "nodepth"]])
        self.assertTrue(any("неполный" in t for t in stored()["verdict"]["text"]))
        r = self.run_ab(args)  # a result with a pair missing is not reused
        self.assertNotIn("пропущен", r.stdout)
        self.assertEqual(stored()["pairs_missing"], [])

    def test_dry_run_prints_commands_only(self):
        out = self.work / "dry"
        r = self.run_ab(["--dry-run", "--experiment", "E7", "--start", "4", "--frames", "3", "--out", str(out)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(sum(" process " in x for x in lines), 1)
        nr = [x for x in lines if " nr " in x]
        self.assertEqual(len(nr), 3)
        self.assertTrue(all("--warmup 0" in x and "--params-file" in x for x in nr))
        self.assertEqual(sum(" compare " in x for x in lines), 6)  # 3 pairs and 3 edit compares
        self.assertFalse(out.exists() and any(out.rglob("*")))

    def test_refused_before_any_run(self):
        bad = self.work / "bad.json"
        bad.write_text(json.dumps({"id": "bad", "variants": [{"name": "a", "params": {"no_such_key": 1}}]}), encoding="utf-8")
        switch = self.work / "switch.json"  # an A/B switch in params: dlssvid would refuse it after the baseline ran
        switch.write_text(json.dumps({"id": "switch", "variants": [{"name": "a"}, {"name": "b", "params": {"mv_scale": "auto"}}]}), encoding="utf-8")
        taken = self.work / "E7.json"  # its id is the file name: results.json, folder and verdict of the catalogue's E7
        taken.write_text(json.dumps({"variants": [{"name": "a"}]}), encoding="utf-8")
        many = self.run_ab(["--experiment", "E6f", "--check-features", "18", "--out", str(self.work / "many")])  # INCIDENT 2026-10-25: hung the machine
        self.assertNotEqual(many.returncode, 0)
        self.assertIn("--check-features", many.stderr)
        self.assertFalse((self.work / "many").exists())
        for args, word in ((["--variants", str(bad)], "no_such_key"), (["--experiment", "E2", "--nr-arg=--intensity=1"], "--intensity=1"),
                           (["--variants", str(switch)], "mv_scale"), (["--variants", str(taken)], "taken"),
                           (["--experiment", "E2", "--passes", str(self.work)], "--prepare makes")):
            out = self.work / "bad_out"
            r = self.run_ab(args + ["--out", str(out)])
            self.assertEqual(r.returncode, 1, word)
            self.assertIn(word, r.stderr)
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
