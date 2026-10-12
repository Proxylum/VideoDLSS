#!/usr/bin/env python3
"""Helpers of bench_clips.py that decide what a measurement reads (review of TASK-0038, bench-stale-npz): an output
folder left by an earlier, longer run must not enter the depth statistics, and a motion-vector pass shorter than the
clip must not turn the warped TAE into the static proxy without a word. Standard library only (ctest: scripts.bench_clips)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_clips  # noqa: E402


def make_pass(folder: Path, frames: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"pass": "mv_dlss", "frame_count": frames}), encoding="utf-8")
    return folder


class BenchHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pass_frames(self) -> None:
        self.assertEqual(bench_clips.pass_frames(make_pass(self.root / "mv", 42)), 42)
        self.assertEqual(bench_clips.pass_frames(self.root / "missing"), 0)
        (self.root / "bad").mkdir()
        (self.root / "bad" / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(bench_clips.pass_frames(self.root / "bad"), 0)

    def test_mv_covers_needs_every_frame(self) -> None:
        mv = make_pass(self.root / "flow_ofa" / "mv_dlss", 120)
        self.assertTrue(bench_clips.mv_covers(mv, 120))
        self.assertTrue(bench_clips.mv_covers(mv, 60))
        self.assertFalse(bench_clips.mv_covers(mv, 240))  # before: only the folder's existence was checked
        self.assertFalse(bench_clips.mv_covers(self.root / "none", 1))

    def test_fresh_dir_drops_frames_of_an_earlier_run(self) -> None:
        out = self.root / "depth_da3" / "depth_raw"
        out.mkdir(parents=True)
        for i in range(5):
            (out / f"{i:06d}.npz").write_bytes(b"old")
        cleared = bench_clips.fresh_dir(self.root / "depth_da3")
        self.assertEqual(cleared, self.root / "depth_da3")
        self.assertFalse(cleared.exists())  # the depth run creates it again with its own frames only


if __name__ == "__main__":
    unittest.main()
