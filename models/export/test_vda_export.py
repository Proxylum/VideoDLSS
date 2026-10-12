#!/usr/bin/env python
"""VDA export (TASK-0039): the patched attention equals the original, and export_vda.py writes a graph with external
data that loads by path. CPU only; skipped without torch, the Video-Depth-Anything checkout (VDA_REPO / SDK) or - for
the export - the cached Small checkpoint (nothing is downloaded here). ctest: models.vda_export."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SMALL_FILE = "depth-anything__Metric-Video-Depth-Anything-Small__metric_video_depth_anything_vits.pth"


def models_dir():
    env = os.environ.get("DLSSVID_MODELS_DIR")
    return Path(env) if env else HERE.parent


def vda_available():
    try:
        import torch  # noqa: F401

        import vda_loader

        vda_loader._import_vda()
        return True
    except ImportError:
        return False


@unittest.skipUnless(vda_available(), "torch or the Video-Depth-Anything checkout is missing")
class PatchedAttention(unittest.TestCase):
    def test_motion_module_bmm_equals_baddbmm(self):
        import torch

        import vda_loader
        from video_depth_anything.motion_module import attention as ma

        torch.manual_seed(0)
        original = ma.CrossAttention.__dict__.get("_attention_original", ma.CrossAttention._attention)
        if original is vda_loader._bmm_attention:
            self.skipTest("already patched in this process")
        att = ma.CrossAttention(query_dim=64, heads=8, dim_head=8).eval()
        q, k, v = (torch.randn(16, 32, 8) for _ in range(3))
        self.assertLess((original(att, q, k, v) - vda_loader._bmm_attention(att, q, k, v)).abs().max().item(), 1e-5)

    def test_dinov2_sdpa_equals_explicit_attention(self):
        import torch

        import vda_loader
        from video_depth_anything.dinov2_layers import attention as da

        torch.manual_seed(1)
        att = da.Attention(dim=384, num_heads=6, qkv_bias=True).eval()
        x = torch.randn(2, 100, 384)
        with torch.no_grad():
            diff = (da.Attention.forward(att, x) - vda_loader._sdpa_forward(att, x)).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_window_rule(self):
        import vda_loader

        for ok in (1, 2, 4, 8, 16, 32):
            self.assertEqual(vda_loader.check_window(ok), ok)
        for bad in (0, 6, 10, 30, 36):
            with self.assertRaises(ValueError):
                vda_loader.check_window(bad)


class ExportLockTakeover(unittest.TestCase):
    """A crashed export (0xC0000005 in the trace of Large at 32 frames) left its lock behind: the next export took 3
    hours to call it stale. A lock of a dead process is taken over at once; one of a live process is waited for."""

    def test_lock_of_a_dead_process_is_taken_over(self):
        import export_vda

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.onnx"
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            lock = out.with_name(out.name + ".lock")
            lock.write_text(f"{dead.pid}\n", encoding="utf-8")
            t0 = time.time()
            with export_vda.ExportLock(out, wait=30) as held:
                self.assertFalse(held.waited)
                self.assertEqual(lock.read_text(encoding="utf-8").strip(), str(os.getpid()))
            self.assertLess(time.time() - t0, 5)
            self.assertFalse(lock.exists())

    def test_lock_of_a_live_process_is_waited_for(self):
        import export_vda

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.onnx"
            out.with_name(out.name + ".lock").write_text(f"{os.getpid()}\n", encoding="utf-8")  # this process: alive
            with self.assertRaises(TimeoutError):
                with export_vda.ExportLock(out, wait=1):
                    pass


@unittest.skipUnless(vda_available(), "torch or the Video-Depth-Anything checkout is missing")
class ExportExternalData(unittest.TestCase):
    def test_small_window_2_with_external_data(self):
        cache = models_dir() / "cache"
        ckpt = cache / SMALL_FILE
        if not ckpt.exists():
            self.skipTest(f"no cached {SMALL_FILE} (models/export/fetch.py metric-vda-small)")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "vda_ext.onnx"
            r = subprocess.run([sys.executable, str(HERE / "export_vda.py"), "--model", "metric-vda-small", "--window", "2", "--width", "70",
                                "--height", "42", "--external-data", "on", "--out", str(out), "--cache", str(cache)],
                               capture_output=True, text=True, cwd=str(HERE))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            data = out.with_name(out.name + ".data")
            self.assertTrue(out.exists())
            self.assertTrue(data.exists())
            self.assertLess(out.stat().st_size, 10 << 20)  # the graph only; the tensors are in .data
            # the patch: no ~T^2 zeros of baddbmm - the data is about the weights (116 MB of fp32)
            self.assertLess(data.stat().st_size, 1.1 * ckpt.stat().st_size)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), sorted([out.name, data.name]))  # no tensor litter, no lock
            self.assertIn("onnxruntime vs torch", r.stdout)

            import onnx

            onnx.checker.check_model(str(out))  # by path: with the external data


if __name__ == "__main__":
    unittest.main()
