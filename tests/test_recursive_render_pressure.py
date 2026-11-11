import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import recursive_render as render


class PressureRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "scene"
        (self.dataset / "sparse").mkdir(parents=True)
        self.model = self.dataset / "trained"
        self.checkpoint = self.model / "point_cloud" / "iteration_30000"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "point_cloud.ply").touch()
        self.calls = []

    def add_trace(self, name):
        path = self.checkpoint / f"{name}.json"
        path.write_text("{}")
        return path

    def run_main(self, argv, fail_trace=None):
        original = SimpleNamespace(image_width=100, image_height=60)
        scene = SimpleNamespace(getTestCameras=lambda: [original], getTrainCameras=lambda: [])
        dataset_args = SimpleNamespace(sh_degree=3, white_background=False)

        def load_trace(path):
            if Path(path).stem == fail_trace:
                raise ValueError("invalid trace")
            return [SimpleNamespace(image_width=200, image_height=100)]

        def benchmark(views, gaussians, pipeline, background, args, width, height, warmup, render_dir):
            self.assertFalse(render.torch.is_grad_enabled())
            self.calls.append((width, height, render_dir))
            for view in views:
                view.image_width, view.image_height = width, height
            return dict(frames=1, seconds=0.1, fps=10, avg_frame_latency_ms=100,
                        p50_frame_latency_ms=100, p95_frame_latency_ms=100,
                        min_frame_latency_ms=100, max_frame_latency_ms=100)

        with contextlib.ExitStack() as stack:
            for name, value in {
                "load_model_args": lambda *a: SimpleNamespace(),
                "init_args": lambda *a: None,
                "init_distributed": lambda *a: None,
                "safe_state": lambda *a: None,
                "set_args": lambda *a: None,
                "set_log_file": lambda *a: None,
                "set_cur_iter": lambda *a: None,
                "ModelParams": lambda *a, **k: SimpleNamespace(extract=lambda _: dataset_args),
                "PipelineParams": lambda *a, **k: SimpleNamespace(extract=lambda _: object()),
                "GaussianModel": lambda *a: object(),
                "Scene": lambda *a, **k: scene,
                "load_camera_sequence": load_trace,
                "benchmark_resolution": benchmark,
                "distributed_barrier": lambda: None,
            }.items():
                stack.enter_context(mock.patch.object(render, name, value))
            stack.enter_context(mock.patch.object(render.torch, "tensor", return_value=object()))
            stack.enter_context(mock.patch.object(render.utils, "WORLD_SIZE", 1))
            stack.enter_context(mock.patch.object(render.utils, "GLOBAL_RANK", 0))
            stack.enter_context(mock.patch.object(sys, "argv", ["recursive_render.py"] + argv))
            stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            status = render.main()
        return status, stderr.getvalue()

    def test_recursive_csv_contains_original_and_both_traces(self):
        scan = self.add_trace("pressure_scan")
        orbit = self.add_trace("pressure_orbit")
        status, _ = self.run_main(["--recursive", str(self.root), "--resolutions", "1.0", "0.5"])
        self.assertEqual(status, 0)
        with (self.root / "recursive_render_fps.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["trajectory"] for r in rows],
                         ["original", "original", "pressure_scan", "pressure_scan", "pressure_orbit", "pressure_orbit"])
        self.assertEqual([r["resolution"] for r in rows],
                         ["100x60", "50x30", "200x100", "100x50", "200x100", "100x50"])
        self.assertEqual(rows[0]["camera_sequence"], "")
        self.assertEqual(rows[2]["camera_sequence"], str(scan.resolve()))
        self.assertEqual(rows[4]["camera_sequence"], str(orbit.resolve()))
        self.assertEqual(self.calls[0][2], self.checkpoint / "render" / "100x60")
        self.assertEqual(self.calls[2][2], self.checkpoint / "render" / "pressure_scan" / "200x100")
        self.assertEqual(self.calls[4][2], self.checkpoint / "render" / "pressure_orbit" / "200x100")

    def test_failed_trace_preserves_other_csv_results(self):
        self.add_trace("pressure_scan")
        self.add_trace("pressure_orbit")
        status, stderr = self.run_main([str(self.dataset), str(self.model), "--resolutions", "1.0"],
                                       fail_trace="pressure_scan")
        self.assertEqual(status, 1)
        self.assertIn("[FAILED] scene/pressure_scan", stderr)
        with (self.model / "recursive_render_fps.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["trajectory"] for row in rows], ["original", "pressure_orbit"])

    def test_explicit_pressure_trace_is_not_benchmarked_twice(self):
        scan = self.add_trace("pressure_scan")
        self.add_trace("pressure_orbit")
        status, _ = self.run_main([str(self.dataset), str(self.model), "--camera-sequence", str(scan),
                                   "--resolutions", "1.0", "--no-save-renders"])
        self.assertEqual(status, 0)
        with (self.model / "recursive_render_fps.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["trajectory"] for row in rows], ["pressure_scan", "pressure_orbit"])
        self.assertTrue(all(call[2] is None for call in self.calls))

    def test_missing_or_skipped_traces_keep_original(self):
        status, stderr = self.run_main([str(self.dataset), str(self.model), "--resolutions", "1.0"])
        self.assertEqual(status, 0)
        self.assertIn("none found", stderr)
        self.add_trace("pressure_scan")
        self.calls.clear()
        status, _ = self.run_main([str(self.dataset), str(self.model), "--resolutions", "1.0",
                                   "--skip-pressure-sequences"])
        self.assertEqual(status, 0)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
