import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from plyfile import PlyData, PlyElement

import gen_pressure_view as generator


class PressureViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.ply_path = self.directory / "scene.ply"
        x, y, z = np.meshgrid(
            np.linspace(-4, 4, 9), np.linspace(-2, 2, 7), np.linspace(-1, 1, 5)
        )
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        points = np.vstack((points, [10000, 10000, 10000]))
        vertices = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i, name in enumerate(("x", "y", "z")):
            vertices[name] = points[:, i]
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(self.ply_path))

    def generate(self, trajectory, count, extra=()):
        output = self.directory / f"{trajectory}-{count}.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            generator.main([
                str(self.ply_path), "--output", str(output), "--trajectory", trajectory,
                "--count", str(count), "--quantile", "0.01",
            ] + list(extra))
        return json.loads(output.read_text())

    def test_sampling_density_preserves_path_and_camera_orientation(self):
        for trajectory in ("orbit", "scan"):
            with self.subTest(trajectory=trajectory):
                coarse = self.generate(trajectory, 5)
                fine = self.generate(trajectory, 9)
                self.assertEqual(len(coarse["frames"]), 5)
                center = np.asarray(coarse["metadata"]["look_at"])
                self.assertTrue(np.all(np.abs(center) < 0.1))
                for index, frame in enumerate(coarse["frames"]):
                    pose = np.asarray(frame["transform_matrix"])
                    other = np.asarray(fine["frames"][index * 2]["transform_matrix"])
                    np.testing.assert_allclose(pose, other, atol=1e-12)
                    rotation = pose[:3, :3]
                    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
                    forward = center - pose[:3, 3]
                    forward /= np.linalg.norm(forward)
                    np.testing.assert_allclose(rotation[:, 2], forward, atol=1e-12)
                    self.assertLess(rotation[2, 1], 0)
                np.testing.assert_allclose(
                    coarse["frames"][0]["transform_matrix"],
                    coarse["frames"][-1]["transform_matrix"], atol=1e-12,
                )

    def test_zero_one_and_limit(self):
        self.assertEqual(self.generate("orbit", 0)["frames"], [])
        self.assertEqual(len(self.generate("scan", 1)["frames"]), 1)
        self.assertEqual(len(self.generate("orbit", 3000)["frames"]), 3000)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self.generate("orbit", 3001)

    def test_scan_visits_radius_and_height_corners(self):
        sequence = self.generate("scan", 5)
        center = np.asarray(sequence["metadata"]["look_at"])
        scale = sequence["metadata"]["reference_radius"]
        expected = [(0.75, -0.1), (1.25, -0.1), (1.25, 0.25), (0.75, 0.25), (0.75, -0.1)]
        for frame, (radius, height) in zip(sequence["frames"], expected):
            offset = np.asarray(frame["transform_matrix"])[:3, 3] - center
            self.assertAlmostEqual(np.linalg.norm(offset[:2]) / scale, radius)
            self.assertAlmostEqual(offset[2] / scale, height)

    def test_reference_cameras_set_up_axis_and_anchor_without_changing_sampling(self):
        up = np.array([0.0, 1.0, 0.0])
        cameras = []
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            position = np.array([4 * np.cos(angle), 0.8, 4 * np.sin(angle)])
            pose = generator.look_at(position, np.zeros(3), up)
            cameras.append({"img_name": str(len(cameras)), "position": position.tolist(),
                            "rotation": pose[:3, :3].tolist(), "width": 1200, "fx": 600})
        # A real camera may point away from the scene: it should not be picked
        # automatically, but must remain selectable explicitly.
        position = np.asarray(cameras[0]["position"])
        cameras[0]["rotation"] = generator.look_at(position, 2 * position, up)[:3, :3].tolist()
        (self.directory / "cameras.json").write_text(json.dumps(cameras))
        for trajectory in ("orbit", "scan"):
            with self.subTest(trajectory=trajectory):
                coarse, fine = self.generate(trajectory, 5), self.generate(trajectory, 9)
                metadata = coarse["metadata"]
                self.assertEqual(metadata["up_source"], "camera_right_axes_pca")
                np.testing.assert_allclose(metadata["up_vector"], up, atol=1e-10)
                self.assertEqual(metadata["center_source"], "reference_rays")
                self.assertEqual(metadata["reference_radius_source"], "median_reference_camera_radius")
                self.assertEqual(metadata["fov_source"], "reference_camera")
                selected = metadata["reference_camera"]["index"]
                self.assertNotEqual(selected, 0)
                self.assertEqual(metadata["reference_camera"]["selection"], "auto_scene_center")
                first = np.asarray(coarse["frames"][0]["transform_matrix"])
                np.testing.assert_allclose(first[:3, 3], cameras[selected]["position"], atol=1e-10)
                np.testing.assert_allclose(first[:3, 2], np.asarray(cameras[selected]["rotation"])[:, 2], atol=1e-10)
                for index, frame in enumerate(coarse["frames"]):
                    pose = np.asarray(frame["transform_matrix"])
                    np.testing.assert_allclose(pose, fine["frames"][index * 2]["transform_matrix"], atol=1e-10)
                    self.assertLess(np.dot(pose[:3, 1], up), 0)
                np.testing.assert_allclose(coarse["frames"][0]["transform_matrix"],
                                           coarse["frames"][-1]["transform_matrix"], atol=1e-10)
        explicit = self.generate("orbit", 5, ["--reference-index", "0"])
        self.assertEqual(explicit["metadata"]["reference_camera"]["index"], 0)
        self.assertEqual(explicit["metadata"]["reference_camera"]["selection"], "explicit")

    def test_json_loads_in_recursive_renderer(self):
        import recursive_render

        output = self.generate("orbit", 5)
        path = self.directory / "orbit-5.json"
        with mock.patch.object(recursive_render, "Camera", side_effect=lambda **kw: kw):
            cameras = recursive_render.load_camera_sequence(path)
        self.assertEqual(len(cameras), 5)
        self.assertEqual((cameras[0]["image_width"], cameras[0]["image_height"]), (1920, 1080))
        self.assertIsNone(cameras[0]["image"])
        w2c = np.eye(4)
        w2c[:3, :3] = cameras[0]["R"].T
        w2c[:3, 3] = cameras[0]["T"]
        np.testing.assert_allclose(
            np.linalg.inv(w2c), output["frames"][0]["transform_matrix"], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
