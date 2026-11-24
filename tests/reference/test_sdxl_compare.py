# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical acceptance rejects corrupt tensors and does not hide model errors."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qualify.sdxl_compare import byte_exact_difference, compare, difference, main, save_image, tensor


class ComparisonTests(unittest.TestCase):
    def manifests(self, root, *, reference, captured, native, native_repeat=None):
        paths = []
        for label, values in (("reference", reference), ("captured", captured), ("native", native)):
            np.asarray(values, dtype=np.float32).tofile(root / f"{label}.bin")
            output = dict(name="latent", shape=[len(values)], dtype="F32", file=f"{label}.bin")
            if label != "captured":
                repeat = native_repeat if label == "native" and native_repeat is not None else values
                np.asarray(repeat, dtype=np.float32).tofile(root / f"{label}.repeat.bin")
                output["repeat_file"] = f"{label}.repeat.bin"
            path = root / f"{label}.json"
            path.write_text(json.dumps(dict(outputs=[output], timing=dict(wall_ms=[2, 3, 4]))))
            paths.append(path)
        return paths[0], paths[2], paths[1]

    def test_equal_tensors(self):
        values = np.array([0, 1, -2], dtype=np.float16)
        result = difference(values, values.copy(), atol=0, rtol=0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_absolute_error"], 0)

    def test_decoded_rgb_images_preserve_dimensions_and_pixel_mapping(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for height, width, dtype, label in ((512, 512, np.float16, "F16"),
                                               (1024, 1024, np.float32, "F32"),
                                               (8, 16, np.float32, "F32")):
                values = np.empty((1, 3, height, width), dtype=dtype)
                values[:, 0], values[:, 1], values[:, 2] = -2, 0, 2
                values.tofile(root / "image.bin")
                output = dict(name="decoded_image", shape=list(values.shape), dtype=label, file="image.bin")
                manifest = root / "reference.json"
                manifest.write_text(json.dumps(dict(outputs=[output])))
                save_image(manifest, root / "image.png")
                with Image.open(root / "image.png") as rendered:
                    self.assertEqual(rendered.size, (width, height))
                    self.assertTrue(np.all(np.asarray(rendered) == [0, 128, 255]))

    def test_decoded_images_reject_invalid_layout_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape, value in (((2, 3, 8, 8), 0), ((1, 4, 8, 8), 0),
                                 ((3, 8, 8), 0), ((1, 3, 8, 8), np.nan)):
                values = np.full(shape, value, dtype=np.float32)
                values.tofile(root / "image.bin")
                output = dict(name="decoded_image", shape=list(shape), dtype="F32", file="image.bin")
                manifest = root / "reference.json"
                manifest.write_text(json.dumps(dict(outputs=[output])))
                with self.assertRaisesRegex(ValueError, "Image output"):
                    save_image(manifest, root / "image.png")

    def test_one_ulp_is_not_exact(self):
        values = np.array([1], dtype=np.float16)
        altered = np.nextafter(values, np.float16(2))
        self.assertFalse(difference(values, altered, atol=0, rtol=0)["passed"])

    def test_signed_zero_is_numerically_equal_but_not_byte_equal(self):
        for dtype in (np.float16, np.float32):
            positive = np.array([0.0, 1.0], dtype=dtype)
            negative = np.array([-0.0, 1.0], dtype=dtype)
            self.assertTrue(difference(positive, negative, atol=0, rtol=0)["passed"])
            result = byte_exact_difference(positive, negative)
            self.assertFalse(result["passed"])
            self.assertFalse(result["byte_equal"])
            self.assertEqual(result["max_absolute_error"], 0)
            self.assertTrue(byte_exact_difference(negative, negative.copy())["passed"])

    def test_signed_zero_change_in_native_or_repeat_fails_capture_acceptance(self):
        for native_values, repeated_values in (([-0.0], [0.0]), ([0.0], [-0.0])):
            with tempfile.TemporaryDirectory() as directory:
                reference, native, captured = self.manifests(Path(directory), reference=[0.0],
                    captured=[0.0], native=native_values, native_repeat=repeated_values)
                report = compare(reference, native, captured_path=captured)
                self.assertEqual(report["reference_status"], "passed")
                self.assertEqual(report["native_capture_status"], "failed")
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("reference_over_native_median", report["timing"])

    def test_nonfinite_never_passes(self):
        for value in (np.nan, np.inf, -np.inf):
            values = np.array([value], dtype=np.float32)
            self.assertFalse(difference(values, values, atol=100, rtol=100)["passed"])
            self.assertFalse(byte_exact_difference(values, values.copy())["passed"])

    def test_exact_byte_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tensor.bin").write_bytes(b"\0\0\0")
            spec = dict(name="latent", file="tensor.bin", shape=[2], dtype="F16")
            with self.assertRaises(ValueError):
                tensor(root / "reference.json", spec)

    def test_failed_tensor_suppresses_speed_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([1, 2], dtype=np.float32).tofile(root / "reference.bin")
            np.array([1, 3], dtype=np.float32).tofile(root / "native.bin")
            output = dict(name="latent", shape=[2], dtype="F32", file="reference.bin", repeat_file="reference.bin")
            reference = dict(outputs=[output], timing=dict(wall_ms=[10, 11, 12]))
            native = dict(outputs=[dict(output, file="native.bin", repeat_file="native.bin")], timing=dict(wall_ms=[2, 3, 4]))
            reference_path, native_path = root / "reference.json", root / "native.json"
            reference_path.write_text(json.dumps(reference))
            native_path.write_text(json.dumps(native))
            report = compare(reference_path, native_path)
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("reference_over_native_median", report["timing"])
            self.assertTrue(report["tensors"][0]["reference_repeatability"]["passed"])

    def test_explicit_tolerance(self):
        first = np.array([0, 100], dtype=np.float32)
        second = np.array([0.001, 100.01], dtype=np.float32)
        self.assertTrue(difference(first, second, atol=0.0011, rtol=0.0001)["passed"])

    def test_compiler_drift_is_distinct_from_native_replay_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, native, captured = self.manifests(Path(directory), reference=[1, 2],
                                                        captured=[1, 3], native=[1, 3])
            report = compare(reference, native, captured_path=captured)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["reference_status"], "failed")
            self.assertEqual(report["native_capture_status"], "passed")
            result = report["tensors"][0]
            self.assertFalse(result["captured_vs_reference"]["passed"])
            self.assertTrue(result["native_vs_captured"]["passed"])
            self.assertTrue(result["native_repeat_vs_captured"]["passed"])
            self.assertNotIn("reference_over_native_median", report["timing"])

    def test_native_replay_corruption_is_distinct_from_compiler_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, native, captured = self.manifests(Path(directory), reference=[1, 2],
                                                        captured=[1, 2], native=[1, 3])
            report = compare(reference, native, captured_path=captured)
            self.assertEqual(report["native_capture_status"], "failed")
            self.assertTrue(report["tensors"][0]["captured_vs_reference"]["passed"])
            self.assertFalse(report["tensors"][0]["native_vs_captured"]["passed"])

    def test_native_repeat_must_match_capture_exactly_despite_reference_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, native, captured = self.manifests(root, reference=[1, 2], captured=[1, 2],
                                                        native=[1, 2], native_repeat=[1, 2.125])
            report = compare(reference, native, captured_path=captured, atol=1)
            self.assertEqual(report["reference_status"], "passed")
            self.assertEqual(report["native_capture_status"], "failed")
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["tensors"][0]["native_vs_captured"]["passed"])
            self.assertFalse(report["tensors"][0]["native_repeat_vs_captured"]["passed"])
            self.assertNotIn("reference_over_native_median", report["timing"])
            with patch("sys.argv", ["sdxl_compare", "--reference", str(reference), "--native", str(native),
                                   "--captured", str(captured), "--output", str(root / "comparison.json"), "--atol", "1"]):
                self.assertEqual(main(), 1)

    def test_matching_capture_and_reference_allow_timing_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, native, captured = self.manifests(Path(directory), reference=[1, 2],
                                                        captured=[1, 2], native=[1, 2])
            report = compare(reference, native, captured_path=captured)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["native_capture_status"], "passed")
            self.assertEqual(report["timing"]["reference_over_native_median"], 1)

    def test_capture_metadata_must_match_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, native, captured = self.manifests(Path(directory), reference=[1, 2],
                                                        captured=[1, 2], native=[1, 2])
            manifest = json.loads(captured.read_text())
            manifest["outputs"][0]["shape"] = [1, 2]
            captured.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "captured tensor metadata"):
                compare(reference, native, captured_path=captured)


if __name__ == "__main__":
    unittest.main()
