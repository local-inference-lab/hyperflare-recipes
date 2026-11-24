# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare canonical diffusion tensors with independently loaded native core outputs."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np


_DTYPES = {"F16": "<f2", "F32": "<f4"}


def tensor(manifest_path, spec, field="file"):
    """Read exactly the declared tensor extent; reject truncated or extra bytes."""
    shape = spec["shape"]
    if not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError(f"{spec['name']}: shape must contain positive integers")
    try:
        dtype = np.dtype(_DTYPES[spec["dtype"]])
    except KeyError as error:
        raise ValueError(f"{spec['name']}: unsupported dtype {spec['dtype']}") from error
    path = manifest_path.parent / spec[field]
    expected = math.prod(shape) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {path.stat().st_size}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def difference(reference, actual, *, atol, rtol):
    """Report every element outside an explicit bound, including nonfinite values."""
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("Compared tensors must have identical shapes and dtypes")
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    if not finite:
        return {"passed": False, "finite": False, "elements": reference.size,
                "reason": "Reference or native output contains NaN or infinity"}
    left = reference.astype(np.float64)
    right = actual.astype(np.float64)
    delta = np.abs(right - left)
    outside = delta > atol + rtol * np.abs(left)
    norm = float(np.linalg.norm(left.ravel()))
    delta_norm = float(np.linalg.norm(delta.ravel()))
    return {
        "passed": not bool(outside.any()), "finite": True,
        "elements": reference.size, "different_elements": int(np.count_nonzero(delta)),
        "outside_tolerance": int(np.count_nonzero(outside)),
        "max_absolute_error": float(delta.max()), "mean_absolute_error": float(delta.mean()),
        "root_mean_square_error": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2_error": delta_norm / norm if norm else (0.0 if delta_norm == 0 else None),
    }


def byte_exact_difference(reference, actual):
    """Require finite, identical tensor bytes while retaining numerical diagnostics."""
    result = difference(reference, actual, atol=0.0, rtol=0.0)
    result["byte_equal"] = bool(np.array_equal(
        np.ascontiguousarray(reference).view(np.uint8),
        np.ascontiguousarray(actual).view(np.uint8)))
    result["passed"] = result["passed"] and result["byte_equal"]
    return result


def named_outputs(manifest):
    outputs = manifest["outputs"]
    indexed = {spec["name"]: spec for spec in outputs}
    if not outputs or len(indexed) != len(outputs):
        raise ValueError("Output names must be nonempty and unique")
    return indexed


def samples(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing samples must be finite and positive")
    return {"samples_ms": values, "median_ms": statistics.median(values),
            "minimum_ms": min(values), "maximum_ms": max(values), "count": len(values)}


def compare(reference_path, native_path, *, atol=0.0, rtol=0.0, captured_path=None):
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    reference = json.loads(reference_path.read_text())
    native = json.loads(native_path.read_text())
    expected, observed = named_outputs(reference), named_outputs(native)
    if set(expected) != set(observed):
        raise ValueError(f"Output sets differ: reference={sorted(expected)}, native={sorted(observed)}")
    captured = None if captured_path is None else named_outputs(json.loads(captured_path.read_text()))
    if captured is not None and set(expected) != set(captured):
        raise ValueError(f"Output sets differ: reference={sorted(expected)}, captured={sorted(captured)}")
    comparisons = []
    for name, spec in expected.items():
        other = observed[name]
        if (spec["shape"], spec["dtype"]) != (other["shape"], other["dtype"]):
            raise ValueError(f"{name}: native tensor metadata differs from the reference")
        left = tensor(reference_path, spec)
        right = tensor(native_path, other)
        repeat = tensor(reference_path, spec, "repeat_file")
        native_repeat = tensor(native_path, other, "repeat_file")
        comparisons.append({"name": name, "shape": spec["shape"], "dtype": spec["dtype"],
                            "native_vs_reference": difference(left, right, atol=atol, rtol=rtol),
                            "native_repeat_vs_reference": difference(left, native_repeat, atol=atol, rtol=rtol),
                            "native_repeatability": difference(right, native_repeat, atol=0.0, rtol=0.0),
                            "reference_repeatability": difference(left, repeat, atol=0.0, rtol=0.0)})
        if captured is not None:
            recorded = captured[name]
            if (spec["shape"], spec["dtype"]) != (recorded["shape"], recorded["dtype"]):
                raise ValueError(f"{name}: captured tensor metadata differs from the reference")
            compiled = tensor(captured_path, recorded)
            comparisons[-1].update({
                "captured_vs_reference": difference(left, compiled, atol=atol, rtol=rtol),
                "native_vs_captured": byte_exact_difference(compiled, right),
                "native_repeat_vs_captured": byte_exact_difference(compiled, native_repeat),
            })
    passed = all(item[key]["passed"] for item in comparisons
                 for key in ("native_vs_reference", "native_repeat_vs_reference"))
    capture_passed = captured is None or all(item[key]["passed"] for item in comparisons
                    for key in ("native_vs_captured", "native_repeat_vs_captured"))
    timing = {
        "reference": samples(reference["timing"]["wall_ms"]),
        "native": samples(native["timing"]["wall_ms"]),
        "reference_boundary": reference["timing"].get("boundary"),
        "native_boundary": native["timing"].get("boundary", native["timing"].get("scope")),
        "reference_vae_precision": reference["timing"].get("vae_precision"),
        "comparison": "Warmed single-request computation; excludes loading, input upload and output download. Exact boundaries are recorded for each implementation.",
        "interpretation": "Whole prepared execution versus the eager pipeline; does not isolate graph-launch overhead or individual optimizations.",
    }
    if native["timing"].get("gpu_ms"):
        timing["native_gpu"] = samples(native["timing"]["gpu_ms"])
    if reference["timing"].get("gpu_ms"):
        timing["reference_gpu"] = samples(reference["timing"]["gpu_ms"])
    if passed and capture_passed:
        timing["reference_over_native_median"] = timing["reference"]["median_ms"] / timing["native"]["median_ms"]
    else:
        timing["ratio_withheld"] = ("At least one tensor failed the numerical comparison." if not passed
                                    else "Native output differs from the captured compiled output.")
    report = {
        "status": "passed" if passed and capture_passed else "failed", "reference": str(reference_path),
        "native": str(native_path), "tolerance": {"atol": atol, "rtol": rtol},
        "tensors": comparisons, "timing": timing,
        "scope": "Fixed-profile diffusion in eager PyTorch versus separately loaded Flare CUDA graph execution",
        "limitations": ["No HTTP latency, concurrent serving throughput or broader model/shape qualification.",
                        "An exact reference repeatability check is reported separately from native acceptance."],
    }
    if captured is not None:
        report.update(captured=str(captured_path), reference_status="passed" if passed else "failed",
                      native_capture_status="passed" if capture_passed else "failed",
                      native_capture_tolerance={"atol": 0.0, "rtol": 0.0},
                      native_capture_contract="Finite tensors with identical bytes, including signed zero.")
    return report


def save_image(manifest_path, destination):
    """Map a raw decoded RGB image from [-1, 1] to byte pixels on the CPU."""
    from PIL import Image
    manifest = json.loads(manifest_path.read_text())
    array = tensor(manifest_path, named_outputs(manifest)["decoded_image"])
    if array.ndim != 4 or array.shape[:2] != (1, 3) or not np.isfinite(array).all():
        raise ValueError("Image output must be a finite floating-point tensor with shape [1,3,height,width]")
    pixels = np.clip(array[0] / 2 + 0.5, 0, 1).transpose(1, 2, 0)
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--captured", type=Path, help="Compiled replay tensors recorded before native export")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    report = compare(args.reference, args.native, atol=args.atol, rtol=args.rtol, captured_path=args.captured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.images:
        save_image(args.reference, args.output.parent / "pytorch.png")
        save_image(args.native, args.output.parent / "flare.png")
    summary = {"status": report["status"], "output": str(args.output), "compared_tensors": len(report["tensors"])}
    if "native_capture_status" in report:
        summary["reference_status"] = report["reference_status"]
        summary["native_capture_status"] = report["native_capture_status"]
    print(json.dumps(summary))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
