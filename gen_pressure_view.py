#!/usr/bin/env python3
"""Generate repeatable scene-aware camera trajectories from a Gaussian PLY.

Examples:
    python gen_pressure_view.py scene/point_cloud.ply --trajectory orbit \
        --count 600 --output orbit.json
    python gen_pressure_view.py scene/point_cloud.ply --trajectory scan \
        --count 1200 --up-axis z --output elevation_radius.json

The output uses camera-to-world matrices in Grendel/COLMAP convention
(+X right, +Y down, +Z forward), as consumed by recursive_render.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


MAX_CAMERAS = 3000


def _count(value):
    count = int(value)
    if not 0 <= count <= MAX_CAMERAS:
        raise argparse.ArgumentTypeError(f"camera count must be between 0 and {MAX_CAMERAS}")
    return count


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def axis_basis(name):
    """Return a right-handed horizontal basis and the requested world-up axis."""
    sign = -1.0 if name.startswith("-") else 1.0
    axis = name.lstrip("-")
    up = np.zeros(3, dtype=np.float64)
    up["xyz".index(axis)] = sign
    first = np.array([0.0, 1.0, 0.0] if axis == "x" else [1.0, 0.0, 0.0])
    second = np.cross(up, first)
    return first, second, up


def basis_from_up(up, first_hint=None):
    """Construct an orthonormal horizontal basis for an arbitrary up vector."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    if first_hint is None:
        first_hint = np.eye(3)[np.argmin(np.abs(up))]
    first = np.asarray(first_hint, dtype=np.float64)
    first = first - np.dot(first, up) * up
    length = np.linalg.norm(first)
    if length <= 1e-10:
        raise ValueError("Reference camera lies on the vertical axis through the trajectory center")
    first /= length
    return first, np.cross(up, first), up


def load_reference_cameras(ply_path, reference_path=None):
    """Find the training cameras.json next to the model, if available."""
    if reference_path is None:
        ply = Path(ply_path).expanduser()
        candidates = [parent / "cameras.json" for parent in list(ply.parents)[:3]]
        reference_path = next((path for path in candidates if path.is_file()), None)
        if reference_path is None:
            return None
    path = Path(reference_path).expanduser().resolve()
    contents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contents, list) or not contents:
        raise ValueError(f"Reference cameras must be a non-empty cameras.json list: {path}")
    positions, rotations = [], []
    for index, camera in enumerate(contents):
        position = np.asarray(camera.get("position"), dtype=np.float64)
        rotation = np.asarray(camera.get("rotation"), dtype=np.float64)
        if (position.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or np.linalg.det(rotation) <= 0):
            raise ValueError(f"Invalid reference camera pose at index {index}: {path}")
        positions.append(position)
        rotations.append(rotation)
    return {
        "path": str(path),
        "cameras": contents,
        "positions": np.asarray(positions),
        "rotations": np.asarray(rotations),
    }


def infer_reference_up(rotations):
    """Estimate upright from the common normal to camera image-right axes."""
    right = rotations[:, :, 0]
    camera_up = -rotations[:, :, 1]
    mean_up = camera_up.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(right.T @ right / len(right))
    up = eigenvectors[:, 0]
    alignment = float(np.dot(up, mean_up))
    if eigenvalues[1] - eigenvalues[0] < 0.02 or abs(alignment) < 0.05:
        length = np.linalg.norm(mean_up)
        if length < 0.2:
            raise ValueError("Cannot infer world up from the reference cameras; specify --up-axis")
        return mean_up / length, "camera_mean_up"
    if alignment < 0:
        up = -up
    return up, "camera_right_axes_pca"


def reference_focus(reference):
    """Return a shared camera-ray focus only when cameras agree on it."""
    positions = reference["positions"]
    forward = reference["rotations"][:, :, 2]
    if len(positions) < 3:
        return None
    projectors = np.eye(3)[None, :, :] - forward[:, :, None] * forward[:, None, :]
    matrix = projectors.sum(axis=0)
    if np.linalg.cond(matrix) > 1e6:
        return None
    target = np.linalg.solve(matrix, np.einsum("nij,nj->i", projectors, positions))
    if not np.isfinite(target).all():
        return None
    front = np.sum((target - positions) * forward, axis=1) > 0
    return target if np.mean(front) >= 0.8 else None


def select_reference_index(reference, center):
    """Prefer a real camera looking near the scene center at a typical distance."""
    offsets = center - reference["positions"]
    local = np.einsum("nij,ni->nj", reference["rotations"], offsets)
    distances = np.linalg.norm(offsets, axis=1)
    valid = (local[:, 2] > 1e-10) & (distances > 1e-10)
    if not valid.any():
        return 0, "fallback_first"
    median = float(np.median(distances[distances > 1e-10]))
    scores = np.full(len(distances), np.inf)
    for i in np.flatnonzero(valid):
        camera = reference["cameras"][i]
        try:
            tan_x = float(camera["width"]) / (2 * float(camera["fx"]))
            tan_y = float(camera["height"]) / (2 * float(camera["fy"]))
            if not (math.isfinite(tan_x) and math.isfinite(tan_y) and tan_x > 0 and tan_y > 0):
                raise ValueError("invalid camera intrinsics")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            tan_x = tan_y = math.tan(math.radians(30))
        offset = math.hypot(local[i, 0] / (local[i, 2] * tan_x),
                            local[i, 1] / (local[i, 2] * tan_y))
        scores[i] = offset + 0.25 * abs(math.log(distances[i] / median))
    return int(np.argmin(scores)), "auto_scene_center"


def scene_statistics(ply_path, max_samples=500000, quantile=0.01, seed=0):
    """Estimate robust bounds, center and PCA from deterministic PLY samples.

    Binary scalar PLYs are memory-mapped by plyfile, so only sampled positions
    are copied into RAM. Samples are stratified over the vertex order.
    """
    if max_samples <= 0 or not 0.0 <= quantile < 0.5:
        raise ValueError("max_samples must be positive and quantile must be in [0, 0.5)")
    ply = PlyData.read(str(ply_path), mmap="r")
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    vertices = ply["vertex"].data
    if not all(name in vertices.dtype.names for name in ("x", "y", "z")):
        raise ValueError("PLY vertex element must contain x, y and z")
    total = len(vertices)
    if total == 0:
        raise ValueError("PLY contains no Gaussian centers")

    sample_count = min(total, max_samples)
    if sample_count == total:
        indices = np.arange(total, dtype=np.int64)
    else:
        # One reproducibly jittered index per equal-sized stratum. Keeping
        # indices ordered also avoids random access across the entire file.
        strata = np.arange(sample_count, dtype=np.int64)
        lower = strata * total // sample_count
        upper = (strata + 1) * total // sample_count
        rng = np.random.default_rng(seed)
        indices = lower + (rng.random(sample_count) * (upper - lower)).astype(np.int64)

    points = np.column_stack(
        [np.asarray(vertices[name][indices], dtype=np.float64) for name in ("x", "y", "z")]
    )
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("PLY sample contains no finite Gaussian centers")
    lower = np.quantile(points, quantile, axis=0)
    upper = np.quantile(points, 1.0 - quantile, axis=0)
    inside = ((points >= lower) & (points <= upper)).all(axis=1)
    robust_points = points[inside]
    if not len(robust_points):
        raise ValueError("No sampled points remain inside the quantile bounds")
    center = robust_points.mean(axis=0)
    centered = robust_points - center
    covariance = centered.T @ centered / max(len(robust_points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for column in range(3):
        pivot = np.argmax(np.abs(eigenvectors[:, column]))
        if eigenvectors[pivot, column] < 0:
            eigenvectors[:, column] *= -1
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1

    return {
        "total_gaussians": total,
        "sampled_gaussians": sample_count,
        "finite_samples": len(points),
        "inlier_samples": len(robust_points),
        "quantile": quantile,
        "sample_seed": seed,
        "bounds_min": lower.tolist(),
        "bounds_max": upper.tolist(),
        "center": center.tolist(),
        "pca_eigenvalues": eigenvalues.tolist(),
        "pca_axes": eigenvectors.T.tolist(),
    }


def look_at(position, target, up):
    """Camera-to-world pose with COLMAP right/down/forward camera axes."""
    forward = np.asarray(target, dtype=np.float64) - position
    length = np.linalg.norm(forward)
    if length <= 1e-12:
        raise ValueError("Camera position coincides with its look-at target")
    forward /= length
    right = np.cross(forward, up)
    right_length = np.linalg.norm(right)
    if right_length <= 1e-12:
        raise ValueError("Camera view direction is parallel to the up axis")
    right /= right_length
    down = np.cross(forward, right)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = position
    return transform


def trajectory_position(t, trajectory, center, reference_radius, basis, args):
    """Evaluate the same continuous trajectory independently of frame count."""
    first, second, up = basis
    azimuth = math.radians(args.start_angle) + 2.0 * math.pi * args.turns * t
    horizontal = math.cos(azimuth) * first + math.sin(azimuth) * second

    if trajectory == "orbit":
        low, high = args.orbit_elevation
        elevation = math.radians((low + high) / 2.0 + (high - low) / 2.0 * math.sin(2.0 * math.pi * t))
        radius = reference_radius * args.orbit_radius
        return center + radius * (math.cos(elevation) * horizontal + math.sin(elevation) * up)

    # Walk all four (radius, height) corners and return to the start. Smoothstep
    # gives zero radial/vertical velocity at each corner and closes the path.
    r0, r1 = args.scan_radius
    h0, h1 = args.scan_height
    corners = ((r0, h0), (r1, h0), (r1, h1), (r0, h1), (r0, h0))
    phase = 4.0 * t
    segment = min(int(phase), 3)
    fraction = phase - segment
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    radius = (1.0 - blend) * corners[segment][0] + blend * corners[segment + 1][0]
    height = (1.0 - blend) * corners[segment][1] + blend * corners[segment + 1][1]
    return center + reference_radius * (radius * horizontal + height * up)


def reference_trajectory_pose(t, center, reference_radius, basis, args, anchor):
    """Join a known camera to the scene-aware loop without an abrupt jump."""
    first, second, up = basis
    fade = 0.15
    distance_to_endpoint = min(t, 1.0 - t)
    fraction = min(max(distance_to_endpoint / fade, 0.0), 1.0)
    anchor_weight = 1.0 - fraction * fraction * (3.0 - 2.0 * fraction)
    azimuth = math.radians(args.start_angle) + 2.0 * math.pi * args.turns * t
    horizontal = math.cos(azimuth) * first + math.sin(azimuth) * second

    if args.trajectory == "orbit":
        low, high = args.orbit_elevation
        elevation = math.radians((low + high) / 2.0 + (high - low) / 2.0 * math.sin(2.0 * math.pi * t))
        nominal_radius = reference_radius * args.orbit_radius * math.cos(elevation)
        nominal_height = reference_radius * args.orbit_radius * math.sin(elevation)
    else:
        # The scan completes all four corners during the middle 70% of the
        # path; the remaining sections smoothly join the reference camera.
        phase = 4.0 * min(max((t - fade) / (1.0 - 2.0 * fade), 0.0), 1.0)
        segment = min(int(phase), 3)
        fraction = phase - segment
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        r0, r1 = args.scan_radius
        h0, h1 = args.scan_height
        corners = ((r0, h0), (r1, h0), (r1, h1), (r0, h1), (r0, h0))
        radius_factor = (1.0 - blend) * corners[segment][0] + blend * corners[segment + 1][0]
        height_factor = (1.0 - blend) * corners[segment][1] + blend * corners[segment + 1][1]
        nominal_radius = reference_radius * radius_factor
        nominal_height = anchor["median_height"] + reference_radius * height_factor

    radius = anchor_weight * anchor["radius"] + (1.0 - anchor_weight) * nominal_radius
    height = anchor_weight * anchor["height"] + (1.0 - anchor_weight) * nominal_height
    position = center + radius * horizontal + height * up
    target = anchor_weight * anchor["target"] + (1.0 - anchor_weight) * center
    return position, target


def generate_sequence(stats, args, reference=None):
    if reference is None and args.up_axis == "auto":
        print("warning: no reference cameras.json found; falling back to world +Z up. "
              "Use --reference-cameras or --up-axis for other datasets.", file=sys.stderr)
    if args.up_axis == "auto" and reference is not None:
        up, up_source = infer_reference_up(reference["rotations"])
    else:
        axis = "z" if args.up_axis == "auto" else args.up_axis
        up = axis_basis(axis)[2]
        up_source = "fallback_z" if args.up_axis == "auto" else "explicit_axis"

    focus = reference_focus(reference) if reference is not None and args.center is None else None
    center = np.asarray(
        args.center if args.center is not None else focus if focus is not None else stats["center"],
        dtype=np.float64,
    )
    center_source = "explicit" if args.center is not None else "reference_rays" if focus is not None else "gaussian_statistics"
    span = np.asarray(stats["bounds_max"]) - np.asarray(stats["bounds_min"])
    basis = basis_from_up(up)
    horizontal_extent = [float(np.dot(np.abs(axis), span)) for axis in basis[:2]]
    reference_radius = 0.5 * math.hypot(*horizontal_extent)
    radius_source = "gaussian_bounds"
    anchor = None
    reference_info = None
    if reference is not None:
        selection = "explicit"
        if args.reference_index is None:
            args.reference_index, selection = select_reference_index(reference, center)
            if selection == "fallback_first":
                print("warning: no reference camera faces the estimated scene center; using camera 0. "
                      "Consider specifying --center or --reference-index.", file=sys.stderr)
        if not 0 <= args.reference_index < len(reference["positions"]):
            raise ValueError("--reference-index is outside the reference camera list")
        position = reference["positions"][args.reference_index]
        forward = reference["rotations"][args.reference_index, :, 2]
        forward = forward / np.linalg.norm(forward)
        delta = position - center
        basis = basis_from_up(up, delta)
        height = float(np.dot(delta, up))
        radius = float(np.linalg.norm(delta - height * up))
        offsets = reference["positions"] - center
        heights = offsets @ up
        radii = np.linalg.norm(offsets - heights[:, None] * up, axis=1)
        usable = radii > 1e-10
        if not usable.any():
            raise ValueError("Reference cameras have zero horizontal distance from the trajectory center")
        reference_radius = float(np.median(radii[usable]))
        radius_source = "median_reference_camera_radius"
        if args.orbit_elevation is None:
            elevations = np.degrees(np.arctan2(heights[usable], radii[usable]))
            low, high = np.percentile(elevations, [10, 90])
            middle = float(np.median(elevations))
            low, high = min(low, middle - 5.0), max(high, middle + 5.0)
            args.orbit_elevation = (max(float(low), -80.0), min(float(high), 80.0))
        distance = max(float(np.linalg.norm(delta)), reference_radius)
        anchor = {
            "radius": radius,
            "height": height,
            "median_height": float(np.median(heights)),
            "target": position + forward * distance,
        }
        reference_info = {
            "path": reference["path"],
            "count": len(reference["positions"]),
            "index": args.reference_index,
            "selection": selection,
            "image_name": reference["cameras"][args.reference_index].get("img_name"),
            "start_position": position.tolist(),
            "start_forward": forward.tolist(),
            "median_height": anchor["median_height"],
        }
    elif args.orbit_elevation is None:
        args.orbit_elevation = (-10.0, 45.0)

    if not math.isfinite(reference_radius) or reference_radius <= 1e-12:
        raise ValueError("Scene has zero horizontal extent; choose a different up axis or input scene")
    if not np.isfinite(center).all():
        raise ValueError("Scene center must be finite")
    if not math.isfinite(args.orbit_radius) or args.orbit_radius <= 0:
        raise ValueError("--orbit-radius must be positive")
    if not (0 < args.scan_radius[0] <= args.scan_radius[1] and all(math.isfinite(x) for x in args.scan_radius)):
        raise ValueError("--scan-radius must contain positive minimum and maximum values")
    if not (all(math.isfinite(x) for x in args.scan_height) and args.scan_height[0] <= args.scan_height[1]):
        raise ValueError("--scan-height must contain finite minimum and maximum values")
    if not (-89.9 < args.orbit_elevation[0] <= args.orbit_elevation[1] < 89.9):
        raise ValueError("--orbit-elevation must be ordered and strictly between -89.9 and 89.9 degrees")
    if not math.isfinite(args.start_angle) or args.turns <= 0:
        raise ValueError("--start-angle must be finite and --turns must be positive")
    fov_source = "explicit"
    if args.fov_x is None:
        camera = reference["cameras"][args.reference_index] if reference is not None else {}
        if camera.get("width") and camera.get("fx") and float(camera["fx"]) > 0:
            args.fov_x = math.degrees(2.0 * math.atan(float(camera["width"]) / (2.0 * float(camera["fx"]))))
            fov_source = "reference_camera"
        else:
            args.fov_x = 60.0
            fov_source = "default"
    if not 0 < args.fov_x < 180:
        raise ValueError("--fov-x must be between 0 and 180 degrees")

    fov_x = math.radians(args.fov_x)
    focal = args.width / (2.0 * math.tan(fov_x / 2.0))
    # The same normalized path is sampled for every count. For closed paths,
    # n >= 2 includes both endpoints; n == 1 uses the start; n == 0 is empty.
    times = np.linspace(0.0, 1.0, args.count) if args.count > 1 else np.zeros(args.count)
    frames = []
    for index, t in enumerate(times):
        if anchor is not None:
            position, target = reference_trajectory_pose(float(t), center, reference_radius, basis, args, anchor)
        else:
            position = trajectory_position(float(t), args.trajectory, center, reference_radius, basis, args)
            target = center
        frames.append({
            "file_path": f"{index + 1:06d}.png",
            "transform_matrix": look_at(position, target, basis[2]).tolist(),
        })

    return {
        "camera_angle_x": fov_x,
        "fl_x": focal,
        "fl_y": focal,
        "cx": args.width / 2.0,
        "cy": args.height / 2.0,
        "w": args.width,
        "h": args.height,
        "frames": frames,
        "metadata": {
            "coordinate_convention": "COLMAP camera-to-world: +X right, +Y down, +Z forward",
            "trajectory": args.trajectory,
            "scene_ply": str(Path(args.ply).expanduser().resolve()),
            "scene_statistics": stats,
            "look_at": center.tolist(),
            "center_source": center_source,
            "reference_radius": reference_radius,
            "reference_radius_source": radius_source,
            "reference_camera": reference_info,
            "up_axis": args.up_axis,
            "up_vector": up.tolist(),
            "up_source": up_source,
            "fov_source": fov_source,
            "turns": args.turns,
            "start_angle_degrees": args.start_angle,
            "orbit_radius_factor": args.orbit_radius,
            "orbit_elevation_degrees": list(args.orbit_elevation),
            "scan_radius_factors": list(args.scan_radius),
            "scan_height_factors": list(args.scan_height),
            "sampling": "uniform t in [0,1]; both endpoints included for count >= 2",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", help="Merged Gaussian point_cloud.ply")
    parser.add_argument("--output", required=True, help="Output camera sequence JSON")
    parser.add_argument("--trajectory", choices=("orbit", "scan"), default="orbit", help="Orbit (A) or elevation/radius scan (B)")
    parser.add_argument("--count", type=_count, default=600, help="Number of cameras, 0 to 3000 (default: 600)")
    parser.add_argument("--width", type=_positive_int, default=1920)
    parser.add_argument("--height", type=_positive_int, default=1080)
    parser.add_argument("--fov-x", type=float, default=None, help="Horizontal FOV in degrees (default: reference camera, else 60)")
    parser.add_argument("--up-axis", choices=("auto", "x", "y", "z", "-x", "-y", "-z"), default="auto", help="World up axis (default: infer from cameras.json, else +Z)")
    parser.add_argument("--reference-cameras", help="Training cameras.json (default: find it in the PLY model directory)")
    parser.add_argument("--reference-index", type=int, default=None, help="Reference camera used as the trajectory start (default: select automatically)")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override the robust scene center")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Starting azimuth in degrees")
    parser.add_argument("--turns", type=_positive_int, default=1, help="Number of complete azimuth revolutions")
    parser.add_argument("--orbit-radius", type=float, default=1.0, help="Orbit radius in reference radii")
    parser.add_argument("--orbit-elevation", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Orbit elevation range in degrees (default: infer from cameras)")
    parser.add_argument("--scan-radius", type=float, nargs=2, default=(0.75, 1.25), metavar=("MIN", "MAX"), help="Horizontal scan radii in reference radii")
    parser.add_argument("--scan-height", type=float, nargs=2, default=(-0.1, 0.25), metavar=("MIN", "MAX"), help="Scan heights relative to median reference height, in reference radii")
    parser.add_argument("--quantile", type=float, default=0.01, help="Per-axis lower/upper quantile for robust bounds")
    parser.add_argument("--max-samples", type=_positive_int, default=500000, help="Maximum sampled Gaussian centers for scene statistics")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sample seed")
    args = parser.parse_args(argv)
    try:
        stats = scene_statistics(args.ply, args.max_samples, args.quantile, args.seed)
        reference = load_reference_cameras(args.ply, args.reference_cameras)
        if reference is None and args.reference_index is not None:
            raise ValueError("--reference-index requires reference cameras")
        sequence = generate_sequence(stats, args, reference)
        output = Path(args.output).expanduser()
        if output.resolve() == Path(args.ply).expanduser().resolve():
            raise ValueError("Output JSON path must differ from the input PLY")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(sequence, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Wrote {len(sequence['frames'])} {args.trajectory} cameras to {output}")
    print(f"Scene center: {sequence['metadata']['look_at']}; reference radius: {sequence['metadata']['reference_radius']:.6g}")
    print(f"World up: {sequence['metadata']['up_vector']} ({sequence['metadata']['up_source']}); "
          f"reference camera: {sequence['metadata']['reference_camera']}")
    if not args.count:
        print("An empty sequence cannot be rendered; use --count 1 or greater for recursive_render.py.")


if __name__ == "__main__":
    main()
