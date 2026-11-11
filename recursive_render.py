#!/usr/bin/env python3

"""Render merged checkpoints recursively, save images, and write an FPS CSV.

The benchmark launches the same distributed Grendel rendering path used by
render.py.  Render images are saved after the timed benchmark so PNG I/O does
not affect the reported FPS or frame-latency statistics.

e.g.
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
recursive_render.py \
--recursive /home/ps/hjf/3dgs/res/training/mipnerf360 \
--resolutions 1.0 0.5 3840x2160

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
recursive_render.py /home/ps/hjf/3dgs/res/training/building /home/ps/hjf/3dgs/res/training/building/output/grendel_all \
--iteration 70000 \
--resolutions 1.0 0.5 640x480
Single-GPU: 
torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
--recursive /home/ps/hjf/3dgs/res/training/deep-blending \
--resolutions 1.0 0.5 3840x2160 2>&1 | tee recursive_render.log

Render an explicit NeRF camera trace:
torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
/home/ps/hjf/3dgs/res/training/building /home/ps/hjf/3dgs/res/training/building/output/grendel_all \
--iteration 70000 \
--camera-sequence /home/ps/hjf/3dgs/res/training/building/stress_nerf/fov_1.1.json \
--resolutions 1.0 0.5 3840x2160 \
--no-save-renders
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
import numpy as np

import utils.general_utils as utils
from arguments import (
    AuxiliaryParams,
    BenchmarkParams,
    DebugParams,
    DistributionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    init_args,
)
from gaussian_renderer import GaussianModel
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
from gaussian_renderer.workload_division import (
    DivisionStrategyHistoryFinal,
    start_strategy_final,
)
from scene import Scene, SceneDataset
from scene.cameras import Camera
from utils.general_utils import (
    init_distributed,
    safe_state,
    set_args,
    set_cur_iter,
    set_log_file,
)


# Keep the target checkpoint convention aligned with recursive_merge.sh.
TARGET_ITERATION = 30000

# Real-time rendering benchmark: one COLMAP camera trace frame per render.
# Do not inherit the training batch size from cfg_args/args.json.
INFERENCE_BATCH_SIZE = 1

# Edit this list for the default benchmark resolutions.  Values without an
# explicit size are scale factors relative to the dataset's original size.
DEFAULT_RESOLUTIONS = ["1.0", "0.5", "0.25"]

# Optional camera traces generated next to the merged point cloud.
PRESSURE_SEQUENCES = ("pressure_scan", "pressure_orbit")


def build_model_parser():
    parser = argparse.ArgumentParser(add_help=False)
    AuxiliaryParams(parser)
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)
    DistributionParams(parser)
    BenchmarkParams(parser)
    DebugParams(parser)
    parser.add_argument("--iteration", default=TARGET_ITERATION, type=int)
    return parser


def load_model_args(source_path, model_path, iteration):
    """Load a model's training configuration and apply render-time overrides."""
    parser = build_model_parser()
    args = parser.parse_args([])

    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.is_file():
        cfg_args = eval(cfg_path.read_text(), {"Namespace": Namespace})
        args_dict = vars(args).copy()
        args_dict.update(vars(cfg_args))
        args = Namespace(**args_dict)

    args.source_path = os.path.abspath(str(source_path))
    args.model_path = os.path.abspath(str(model_path))
    args.iteration = iteration

    return args


def parse_resolution_specs(specs):
    values = []
    for spec in specs:
        values.extend(value.strip() for value in str(spec).split(","))
    values = [value for value in values if value]
    if not values:
        raise ValueError("At least one resolution must be configured.")
    return values


def resolve_resolution(spec, base_width, base_height):
    normalized = spec.lower().replace(" ", "")
    if "x" in normalized:
        width_text, height_text = normalized.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        label = f"{width}x{height}"
    else:
        scale = float(normalized)
        if scale <= 0:
            raise ValueError(f"Resolution scale must be positive: {spec}")
        width = max(1, round(base_width * scale))
        height = max(1, round(base_height * scale))
        label = f"{width}x{height}"

    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive: {spec}")
    return label, width, height


def _positive_dimension(value):
    if value is None:
        return None
    dimension = int(round(float(value)))
    return dimension if dimension > 0 else None


def load_camera_sequence(sequence_path):
    """Load stress-nerf camera poses from a transforms-style JSON file."""
    sequence_path = Path(sequence_path).expanduser().resolve()
    contents = json.loads(sequence_path.read_text())
    frames = contents.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Camera sequence has no frames: {sequence_path}")

    width = _positive_dimension(contents.get("w"))
    height = _positive_dimension(contents.get("h"))
    cx = contents.get("cx")
    cy = contents.get("cy")
    # stress-nerf sequences may leave w/h null but provide a centered principal
    # point, from which the native image dimensions can be recovered.
    if width is None and cx is not None:
        width = _positive_dimension(2.0 * float(cx))
    if height is None and cy is not None:
        height = _positive_dimension(2.0 * float(cy))
    if width is None or height is None:
        raise ValueError(
            f"Camera sequence must specify positive w/h or centered cx/cy: {sequence_path}"
        )

    if cx is not None and abs(float(cx) - width / 2.0) > 1.0:
        raise ValueError("Off-center cx is unsupported by the current pinhole renderer.")
    if cy is not None and abs(float(cy) - height / 2.0) > 1.0:
        raise ValueError("Off-center cy is unsupported by the current pinhole renderer.")

    for distortion_name in ("k1", "k2", "k3", "k4", "p1", "p2"):
        distortion = float(contents.get(distortion_name, 0.0) or 0.0)
        if abs(distortion) > 1e-8:
            raise ValueError(
                f"Non-zero {distortion_name} distortion is unsupported by the current renderer."
            )

    fx = contents.get("fl_x")
    fy = contents.get("fl_y", fx)
    fov_x = contents.get("camera_angle_x")
    if fov_x is None:
        if fx is None or float(fx) <= 0:
            raise ValueError("Camera sequence requires camera_angle_x or positive fl_x.")
        fov_x = 2.0 * math.atan(width / (2.0 * float(fx)))
    else:
        fov_x = float(fov_x)

    if fy is not None:
        if float(fy) <= 0:
            raise ValueError("fl_y must be positive.")
        fov_y = 2.0 * math.atan(height / (2.0 * float(fy)))
    else:
        focal = width / (2.0 * math.tan(fov_x / 2.0))
        fov_y = 2.0 * math.atan(height / (2.0 * focal))

    if not (0.0 < fov_x < math.pi and 0.0 < fov_y < math.pi):
        raise ValueError(f"Invalid camera FOV: fov_x={fov_x}, fov_y={fov_y}")

    cameras = []
    for index, frame in enumerate(frames):
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(
                f"Frame {index} must have a finite 4x4 transform_matrix."
            )

        # These stress-nerf matrices are already in Grendel/COLMAP camera
        # coordinates (they match camera_to_JSON output), unlike Blender's
        # OpenGL transforms that require flipping the Y and Z axes.
        c2w = transform
        try:
            w2c = np.linalg.inv(c2w)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Frame {index} has a singular transform_matrix.") from exc

        file_path = str(frame.get("file_path", f"frame_{index:06d}"))
        cameras.append(
            Camera(
                colmap_id=index,
                R=np.transpose(w2c[:3, :3]),
                T=w2c[:3, 3],
                FoVx=fov_x,
                FoVy=fov_y,
                image=None,
                gt_alpha_mask=None,
                image_name=Path(file_path).stem,
                uid=index,
                image_path=None,
                image_width=width,
                image_height=height,
            )
        )

    return cameras


def is_dataset(path):
    return (path / "sparse").is_dir() or (path / "transforms_train.json").is_file()


def checkpoint_path(model_path, iteration):
    return model_path / "point_cloud" / f"iteration_{iteration}"


def discover_camera_traces(model_checkpoint, camera_sequence=None, include_pressure=True):
    """Return the original trace followed by any distinct pressure traces."""
    pressure_paths = {
        name: model_checkpoint / f"{name}.json" for name in PRESSURE_SEQUENCES
    }
    explicit = Path(camera_sequence).expanduser().resolve() if camera_sequence else None
    primary_name = "original"
    if explicit is not None:
        primary_name = next(
            (name for name, path in pressure_paths.items() if path.resolve() == explicit),
            "camera_sequence",
        )
    traces = [(primary_name, explicit)]
    if include_pressure:
        for name, path in pressure_paths.items():
            if path.is_file() and (explicit is None or path.resolve() != explicit):
                traces.append((name, path))
    return traces


def distributed_barrier():
    if utils.WORLD_SIZE > 1:
        dist.barrier(group=utils.DEFAULT_GROUP)


def render_batch(
    batched_cameras,
    gaussians,
    pipeline,
    background,
    strategy_history,
    render_dir=None,
    frame_indices=None,
):
    batched_strategies, gpuid2tasks = start_strategy_final(
        batched_cameras, strategy_history
    )

    # The benchmark has no need for ground-truth images.  Rendering only uses
    # the camera geometry, so do not call the regular GT-loading path here.
    screenspace = distributed_preprocess3dgs_and_all2all_final(
        batched_cameras,
        gaussians,
        pipeline,
        background,
        batched_strategies=batched_strategies,
        mode="test",
    )
    rendered_images, _ = render_final(screenspace, batched_strategies)

    # render_final returns None/scalar on ranks that do not own a camera/tile.
    # Materialize a full-size tensor before the reduction, exactly as render.py
    # does before saving its final image.
    for camera_index, (image, camera) in enumerate(zip(rendered_images, batched_cameras)):
        if image is None or len(image.shape) == 0:
            image = torch.zeros(
                (3, camera.image_height, camera.image_width),
                device="cuda",
                dtype=torch.float32,
            )
        if utils.WORLD_SIZE > 1:
            dist.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)

        if render_dir is not None and utils.GLOBAL_RANK == 0:
            if frame_indices is None:
                frame_index = camera_index
            else:
                frame_index = frame_indices[camera_index]
            torchvision.utils.save_image(
                torch.clamp(image, 0.0, 1.0),
                render_dir / f"{frame_index + 1:05d}.png",
            )

    return len(batched_cameras)


def benchmark_resolution(
    views,
    gaussians,
    pipeline,
    background,
    args,
    width,
    height,
    warmup_batches,
    render_dir=None,
):
    if not views:
        raise ValueError("No cameras available for rendering.")

    utils.set_img_size(height, width)
    for camera in views:
        camera.image_width = width
        camera.image_height = height

    dataset = SceneDataset(views)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    batch_size = max(1, min(INFERENCE_BATCH_SIZE, len(views)))
    batches = [views[index : index + batch_size] for index in range(0, len(views), batch_size)]

    warmup_batches = min(max(warmup_batches, 0), len(batches))
    for batch in batches[:warmup_batches]:
        render_batch(batch, gaussians, pipeline, background, strategy_history)
    torch.cuda.synchronize()
    distributed_barrier()

    start_time = time.perf_counter()
    frame_count = 0
    frame_latencies_ms = []
    for batch in batches:
        # bsz is fixed to one for the real-time benchmark.  Synchronizing here
        # makes this a true end-to-end per-frame latency measurement rather
        # than only a CPU enqueue interval.
        torch.cuda.synchronize()
        frame_start = time.perf_counter()
        frame_count += render_batch(
            batch, gaussians, pipeline, background, strategy_history
        )
        torch.cuda.synchronize()
        frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
    torch.cuda.synchronize()
    distributed_barrier()
    elapsed = time.perf_counter() - start_time

    # Rank 0 measures the wall-clock interval after all ranks have completed.
    # Use the slowest rank's elapsed time for the distributed FPS result.
    elapsed_tensor = torch.tensor([elapsed], device="cuda", dtype=torch.float64)
    if utils.WORLD_SIZE > 1:
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=utils.DEFAULT_GROUP)
    elapsed = float(elapsed_tensor.item())

    # Use the slowest rank's latency for each frame.  This collective happens
    # after the timed loop and therefore does not inflate the FPS measurement.
    latency_tensor = torch.tensor(frame_latencies_ms, device="cuda", dtype=torch.float64)
    if utils.WORLD_SIZE > 1:
        dist.all_reduce(latency_tensor, op=dist.ReduceOp.MAX, group=utils.DEFAULT_GROUP)
    frame_latencies_ms = latency_tensor.cpu().tolist()
    sorted_latencies = sorted(frame_latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, max(0, int(len(sorted_latencies) * 0.95) - 1))

    # Save a separate pass after timing.  Disk I/O is intentionally excluded
    # from the real-time FPS and latency measurements above.
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        save_dataset = SceneDataset(views)
        save_strategy_history = DivisionStrategyHistoryFinal(
            save_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
        )
        for start in range(0, len(views), batch_size):
            batch = views[start : start + batch_size]
            render_batch(
                batch,
                gaussians,
                pipeline,
                background,
                save_strategy_history,
                render_dir=render_dir,
                frame_indices=range(start, start + len(batch)),
            )
        distributed_barrier()

    return {
        "frames": frame_count,
        "seconds": elapsed,
        "fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "avg_frame_latency_ms": statistics.mean(frame_latencies_ms),
        "p50_frame_latency_ms": statistics.median(frame_latencies_ms),
        "p95_frame_latency_ms": sorted_latencies[p95_index],
        "min_frame_latency_ms": sorted_latencies[0],
        "max_frame_latency_ms": sorted_latencies[-1],
    }


def render_dataset(
    source_path,
    model_path,
    iteration,
    resolution_specs,
    split,
    warmup_batches,
    camera_sequence=None,
    save_renders=True,
    include_pressure=True,
):
    model_checkpoint = checkpoint_path(model_path, iteration)
    merged_point_cloud = model_checkpoint / "point_cloud.ply"
    if not merged_point_cloud.is_file():
        raise FileNotFoundError(
            f"Merged point cloud not found: {merged_point_cloud}. "
            "Run recursive_merge.sh first."
        )

    args = load_model_args(source_path, model_path, iteration)
    # recursive_render.py has its own explicit --resolutions benchmark
    # control.  Keep Scene at the dataset's native camera size here so that
    # those benchmark resolutions remain relative to the original images.
    args.resolution = 1
    init_args(args)

    # Force the Grendel distributed rendering path for multi-GPU execution.
    # The merged point cloud is sharded by load_raw_ply when this is enabled.
    if utils.WORLD_SIZE > 1:
        args.gaussians_distribution = True
        args.image_distribution = True
        args.distributed_dataset_storage = False
        args.local_sampling = False
    args.lazy_image_loading = True
    args.image_loading_mode = "lazy"
    args.preload_dataset_to_gpu = False
    args.multiprocesses_image_loading = False
    args.bsz = INFERENCE_BATCH_SIZE
    set_args(args)

    log_file = open(os.devnull, "w")
    set_log_file(log_file)
    set_cur_iter(iteration)

    dataset_args = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(args)
    pipeline_args = PipelineParams(argparse.ArgumentParser()).extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset_args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    background_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

    results = []
    failures = []
    traces = discover_camera_traces(model_checkpoint, camera_sequence, include_pressure)
    if include_pressure and utils.GLOBAL_RANK == 0:
        found = [name for name, _ in traces if name in PRESSURE_SEQUENCES]
        print(
            f"[INFO] {source_path.name}: pressure sequences: {', '.join(found) or 'none found'}",
            file=sys.stderr,
            flush=True,
        )
    for trajectory, sequence_path in traces:
        try:
            if sequence_path is not None:
                views = load_camera_sequence(sequence_path)
                actual_split = "camera_sequence"
            else:
                actual_split = split
                if split == "test":
                    views = scene.getTestCameras()
                    if not views:
                        views = scene.getTrainCameras()
                        actual_split = "train"
                else:
                    views = scene.getTrainCameras()
            if not views:
                raise ValueError(f"No cameras found for dataset: {source_path}")

            base_width = int(views[0].image_width)
            base_height = int(views[0].image_height)
            for camera in views:
                if camera.image_width != base_width or camera.image_height != base_height:
                    raise ValueError("All cameras must have the same resolution.")

            for spec in resolution_specs:
                label, width, height = resolve_resolution(spec, base_width, base_height)
                render_root = model_checkpoint / "render"
                if trajectory in PRESSURE_SEQUENCES:
                    render_root = render_root / trajectory
                render_dir = render_root / label if save_renders else None
                # Inference only: do not retain training autograd buffers.
                with torch.no_grad():
                    benchmark = benchmark_resolution(
                        views,
                        gaussians,
                        pipeline_args,
                        background,
                        args,
                        width,
                        height,
                        warmup_batches,
                        render_dir,
                    )
                results.append(
                    {
                        "trajectory": trajectory,
                        "split": actual_split,
                        "camera_sequence": str(sequence_path.resolve()) if sequence_path else "",
                        "resolution": label,
                        "width": width,
                        "height": height,
                        **benchmark,
                    }
                )
        except Exception as exc:
            failures.append((f"{source_path.name}/{trajectory}", str(exc)))
            if utils.GLOBAL_RANK == 0:
                print(
                    f"[FAILED] {source_path.name}/{trajectory}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)

    log_file.close()
    return results, failures


def discover_datasets(dataset_root):
    return sorted(
        path for path in dataset_root.iterdir() if path.is_dir()
    )


def resolve_paths(cli_args):
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "../res/training/tanksandtemples/train"

    if cli_args.recursive:
        dataset_root_text = cli_args.path1 or os.environ.get("SOURCE_PATH", "")
        if not dataset_root_text:
            raise ValueError("dataset root is required")
        dataset_root = Path(dataset_root_text)
        model_root_text = cli_args.path2 or os.environ.get("MODEL_PATH", "")
        if not model_root_text:
            model_root = dataset_root
        else:
            model_root = Path(model_root_text)
        return True, dataset_root, model_root

    source_path = Path(
        cli_args.path1
        or os.environ.get("SOURCE_PATH", str(default_source))
    )
    model_path = Path(
        cli_args.path2
        or os.environ.get("MODEL_PATH", str(source_path / "trained"))
    )
    return False, source_path, model_path


def main():
    parser = argparse.ArgumentParser(
        description="Recursively benchmark Grendel rendering and save rendered images."
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("path1", nargs="?")
    parser.add_argument("path2", nargs="?")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=None,
        help="Scale factors or WxH values, e.g. 1.0 0.5 1920x1080.",
    )
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--iteration", type=int, default=TARGET_ITERATION)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--camera-sequence",
        default=None,
        help="Render frames from a transforms-style NeRF JSON instead of the COLMAP split.",
    )
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--no-save-renders",
        action="store_true",
        help="Only benchmark rendering; do not save PNG outputs.",
    )
    parser.add_argument(
        "--skip-pressure-sequences",
        action="store_true",
        help="Do not automatically benchmark pressure_scan.json and pressure_orbit.json.",
    )
    cli_args = parser.parse_args()

    recursive, first_path, second_path = resolve_paths(cli_args)
    if recursive and cli_args.camera_sequence:
        raise ValueError(
            "--camera-sequence is only supported in non-recursive mode because a camera trace belongs to one scene."
        )
    resolution_specs = parse_resolution_specs(
        cli_args.resolutions or DEFAULT_RESOLUTIONS
    )

    if recursive:
        dataset_root = first_path
        model_root = second_path
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        dataset_paths = discover_datasets(dataset_root)
        if not dataset_paths:
            raise FileNotFoundError(
                f"No dataset directories found under: {dataset_root}"
            )
        output_root = model_root
    else:
        dataset_paths = [first_path]
        output_root = second_path

    # Initialize distributed execution once.  Per-model training arguments are
    # loaded later because each trained model has its own cfg_args file.
    bootstrap_args = Namespace()
    init_distributed(bootstrap_args)
    safe_state(True)

    rows = []
    failures = []
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        if not is_dataset(dataset_path):
            if recursive:
                if utils.GLOBAL_RANK == 0:
                    print(f"Skipping non-dataset directory: {dataset_name}")
                continue
            raise ValueError(f"Unrecognized dataset layout: {dataset_path}")

        model_path = (
            dataset_path / "trained"
            if recursive and second_path == first_path
            else (second_path / dataset_name if recursive else second_path)
        )

        try:
            results, trace_failures = render_dataset(
                dataset_path,
                model_path,
                cli_args.iteration,
                resolution_specs,
                cli_args.split,
                cli_args.warmup_batches,
                camera_sequence=cli_args.camera_sequence,
                save_renders=not cli_args.no_save_renders,
                include_pressure=not cli_args.skip_pressure_sequences,
            )
            failures.extend(trace_failures)
            if utils.GLOBAL_RANK == 0:
                for result in results:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "model_path": str(model_path),
                            "iteration": cli_args.iteration,
                            "bsz": INFERENCE_BATCH_SIZE,
                            "split": result["split"],
                            "trajectory": result["trajectory"],
                            "camera_sequence": result["camera_sequence"],
                            "resolution": result["resolution"],
                            "width": result["width"],
                            "height": result["height"],
                            "frames": result["frames"],
                            "seconds": f'{result["seconds"]:.6f}',
                            "fps": f'{result["fps"]:.6f}',
                            "avg_frame_latency_ms": f'{result["avg_frame_latency_ms"]:.6f}',
                            "p50_frame_latency_ms": f'{result["p50_frame_latency_ms"]:.6f}',
                            "p95_frame_latency_ms": f'{result["p95_frame_latency_ms"]:.6f}',
                            "min_frame_latency_ms": f'{result["min_frame_latency_ms"]:.6f}',
                            "max_frame_latency_ms": f'{result["max_frame_latency_ms"]:.6f}',
                            "world_size": utils.WORLD_SIZE,
                        }
                    )
        except Exception as exc:
            failures.append((dataset_name, str(exc)))
            if utils.GLOBAL_RANK == 0:
                print(
                    f"[FAILED] {dataset_name}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)

    if utils.GLOBAL_RANK == 0:
        csv_path = Path(cli_args.csv_path) if cli_args.csv_path else output_root / "recursive_render_fps.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset",
            "model_path",
            "iteration",
            "bsz",
            "split",
            "trajectory",
            "camera_sequence",
            "resolution",
            "width",
            "height",
            "frames",
            "seconds",
            "fps",
            "avg_frame_latency_ms",
            "p50_frame_latency_ms",
            "p95_frame_latency_ms",
            "min_frame_latency_ms",
            "max_frame_latency_ms",
            "world_size",
        ]
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"FPS CSV written to: {csv_path}")
        if failures:
            print("Failed datasets:")
            for dataset_name, error in failures:
                print(f"  - {dataset_name}: {error}")

    distributed_barrier()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
