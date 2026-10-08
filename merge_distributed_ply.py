#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge rank-sharded Gaussian PLY files into a single point_cloud.ply.'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing point_cloud_rk*_ws*.ply files.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output PLY path. Defaults to <input_dir>/point_cloud.ply.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output file if it already exists.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f'Input directory not found: {input_dir}')

    shard_paths = sorted(input_dir.glob('point_cloud_rk*_ws*.ply'))
    if not shard_paths:
        raise SystemExit(
            f'No distributed shards found in {input_dir}. Expected point_cloud_rk*_ws*.ply.'
        )

    output_path = args.output.expanduser().resolve() if args.output else input_dir / 'point_cloud.ply'
    if output_path.exists() and not args.force:
        raise SystemExit(
            f'Output already exists: {output_path}. Use --force to overwrite.'
        )

    merged_arrays = []
    dtype_ref = None
    total_points = 0

    for shard_path in shard_paths:
        ply = PlyData.read(str(shard_path))
        vertex_array = ply['vertex'].data
        if dtype_ref is None:
            dtype_ref = vertex_array.dtype
        elif vertex_array.dtype != dtype_ref:
            raise SystemExit(
                f'Attribute layout mismatch in {shard_path}. All shards must share the same dtype.'
            )

        merged_arrays.append(vertex_array)
        total_points += len(vertex_array)
        print(f'{shard_path.name}: {len(vertex_array)} points')

    merged_vertex_array = np.concatenate(merged_arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged_vertex_array, 'vertex')]).write(str(output_path))

    print(f'Wrote {output_path}')
    print(f'Total points: {total_points}')


if __name__ == '__main__':
    main()
