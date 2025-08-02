#!/usr/bin/env python3
"""
Wrapper for the Rerun-enabled MASt3R-SLAM entry point.

1) Parses the same flags as the Rerun demo:
     --dataset, --config, --save-as, --img-size, --no-viz, [--calib]
2) Calls mast3r_slam_inference(...) under the hood
3) Reads the resulting trajectory .txt and dumps your JSON
"""

import numpy as _np

# Monkey-patch np.asarray to accept copy=… by routing to np.array
_orig_asarray = _np.asarray
def _patched_asarray(a, dtype=None, copy=False):
    if copy:
        return _np.array(a, dtype=dtype, copy=copy)
    return _orig_asarray(a, dtype=dtype)
_np.asarray = _patched_asarray

import os
os.environ.setdefault("RERUN_SPAWN", "false")  

import json
import math
import yaml
from pathlib import Path

import tyro
from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference

def main():
    # 1) Parse CLI flags into their dataclass.
    cfg = tyro.cli(InferenceConfig)

    # _this_ kicks off gRPC + SLAM logging
    mast3r_slam_inference(cfg)

    # 3) Read back the dumped trajectory:
    seq = Path(cfg.dataset).stem
    traj = Path("logs") / cfg.save_as / f"{seq}.txt"
    if not traj.exists():
        raise FileNotFoundError(f"Expected trajectory at {traj}")

    positions, yaws = [], []
    for line in traj.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        x, y = float(parts[1]), float(parts[2])
        qx, qy, qz, qw = map(float, parts[4:8])
        # yaw = atan2(2*(w*z + x*y), 1 − 2*(y²+z²))
        t0 = 2 * (qw * qz + qx * qy)
        t1 = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(t0, t1)
        positions.append([x, y])
        yaws.append(yaw)

    out = {"position": positions, "yaw": yaws}
    json_path = Path(cfg.save_as + "_traj_data.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {{'position':{len(positions)}, 'yaw':{len(yaws)}}} to {json_path}")
    

if __name__ == "__main__":
    main() # /workspace/mnt/sati-data/unstructured_datasets/Lelan/Lelan/dataset_LeLaN/dataset_LeLaN_sacson/Feb-27-2023-soda3_prop/00000015/image