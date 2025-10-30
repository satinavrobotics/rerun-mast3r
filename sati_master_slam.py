#!/usr/bin/env python3
"""
Wrapper for the Rerun-enabled MASt3R-SLAM entry point.

1) Parses the same flags as the Rerun demo:
     --dataset, --config, --save-as, --img-size, --no-viz, [--calib]
2) Calls mast3r_slam_inference(...) under the hood
3) Reads the resulting trajectory .txt and dumps your JSON
"""

import numpy as _np
_orig_asarray = _np.asarray
def _patched_asarray(a, dtype=None, copy=False):
    if copy:
        return _np.array(a, dtype=dtype, copy=copy)
    return _orig_asarray(a, dtype=dtype)
_np.asarray = _patched_asarray

import os, json, math
from pathlib import Path
import tyro

# Disable rerun spawn unless explicitly allowed
os.environ.setdefault("RERUN_SPAWN", "false")

def main():
    # Import here to avoid import errors when loading as FastAPI app
    from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference

    cfg = tyro.cli(InferenceConfig)
    mast3r_slam_inference(cfg)

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
        t0 = 2 * (qw * qz + qx * qy)
        t1 = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(t0, t1)
        positions.append([x, y])
        yaws.append(yaw)

    out = {"position": positions, "yaw": yaws}
    json_path = Path(cfg.save_as + "_traj_data.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {{'position':{len(positions)}, 'yaw':{len(yaws)}}} to {json_path}")

# ------------------------------------------------------------------
# FastAPI server wrapper (no separate file needed) - CImbi
# ------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Form
import tempfile, subprocess

app = FastAPI(title="MASt3R-SLAM Server")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/estimate_pose")
async def estimate_pose(
    image: UploadFile = File(...),
    config: str = Form("default"),
    save_as: str = Form("api_req")
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(await image.read())
    tmp.close()

    # Call our own CLI logic for consistency
    cmd = [
        "python", "/workspace/rerun_mast3r/sati_master_slam.py",
        "--dataset", tmp.name,
        "--config", config,
        "--save-as", save_as,
        "--no-viz",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out_file = Path(f"{save_as}_traj_data.json")
    if out_file.exists():
        data = json.loads(out_file.read_text())
    else:
        data = {"stdout": result.stdout, "stderr": result.stderr}

    os.unlink(tmp.name)
    return data

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()