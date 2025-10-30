#!/usr/bin/env python3
"""
Wrapper for the Rerun-enabled MASt3R-SLAM entry point.

1) Parses the same flags as the Rerun demo:
     --dataset, --config, --save-as, --img-size, --no-viz, [--calib]
2) Calls mast3r_slam_inference(...) under the hood
3) Reads the resulting trajectory .txt and dumps your JSON
"""

# Patch numpy.asarray to support 'copy' parameter for numpy < 2.0
# This is needed for rerun-sdk 0.23.1 compatibility with numpy 1.26.4
import numpy as _np
_orig_asarray = _np.asarray
def _patched_asarray(a, dtype=None, order=None, *, like=None, copy=None):
    """Backport copy parameter support for numpy < 2.0"""
    # Build kwargs, filtering out None values
    kwargs = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if order is not None:
        kwargs['order'] = order
    # Note: 'like' parameter not supported in numpy 1.26.4, skip it

    if copy is True:
        # Explicit copy requested - use np.array which always copies
        return _np.array(a, copy=True, **kwargs)
    else:
        # Default behavior - no copy or copy=False/None
        return _orig_asarray(a, **kwargs)
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
 
'''  
docker exec -it master_slam_api bash
cd rerun_mast3r
conda activate mast3r-slam
python sati_master_slam.py \
  --dataset /workspace/dataset/rgb_no23vcF_69_0 \
  --config config/base.yaml \
  --save-as stanford_out \
  --img-size 512 \
  --all-frames \
  --rr-config.headless \
  --rerun-server-addr master_slam_cli:9878
''' 