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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import subprocess
import base64
import tempfile
import shutil

app = FastAPI(
    title="MASt3R-SLAM Server",
    description="SLAM server supporting both batch processing and real-time streaming",
    version="1.0.0"
)

# Session storage for real-time streaming
active_sessions: Dict[str, Dict] = {}

# ============================================================
# Request/Response Models
# ============================================================

class EstimatePoseRequest(BaseModel):
    """Batch processing: Process entire dataset folder"""
    dataset_path: str
    config_path: str = "config/base.yaml"
    save_as: str = "api_req"
    img_size: int = 512
    all_frames: bool = False
    rerun_server_addr: Optional[str] = None

class SlamInitRequest(BaseModel):
    """Real-time streaming: Initialize SLAM session"""
    session_id: str
    config_path: str = "config/base.yaml"
    img_size: int = 512
    rerun_server_addr: Optional[str] = None

class SlamFrameRequest(BaseModel):
    """Real-time streaming: Process single frame"""
    session_id: str
    frame_id: int
    image_base64: str  # Base64 encoded image

class SlamFinalizeRequest(BaseModel):
    """Real-time streaming: Finalize session and get trajectory"""
    session_id: str
    save_as: str = "slam_session"
    all_frames: bool = True

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}

@app.post("/estimate_pose")
async def estimate_pose(request: EstimatePoseRequest):
    """
    Run MASt3R-SLAM inference on a dataset directory.

    Args:
        dataset_path: Path to dataset directory (e.g., /workspace/dataset/rgb_no23vcF_69_0)
        config_path: Path to config YAML file (default: config/base.yaml)
        save_as: Output name for results (default: api_req)
        img_size: Image size for processing - 224 or 512 (default: 512)
        all_frames: Save poses for all frames, not just keyframes (default: False)
        rerun_server_addr: Optional rerun server address (e.g., master_slam_cli:9878)

    Returns:
        JSON with trajectory data: {"position": [[x,y], ...], "yaw": [yaw, ...]}
    """
    # Build command
    cmd = [
        "python", "/workspace/rerun_mast3r/sati_master_slam.py",
        "--dataset", request.dataset_path,
        "--config", request.config_path,
        "--save-as", request.save_as,
        "--img-size", str(request.img_size),
        "--rr-config.headless",
    ]

    if request.all_frames:
        cmd.append("--all-frames")

    if request.rerun_server_addr:
        cmd.extend(["--rerun-server-addr", request.rerun_server_addr])

    # Run inference
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/rerun_mast3r")

    # Check for output file
    out_file = Path(f"/workspace/rerun_mast3r/{request.save_as}_traj_data.json")
    if out_file.exists():
        trajectory = json.loads(out_file.read_text())
        return {
            "status": "success",
            "trajectory": trajectory,
            "num_frames": len(trajectory.get("position", [])),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Inference failed - no output file generated",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        )

# ============================================================
# Real-time Streaming Endpoints
# ============================================================

@app.post("/slam/init")
async def slam_init(request: SlamInitRequest):
    """
    Initialize a real-time SLAM session for frame-by-frame processing.

    Creates a temporary directory to store incoming frames.
    """
    if request.session_id in active_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session {request.session_id} already exists. Use a different session_id or finalize the existing session."
        )

    # Create temporary directory for this session
    session_dir = Path(f"/tmp/slam_session_{request.session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)

    active_sessions[request.session_id] = {
        "session_dir": str(session_dir),
        "config_path": request.config_path,
        "img_size": request.img_size,
        "rerun_server_addr": request.rerun_server_addr,
        "frame_count": 0,
        "status": "ready"
    }

    return {
        "status": "success",
        "session_id": request.session_id,
        "message": "SLAM session initialized. Ready to receive frames."
    }

@app.post("/slam/process_frame")
async def slam_process_frame(request: SlamFrameRequest):
    """
    Process a single frame in real-time streaming mode.

    Saves the frame to the session directory. Actual SLAM processing
    happens when finalize is called.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found. Initialize session first with /slam/init"
        )

    session = active_sessions[request.session_id]
    session_dir = Path(session["session_dir"])

    # Decode base64 image
    try:
        image_data = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    # Save frame with zero-padded filename (e.g., 00000.png, 00001.png)
    frame_path = session_dir / f"{request.frame_id:05d}.png"
    frame_path.write_bytes(image_data)

    session["frame_count"] = max(session["frame_count"], request.frame_id + 1)

    return {
        "status": "success",
        "session_id": request.session_id,
        "frame_id": request.frame_id,
        "total_frames": session["frame_count"]
    }

@app.post("/slam/finalize")
async def slam_finalize(request: SlamFinalizeRequest):
    """
    Finalize SLAM session and run inference on all collected frames.

    Returns the complete trajectory.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found"
        )

    session = active_sessions[request.session_id]
    session_dir = session["session_dir"]

    # Build command to run SLAM on collected frames
    cmd = [
        "python", "/workspace/rerun_mast3r/sati_master_slam.py",
        "--dataset", session_dir,
        "--config", session["config_path"],
        "--save-as", request.save_as,
        "--img-size", str(session["img_size"]),
        "--rr-config.headless",
    ]

    if request.all_frames:
        cmd.append("--all-frames")

    if session["rerun_server_addr"]:
        cmd.extend(["--rerun-server-addr", session["rerun_server_addr"]])

    # Run inference
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace/rerun_mast3r")

    # Check for output file
    out_file = Path(f"/workspace/rerun_mast3r/{request.save_as}_traj_data.json")

    # Cleanup session
    try:
        shutil.rmtree(session_dir)
    except Exception as e:
        print(f"Warning: Failed to cleanup session directory: {e}")

    del active_sessions[request.session_id]

    if out_file.exists():
        trajectory = json.loads(out_file.read_text())
        return {
            "status": "success",
            "session_id": request.session_id,
            "trajectory": trajectory,
            "num_frames": len(trajectory.get("position", [])),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "SLAM inference failed - no output file generated",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        )

@app.get("/slam/sessions")
async def list_sessions():
    """List all active SLAM sessions"""
    return {
        "active_sessions": [
            {
                "session_id": sid,
                "frame_count": session["frame_count"],
                "status": session["status"]
            }
            for sid, session in active_sessions.items()
        ]
    }

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
 
'''  
docker exec -it master_slam_api bash
cd rerun_mast3r
conda activate mast3r-slam

# On local pc:
rerun --connect rerun+http://0.0.0.0:9878/proxy

# In api or via client call:
python sati_master_slam.py \
  --dataset /workspace/dataset/rgb_no23vcF_69_0 \
  --config config/base.yaml \
  --save-as stanford_out \
  --img-size 512 \
  --all-frames \
  --rr-config.headless \
  --rerun-server-addr master_slam_cli:9878
''' 