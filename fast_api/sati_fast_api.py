"""
MASt3R-SLAM FastAPI Server

Provides two modes of operation:
1. Batch Processing: Process entire dataset folders
2. Real-time Streaming: Process frames iteratively from simulation

Author: CImbi
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import subprocess
import base64
import json
import shutil
from pathlib import Path

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
# Batch Processing Endpoints
# ============================================================

@app.get("/health")
def health():
    """Health check endpoint"""
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

