"""
MASt3R-SLAM FastAPI Server

Provides two modes of operation:
1. Batch Processing: Process entire dataset folders
2. Real-time Streaming: Process frames iteratively from simulation

Uses in-memory SLAM sessions for efficient real-time processing.

Author: CImbi
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import subprocess
import base64
import json
import shutil
import numpy as np
import cv2
from pathlib import Path

# Import SLAM session manager
import sys
sys.path.insert(0, '/workspace/rerun_mast3r')
from mast3r_slam.slam_session import SLAMSession, PoseEstimate

app = FastAPI(
    title="MASt3R-SLAM Server",
    description="SLAM server supporting both batch processing and real-time streaming",
    version="2.0.0"
)

# Active SLAM sessions (in-memory)
active_sessions: Dict[str, SLAMSession] = {}

# ============================================================
# Request/Response Models
# ============================================================

class EstimatePoseRequest(BaseModel):
    """Batch processing: Process entire dataset folder using sati_master_slam.py"""
    dataset_path: str
    config_path: str = "config/base.yaml"
    save_as: str = "api_req"
    img_size: int = 512
    all_frames: bool = True  # Default to True - save all frames, not just keyframes
    rerun_server_addr: Optional[str] = "master_slam_cli:9878"  # Default to CLI server

class SlamInitRequest(BaseModel):
    """Real-time streaming: Initialize SLAM session"""
    session_id: str
    config_path: str = "config/base.yaml"
    img_size: int = 512
    rerun_server_addr: Optional[str] = "master_slam_cli:9878"
    enable_rerun: bool = True  # Enable/disable rerun visualization
    real_time: bool = True  # Real-time mode: process frames incrementally

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
# Batch Processing Endpoints (using sati_master_slam.py)
# ============================================================

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "active_sessions": len(active_sessions)}

@app.post("/estimate_pose")
async def estimate_pose(request: EstimatePoseRequest):
    """
    Run MASt3R-SLAM inference on a dataset directory (batch mode).
    Uses sati_master_slam.py to process entire folder of images.

    Args:
        dataset_path: Path to dataset directory (e.g., /workspace/dataset/rgb_no23vcF_69_0)
        config_path: Path to config YAML file (default: config/base.yaml)
        save_as: Output name for results (default: api_req)
        img_size: Image size for processing - 224 or 512 (default: 512)
        all_frames: Save poses for all frames, not just keyframes (default: True)
        rerun_server_addr: Optional rerun server address (default: master_slam_cli:9878)

    Returns:
        JSON with trajectory data: {"position": [[x,y], ...], "yaw": [yaw, ...]}
    """
    # Build command to run sati_master_slam.py
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

    # Check for output file (written to current directory by sati_master_slam.py)
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
    Initialize a SLAM session for real-time frame-by-frame processing.

    Creates an in-memory SLAM session that persists until finalized.
    """
    sid = request.session_id

    if sid in active_sessions:
        old = active_sessions[sid]
        # Treat zero-frame, zero-pose sessions as stale and overwrite them
        if getattr(old, "frame_count", 0) == 0 and len(getattr(old, "poses", [])) == 0:
            print(f"[SLAM API] Overwriting stale empty session {sid}")
            try:
                old.finalize(save_as=f"{sid}_stale")
            except Exception as e:
                print(f"[SLAM API] Failed to finalize stale session {sid}: {e}")
            finally:
                try:
                    del active_sessions[sid]
                except KeyError:
                    pass
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Session {sid} already exists and has data. "
                       f"Use a different session_id or finalize the existing session."
            )

    try:
        # Create in-memory SLAM session
        session = SLAMSession(
            session_id=sid,
            config_path=request.config_path,
            img_size=request.img_size,
            real_time=request.real_time,
            rerun_server_addr=request.rerun_server_addr,
            enable_rerun=request.enable_rerun,
            output_dir="/workspace/rerun_mast3r/logs"
        )

        # Initialize for real-time mode
        session.initialize_real_time_mode()

        # Store session
        active_sessions[sid] = session

        return {
            "status": "success",
            "session_id": sid,
            "mode": "real-time" if request.real_time else "batch",
            "message": "SLAM session initialized. Ready to receive frames."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize SLAM session: {str(e)}"
        )

@app.post("/slam/process_frame")
async def slam_process_frame(request: SlamFrameRequest):
    """
    Process a single frame in real-time streaming mode.

    Returns pose estimate immediately after processing.
    """
    if request.session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.session_id} not found. Initialize session first with /slam/init"
        )

    session = active_sessions[request.session_id]

    # Check if session has crashed
    if hasattr(session, 'crashed') and session.crashed:
        # Auto-cleanup crashed session
        print(f"[SLAM API] Session {request.session_id} has crashed, auto-cleaning up...")
        try:
            session.finalize(save_as=f"{request.session_id}_crashed")
        except Exception as e:
            print(f"[SLAM API] Failed to finalize crashed session: {e}")
        finally:
            del active_sessions[request.session_id]

        raise HTTPException(
            status_code=500,
            detail=f"Session {request.session_id} has crashed and been cleaned up. Please create a new session."
        )

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Process frame through SLAM
        pose = session.add_frame(image)

        if pose is None:
            # Still initializing, no pose yet
            return {
                "status": "initializing",
                "session_id": request.session_id,
                "frame_id": request.frame_id,
                "message": "Frame received, SLAM still initializing"
            }

        # Return pose immediately
        return {
            "status": "success",
            "session_id": request.session_id,
            "frame_id": request.frame_id,
            "pose": {
                "position": list(pose.position),
                "yaw": float(pose.yaw),
                "timestamp": pose.timestamp
            },
            "total_frames": session.frame_count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing frame: {str(e)}"
        )

@app.post("/slam/finalize")
async def slam_finalize(request: SlamFinalizeRequest):
    """
    Finalize SLAM session and return complete trajectory.

    Saves results to disk and cleans up session.
    """
    if request.session_id not in active_sessions:
        # Session already finalized or never existed
        # Return success to make this endpoint idempotent
        print(f"[SLAM API] Session {request.session_id} already finalized or not found")
        return {
            "status": "already_finalized",
            "session_id": request.session_id,
            "message": "Session already finalized or not found"
        }

    session = active_sessions[request.session_id]

    try:
        # Finalize session (saves trajectory to disk)
        result = session.finalize(save_as=request.save_as)

        # Remove from active sessions
        del active_sessions[request.session_id]

        return {
            "status": "success",
            "session_id": result["session_id"],
            "trajectory": result["trajectory"],
            "num_frames": result["num_frames"],
            "duration": result["duration"],
            "output_file": result["output_file"]
        }

    except Exception as e:
        # Even if finalization fails, remove from active sessions to prevent memory leak
        if request.session_id in active_sessions:
            try:
                del active_sessions[request.session_id]
                print(f"[SLAM API] Removed session {request.session_id} from active sessions after error")
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Error finalizing session: {str(e)}"
        )

@app.get("/slam/sessions")
async def list_sessions():
    """List all active SLAM sessions"""
    return {
        "active_sessions": [
            {
                "session_id": sid,
                "frame_count": session.frame_count,
                "mode": "real-time" if session.real_time else "batch",
                "num_poses": len(session.poses)
            }
            for sid, session in active_sessions.items()
        ]
    }

@app.get("/slam/current_pose/{session_id}")
async def get_current_pose(session_id: str):
    """Get the most recent pose estimate for a session"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )

    session = active_sessions[session_id]
    pose = session.get_current_pose()

    if pose is None:
        return {
            "status": "no_pose",
            "message": "No pose available yet (still initializing)"
        }

    return {
        "status": "success",
        "session_id": session_id,
        "pose": {
            "frame_id": pose.frame_id,
            "position": list(pose.position),
            "yaw": float(pose.yaw),
            "timestamp": pose.timestamp
        }
    }

@app.delete("/slam/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Force cleanup of a SLAM session (useful for freeing GPU memory)"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )

    session = active_sessions[session_id]

    try:
        # Finalize without saving
        session.finalize()
        del active_sessions[session_id]

        return {
            "status": "success",
            "message": f"Session {session_id} cleaned up successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.delete("/slam/cleanup_all")
async def cleanup_all_sessions():
    """Force cleanup of ALL SLAM sessions (useful for freeing GPU memory)"""
    import torch
    import gc

    session_ids = list(active_sessions.keys())
    cleaned = []
    errors = []

    for session_id in session_ids:
        try:
            session = active_sessions[session_id]
            session.finalize()
            del active_sessions[session_id]
            cleaned.append(session_id)
        except Exception as e:
            errors.append({"session_id": session_id, "error": str(e)})

    # Force aggressive cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        "status": "success",
        "cleaned_sessions": cleaned,
        "errors": errors,
        "remaining_sessions": len(active_sessions)
    }

