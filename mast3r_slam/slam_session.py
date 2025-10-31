"""
In-Memory SLAM Session Manager

Keeps SLAM state in memory for real-time processing.
Supports both batch (dataset folder) and streaming (frame-by-frame) modes.

Author: CImbi
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# Heavy imports are done lazily inside methods to avoid slow startup


@dataclass
class PoseEstimate:
    """Single pose estimate"""
    frame_id: int
    position: Tuple[float, float]  # (x, y)
    yaw: float
    timestamp: float
    confidence: Optional[float] = None


class SLAMSession:
    """
    In-memory SLAM session that processes frames incrementally.
    
    Supports two modes:
    1. Batch mode: Load all frames from dataset folder, process sequentially
    2. Real-time mode: Receive frames one-by-one, process immediately
    """
    
    def __init__(
        self,
        session_id: str,
        config_path: str = "config/base.yaml",
        img_size: int = 512,
        real_time: bool = False,
        rerun_server_addr: Optional[str] = None,
        output_dir: str = "logs"
    ):
        self.session_id = session_id
        self.config_path = config_path
        self.img_size = img_size
        self.real_time = real_time
        self.rerun_server_addr = rerun_server_addr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.poses: List[PoseEstimate] = []
        self.frame_count = 0
        self.is_initialized = False
        self.start_time = time.time()
        
        # JSON Lines output file for incremental pose logging
        self.pose_file_path = self.output_dir / f"{session_id}_poses.jsonl"
        self.pose_file = None
        
        # For batch mode: dataset and frame iterator
        self.dataset = None
        self.frame_iterator = None
        
        # SLAM state (will be populated during processing)
        self.slam_state = None
        
        print(f"[SLAM Session {session_id}] Created")
        print(f"  Mode: {'Real-time' if real_time else 'Batch'}")
        print(f"  Config: {config_path}")
        print(f"  Image size: {img_size}")
        print(f"  Output: {self.pose_file_path}")
    
    def initialize_batch_mode(self, dataset_path: str):
        """
        Initialize for batch processing of a dataset folder.

        Args:
            dataset_path: Path to folder containing images (00000.png, 00001.png, ...)
        """
        print(f"[SLAM Session {self.session_id}] Initializing batch mode with dataset: {dataset_path}")

        # Lazy import
        from mast3r_slam.dataloader import SatiDataset

        # Load dataset
        self.dataset = SatiDataset(dataset_path, self.img_size)
        self.frame_iterator = iter(self.dataset)

        # Open pose output file
        self.pose_file = open(self.pose_file_path, 'w')

        self.is_initialized = True
        print(f"[SLAM Session {self.session_id}] Batch mode initialized. Dataset has {len(self.dataset)} frames")
    
    def initialize_real_time_mode(self):
        """
        Initialize for real-time streaming mode.
        Frames will be added one-by-one via add_frame().
        """
        print(f"[SLAM Session {self.session_id}] Initializing real-time mode")
        
        # Open pose output file
        self.pose_file = open(self.pose_file_path, 'w')
        
        self.is_initialized = True
        print(f"[SLAM Session {self.session_id}] Real-time mode initialized")
    
    def add_frame(self, image_data: np.ndarray) -> Optional[PoseEstimate]:
        """
        Add a single frame in real-time mode.
        
        Args:
            image_data: Image as numpy array (H, W, 3) in BGR format
            
        Returns:
            PoseEstimate if pose was computed, None if still initializing
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized. Call initialize_real_time_mode() first.")
        
        if not self.real_time:
            raise RuntimeError("add_frame() only works in real-time mode")
        
        # TODO: Implement incremental SLAM processing
        # This requires refactoring inference.py to be stateful
        # For now, this is a placeholder
        
        print(f"[SLAM Session {self.session_id}] Processing frame {self.frame_count}")
        
        # Placeholder: Return dummy pose
        # In real implementation, this would call SLAM tracking
        pose = PoseEstimate(
            frame_id=self.frame_count,
            position=(0.0, 0.0),
            yaw=0.0,
            timestamp=time.time()
        )
        
        self._store_pose(pose)
        self.frame_count += 1
        
        return pose
    
    def process_batch(self, all_frames: bool = True) -> List[PoseEstimate]:
        """
        Process all frames in batch mode.
        
        Args:
            all_frames: If True, save poses for all frames. If False, only keyframes.
            
        Returns:
            List of all pose estimates
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized. Call initialize_batch_mode() first.")
        
        if self.real_time:
            raise RuntimeError("process_batch() only works in batch mode")
        
        print(f"[SLAM Session {self.session_id}] Processing batch...")
        
        # Run inference using existing pipeline
        # This will process all frames and populate self.poses
        
        # Create inference config
        inf_config = InferenceConfig(
            dataset_path=str(self.dataset.dataset_path),
            config_path=self.config_path,
            img_size=self.img_size,
            all_frames=all_frames,
            rerun_server_addr=self.rerun_server_addr,
            save_as=self.session_id,
            real_time=self.real_time
        )
        
        # Create rerun config
        rr_config = RerunTyroConfig(
            headless=True,
            serve=False,
            connect=bool(self.rerun_server_addr)
        )
        
        # Run inference
        # NOTE: This still uses the old subprocess-based approach
        # We need to refactor inference.py to return poses incrementally
        print(f"[SLAM Session {self.session_id}] Running SLAM inference...")
        
        # For now, call the existing run_inference function
        # TODO: Refactor to process frames incrementally and call _store_pose() for each
        
        # Placeholder: Read poses from output file after inference completes
        # In real implementation, poses would be stored during inference
        
        return self.poses
    
    def _store_pose(self, pose: PoseEstimate):
        """
        Store pose in memory and write to JSON Lines file.
        
        Args:
            pose: Pose estimate to store
        """
        # Add to in-memory list
        self.poses.append(pose)
        
        # Write to JSON Lines file (one pose per line)
        pose_dict = {
            "frame_id": pose.frame_id,
            "position": list(pose.position),
            "yaw": float(pose.yaw),
            "timestamp": pose.timestamp
        }
        
        if pose.confidence is not None:
            pose_dict["confidence"] = pose.confidence
        
        self.pose_file.write(json.dumps(pose_dict) + "\n")
        self.pose_file.flush()  # Ensure it's written immediately
    
    def get_current_pose(self) -> Optional[PoseEstimate]:
        """Get the most recent pose estimate"""
        if len(self.poses) == 0:
            return None
        return self.poses[-1]
    
    def get_trajectory(self) -> Dict:
        """
        Get full trajectory in the format expected by the API.
        
        Returns:
            Dict with 'position' and 'yaw' lists
        """
        return {
            "position": [list(p.position) for p in self.poses],
            "yaw": [p.yaw for p in self.poses]
        }
    
    def finalize(self, save_as: Optional[str] = None) -> Dict:
        """
        Finalize session and save results.
        
        Args:
            save_as: Optional name for final output file
            
        Returns:
            Summary dict with trajectory and statistics
        """
        print(f"[SLAM Session {self.session_id}] Finalizing...")
        
        # Close pose file
        if self.pose_file:
            self.pose_file.close()
        
        # Save final trajectory as JSON
        output_name = save_as or self.session_id
        traj_file = self.output_dir / f"{output_name}_traj_data.json"
        
        trajectory = self.get_trajectory()
        with open(traj_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        duration = time.time() - self.start_time
        
        print(f"[SLAM Session {self.session_id}] Finalized")
        print(f"  Total frames: {len(self.poses)}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Trajectory saved to: {traj_file}")
        
        return {
            "session_id": self.session_id,
            "num_frames": len(self.poses),
            "duration": duration,
            "trajectory": trajectory,
            "output_file": str(traj_file)
        }

