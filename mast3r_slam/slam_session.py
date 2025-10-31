"""
In-Memory SLAM Session Manager

Simplified version that creates a streaming dataset and calls mast3r_slam_inference().
The only difference from batch mode is that we feed images incrementally instead of from a folder.

Author: CImbi
"""

import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal
import numpy as np
from dataclasses import dataclass
import cv2
import yaml

@dataclass
class PoseEstimate:
    """Single pose estimate"""
    frame_id: int
    position: Tuple[float, float]  # (x, y)
    yaw: float
    timestamp: float
    confidence: Optional[float] = None


class StreamingDataset:
    """
    A dataset that yields images one-by-one from memory instead of from disk.
    Blocks when waiting for new images (for real-time SLAM).
    Compatible with the existing SLAM pipeline.
    """
    def __init__(self, img_size: Literal[224, 512] = 512):
        print(f"[StreamingDataset] Initializing with img_size={img_size}")
        self.img_size = img_size
        self.images = []  # List of numpy arrays (H, W, 3) in BGR format
        self.timestamps = []
        self.dtype = np.float32
        self.save_results = False
        self.terminated = False  # Flag to stop SLAM loop

        # Load camera intrinsics from config/intrinsics.yaml
        cfg_path = Path(__file__).parents[1] / "config" / "intrinsics.yaml"
        print(f"[StreamingDataset] Loading intrinsics from: {cfg_path}")

        try:
            with open(cfg_path, "r") as f:
                data = yaml.safe_load(f)
            print(f"[StreamingDataset] ✓ Loaded intrinsics YAML")
        except Exception as e:
            print(f"[StreamingDataset] ✗ FAILED to load intrinsics YAML: {e}")
            raise

        W, H = data["width"], data["height"]
        calib = data["calibration"]  # [fx, fy, cx, cy, k1, k2, p1, p2]
        print(f"[StreamingDataset] Camera: {W}x{H}, calib={calib}")

        # Import Intrinsics class
        try:
            from mast3r_slam.dataloader import Intrinsics
            print(f"[StreamingDataset] ✓ Imported Intrinsics")
        except Exception as e:
            print(f"[StreamingDataset] ✗ FAILED to import Intrinsics: {e}")
            raise

        try:
            self.camera_intrinsics = Intrinsics.from_calib(
                img_size, W, H, calib, always_undistort=True
            )
            print(f"[StreamingDataset] ✓ Created camera intrinsics")
        except Exception as e:
            print(f"[StreamingDataset] ✗ FAILED to create intrinsics: {e}")
            import traceback
            traceback.print_exc()
            raise

    def add_image(self, img: np.ndarray, timestamp: float):
        """Add a new image to the dataset (called by API when frame arrives)"""
        self.images.append(img)
        self.timestamps.append(timestamp)
        print(f"[StreamingDataset] Added frame {len(self.images)-1}, total frames: {len(self.images)}")

    def terminate(self):
        """Signal that no more images will be added"""
        self.terminated = True

    def __len__(self):
        """
        Return a very large number so SLAM loop doesn't terminate.
        The loop will block in __getitem__ waiting for new images.
        """
        if self.terminated:
            return len(self.images)
        return 999999  # Effectively infinite

    def __getitem__(self, idx):
        """
        Get image at index (compatible with SLAM pipeline).
        Blocks if image not available yet (waiting for ROS node to send it).
        """
        # Wait for image to arrive
        while idx >= len(self.images):
            if self.terminated:
                raise IndexError(f"Dataset terminated, no image at index {idx}")
            time.sleep(0.01)  # Wait 10ms and check again

        timestamp = self.timestamps[idx]
        img = self.get_image(idx)
        return timestamp, img

    def get_image(self, idx):
        """Get preprocessed image"""
        img = self.images[idx]
        # Apply camera calibration (undistortion)
        if self.camera_intrinsics is not None:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        """Get image shape"""
        from mast3r_slam.dataloader import resize_img
        # Wait for first image
        while len(self.images) == 0:
            time.sleep(0.01)
        img = self.images[0]
        raw_img_shape = img.shape
        img = resize_img(img, self.img_size)
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def subsample(self, subsample):
        """Subsample dataset (no-op for streaming)"""
        pass

    def has_calib(self):
        """Check if calibration is available"""
        return self.camera_intrinsics is not None

    def get_timestamp(self, idx):
        """Get timestamp at index"""
        return self.timestamps[idx]


class SLAMSession:
    """
    SLAM session that processes frames incrementally using a streaming dataset.
    Much simpler than before - just wraps mast3r_slam_inference() with a streaming dataset.
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

        # Validate mode
        if not real_time:
            raise ValueError("slam_session.py only supports real_time=True. For batch processing, use sati_master_slam.py directly.")

        # Streaming dataset
        print(f"[SLAM Session {session_id}] Creating StreamingDataset...")
        try:
            self.dataset = StreamingDataset(img_size=img_size)
            print(f"[SLAM Session {session_id}] ✓ StreamingDataset created")
        except Exception as e:
            print(f"[SLAM Session {session_id}] ✗ FAILED to create StreamingDataset: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[SLAM Session {session_id}] ✓ Created successfully")
        print(f"  Mode: Real-time streaming")
        print(f"  Config: {config_path}")
        print(f"  Image size: {img_size}")
        print(f"  Rerun server: {rerun_server_addr}")
        print(f"  Output: {self.pose_file_path}")
    
    def initialize_real_time_mode(self):
        """
        Initialize for real-time streaming mode.
        Starts the SLAM inference thread that will process frames as they arrive.
        """
        print(f"[SLAM Session {self.session_id}] Initializing real-time mode")

        # Ensure output directory exists
        self.pose_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[SLAM Session {self.session_id}] Output directory: {self.pose_file_path.parent}")

        # Open pose output file
        try:
            self.pose_file = open(self.pose_file_path, 'w')
            print(f"[SLAM Session {self.session_id}] Opened pose file: {self.pose_file_path}")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ERROR: Failed to open pose file: {e}")
            raise

        # Start SLAM inference in a background thread
        import threading
        self.slam_thread = threading.Thread(target=self._run_slam_inference, daemon=True)
        self.slam_thread.start()
        print(f"[SLAM Session {self.session_id}] SLAM thread started (daemon={self.slam_thread.daemon})")

        # Give thread a moment to start and check for immediate crashes
        time.sleep(0.5)
        if not self.slam_thread.is_alive():
            print(f"[SLAM Session {self.session_id}] WARNING: SLAM thread died immediately after start!")
            raise RuntimeError("SLAM thread failed to start - check logs for import errors")

        self.is_initialized = True
        print(f"[SLAM Session {self.session_id}] Real-time mode initialized successfully")
    
    def _extract_pose_from_frame(self, frame) -> PoseEstimate:
        """
        Extract pose from a Frame object (called by SLAM thread after processing each frame).

        Args:
            frame: Frame object with T_WC transformation

        Returns:
            PoseEstimate
        """
        import lietorch

        # Get transformation (world to camera)
        T_WC = frame.T_WC

        # Extract translation
        translation = T_WC.translation().cpu().numpy()[0]  # (3,)
        x, y, z = translation

        # Extract rotation matrix to get yaw
        rotation = T_WC.rotation().matrix().cpu().numpy()[0]  # (3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])

        pose = PoseEstimate(
            frame_id=frame.idx,
            position=(float(x), float(y)),
            yaw=float(yaw),
            timestamp=time.time()
        )

        return pose

    def _run_slam_inference(self):
        """
        Run SLAM inference in background thread.
        This is the continuous SLAM process that processes frames as they arrive.
        """
        print(f"[SLAM Session {self.session_id}] SLAM thread started, attempting imports...")

        try:
            from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference
            print(f"[SLAM Session {self.session_id}] ✓ Imported InferenceConfig, mast3r_slam_inference")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to import InferenceConfig: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            from simplecv.rerun_log_utils import RerunTyroConfig
            print(f"[SLAM Session {self.session_id}] ✓ Imported RerunTyroConfig")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to import RerunTyroConfig: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"[SLAM Session {self.session_id}] Creating inference config...")
        print(f"  - rerun_server_addr: {self.rerun_server_addr}")
        print(f"  - config_path: {self.config_path}")
        print(f"  - img_size: {self.img_size}")

        # Create inference config
        try:
            inf_config = InferenceConfig(
                rr_config=RerunTyroConfig(
                    headless=True,
                    serve=False,
                    connect=bool(self.rerun_server_addr)
                ),
                dataset="streaming",  # Dummy path, we use self.dataset instead
                config=self.config_path,
                save_as=self.session_id,
                img_size=self.img_size,
                all_frames=True,  # Save all frame poses
                rerun_server_addr=self.rerun_server_addr,
                real_time=True
            )
            print(f"[SLAM Session {self.session_id}] ✓ Created InferenceConfig")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to create InferenceConfig: {e}")
            import traceback
            traceback.print_exc()
            return

        # Monkey-patch the dataset loading to use our streaming dataset
        print(f"[SLAM Session {self.session_id}] Setting up dataset monkey-patch...")
        try:
            import mast3r_slam.api.inference as inf_module
            original_load_dataset = inf_module.load_dataset

            def patched_load_dataset(dataset_path, img_size):
                print(f"[SLAM Session {self.session_id}] Using streaming dataset instead of {dataset_path}")
                return self.dataset

            inf_module.load_dataset = patched_load_dataset
            print(f"[SLAM Session {self.session_id}] ✓ Dataset monkey-patch installed")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to monkey-patch dataset: {e}")
            import traceback
            traceback.print_exc()
            return

        # Monkey-patch the inference loop to extract poses in real-time
        original_mast3r_slam_inference = mast3r_slam_inference

        def patched_mast3r_slam_inference(cfg):
            print(f"[SLAM Session {self.session_id}] Running patched SLAM inference...")
            # We need to hook into the SLAM loop to extract poses
            # For now, just call the original and read poses from file at the end
            result = original_mast3r_slam_inference(cfg)

            # After SLAM completes, read the trajectory file and populate poses
            print(f"[SLAM Session {self.session_id}] SLAM inference finished, loading poses from trajectory...")
            self._load_poses_from_trajectory()

            return result

        try:
            print(f"[SLAM Session {self.session_id}] Starting SLAM inference (will block waiting for images)...")
            # Run SLAM inference (blocks until dataset.terminated = True)
            patched_mast3r_slam_inference(inf_config)
            print(f"[SLAM Session {self.session_id}] ✓ SLAM inference completed successfully")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ SLAM inference error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original function
            print(f"[SLAM Session {self.session_id}] Restoring original load_dataset function")
            inf_module.load_dataset = original_load_dataset

    def _load_poses_from_trajectory(self):
        """Load poses from the trajectory file written by SLAM"""
        traj_file = self.output_dir / self.session_id / "streaming.txt"
        if not traj_file.exists():
            print(f"[SLAM Session {self.session_id}] Trajectory file not found: {traj_file}")
            return

        print(f"[SLAM Session {self.session_id}] Loading poses from {traj_file}")

        for line in traj_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue

            # Parse TUM format: timestamp x y z qx qy qz qw
            timestamp = float(parts[0])
            x, y = float(parts[1]), float(parts[2])
            qx, qy, qz, qw = map(float, parts[4:8])

            # Convert quaternion to yaw
            t0 = 2 * (qw * qz + qx * qy)
            t1 = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(t0, t1)

            pose = PoseEstimate(
                frame_id=len(self.poses),
                position=(x, y),
                yaw=yaw,
                timestamp=timestamp
            )

            self._store_pose(pose)

    def add_frame(self, image_data: np.ndarray) -> Optional[PoseEstimate]:
        """
        Add a single frame in real-time mode.
        Simply adds the image to the streaming dataset - the SLAM thread processes it.

        Args:
            image_data: Image as numpy array (H, W, 3) in BGR format

        Returns:
            PoseEstimate with current best estimate (may be from previous frame if SLAM is still processing)
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized. Call initialize_real_time_mode() first.")

        if not self.real_time:
            raise RuntimeError("add_frame() only works in real-time mode")

        # Add image to streaming dataset (SLAM thread will pick it up)
        timestamp = time.time()
        self.dataset.add_image(image_data, timestamp)

        print(f"[SLAM Session {self.session_id}] Frame {self.frame_count} added to dataset")

        # First frame: return origin pose immediately
        if self.frame_count == 0:
            pose = PoseEstimate(
                frame_id=self.frame_count,
                position=(0.0, 0.0),
                yaw=0.0,
                timestamp=timestamp
            )
            self._store_pose(pose)
            self.frame_count += 1
            return pose

        # For subsequent frames: wait a bit for SLAM to process, then return latest pose
        # Give SLAM thread time to process (adjust based on performance)
        time.sleep(0.1)

        # Get the latest pose from our stored poses
        # The SLAM thread updates poses via the trajectory file
        pose = self.get_current_pose()

        if pose is None:
            # SLAM hasn't computed a pose yet, return last known pose
            print(f"[SLAM Session {self.session_id}] SLAM still processing, returning last pose")
            pose = self.poses[-1] if self.poses else PoseEstimate(
                frame_id=self.frame_count,
                position=(0.0, 0.0),
                yaw=0.0,
                timestamp=timestamp
            )

        self.frame_count += 1
        return pose
    
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

