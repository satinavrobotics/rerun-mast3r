"""
Real-Time SLAM Session Manager

Architecture:
1. Creates a StreamingDataset that holds images in memory (blocking when waiting for new images)
2. Starts SLAM in a background thread running mast3r_slam_inference()
3. SLAM runs continuously: while True: timestamp, img = dataset[i]
4. ROS node controls frame rate by only sending images when robot travels X meters
5. Poses are extracted from live SLAM states (states.get_frame().T_WC) after each frame is processed
6. Rerun visualization happens automatically (SLAM logs to rerun server at master_slam_cli:9878)

Key Insight: We don't reimplement SLAM - we just feed it a streaming dataset and extract poses in real-time.

Author: CImbi
"""

# Patch numpy.asarray to support 'copy' parameter for numpy < 2.0
# This is needed for rerun-sdk 0.23.1 compatibility with numpy 1.26.4
import numpy as np
_orig_asarray = np.asarray
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
        return np.array(a, copy=True, **kwargs)
    else:
        # Default behavior - no copy or copy=False/None
        return _orig_asarray(a, **kwargs)
np.asarray = _patched_asarray

import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal
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
        print(f"[StreamingDataset] get_img_shape() called, waiting for first image...")
        # Wait for first image
        while len(self.images) == 0:
            time.sleep(0.01)
        print(f"[StreamingDataset] ✓ First image available, computing shape...")
        img = self.images[0]
        raw_img_shape = img.shape
        img = resize_img(img, self.img_size)
        shape_result = img["img"][0].shape[1:], raw_img_shape[:2]
        print(f"[StreamingDataset] ✓ Image shape: {shape_result}")
        return shape_result

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
        enable_rerun: bool = True,  # New parameter to control rerun visualization
        output_dir: str = "logs"
    ):
        self.session_id = session_id
        self.config_path = config_path
        self.img_size = img_size
        self.real_time = real_time
        self.rerun_server_addr = rerun_server_addr if enable_rerun else None
        self.enable_rerun = enable_rerun
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

        # Reference to SLAM states object (set by SLAM thread when it starts)
        self.slam_states = None

        # Validate mode
        if not real_time:
            raise ValueError("slam_session.py only supports real_time=True. For batch processing, use sati_master_slam.py directly.")

        # Load SLAM config (populates global config variable needed by Intrinsics.from_calib)
        print(f"[SLAM Session {session_id}] Loading SLAM config from {config_path}...")
        try:
            from mast3r_slam.config import load_config
            load_config(config_path)
            print(f"[SLAM Session {session_id}] ✓ SLAM config loaded")
        except Exception as e:
            print(f"[SLAM Session {session_id}] ✗ FAILED to load SLAM config: {e}")
            import traceback
            traceback.print_exc()
            raise

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
        print(f"  Rerun: {'Enabled' if self.enable_rerun else 'Disabled'}")
        if self.enable_rerun:
            print(f"  Rerun server: {self.rerun_server_addr}")
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
            frame_id=frame.frame_id,
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
            import tyro
            from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference
            print(f"[SLAM Session {self.session_id}] ✓ Imported tyro, InferenceConfig, mast3r_slam_inference")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to import: {e}")
            import traceback
            traceback.print_exc()
            return

        # Use tyro.cli() the same way as sati_master_slam.py, but with programmatic args
        print(f"[SLAM Session {self.session_id}] Creating inference config with tyro.cli()...")
        print(f"  - rerun_server_addr: {self.rerun_server_addr}")
        print(f"  - config_path: {self.config_path}")
        print(f"  - img_size: {self.img_size}")

        try:
            # Build CLI arguments programmatically (same as sati_master_slam.py would receive)
            args = [
                "--dataset", "streaming",  # Dummy path, we use self.dataset instead
                "--config", self.config_path,
                "--save-as", self.session_id,
                "--img-size", str(self.img_size),
                # NOTE: --all-frames is NOT used in real-time mode to save memory
                # Real-time SLAM only needs keyframe poses, not every single frame
                "--real-time",
                "--rr-config.headless",
            ]

            if self.rerun_server_addr:
                args.extend(["--rerun-server-addr", self.rerun_server_addr])

            print(f"[SLAM Session {self.session_id}] tyro.cli args: {args}")

            # Use tyro.cli() with programmatic args (same as sati_master_slam.py)
            inf_config = tyro.cli(InferenceConfig, args=args)
            print(f"[SLAM Session {self.session_id}] ✓ Created InferenceConfig via tyro.cli()")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to create InferenceConfig: {e}")
            import traceback
            traceback.print_exc()
            return

        # Monkey-patch the dataset loading and multiprocessing to use our streaming dataset
        print(f"[SLAM Session {self.session_id}] Setting up monkey-patches...")
        try:
            import mast3r_slam.api.inference as inf_module
            original_load_dataset = inf_module.load_dataset

            # Patch the mp module that inference.py imported
            original_mp_set_start_method = inf_module.mp.set_start_method

            def patched_load_dataset(dataset_path, img_size):
                print(f"[SLAM Session {self.session_id}] Using streaming dataset instead of {dataset_path}")
                return self.dataset

            def patched_set_start_method(method, force=False):
                """Patch to avoid 'context has already been set' error"""
                try:
                    original_mp_set_start_method(method, force=True)
                    print(f"[SLAM Session {self.session_id}] Set multiprocessing start method: {method}")
                except RuntimeError as e:
                    if "context has already been set" in str(e):
                        print(f"[SLAM Session {self.session_id}] Multiprocessing context already set, skipping")
                    else:
                        raise

            inf_module.load_dataset = patched_load_dataset
            inf_module.mp.set_start_method = patched_set_start_method
            print(f"[SLAM Session {self.session_id}] ✓ Dataset and multiprocessing monkey-patches installed")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to monkey-patch: {e}")
            import traceback
            traceback.print_exc()
            return

        # Monkey-patch SharedStates.__init__ to capture the states object
        print(f"[SLAM Session {self.session_id}] Setting up SharedStates monkey-patch...")
        try:
            from mast3r_slam.frame import SharedStates
            original_shared_states_init = SharedStates.__init__

            def patched_shared_states_init(states_self, *args, **kwargs):
                # Call original init
                original_shared_states_init(states_self, *args, **kwargs)
                # Capture the states object
                self.slam_states = states_self
                print(f"[SLAM Session {self.session_id}] ✓ Captured SLAM states object")

            # Apply the patch
            SharedStates.__init__ = patched_shared_states_init
            print(f"[SLAM Session {self.session_id}] ✓ SharedStates monkey-patch installed")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ FAILED to monkey-patch SharedStates: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # Change to rerun_mast3r directory so relative paths work (same as batch API)
            import os
            original_cwd = os.getcwd()
            os.chdir("/workspace/rerun_mast3r")
            print(f"[SLAM Session {self.session_id}] Changed working directory to: {os.getcwd()}")

            print(f"[SLAM Session {self.session_id}] Starting SLAM inference (will block waiting for images)...")
            mast3r_slam_inference(inf_config)
            print(f"[SLAM Session {self.session_id}] ✓ SLAM inference completed successfully")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] ✗ SLAM inference error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original working directory and functions
            print(f"[SLAM Session {self.session_id}] Restoring original functions")
            os.chdir(original_cwd)
            inf_module.load_dataset = original_load_dataset
            SharedStates.__init__ = original_shared_states_init



    def add_frame(self, image_data: np.ndarray) -> Optional[PoseEstimate]:
        """
        Add a single frame in real-time mode.
        Adds the image to the streaming dataset and waits for SLAM to process it.

        Args:
            image_data: Image as numpy array (H, W, 3) in BGR format

        Returns:
            PoseEstimate extracted from SLAM states after processing
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized. Call initialize_real_time_mode() first.")

        if not self.real_time:
            raise RuntimeError("add_frame() only works in real-time mode")

        # Add image to streaming dataset (SLAM thread will pick it up)
        timestamp = time.time()
        self.dataset.add_image(image_data, timestamp)

        print(f"[SLAM Session {self.session_id}] Frame {self.frame_count} added to dataset, waiting for SLAM to process...")

        # First frame: return origin pose immediately (SLAM initializes at origin)
        if self.frame_count == 0:
            pose = PoseEstimate(
                frame_id=self.frame_count,
                position=(0.0, 0.0),
                yaw=0.0,
                timestamp=timestamp
            )
            self._store_pose(pose)
            self.frame_count += 1

            # Wait for SLAM to initialize and capture states object
            max_wait = 10.0  # seconds
            wait_start = time.time()
            while self.slam_states is None and (time.time() - wait_start) < max_wait:
                time.sleep(0.1)

            if self.slam_states is None:
                print(f"[SLAM Session {self.session_id}] WARNING: SLAM states not captured yet")
            else:
                print(f"[SLAM Session {self.session_id}] ✓ SLAM states available")

            return pose

        # For subsequent frames: wait for SLAM to process, then extract pose from states
        # Wait for SLAM to process this frame (check that frame_id in states matches)
        max_wait = 5.0  # seconds
        wait_start = time.time()
        pose = None

        while (time.time() - wait_start) < max_wait:
            if self.slam_states is not None:
                try:
                    # Get current frame from SLAM states
                    current_frame = self.slam_states.get_frame()

                    # Check if SLAM has processed our frame
                    if current_frame.frame_id >= self.frame_count:
                        # Extract pose from the frame
                        pose = self._extract_pose_from_frame(current_frame)
                        print(f"[SLAM Session {self.session_id}] ✓ Extracted pose for frame {self.frame_count}: pos={pose.position}, yaw={pose.yaw:.3f}")
                        break
                except Exception as e:
                    print(f"[SLAM Session {self.session_id}] Error extracting pose: {e}")

            time.sleep(0.05)  # Check every 50ms

        if pose is None:
            # SLAM hasn't computed a pose yet, return last known pose
            print(f"[SLAM Session {self.session_id}] WARNING: SLAM still processing, returning last pose")
            pose = self.poses[-1] if self.poses else PoseEstimate(
                frame_id=self.frame_count,
                position=(0.0, 0.0),
                yaw=0.0,
                timestamp=timestamp
            )
        else:
            # Update pose metadata
            pose.frame_id = self.frame_count
            pose.timestamp = timestamp
            self._store_pose(pose)

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

        # Terminate dataset to stop SLAM loop
        if hasattr(self, 'dataset') and self.dataset:
            print(f"[SLAM Session {self.session_id}] Terminating dataset...")
            self.dataset.terminate()

        # Wait for SLAM thread to finish (with timeout)
        if hasattr(self, 'slam_thread') and self.slam_thread and self.slam_thread.is_alive():
            print(f"[SLAM Session {self.session_id}] Waiting for SLAM thread to finish...")
            self.slam_thread.join(timeout=5.0)
            if self.slam_thread.is_alive():
                print(f"[SLAM Session {self.session_id}] WARNING: SLAM thread did not terminate within timeout")

        # Close pose file
        if self.pose_file:
            self.pose_file.close()

        # Free CUDA memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[SLAM Session {self.session_id}] Cleared CUDA cache")
        except Exception as e:
            print(f"[SLAM Session {self.session_id}] Failed to clear CUDA cache: {e}")

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

