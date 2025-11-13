#!/usr/bin/env python3
"""
Wrapper for the Rerun-enabled MASt3R-SLAM entry point.

1) Parses the same flags as the Rerun demo:
     --dataset, --config, --save-as, --img-size, --no-viz, [--calib]
2) Calls mast3r_slam_inference(...) under the hood
3) Reads the resulting trajectory .txt and dumps your JSON
4) If --full-slam flag is set, builds and exports global fused pointcloud
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

import os, json, math, sys
from pathlib import Path
import tyro

# Disable rerun spawn unless explicitly allowed
os.environ.setdefault("RERUN_SPAWN", "false")

def main():
    # Import here to avoid import errors when loading as FastAPI app
    from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference, cleanup_slam_processes
    import torch
    import gc

    cfg = tyro.cli(InferenceConfig)

    keyframes = None
    try:
        # Run SLAM inference
        keyframes = mast3r_slam_inference(cfg)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted by user, cleaning up...")
        # Cleanup stray processes and GPU memory
        cleanup_slam_processes(cfg.save_as)

        # Force GPU cleanup with CUDA IPC cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # Clean up IPC handles
                for _ in range(3):
                    gc.collect()
                torch.cuda.empty_cache()
                print("[Ctrl+C] ✓ GPU memory and CUDA context cleaned")
            except Exception as e:
                print(f"[Ctrl+C] WARNING: CUDA cleanup failed: {e}")

        print("[Ctrl+C] ✓ Cleanup complete, exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] SLAM inference failed: {e}")
        cleanup_slam_processes(cfg.save_as)
        raise

    # Full SLAM: Build and export global fused pointcloud
    if cfg.full_slam and keyframes is not None:
        print(f"\n[Full SLAM] Building global reconstruction with conf_thresh={cfg.conf_thresh}...")

        # Import save_reconstruction_ply from local evaluate.py
        from mast3r_slam.evaluate import save_reconstruction_ply

        save_dir = Path("logs") / cfg.save_as
        seq = Path(cfg.dataset).stem

        # Save global fused pointcloud to PLY file
        save_reconstruction_ply(
            savedir=save_dir,
            filename=f"{seq}.ply",
            keyframes=keyframes,
            c_conf_threshold=cfg.conf_thresh
        )
        print(f"[Full SLAM] ✓ Saved global reconstruction to {save_dir}/{seq}.ply")

        # Log final fused pointcloud to Rerun viewer (if visualization is enabled)
        # This uses the original rerun-master approach: final pointcloud from nerfstudio export
        if not cfg.no_viz and cfg.rerun_server_addr:
            import rerun as rr
            from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio

            # Generate final fused pointcloud (same as nerfstudio export)
            # Only include keyframes with N_updates >= 2 (refined by backend)
            # NOTE: N_updates=1 means initial observation, N_updates=2+ means refined by tracking/optimization
            pcd = save_kf_to_nerfstudio(
                ns_save_path=save_dir / "nerfstudio-output",
                keyframes=keyframes,
                confidence_thresh=cfg.conf_thresh,  # Use same threshold as PLY export
                min_updates=2,  # Only include keyframes refined by backend (not just initialized)
            )

            # Log final pointcloud to Rerun
            rr.log(
                "/world/final_pointcloud",
                rr.Points3D(positions=pcd.points, colors=pcd.colors),
            )
            print(f"[Full SLAM] ✓ Logged final fused pointcloud to Rerun viewer ({len(pcd.points):,} points)")

    # Custom Shaders Mode: Log mesh-based reconstruction (experimental)
    if cfg.custom_shaders and keyframes is not None:
        print(f"\n[Custom Shaders] Building mesh-based reconstruction (experimental)...")
        print(f"\nTODO: Waiting for custom shaders support to be added to rerun-sdk")

        if not cfg.no_viz and cfg.rerun_server_addr:
            import rerun as rr
            from mast3r_slam.rerun_log_utils import RerunLogger

            rr_logger = RerunLogger(parent_log_path=Path("/world"))
            rr_logger.log_global_map(keyframes, conf_thresh=cfg.conf_thresh)
            print(f"[Custom Shaders] ✓ Logged mesh-based global map to Rerun viewer")

    # Export trajectory to JSON (optional, for debugging/analysis)
    if cfg.log_trajectory_json:
        seq = Path(cfg.dataset).stem
        traj = Path("logs") / cfg.save_as / f"{seq}.txt"
        if not traj.exists():
            print(f"[Warning] Expected trajectory at {traj}, skipping JSON export")
        else:
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
            print(f"[Trajectory JSON] Wrote {{'position':{len(positions)}, 'yaw':{len(yaws)}}} to {json_path}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()


