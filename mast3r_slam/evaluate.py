import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_ATE(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction_ply(savedir, filename, keyframes: SharedKeyframes, c_conf_threshold, voxel_size=0.03, keyframe_indices=None, min_depth=0.1, max_depth=5.0):
    """Save global fused pointcloud to .ply file (official MASt3R-SLAM version adapted for SharedKeyframes)

    Args:
        savedir: Directory to save the PLY file
        filename: Name of the PLY file
        keyframes: SharedKeyframes object containing all keyframes
        c_conf_threshold: Confidence threshold for filtering points
        voxel_size: Voxel size for downsampling (default: 0.03m = 1cm). Set to None to disable downsampling.
        keyframe_indices: Optional list of keyframe indices to include (default: None = all keyframes)
                         If provided, only these keyframes will be exported (useful for matching runtime visualization)
        min_depth: Minimum depth in meters (default: 0.1m). Points closer than this are filtered out.
        max_depth: Maximum depth in meters (default: 5.0m). Points farther than this are filtered out.
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []

    # Determine which keyframes to process
    if keyframe_indices is not None:
        indices_to_process = keyframe_indices
        print(f"[PLY Export] Using {len(indices_to_process)} keyframes from logged list (total keyframes: {len(keyframes)})")
    else:
        indices_to_process = range(len(keyframes))
        print(f"[PLY Export] Using all {len(keyframes)} keyframes")

    for i in indices_to_process:
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)

        # Get positions in camera frame (before world transform)
        positions_cam = keyframe.X_canon.cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)

        # Filter by confidence threshold FIRST (matches runtime visualization order)
        conf_valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )

        # Apply confidence mask first
        positions_cam_conf = positions_cam[conf_valid]
        color_conf = color[conf_valid]

        # Filter by depth (Z-axis in camera frame) on confidence-filtered points
        # This matches the filtering applied during runtime visualization
        depth_values = positions_cam_conf[:, 2]  # Z-coordinate is depth
        depth_valid = (depth_values >= min_depth) & (depth_values <= max_depth)

        # Apply depth filter
        positions_cam_filtered = positions_cam_conf[depth_valid]
        color_filtered = color_conf[depth_valid]

        # Transform to world frame using T_WC
        pW = keyframe.T_WC.act(torch.from_numpy(positions_cam_filtered).to(keyframe.T_WC.device)).cpu().numpy()

        pointclouds.append(pW)
        colors.append(color_filtered)

    # Concatenate all keyframes into one global pointcloud
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    print(f"[PLY Export] Raw pointcloud: {len(pointclouds):,} points")

    # Voxel downsample to remove duplicate/overlapping points and reduce blur
    if voxel_size is not None and voxel_size > 0:
        import open3d as o3d

        # Create Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointclouds)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

        # Voxel downsample (averages points within each voxel)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Extract downsampled points and colors
        pointclouds = np.asarray(pcd_downsampled.points)
        colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)

        print(f"[PLY Export] Downsampled pointcloud (voxel_size={voxel_size}m): {len(pointclouds):,} points")

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )

def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
