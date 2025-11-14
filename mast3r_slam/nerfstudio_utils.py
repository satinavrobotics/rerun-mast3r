import cv2
import numpy as np
import open3d as o3d
import torch
import tqdm
from pathlib import Path
from jaxtyping import Float32, UInt8
from typing import Literal
from serde import serde
from serde.json import to_json
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
import lietorch
from simplecv.ops import conventions
from mast3r_slam.mast3r_utils import frame_to_intir


@serde
class NSFrame:
    file_path: str
    transform_matrix: list[list[float]]  # 4x4 camera transformation matrix in OpenGL format (as nested list)
    colmap_im_id: int


@serde
class NerfstudioData:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    camera_model: Literal["OPENCV"]
    frames: list[NSFrame]
    applied_transform: list[list[float]]  # 3x4 matrix as nested list
    ply_file_path: Literal["sparse_pc.ply"]


def save_kf_to_nerfstudio(
    ns_save_path: Path,
    keyframes: SharedKeyframes,
    confidence_thresh: float = 1.0,
    min_updates: int = 1,
    keyframe_indices: list = None,
    voxel_size: float = 2.0,
    min_depth: float = 0.1,
    max_depth: float = 4.2,
):
    """
    Save keyframes to NerfStudio format
    :param ns_save_path: Path to save the NerfStudio data
    :param keyframes: SharedKeyframes object
    :param confidence_thresh: Confidence threshold to apply to the keyframes (float, typically 0.1-5.0)
    :param min_updates: Minimum number of pose updates required to include keyframe (default: 1)
                        Filters out keyframes that were never optimized by the backend
    :param keyframe_indices: Optional list of keyframe indices to include (default: None = all keyframes)
                             If provided, only these keyframes will be exported (useful for matching runtime visualization)
    :param voxel_size: Voxel size for downsampling the fused pointcloud (default: 0.01m = 1cm)
                       Smaller = more detail but more points, larger = smoother but less detail
    :param min_depth: Minimum depth in meters (default: 0.1m). Points closer than this are filtered out.
    :param max_depth: Maximum depth in meters (default: 4.2m). Points farther than this are filtered out.

    :return: Open3D point cloud object
    """
    ns_save_path.mkdir(parents=True, exist_ok=True)
    # Create images subdirectory
    images_dir = ns_save_path / "images"
    images_dir.mkdir(exist_ok=True)

    # Process keyframes with ceiling filtering
    # Each keyframe's pointcloud is filtered based on its own local Y-range
    ns_frames_list = []
    pcd_positions = []
    pcd_colors = []
    skipped_count = 0

    # Determine which keyframes to process
    if keyframe_indices is not None:
        # Use only specified keyframe indices (e.g., those logged during runtime)
        indices_to_process = keyframe_indices
        print(f"[NerfStudio Export] Using {len(indices_to_process)} keyframes from logged list")
    else:
        # Use all keyframes
        indices_to_process = range(len(keyframes))

    for i in tqdm.tqdm(indices_to_process, desc="Processing keyframes"):
        keyframe = keyframes[i]

        # Skip keyframes that haven't been optimized by the backend
        # N_updates tracks how many times the keyframe's pose has been refined
        # Keyframes with N_updates < min_updates have unoptimized poses and should be excluded
        if keyframe.N_updates < min_updates:
            skipped_count += 1
            continue
        rgb_img: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img * 255).numpy().astype(np.uint8)
        bgr_img: UInt8[np.ndarray, "H W 3"] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr_img.shape

        # Save the image with zero-padded numbering
        image_filename = f"frame_{i + 1:05d}.png"  # Format: frame_00001.png
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), bgr_img)
        relative_image_path = f"images/{image_filename}"

        se3_pose: lietorch.SE3 = as_SE3(keyframe.T_WC.cpu())
        matb4x4: Float32[np.ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        # in RDF (OpenCV) Format
        mat4x4_cv: Float32[np.ndarray, "4 4"] = matb4x4[0]

        # in RUB (OpenGL) Format
        mat4x4_gl = conventions.convert_pose(
            mat4x4_cv,
            src_convention=conventions.CC.CV,
            dst_convention=conventions.CC.GL,
        )

        # Get positions in camera frame
        positions: Float32[np.ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
        colors: UInt8[np.ndarray, "num_points 3"] = rgb_img.reshape(-1, 3)

        # Filter by confidence threshold
        conf_mask = keyframe.C.cpu().numpy() > confidence_thresh
        conf_mask = conf_mask.squeeze()  # Remove the trailing dimension to get a 1D boolean array

        # Filter by depth (Z-axis in camera frame)
        # This matches the filtering applied during runtime visualization
        depth_values = positions[:, 2]  # Z-coordinate is depth
        depth_mask = (depth_values >= min_depth) & (depth_values <= max_depth)

        # Combine confidence and depth filters
        combined_mask = conf_mask & depth_mask

        # Apply combined filter
        masked_positions = positions[combined_mask]
        masked_colors = colors[combined_mask]

        # Apply ceiling filter based on LOCAL Y-range in camera coordinates
        # Each pointcloud is filtered independently based on its own Y-distribution
        if len(masked_positions) > 0:
            y_coords = masked_positions[:, 1]  # Y in camera frame (vertical)

            # Remove bottom 42% of points by Y-value (42nd percentile)
            # In camera coords, Y points DOWN, so low Y = ceiling, high Y = floor
            y_threshold = np.percentile(y_coords, 42)

            # Keep points ABOVE threshold (higher Y = floor/walls, remove ceiling)
            ceiling_mask = y_coords > y_threshold
            masked_positions = masked_positions[ceiling_mask]
            masked_colors = masked_colors[ceiling_mask]

            # print(f"[DEBUG] Keyframe {i}: Y range [{y_coords.min():.2f}, {y_coords.max():.2f}], "
                  # f"threshold={y_threshold:.2f}, removed {(~ceiling_mask).sum()}/{len(y_coords)} ceiling points")

        # Convert to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_positions = np.ones(
            (masked_positions.shape[0], 4), dtype=np.float32
        )
        homogeneous_positions[:, :3] = masked_positions

        # Apply transformation (points are column vectors: p_world = T_world_cam * p_cam)
        world_positions = (mat4x4_cv @ homogeneous_positions.T).T[:, :3]

        pcd_positions.append(world_positions)
        pcd_colors.append(masked_colors)

        ns_frames_list.append(
            NSFrame(
                file_path=relative_image_path,
                transform_matrix=mat4x4_gl.tolist(),  # Convert numpy array to list for JSON serialization
                colmap_im_id=i,
            )
        )

    # Print filtering statistics
    total_keyframes = len(keyframes)
    total_processed = len(indices_to_process) if keyframe_indices is not None else total_keyframes
    included_keyframes = total_processed - skipped_count

    if keyframe_indices is not None:
        print(f"[NerfStudio Export] Included {included_keyframes}/{total_processed} logged keyframes "
              f"(skipped {skipped_count} with N_updates < {min_updates}, total keyframes: {total_keyframes})")
    else:
        print(f"[NerfStudio Export] Included {included_keyframes}/{total_keyframes} keyframes "
              f"(skipped {skipped_count} with N_updates < {min_updates})")

    # stack all the point clouds
    pcd_positions: Float32[np.ndarray, "num_points 3"] = np.vstack(pcd_positions)
    pcd_colors: UInt8[np.ndarray, "num_points 3"] = np.vstack(pcd_colors)

    print(f"[NerfStudio Export] Raw fused pointcloud: {len(pcd_positions):,} points")

    # normalize point colors to be between 0 and 1 and a float32
    pcd_colors: Float32[np.ndarray, "num_points 3"] = (
        pcd_colors.astype(np.float32) / 255.0
    )
    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    pcd.points = o3d.utility.Vector3dVector(pcd_positions.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors.astype(np.float64))

    # Voxel downsample to remove duplicate/overlapping points and reduce blur
    # This averages points within each voxel, creating a cleaner reconstruction
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"[NerfStudio Export] Downsampled pointcloud (voxel_size={voxel_size}m): {len(pcd.points):,} points")

    # save point cloud to file
    o3d.io.write_point_cloud(str(ns_save_path / "sparse_pc.ply"), pcd)

    # use the last keyframe to get the focal and principal point
    focal, principal_point = frame_to_intir(keyframe)
    # save to nerfstudio format, assumes no distortion
    ns_data = NerfstudioData(
        w=w,
        h=h,
        fl_x=focal[0],
        fl_y=focal[1],
        cx=principal_point[0],
        cy=principal_point[1],
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        camera_model="OPENCV",
        frames=ns_frames_list,
        applied_transform=np.eye(3, 4, dtype=np.float32).tolist(),  # Convert to list for JSON serialization
        ply_file_path="sparse_pc.ply",
    )
    json_str: str = to_json(ns_data)
    with open(ns_save_path / "transforms.json", "w") as f:
        f.write(json_str)

    return pcd
