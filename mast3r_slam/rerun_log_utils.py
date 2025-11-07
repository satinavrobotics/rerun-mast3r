import numpy as np
import torch
import rerun as rr
from jaxtyping import UInt8, Float32, Int
from pathlib import Path
from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.mast3r_utils import estimate_focal_knowing_depth
import lietorch
from mast3r_slam.lietorch_utils import as_SE3
from simplecv.ops import conventions
import rerun.blueprint as rrb


def create_blueprints(parent_log_path: Path) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=parent_log_path),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=parent_log_path / "current_camera" / "pinhole"
                ),
                rrb.Spatial2DView(origin=parent_log_path / "last_keyframe"),
                rrb.TextDocumentView(origin=parent_log_path),
            ),
            column_shares=(3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint


class RerunLogger:
    def __init__(self, parent_log_path: Path, log_pointclouds: bool = False):
        self.parent_log_path: Path = parent_log_path
        # Only log per-keyframe pointclouds if --full-slam is enabled WITHOUT --custom-shaders
        # Custom shaders mode uses log_global_map() instead (mesh-based global reconstruction)
        self.log_pointclouds = log_pointclouds
        # Create a 3x3 rotation matrix for 90-degree rotation around X-axis
        rr.log(f"{self.parent_log_path}", rr.ViewCoordinates.RDF, static=True)
        # this does not work and I don't know why
        rr.log(
            f"{parent_log_path}",
            rr.Transform3D(
                rotation=rr.RotationAxisAngle(axis=(0, 0, 1), radians=-np.pi / 4)
            ),
            static=True,
        )

        self.path_list = []
        self.keyframe_logged_list = []
        self.global_map_logged_list = []  # Track which keyframes have been logged as meshes (custom shaders mode)
        self.num_keyframes_logged = 0
        self.conf_thresh = 1.0  # Confidence threshold for point filtering
        self.image_plane_distance = 0.2

        # Depth filtering: Only log points within this depth range (in camera frame)
        # Z-axis in camera frame is depth (forward direction)
        # This prevents long streaks extending far behind the camera
        self.min_depth = 0.1  # Minimum depth in meters (avoid points too close/behind camera)
        self.max_depth = 3.0  # Maximum depth in meters (tighter constraint for cleaner reconstruction)

        # Localization filtering: Only show pointclouds for well-localized keyframes
        # Keyframes with N_updates >= min_updates have been refined by tracking/optimization
        self.min_updates_for_display = 1  # Require at least 1 update (involved in tracking)

    def _filter_ceiling_local(self, positions, colors, mat4x4):
        """
        Memory-efficient ceiling filter: removes top 30% of points by Y-value.
        Filters based on LOCAL Y-range of THIS pointcloud only (in camera frame).
        Y-axis is vertical in camera coordinates.

        Args:
            positions: Point positions in camera frame
            colors: Point colors
            mat4x4: Camera pose transformation matrix (not used for filtering)

        Returns:
            Filtered positions and colors
        """
        if len(positions) == 0:
            return positions, colors

        # Use LOCAL camera-frame Y coordinates (vertical axis in camera frame)
        # In camera coordinates: X=right, Y=up, Z=forward (depth)
        # This ensures each pointcloud is filtered based on its OWN Y-range
        y_coords = positions[:, 1]  # Y in camera frame (vertical)

        y_min = y_coords.min()
        y_max = y_coords.max()
        y_range = y_max - y_min

        # Strategy: Remove bottom 42% of points by Y-value (42nd percentile)
        # In camera coords, Y often points DOWN, so low Y = ceiling, high Y = floor
        # We want to remove ceiling, so we remove the BOTTOM 42% (lowest Y values)
        y_threshold = np.percentile(y_coords, 42)

        # print(f"[DEBUG] LOCAL Y range: [{y_min:.2f}, {y_max:.2f}], range={y_range:.2f}m")
        # print(f"[DEBUG] Y threshold (42nd percentile): {y_threshold:.2f}")

        # Filter by Y threshold - keep points ABOVE threshold (higher Y = floor/walls)
        height_mask = y_coords > y_threshold
        filtered_positions = positions[height_mask]
        filtered_colors = colors[height_mask]

        # Debug info
        points_removed = len(positions) - len(filtered_positions)
        # print(f"[DEBUG] Removed {points_removed}/{len(positions)} points ({100*points_removed/len(positions):.1f}%)")

        return filtered_positions, filtered_colors

    def log_frame(
        self, current_frame: Frame, keyframes: SharedKeyframes, states: SharedStates
    ):
        # Add your rerun logging logic here
        H = current_frame.img_shape.squeeze()[0].item()
        W = current_frame.img_shape.squeeze()[1].item()

        pp: Float32[torch.Tensor, "2"] = torch.tensor((W / 2, H / 2))
        pts3d: Float32[torch.Tensor, "H W 3"] = (
            current_frame.X_canon.clone().cpu().reshape(H, W, 3)
        )
        focal: float = float(
            estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld")
        )

        rgb_img: Float32[torch.Tensor, "H W 3"] = current_frame.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img * 255).numpy().astype(np.uint8)

        # Debug: Check image data before logging
        # print(f"[RerunLogger] Logging frame {current_frame.frame_id}: img shape={rgb_img.shape}, dtype={rgb_img.dtype}, range=[{rgb_img.min()}, {rgb_img.max()}]")

        se3_pose: lietorch.SE3 = as_SE3(current_frame.T_WC.cpu())
        matb4x4: Float32[np.ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        mat4x4: Float32[np.ndarray, "4 4"] = matb4x4[
            0
        ]  # Extract the first batch element

        mat4x4 = conventions.convert_pose(
            mat4x4, src_convention=conventions.CC.CV, dst_convention=conventions.CC.GL
        )

        # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
        rotation_matrix: Float32[np.ndarray, "3 3"] = mat4x4[
            :3, :3
        ]  # Top-left 3x3 block
        translation_vector: Float32[np.ndarray, "3"] = mat4x4[
            :3, 3
        ]  # Right column, first 3 elements

        # Debug: Log camera pose and orientation
        import math
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # print(f"[RerunLogger] Frame {current_frame.frame_id}: pos=({translation_vector[0]:.4f}, {translation_vector[1]:.4f}, {translation_vector[2]:.4f}), yaw={yaw:.4f} rad ({math.degrees(yaw):.1f}°)")
        # print(f"[RerunLogger] Frame {current_frame.frame_id}: rotation_matrix[0,0]={rotation_matrix[0,0]:.4f}, rotation_matrix[1,0]={rotation_matrix[1,0]:.4f}")

        cam_log_path = self.parent_log_path / "current_camera"
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                focal_length=focal,
                principal_point=pp.numpy(),
                height=H,
                width=W,
                camera_xyz=rr.ViewCoordinates.RUB,
                image_plane_distance=self.image_plane_distance * 2,
            ),
        )
        try:
            rr.log(
                f"{cam_log_path}/pinhole/image",
                rr.Image(image=rgb_img, color_model=rr.ColorModel.RGB).compress(
                    jpeg_quality=75
                ),
            )
            # print(f"[RerunLogger] ✓ Logged image to rerun: {cam_log_path}/pinhole/image")
        except Exception as e:
            print(f"[RerunLogger] ✗ Failed to log image: {e}")
        self.path_list.append(translation_vector.tolist())
        rr.log(
            f"{self.parent_log_path}/path",
            rr.LineStrips3D(
                strips=self.path_list,
                colors=(255, 0, 0),
                labels=("Camera Path"),
            ),
        )

        with keyframes.lock:
            N_keyframes = len(keyframes)
            dirty_idx = keyframes.get_dirty_idx()

        # Convert dirty_idx to a set for faster lookup
        dirty_set = set(dirty_idx.cpu().numpy().tolist()) if len(dirty_idx) > 0 else set()

        # Process ALL keyframes to update Transform3D with latest optimized poses
        # Re-log pointcloud for dirty keyframes to remove "ghost" clouds at old positions
        for kf_idx in range(N_keyframes):
            keyframe: Frame = keyframes[kf_idx]
            se3_pose: lietorch.SE3 = as_SE3(keyframe.T_WC.cpu())
            matb4x4: Float32[np.ndarray, "1 4 4"] = (
                se3_pose.matrix().numpy().astype(dtype=np.float32)
            )
            mat4x4: Float32[np.ndarray, "4 4"] = matb4x4[
                0
            ]  # Extract the first batch element

            # Keep in OpenCV (RDF) convention - no conversion needed
            # World is RDF, points are in camera frame (RDF), so pose should also be RDF

            # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
            rotation_matrix: Float32[np.ndarray, "3 3"] = mat4x4[
                :3, :3
            ]  # Top-left 3x3 block
            translation_vector: Float32[np.ndarray, "3"] = mat4x4[
                :3, 3
            ]  # Right column, first 3 elements
            cam_log_path = self.parent_log_path / "keyframes" / f"keyframe-{kf_idx}"

            is_new_keyframe = kf_idx not in self.keyframe_logged_list
            is_dirty_keyframe = kf_idx in dirty_set

            # Check if keyframe is well-localized (has been updated by tracking/optimization)
            is_localized = keyframe.N_updates >= self.min_updates_for_display

            # Log static content for new keyframes OR re-log pointcloud for dirty keyframes
            if is_new_keyframe or (is_dirty_keyframe and self.log_pointclouds):
                # Get image (needed for colors)
                kf_img: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
                kf_img: UInt8[np.ndarray, "H W 3"] = (
                    (kf_img * 255).numpy().astype(np.uint8)
                )

                # Log image only for new keyframes (not dirty)
                if is_new_keyframe:
                    rr.log(
                        f"{cam_log_path}/pinhole/image",
                        rr.Image(image=kf_img, color_model=rr.ColorModel.RGB).compress(),
                    )

                # Log/re-log pointcloud ONLY for well-localized keyframes (if --full-slam enabled)
                # This prevents showing uncertain/unoptimized pointclouds that cause slipping
                if self.log_pointclouds and is_localized:
                    # Create a mask based on the confidence values
                    conf_mask = keyframe.C.cpu().numpy() > self.conf_thresh
                    conf_mask = conf_mask.squeeze()

                    # Get positions in camera frame (may have been updated by weighted_pointmap)
                    positions: Float32[np.ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
                    colors: UInt8[np.ndarray, "num_points 3"] = kf_img.reshape(-1, 3)

                    # Apply confidence mask
                    masked_positions = positions[conf_mask]
                    masked_colors = colors[conf_mask]

                    # CRITICAL: Filter by depth (Z-axis in camera frame)
                    # Only keep points within robot's perimeter (e.g., 0.1m to 5m)
                    # This prevents long streaks extending far from camera
                    depth_values = masked_positions[:, 2]  # Z-coordinate is depth
                    depth_mask = (depth_values >= self.min_depth) & (depth_values <= self.max_depth)

                    depth_filtered_positions = masked_positions[depth_mask]
                    depth_filtered_colors = masked_colors[depth_mask]

                    # Filter ceiling based on local Y-range of this pointcloud (in camera frame)
                    filtered_positions, filtered_colors = self._filter_ceiling_local(
                        depth_filtered_positions, depth_filtered_colors, mat4x4
                    )

                    # Log pointcloud in camera frame (Transform3D will handle world transform)
                    # Re-logging to same path replaces old pointcloud, removing "ghost" at old position
                    if len(filtered_positions) > 0:
                        rr.log(
                            f"{cam_log_path}/pointcloud",
                            rr.Points3D(
                                positions=filtered_positions,
                                colors=filtered_colors,
                            ),
                        )

                # Log pinhole camera parameters only for new keyframes
                if is_new_keyframe:
                    rr.log(
                        f"{cam_log_path}/pinhole",
                        rr.Pinhole(
                            focal_length=focal,
                            principal_point=pp.numpy(),
                            height=H,
                            width=W,
                            camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV convention (matches world coordinate system)
                            image_plane_distance=self.image_plane_distance,
                        ),
                    )

                    self.keyframe_logged_list.append(kf_idx)

            # Hide pointcloud for unlocalized keyframes (clear by logging empty pointcloud)
            elif self.log_pointclouds and not is_localized and kf_idx in self.keyframe_logged_list:
                # Keyframe exists but is not yet well-localized - hide its pointcloud
                rr.log(
                    f"{cam_log_path}/pointcloud",
                    rr.Points3D(positions=[], colors=[]),
                )

            # ALWAYS update Transform3D with latest optimized pose (even for existing keyframes)
            # This is critical: when backend optimizes poses, we need to update the transform
            # so the pointcloud (which stays in camera frame) gets positioned correctly
            rr.log(
                f"{cam_log_path}",
                rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
            )

        # log the last keyframe image
        if N_keyframes > 0:
            last_kf: Frame = keyframes[N_keyframes - 1]
            last_kf_img: Float32[torch.Tensor, "H W 3"] = last_kf.uimg
            last_kf_img: UInt8[np.ndarray, "H W 3"] = (
                (last_kf_img * 255).numpy().astype(np.uint8)
            )
            rr.log(
                f"{self.parent_log_path}/last_keyframe",
                rr.Image(image=last_kf_img, color_model=rr.ColorModel.RGB).compress(),
            )

        # Log the edges
        with states.lock:
            ii: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_ii, dtype=torch.long
            )
            jj: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_jj, dtype=torch.long
            )
            if ii.numel() > 0 and jj.numel() > 0:
                T_WCi = lietorch.Sim3(keyframes.T_WC[ii, 0])
                T_WCj = lietorch.Sim3(keyframes.T_WC[jj, 0])
        if ii.numel() > 0 and jj.numel() > 0:
            t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
            t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
            line_strips = []
            for t_i, t_j in zip(t_WCi, t_WCj):
                line_strips.append(t_i.tolist())
                line_strips.append(t_j.tolist())
            rr.log(
                f"{self.parent_log_path}/edges",
                rr.LineStrips3D(
                    strips=line_strips, colors=(0, 255, 0), labels=("Factor Graph")
                ),
            )

    def log_global_map(self, keyframes: SharedKeyframes, conf_thresh: float = 0.0):
        """
        [CUSTOM SHADERS MODE - EXPERIMENTAL]
        Log mesh-based global reconstruction to Rerun viewer.

        This attempts to replicate the original MASt3R-SLAM OpenGL shader visualization:
        - Each keyframe's pointmap (X_canon in camera frame) is triangulated into a mesh
        - Transformation to world frame happens via Rerun's Transform3D (like OpenGL's m_model matrix)

        NOTE: This is experimental and only used when --custom_shaders flag is enabled.
        The default behavior (--full-slam without --custom_shaders) uses per-keyframe pointclouds
        during streaming and a final fused pointcloud at the end (original rerun-master approach).
        - Pointmap is triangulated by connecting neighboring pixels (like trianglemap.glsl)
        - Confidence filtering is applied per-quad (like the shader)

        Args:
            keyframes: SharedKeyframes object containing all keyframes
            conf_thresh: Confidence threshold for filtering points (default: 0.0)
        """
        from mast3r_slam.config import config
        from mast3r_slam.geometry import get_pixel_coords

        # Only process NEW keyframes (not already logged)
        # This achieves O(N) complexity like the original OpenGL implementation
        total_vertices = 0
        total_triangles = 0
        num_new_keyframes = 0

        for i in range(len(keyframes)):
            # Skip keyframes that have already been logged
            if i in self.global_map_logged_list:
                continue

            keyframe = keyframes[i]
            num_new_keyframes += 1

            # Get image dimensions
            h, w = keyframe.img_shape.flatten()[:2].cpu().numpy().astype(int)

            # Get pointmap in camera frame (same as frame_X() in visualization.py)
            X_canon = self._frame_X(keyframe)

            # Reshape to H×W×3 grid (camera frame)
            X_grid = X_canon.reshape(h, w, 3)

            # Convert points from OpenCV (RDF) to OpenGL (RUB) convention
            # RDF: Right=+X, Down=+Y, Forward=+Z
            # RUB: Right=+X, Up=+Y, Back=+Z
            # Conversion: flip Y and Z
            # X_grid = X_grid * np.array([1, -1, -1], dtype=np.float32)

            # Get colors (H×W×3)
            colors = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(h, w, 3)

            # Get confidence (H×W)
            avg_conf = keyframe.get_average_conf()
            if avg_conf is not None:
                conf_grid = avg_conf.cpu().numpy().astype(np.float32).reshape(h, w)
            else:
                conf_grid = np.ones((h, w), dtype=np.float32)

            # Create mesh by triangulating the grid (similar to trianglemap.glsl)
            # Points are in CAMERA FRAME, will be transformed to world via Transform3D
            # Use conservative thresholds to avoid blurry/stretched mesh:
            # - slant_thresh=0.2 (higher = more aggressive filtering of grazing angles)
            # - depth_discontinuity_thresh=0.3 (30% max depth variation in a quad)
            vertices, vertex_colors, triangles = self._create_mesh_from_pointmap(
                X_grid, colors, conf_grid, conf_thresh,
                slant_thresh=0.2,
                depth_discontinuity_thresh=0.3
            )

            if len(vertices) > 0:
                # Apply ceiling filter based on LOCAL Y-range in camera coordinates
                # Each keyframe's mesh is filtered independently based on its own Y-distribution
                y_coords = vertices[:, 1]  # Y in camera frame (vertical)

                # Remove bottom 42% of points by Y-value (42nd percentile)
                # In camera coords, Y points DOWN, so low Y = ceiling, high Y = floor
                y_threshold = np.percentile(y_coords, 42)

                # Keep vertices ABOVE threshold (higher Y = floor/walls, remove ceiling)
                vertex_mask = y_coords > y_threshold

                # Filter vertices and colors
                filtered_vertices = vertices[vertex_mask]
                filtered_vertex_colors = vertex_colors[vertex_mask]

                # Create a mapping from old vertex indices to new vertex indices
                old_to_new_idx = np.full(len(vertices), -1, dtype=np.int32)
                old_to_new_idx[vertex_mask] = np.arange(np.sum(vertex_mask))

                # Filter triangles: keep only triangles where ALL 3 vertices are kept
                valid_triangles = []
                for tri in triangles:
                    new_tri = old_to_new_idx[tri]
                    if np.all(new_tri >= 0):  # All 3 vertices survived filtering
                        valid_triangles.append(new_tri)

                if len(valid_triangles) > 0:
                    filtered_triangles = np.array(valid_triangles, dtype=np.uint32)

                    # Log camera pose transformation (like m_model in OpenGL shader)
                    cam_log_path = f"{self.parent_log_path}/global_map/keyframe_{i}"

                    # Transform from camera frame to world frame using T_WC
                    se3_pose = as_SE3(keyframe.T_WC.cpu())
                    mat4x4 = se3_pose.matrix().numpy().astype(np.float32)[0]

                    # Keep in OpenCV (RDF) convention - no conversion needed
                    # This matches the full-SLAM mode behavior (see log_keyframes method)
                    # World is RDF, points are in camera frame (RDF), so pose should also be RDF

                    # Extract rotation (3x3) and translation (3,) from 4x4 matrix
                    rotation_matrix = mat4x4[:3, :3]
                    translation_vector = mat4x4[:3, 3]

                    rr.log(
                        cam_log_path,
                        rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix)
                    )

                    # Log filtered mesh in camera frame (will be transformed by parent Transform3D)
                    rr.log(
                        f"{cam_log_path}/mesh",
                        rr.Mesh3D(
                            vertex_positions=filtered_vertices,
                            vertex_colors=filtered_vertex_colors,
                            triangle_indices=filtered_triangles,
                        )
                    )
                    total_vertices += len(filtered_vertices)
                    total_triangles += len(filtered_triangles)

                    # Mark this keyframe as logged
                    self.global_map_logged_list.append(i)

        if num_new_keyframes > 0:
            print(f"[RerunLogger] ✓ Logged {num_new_keyframes} new keyframe mesh(es): {total_vertices:,} vertices, {total_triangles:,} triangles (total keyframes: {len(keyframes)})")
        # else:
        #     print(f"[RerunLogger] No new keyframes to log")

    def _frame_X(self, frame):
        """
        Get pointmap in camera frame, applying calibration constraint if enabled.
        This replicates the frame_X() method from visualization.py lines 358-380.

        Args:
            frame: Keyframe object

        Returns:
            X: (H*W, 3) array of 3D points in camera frame
        """
        from mast3r_slam.config import config
        from mast3r_slam.geometry import get_pixel_coords

        if config["use_calib"]:
            # Constrain points to camera rays (depth * ray_direction)
            Xs = frame.X_canon[None]
            img_size = frame.img_shape.flatten()[:2]
            K = frame.K

            # Get pixel coordinates
            p = get_pixel_coords(
                Xs.shape[0], img_size, device=Xs.device, dtype=Xs.dtype
            ).view(*Xs.shape[:-1], 2)

            # Compute ray directions from camera center
            tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
            tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
            dP_dz = torch.empty(
                p.shape[:-1] + (3, 1), device=Xs.device, dtype=Xs.dtype
            )
            dP_dz[..., 0, 0] = tmp1
            dP_dz[..., 1, 0] = tmp2
            dP_dz[..., 2, 0] = 1.0
            dP_dz = dP_dz[..., 0]

            # Constrain to rays: depth * ray_direction
            X = (Xs[..., 2:3] * dP_dz)[0].cpu().numpy().astype(np.float32)
            return X

        return frame.X_canon.cpu().numpy().astype(np.float32)

    def _create_mesh_from_pointmap(self, points, colors, conf, conf_thresh, slant_thresh=0.1, depth_discontinuity_thresh=0.5):
        """
        Create a triangle mesh from a H×W pointmap by connecting neighboring points.
        This replicates the trianglemap.glsl shader logic (lines 41-93).

        The shader creates quads (2 triangles) for each pixel, filtering by:
        1. Border pixels (10 pixel margin)
        2. Confidence threshold (all 4 corners must pass)
        3. Slant threshold (surface angle relative to camera ray)
        4. Depth discontinuity (reject quads with large depth jumps)

        Args:
            points: (H, W, 3) array of 3D positions in camera frame
            colors: (H, W, 3) array of RGB colors
            conf: (H, W) array of confidence values
            conf_thresh: confidence threshold for filtering
            slant_thresh: slant threshold for filtering grazing angles (default: 0.1)
            depth_discontinuity_thresh: max relative depth difference between quad corners (default: 0.5 = 50%)

        Returns:
            vertices: (N, 3) array of vertex positions
            vertex_colors: (N, 3) array of vertex colors
            triangles: (M, 3) array of triangle indices
        """
        h, w = points.shape[:2]

        # Border margin (line 44 in trianglemap.glsl)
        border = 10

        # Create vertex list and index mapping
        vertex_map = np.full((h, w), -1, dtype=np.int32)
        vertices_list = []
        colors_list = []

        # First pass: collect all vertices that might be used
        for y in range(border, h - border):
            for x in range(border, w - border):
                if vertex_map[y, x] == -1:
                    vertex_map[y, x] = len(vertices_list)
                    vertices_list.append(points[y, x])
                    colors_list.append(colors[y, x])

        if len(vertices_list) == 0:
            return np.array([]), np.array([]), np.array([])

        # Second pass: create triangles (quads) with confidence filtering
        triangles = []
        for y in range(border, h - border - 1):
            for x in range(border, w - border - 1):
                # Get indices of 4 neighboring points: TL, TR, BL, BR (line 50-51)
                tl = vertex_map[y, x]
                tr = vertex_map[y, x + 1]
                bl = vertex_map[y + 1, x]
                br = vertex_map[y + 1, x + 1]

                # Check if all 4 vertices exist
                if tl < 0 or tr < 0 or bl < 0 or br < 0:
                    continue

                # Confidence filtering: only check top-left corner (line 57 + 75-78 in trianglemap.glsl)
                # The shader fetches conf from (x,y) for all 4 pixels, so it only checks TL confidence
                if conf[y, x] <= conf_thresh:
                    continue

                # Get 4 corner points
                p_tl = points[y, x]
                p_tr = points[y, x + 1]
                p_bl = points[y + 1, x]
                p_br = points[y + 1, x + 1]

                # Depth discontinuity filtering: reject quads with large depth jumps
                # Compute depths (Z-coordinate in camera frame)
                depths = np.array([                  
                    p_tl[2],  # Z-coordinate = depth along camera axis
                    p_tr[2],
                    p_bl[2],
                    p_br[2]
                ])

                # Check relative depth variation: (max - min) / mean
                depth_mean = np.mean(depths)
                depth_variation = (np.max(depths) - np.min(depths)) / (depth_mean + 1e-8)

                if depth_variation > depth_discontinuity_thresh:
                    continue  # Skip this quad - depth discontinuity too large

                # Slant threshold filtering (line 60-70 in trianglemap.glsl)
                # Compute surface normals for the 2 triangles

                # Normal for triangle 1 (TL, BL, TR)
                n1 = np.cross(p_bl - p_tl, p_tr - p_tl)
                n1_norm = np.linalg.norm(n1)
                if n1_norm > 1e-8:
                    n1 = n1 / n1_norm
                else:
                    continue  # Degenerate triangle

                # Normal for triangle 2 (TR, BL, BR)
                n2 = np.cross(p_bl - p_tr, p_br - p_tr)
                n2_norm = np.linalg.norm(n2)
                if n2_norm > 1e-8:
                    n2 = n2 / n2_norm
                else:
                    continue  # Degenerate triangle

                # Ray directions from camera (normalized positions in camera frame)
                ray1 = p_tl / (np.linalg.norm(p_tl) + 1e-8)
                ray2 = p_tr / (np.linalg.norm(p_tr) + 1e-8)

                # Check if surface is too slanted (grazing angle)
                # abs(dot(normal, ray)) < threshold means surface is nearly parallel to viewing direction
                if abs(np.dot(n1, ray1)) < slant_thresh:
                    continue
                if abs(np.dot(n2, ray2)) < slant_thresh:
                    continue

                # Create 2 triangles in CCW order (line 82-92)
                # Triangle 1: TL, BL, TR
                # Triangle 2: TR, BL, BR
                triangles.append([tl, bl, tr])
                triangles.append([tr, bl, br])

        if len(triangles) == 0:
            return np.array(vertices_list, dtype=np.float32), np.array(colors_list, dtype=np.uint8), np.array([])

        vertices = np.array(vertices_list, dtype=np.float32)
        vertex_colors = np.array(colors_list, dtype=np.uint8)
        triangles = np.array(triangles, dtype=np.uint32)

        return vertices, vertex_colors, triangles