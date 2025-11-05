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
        self.log_pointclouds = log_pointclouds  # Only log pointclouds if --full-slam is enabled
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
        self.num_keyframes_logged = 0
        self.conf_thresh = 1.5  # Lowered from 7 to 1.5 for denser pointclouds
        self.image_plane_distance = 0.2

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

        # Only process new keyframes or dirty (updated) keyframes to avoid O(N²) complexity
        # New keyframes: not yet logged (not in self.keyframe_logged_list)
        # Dirty keyframes: poses updated by backend optimization (in dirty_idx)
        keyframes_to_process = []
        for kf_idx in range(N_keyframes):
            if kf_idx not in self.keyframe_logged_list or kf_idx in dirty_idx:
                keyframes_to_process.append(kf_idx)

        for kf_idx in keyframes_to_process:
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

            # IMPORTANT: Log the transform FIRST, before logging any child entities (image, pointcloud)
            # This ensures that when the pointcloud is logged, it's already under the correct transform
            rr.log(
                f"{cam_log_path}",
                rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
            )

            if kf_idx not in self.keyframe_logged_list:
                kf_img: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
                kf_img: UInt8[np.ndarray, "H W 3"] = (
                    (kf_img * 255).numpy().astype(np.uint8)
                )
                rr.log(
                    f"{cam_log_path}/pinhole/image",
                    rr.Image(image=kf_img, color_model=rr.ColorModel.RGB).compress(),
                )

                # Log per-keyframe pointcloud only if --full-slam is enabled
                if self.log_pointclouds:
                    # Create a mask based on the confidence values
                    conf_mask = keyframe.C.cpu().numpy() > self.conf_thresh

                    # Convert the mask from shape (h*w, 1) to shape (h*w,)
                    conf_mask = conf_mask.squeeze()  # Remove the trailing dimension to get a 1D boolean array

                    # Now apply the mask to both positions and colors
                    positions: Float32[np.ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
                    colors: UInt8[np.ndarray, "num_points 3"] = kf_img.reshape(-1, 3)

                    # Apply confidence mask first
                    masked_positions = positions[conf_mask]
                    masked_colors = colors[conf_mask]

                    # Filter out ceiling: keep only bottom 90% by height (Z-coordinate in camera frame)
                    # In camera frame (RDF), Z points forward, Y points down, X points right
                    # We want to filter by Y (vertical) coordinate to remove ceiling
                    if len(masked_positions) > 0:
                        y_coords = masked_positions[:, 1]  # Y is vertical in camera frame
                        # Calculate 90th percentile of Y (higher Y = lower in scene since Y points down)
                        # We want to keep points with Y >= 10th percentile (remove top 10% = ceiling)
                        y_threshold = np.percentile(y_coords, 10)
                        height_mask = y_coords >= y_threshold

                        masked_positions = masked_positions[height_mask]
                        masked_colors = masked_colors[height_mask]

                    if len(masked_positions) > 0:
                        rr.log(
                            f"{cam_log_path}/pointcloud",
                            rr.Points3D(
                                positions=masked_positions,
                                colors=masked_colors,
                            ),
                        )
                self.keyframe_logged_list.append(kf_idx)
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

        # print(f"[RerunLogger] Building global mesh from {len(keyframes)} keyframes with conf_thresh={conf_thresh}...")

        total_vertices = 0
        total_triangles = 0

        for i in range(len(keyframes)):
            keyframe = keyframes[i]

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
            vertices, vertex_colors, triangles = self._create_mesh_from_pointmap(
                X_grid, colors, conf_grid, conf_thresh
            )

            if len(vertices) > 0:
                # Log camera pose transformation (like m_model in OpenGL shader)
                cam_log_path = f"{self.parent_log_path}/global_map/keyframe_{i}"

                # Transform from camera frame to world frame using T_WC
                se3_pose = as_SE3(keyframe.T_WC.cpu())
                mat4x4_cv = se3_pose.matrix().numpy().astype(np.float32)[0]

                # Convert from OpenCV (RDF) to OpenGL (RUB) convention (same as keyframe logging)
                mat4x4_gl = conventions.convert_pose(
                    mat4x4_cv, src_convention=conventions.CC.CV, dst_convention=conventions.CC.GL
                )

                # Extract rotation (3x3) and translation (3,) from 4x4 matrix
                rotation_matrix = mat4x4_gl[:3, :3]
                translation_vector = mat4x4_gl[:3, 3]

                rr.log(
                    cam_log_path,
                    rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix)
                )

                # Log mesh in camera frame (will be transformed by parent Transform3D)
                rr.log(
                    f"{cam_log_path}/mesh",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        vertex_colors=vertex_colors,
                        triangle_indices=triangles,
                    )
                )
                total_vertices += len(vertices)
                total_triangles += len(triangles)

        if total_vertices > 0:
            print(f"[RerunLogger] ✓ Logged global mesh: {total_vertices:,} vertices, {total_triangles:,} triangles from {len(keyframes)} keyframes")
        else:
            print(f"[RerunLogger] ✗ No mesh to log (all filtered out by conf_thresh={conf_thresh})")

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

    def _create_mesh_from_pointmap(self, points, colors, conf, conf_thresh, slant_thresh=0.1):
        """
        Create a triangle mesh from a H×W pointmap by connecting neighboring points.
        This replicates the trianglemap.glsl shader logic (lines 41-93).

        The shader creates quads (2 triangles) for each pixel, filtering by:
        1. Border pixels (10 pixel margin)
        2. Confidence threshold (all 4 corners must pass)
        3. Slant threshold (surface angle relative to camera ray)

        Args:
            points: (H, W, 3) array of 3D positions in camera frame
            colors: (H, W, 3) array of RGB colors
            conf: (H, W) array of confidence values
            conf_thresh: confidence threshold for filtering
            slant_thresh: slant threshold for filtering grazing angles (default: 0.1)

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

                # Slant threshold filtering (line 60-70 in trianglemap.glsl)
                # Compute surface normals for the 2 triangles
                p_tl = points[y, x]
                p_tr = points[y, x + 1]
                p_bl = points[y + 1, x]
                p_br = points[y + 1, x + 1]

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