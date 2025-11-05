from dataclasses import dataclass
import sys
import time
import lietorch
import torch
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker
import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager
from timeit import default_timer as timer
import rerun as rr


from simplecv.rerun_log_utils import RerunTyroConfig
from pathlib import Path
from typing import Literal
import rerun.blueprint as rrb

from mast3r_slam.rerun_log_utils import create_blueprints, RerunLogger
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio

# Track active subprocesses for cleanup
# session_id -> {'backend': Process, 'frontend': Process}
_active_processes = {}


def format_time(seconds):
    """Format time in minutes:seconds format (mm:ss)."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class InferenceConfig:
    rr_config: RerunTyroConfig
    dataset: str = "data/normal-apt-tour.MOV"
    config: str = "config/base.yaml"
    save_as: str = "default"
    no_viz: bool = False
    img_size: Literal[224, 512] = 512
    ns_save_path: None | Path = None
    all_frames: bool = False  # Save poses for all frames, not just keyframes
    rerun_server_addr: str | None = None  # Optional rerun server address (e.g., "master_slam_cli:9878")
    real_time: bool = False  # Real-time mode: output pose after each frame instead of at the end
    full_slam: bool = False  # Full SLAM mode: build and export global fused pointcloud (dense reconstruction)
    conf_thresh: float = 0.0  # Confidence threshold for pointcloud filtering (C_conf from config)
    custom_shaders: bool = False  # Custom shader mode: replicate OpenGL shader behavior with mesh-based reconstruction (experimental)
    log_trajectory_json: bool = False  # Export trajectory to JSON file (for debugging/analysis)


def mast3r_slam_inference(inf_config: InferenceConfig):
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    ## rerun setup
    # Connect to rerun server if server address is provided (rerun is enabled)
    # NOTE: rr.init() was already called by RerunTyroConfig.__post_init__()
    # DO NOT call rr.init() again - it would create a new recording and break threading!
    if inf_config.rerun_server_addr:
        # Get the existing global recording stream
        # This ensures we use the same recording across all threads
        print(f"[SLAM Inference] Using existing global recording stream")

        # Connect to rerun server
        # CRITICAL: Use flush_timeout_sec=0.1 to ensure data is sent immediately
        # flush_timeout_sec=None causes data to be buffered indefinitely and dropped in daemon threads!
        print(f"[SLAM Inference] Connecting to rerun server at {inf_config.rerun_server_addr}")
        rr.connect_grpc(f"rerun+http://{inf_config.rerun_server_addr}/proxy", flush_timeout_sec=0.1)
        print(f"[SLAM Inference] ✓ Connected with flush_timeout_sec=0.1 (auto-flush enabled)")

    parent_log_path = Path("/world")
    # Only log pointclouds if --full-slam is enabled (for dense reconstruction)
    rr_logger = RerunLogger(parent_log_path, log_pointclouds=inf_config.full_slam)
    # create a blueprint
    blueprint: rrb.Blueprint = create_blueprints(parent_log_path)
    rr.send_blueprint(blueprint)

    load_config(inf_config.config)
    print(inf_config.dataset)
    print(config)

    manager: SyncManager = mp.Manager()

    dataset = load_dataset(inf_config.dataset, img_size=inf_config.img_size)
    dataset.subsample(config["dataset"]["subsample"])

    h, w = dataset.get_img_shape()[0]
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib: bool = dataset.has_calib()
    use_calib: bool = config["use_calib"]
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)
        print(f"Saving results to {save_dir}")
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.pt"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)

    backend = mp.Process(
        target=run_backend, args=(inf_config.config, model, states, keyframes, K)
    )
    backend.start()

    # Store backend process for cleanup
    session_id = inf_config.save_as
    _active_processes[session_id] = {'backend': backend}

    # Collect all frames if --all-frames flag is set
    all_frames = [] if inf_config.all_frames else None

    i = 0
    fps_timer: float = time.time()
    start_time = timer()

    while True:
        rr.set_time_sequence(timeline="frame", sequence=i)
        mode: Mode = states.get_mode()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]

        # Check for graceful shutdown (dataset terminated while waiting)
        if timestamp is None or img is None:
            print(f"[SLAM Inference] Dataset terminated, stopping gracefully...")
            states.set_mode(Mode.TERMINATED)
            break

        # get frames last camera pose
        T_WC: lietorch.Sim3 = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame: Frame = create_frame(
            i, img, T_WC, img_size=dataset.img_size, device=device
        )

        # Set intrinsics K on the frame if using calibrated mode
        if config["use_calib"] and K is not None:
            frame.K = K
            print(f"[Inference] Frame {i}: Set K matrix on frame (use_calib=True)")

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features needed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            rr_logger.log_frame(frame, keyframes, states)

            # Custom Shaders Mode: Log global mesh map after initial keyframe (experimental)
            # This replicates OpenGL shader behavior but is slower and experimental
            if inf_config.custom_shaders and not inf_config.no_viz:
                rr_logger.log_global_map(keyframes, conf_thresh=inf_config.conf_thresh)

            # Collect all frames if --all-frames flag is set
            if inf_config.all_frames:
                all_frames.append(frame)

            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        ## rerun log stuff
        rr_logger.log_frame(frame, keyframes, states)

        # Custom Shaders Mode: Log global mesh map after each new keyframe (experimental)
        # This replicates OpenGL shader behavior but is slower and experimental
        if add_new_kf and inf_config.custom_shaders and not inf_config.no_viz:
            rr_logger.log_global_map(keyframes, conf_thresh=inf_config.conf_thresh)

        # Collect all frames if --all-frames flag is set
        if inf_config.all_frames:
            all_frames.append(frame)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    # Finalize any remaining pointclouds in the batch
    if inf_config.full_slam:
        rr_logger.finalize_pointclouds()

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)

        # Use all_frames if --all-frames flag is set, otherwise use keyframes
        if inf_config.all_frames:
            # Create a temporary SharedKeyframes object to hold all frames
            h, w = dataset.get_img_shape()[0]
            all_frames_shared = SharedKeyframes(manager, h, w, buffer=len(all_frames))
            for frame in all_frames:
                all_frames_shared.append(frame)
            frames_to_save = all_frames_shared
        else:
            frames_to_save = keyframes

        eval.save_ATE(save_dir, f"{seq_name}.txt", dataset.timestamps, frames_to_save)
        # Note: Global PLY reconstruction is saved in sati_master_slam.py if --full-slam is set
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, frames_to_save
        )

    if inf_config.ns_save_path is not None:
        pcd = save_kf_to_nerfstudio(
            ns_save_path=inf_config.ns_save_path,
            keyframes=keyframes,
        )
        rr.log(
            f"{parent_log_path}/final_pointcloud",
            rr.Points3D(positions=pcd.points, colors=pcd.colors),
        )

    print("done")
    print(f"Inference time: {format_time(timer() - start_time)}")
    if inf_config.all_frames:
        print(f"Processed {len(all_frames)} frames (all frames mode)")
    else:
        print(f"Processed {len(keyframes)} keyframes")
    backend.join()
    if not inf_config.no_viz:
        print("All visualization processes terminated")



    # Keep server alive if serving for visualization
    if inf_config.rr_config.serve:
        print("\n" + "="*60)
        print("Rerun server is running for visualization:")
        print(f"  - gRPC: rerun --connect rerun+http://localhost:9876/proxy")
        print(f"  - Web:  http://localhost:9090")
        print("Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down rerun server...")

    # Return keyframes for full SLAM processing
    return keyframes


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(config_path, model, states, keyframes, K):
    load_config(config_path)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success: bool = relocalization(
                frame, keyframes, factor_graph, retrieval_database
            )
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
