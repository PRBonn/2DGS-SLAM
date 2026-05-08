import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from utils.logging_utils import Log
from utils.io_utils import clone_obj
from utils.slam_utils import image_gradient, image_gradient_mask

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append("gaussian_splatting")
from gaussian_renderer import render, render_for_tracking
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from utils.camera_utils import Camera

from utils.eval_utils import (
    eval_ate,
    eval_ate_for_indices,
    eval_rendering,
    write_run_metrics_csv,
)
from utils.mesh_utils import tsdf_fusion

from gui import gui_utils

# Minimum linear scale along the thinnest axis when exporting live.ply for Spark (2DGS).
SPARK_LIVE_MIN_LINEAR_SCALE = 1e-4

@dataclass
class Frame:
    camera_id: int
    rgb: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    mapping_mask: Optional[torch.Tensor] = None


class FrontEnd(mp.Process):
    """Pose tracking and frontend I/O. Calls ``run()`` on the main process; mapping uses ``BackEnd`` in a subprocess."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.dataset = None
        self.dust3r = None
        
        self.initialized = False
        self.iteration_count = 0

        self.requested_init = False
        self.requested_refine = False
        self.requested_keyframe = 0
        self.requested_pgo = 0

        self.last_loop_at_len_kf = 0
        self.last_loop_id = 0

        self.gaussians = None
        self.cameras = dict()
        self.key_frames = dict()
        self.key_frame_ids = []
        self.loop_frame_ids = []

        self.loop_uid_pairs = []

        self.device = "cuda:0"
        self.dtype = torch.float32
        self.pause = False

        bg_color = self.config["Training"]["background_color"]
        if bg_color == 'white':
            self.bg_color = torch.tensor([1.0,1.0,1.0], device=self.device, dtype=self.dtype)
        else :
            self.bg_color = torch.tensor([0.0,0.0,0.0], device=self.device, dtype=self.dtype)

    def set_params(self):
        rs = self.config["Results"]
        dset = self.config["Dataset"]
        self.name_tag = str(rs.get("name_tag", rs.get("test_num", "test")))
        run_folder = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + self.name_tag
        self.save_dir = os.path.join(
            rs["save_dir"], dset["type"], dset["sequence_name"], run_folder
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.use_gui = rs["use_gui"]
        self.save_trj_kf_intv = rs["save_trj_kf_intv"]
        self.map_refine = rs["map_refine"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]

        self.min_depth = self.config["Training"]["depth_min_threshold"]
        self.max_depth = self.config["Training"]["depth_max_threshold"]
        
        depth_type = self.config["Training"]["depth_type"]
        if depth_type == 'expected':
            self.depth_type = 'rend_depth_expected'
        else:
            self.depth_type = 'rend_depth_median'

        # for gradient mask 
        self.edge_threshold = self.config["Training"]["edge_threshold"]
        self.rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]

        self.loop_candidate_score = self.config["Training"]["loop_candidate_score"]
        self.old_than_N_keyframe = self.config["Training"]["old_than_N_keyframe"]

        self.kf_overlap = self.config["Training"]["kf_overlap"]
        self.kf_max_translation = self.config["Training"]["kf_max_translation"]
        self.kf_min_overlap = self.config["Training"]["kf_min_overlap"]

        self.traking_lambda_depth = self.config["Training"]["traking_lambda_depth"]
        self.traking_lambad_grad = self.config["Training"]["traking_lambad_grad"]

        self.rot_lr = self.config["Training"]["lr"]["cam_rot_delta"]
        self.trans_lr = self.config["Training"]["lr"]["cam_trans_delta"]

        # larger than the threshold means this gaussian can be observed.
        self.active_threshold = 0.5
        self.conf_threshold = 1.5

        self.loop_overlap_ratio = self.config["Training"]["loop_overlap_ratio"]
        self.depth_error_threshold = self.config["Training"]["depth_error_threshold"]
        self.reloc_method = self.config["Training"].get("reloc_method", "mast3r")
        self.enable_revisit_loop = self.config["Training"].get("enable_revisit_loop", True)
        self.enable_scale_optimization = self.config["Training"].get("enable_scale_optimization", True)
        self.enable_mast3r_reloc = self.config["Training"].get("enable_mast3r_reloc", True)

        self.resized_w, self.resized_h = self.dust3r.input_img_size
        self.resized_K = self.dust3r.resized_K

        self.frame_begin = max(0, int(dset.get("frame_begin", 0)))
        fe_raw = dset.get("frame_end")
        self.frame_end_exclusive = None if fe_raw is None else int(fe_raw)
        self.frame_step = max(1, int(dset.get("frame_step", 1)))

        self._spark_live_enabled = rs.get("spark_live_enable", False)
        self._spark_live_interval = float(rs.get("spark_live_interval_sec", 5.0))
        self._spark_live_port = int(rs.get("spark_live_port", 8765))
        self._spark_live_dir = os.path.join(self.save_dir, "spark_live")
        self._spark_live_ply_path = os.path.join(self._spark_live_dir, "live.ply")
        self._spark_live_traj_path = os.path.join(self._spark_live_dir, "traj_live.json")
        self._spark_live_min_linear_scale = SPARK_LIVE_MIN_LINEAR_SCALE
        self._last_spark_export_time = 0.0
        self._shutdown_spark_live = None

        self.verbose = bool(rs.get("verbose", False))
        self.log_mesh_cleaning = bool(rs.get("log_mesh_cleaning", False))

    
    def request_init(self, camera, frame):
        msg = ['init', camera, frame]
        self.backend_queue.put(msg)

    def request_map_refine(self):
        msg = ['refine', self._spark_live_dir if self._spark_live_enabled else None]
        self.backend_queue.put(msg)

    def request_key_frame(self, camera, frame, loop_frames):
        msg = ['keyframe', camera, frame, loop_frames]
        self.backend_queue.put(msg)

    def request_pgo(self, cur_camera, cur_frame, loop_camera):
        msg = ['pgo', clone_obj(cur_camera), clone_obj(cur_frame), clone_obj(loop_camera)]
        self.backend_queue.put(msg)

    def sync_backend(self, data):
        del self.gaussians
        self.gaussians = data[1]
        key_cams = data[2]
        for cam in key_cams:
            if cam.uid in self.cameras:
                del self.cameras[cam.uid]
            self.cameras[cam.uid] = cam
        if len(data) > 3:
            lp = data[3] or []
            self.loop_uid_pairs = [(int(a), int(b)) for a, b in lp]

        # Release CUDA IPC bookkeeping for tensors received from backend via multiprocessing;
        # avoids slow teardown / warnings when IPC handle count grows (W502 CudaIPCTypes).
        coll = getattr(torch.cuda, "ipc_collect", None)
        if coll is not None and torch.cuda.is_available():
            coll()

    def _sorted_frame_indices(self):
        return sorted(self.cameras.keys())

    def _full_est_traj_arrays(self):
        """All tracked poses (camera-to-world) every processed frame, not keyframes only."""
        keys = self._sorted_frame_indices()
        if len(keys) < 2:
            return None, None
        poses = []
        ids = []
        for k in keys:
            cam = self.cameras[k]
            cR, ct = cam.get_RT
            W2C = getWorld2View2(cR, ct).detach().cpu().numpy()
            poses.append(np.linalg.inv(W2C))
            ids.append(int(cam.uid))
        return np.stack(poses, axis=0), np.asarray(ids, dtype=np.int64)

    def _full_gt_traj_arrays(self):
        keys = self._sorted_frame_indices()
        if len(keys) < 2:
            return None
        poses = []
        for k in keys:
            cam = self.cameras[k]
            if cam.gt_pose is None:
                return None
            g = cam.gt_pose.detach().cpu()
            cR, ct = g[:3, :3], g[:3, 3]
            W2C = getWorld2View2(cR, ct).detach().cpu().numpy()
            poses.append(np.linalg.inv(W2C))
        return np.stack(poses, axis=0)

    def _vertex_loop_edges_indices(self, traj_frame_ids_np):
        if traj_frame_ids_np is None or len(self.loop_uid_pairs) == 0:
            return None
        fid = traj_frame_ids_np.reshape(-1)
        id_to_idx = {int(x): i for i, x in enumerate(fid)}
        out = []
        for a, b in self.loop_uid_pairs:
            ia, ib = id_to_idx.get(a), id_to_idx.get(b)
            if ia is None or ib is None:
                continue
            out.append([ia, ib])
        if not out:
            return None
        return np.asarray(out, dtype=np.int32)

    def get_mesh(self, file_name):
        tsdf_fusion(
            self.cameras,
            self.gaussians,
            self.depth_type,
            self.key_frame_ids,
            file_name,
            verbose_mesh_clean=self.log_mesh_cleaning,
        )

    def eval_pose(self, cur_frame_idx, quiet=False):
        if not quiet:
            Log("Evaluating ATE at frame: ", cur_frame_idx)
        eval_ate(
            self.cameras,
            self.key_frame_ids,
            cur_frame_idx,
            monocular=False,
            quiet=quiet,
        )

    def eval_pose_keyframes_final(self, quiet=False):
        kf_ids = sorted(self.key_frame_ids)
        if len(kf_ids) < 2:
            if not quiet:
                Log("Skipping keyframe ATE: need at least 2 keyframes", tag="Eval")
            return None
        if not quiet:
            Log(
                f"Final ATE on all keyframes (n={len(kf_ids)})",
                tag="Eval",
            )
        return eval_ate_for_indices(
            self.cameras,
            kf_ids,
            "final_keyframes",
            monocular=False,
            quiet=quiet,
        )

    def eval_pose_all_tracked(self, quiet=False):
        ids = sorted(self.cameras.keys())
        if len(ids) < 2:
            if not quiet:
                Log("Skipping full-tracked ATE: need at least 2 poses", tag="Eval")
            return None
        if not quiet:
            Log(
                f"Final ATE on all tracked frames (n={len(ids)})",
                tag="Eval",
            )
        return eval_ate_for_indices(
            self.cameras,
            ids,
            "final_all_tracked",
            monocular=False,
            quiet=quiet,
        )

    def eval_rendering(self):
        return eval_rendering(
            self.cameras,
            self.gaussians,
            self.dataset,
            self.bg_color,
            self.key_frame_ids,
        )

    def _save_run_metrics_csv(self, ate_kf, ate_all, rend_out):
        run_name = os.path.basename(os.path.normpath(self.save_dir))
        csv_path = os.path.join(self.save_dir, "metrics.csv")
        mp, ms, ml = None, None, None
        if rend_out is not None:
            mp = rend_out.get("mean_psnr")
            ms = rend_out.get("mean_ssim")
            ml = rend_out.get("mean_lpips")
        write_run_metrics_csv(
            csv_path,
            run_name,
            ate_rmse_keyframes_m=ate_kf,
            ate_rmse_all_tracked_m=ate_all,
            mean_psnr=mp,
            mean_ssim=ms,
            mean_lpips=ml,
        )
        Log(f"Wrote metrics CSV: {csv_path}", tag="Eval")
    
    def optimize_scale(self, ref_camera, depth_1, confidence_1, option):
        render_pkg = render(ref_camera, pc = self.gaussians, render_option=option, bg_color=self.bg_color)
        depth_0 = render_pkg[self.depth_type].squeeze(0)
        opactiy = render_pkg["rend_alpha"]
        mask_0 = (opactiy > 0.95).squeeze(0)
        mask_1 = confidence_1 > self.conf_threshold
        mask = torch.logical_and(mask_0, mask_1)
        d0_flat, d1_flat = (d[mask].view(-1) for d in (depth_0, depth_1))
        K = (d0_flat * d1_flat).sum() / (d1_flat ** 2).sum()
        return K
    
    def compute_grad_mask(self, original_image):
        gray_img = original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        median_img_grad_intensity = img_grad_intensity.median()
        grad_mask = (
                img_grad_intensity > median_img_grad_intensity * self.edge_threshold
        )

        _, h, w = original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_pixel_mask = (original_image.sum(dim=0) > self.rgb_boundary_threshold).view(*mask_shape)
        
        rgb_tracking_mask = rgb_pixel_mask * grad_mask

        return rgb_tracking_mask, rgb_pixel_mask
        
        
    def initialize(self, first_id):
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        first_original_img, first_pil_img, first_depth, first_gt_pose = self.dataset[first_id]
        
        cam_start = Camera(first_id, first_gt_pose, first_gt_pose, self.dataset.K, self.dataset.height, self.dataset.width)
        _, mapping_mask = self.compute_grad_mask(first_original_img)

        start_frame = Frame(camera_id=first_id, rgb=first_original_img, depth=first_depth, mapping_mask=mapping_mask)

        resized_first_img = self.dust3r.preprocess(first_pil_img)
        self.dust3r.add_img_to_retriever(resized_first_img, first_id)
        
        self.cameras[first_id] = cam_start
        self.key_frames[first_id] = start_frame
        self.key_frame_ids.append(first_id)

        self.request_init(cam_start, start_frame)
        self.requested_init = True

        self.initialized = True
    

    def tracking(self, camera, gt_img, depth, grad_mask,  tracking_mask=None, render_option='active'):
        opt_params = []
        opt_params.append({"params": [camera.cam_rot_delta], "lr": self.rot_lr})
        opt_params.append({"params": [camera.cam_trans_delta], "lr": self.trans_lr})
        opt_params.append({"params": [camera.exposure_a], "lr": 0.01})
        opt_params.append({"params": [camera.exposure_b], "lr": 0.01})
        pose_optimizer = torch.optim.AdamW(opt_params, weight_decay=0.01)

        self.gaussians.optimizer.zero_grad(set_to_none=True)
        depth_mask = torch.logical_and(depth > self.min_depth, depth < self.max_depth)
        
        if render_option == 'active':
            stat_mask = self.gaussians.get_active_mask.squeeze(1)
        elif render_option == 'inactive':
            stat_mask = ~self.gaussians.get_active_mask.squeeze(1)
        g_xyz = self.gaussians.get_xyz[stat_mask]
        g_opacity = self.gaussians.get_opacity[stat_mask]
        g_scales = self.gaussians.get_scaling[stat_mask]
        g_rotations = self.gaussians.get_rotation[stat_mask]
        g_shs = self.gaussians.get_features[stat_mask]
        ray_vectors = camera.camera_ray_vectors()

        for i in range(self.tracking_itr_num):
            render_pkg = render_for_tracking(camera, g_xyz, g_opacity, g_scales, g_rotations, g_shs,
                                             self.gaussians.active_sh_degree, bg_color=self.bg_color)
            rendered_image = render_pkg["render"]
            rendered_opacity = render_pkg["rend_alpha"]
            rendered_depth = render_pkg[self.depth_type]
            rendered_normal = render_pkg["rend_normal"]

            image_ab = torch.exp(camera.exposure_a) * rendered_image + camera.exposure_b
            normal_weights = -(ray_vectors*rendered_normal).sum(0)

            opacity_mask = rendered_opacity > 0.95
            normal_mask = normal_weights > 0

            valid_mask = torch.logical_and(opacity_mask, normal_mask)
            if tracking_mask is not None:
                valid_mask = torch.logical_and(valid_mask, tracking_mask)

            color_error = torch.abs(gt_img - image_ab).mean(0)
            color_error[~(valid_mask.squeeze(0))] = 0.0
            color_error[grad_mask.squeeze(0)] *= self.traking_lambad_grad
            color_loss = color_error.mean()

            valid_depth_mask = torch.logical_and(depth_mask, valid_mask)
            depth_error = torch.abs(depth - rendered_depth)
            depth_error[~valid_depth_mask] = 0.0
            depth_loss = depth_error.mean()

            loss = color_loss + self.traking_lambda_depth*depth_loss

            loss.backward()
            
            pose_optimizer.step()
            pose_optimizer.zero_grad()
            with torch.no_grad():
                converged = camera.update_pose()
                if converged:
                    break
        
        render_pkg = render(camera, pc = self.gaussians, render_option=render_option, bg_color=self.bg_color)
        rendered_opacity = render_pkg["rend_alpha"]
        rendered_normal = render_pkg["rend_normal"]

        normal_weights = -(ray_vectors*rendered_normal).sum(0)
        opacity_mask = rendered_opacity > 0.95
        normal_mask = normal_weights > 0
        valid_mask = torch.logical_and(opacity_mask, normal_mask)
        if tracking_mask is not None:
            valid_mask = torch.logical_and(valid_mask, tracking_mask)
        valid_depth_mask = torch.logical_and(depth_mask, valid_mask)

        rendered_depth = render_pkg[self.depth_type]
        depth_error = torch.abs(rendered_depth - depth)

        depth_error[~valid_depth_mask] = 0
        depth_avg_error = (depth_error.sum() / valid_depth_mask.sum()).detach()

        return render_pkg, depth_avg_error
    
        
    def caculate_overlap(self, contributions, key_camera, option='active'):
        render_pkg = render(key_camera, pc = self.gaussians, render_option=option, bg_color=self.bg_color)
        key_contributions = render_pkg["contributions"]
        activated = contributions > self.active_threshold
        key_activated = key_contributions > self.active_threshold

        union = torch.logical_or(activated, key_activated).count_nonzero()
        intersection = torch.logical_and(activated, key_activated).count_nonzero()
        overlap_ratio = intersection/union
        return overlap_ratio
    

    def detect_loop_by_featquery(self, query_image, query_camera):
        # detect loop based on image feature
        id_and_scores = self.dust3r.query_from_retriever(query_image, query_camera.uid)
        if id_and_scores is not None:
            ids = id_and_scores[0]
            scores = id_and_scores[1]
            last_n_id = self.key_frame_ids[-self.old_than_N_keyframe:][0]
                
            for idx in range(len(ids)):
                k_id = ids[idx]
                k_score = scores[idx]
                if (k_id < last_n_id) and (k_score > self.loop_candidate_score):
                    return k_id
                if k_score < self.loop_candidate_score:
                    break

        return -1
    

    def detect_loop_by_revisit(self, query_depth, query_camera):
        render_pkg = render(query_camera, pc = self.gaussians, render_option="inactive", bg_color=self.bg_color)
        rendered_opacity = render_pkg["rend_alpha"]
        contributions = render_pkg["contributions"]
        rendered_depth = render_pkg[self.depth_type]
        rendered_normal = render_pkg["rend_normal"]

        ray_vectors = query_camera.camera_ray_vectors()
        normal_weights = -(ray_vectors*rendered_normal).sum(0)
        normal_mask = normal_weights > 0

        max_query_depth = torch.max(query_depth).item()
        max_query_depth = min(max_query_depth, self.max_depth)

        opacity_mask = rendered_opacity > 0.95
        depth_mask = rendered_depth < max_query_depth
        valid_mask = torch.logical_and(opacity_mask, depth_mask)
        valid_mask = torch.logical_and(normal_mask, valid_mask)
        observed_ratio = valid_mask.sum()/(valid_mask.shape[1] * valid_mask.shape[2])

        if observed_ratio > self.loop_overlap_ratio:
            # find the frame who observe most inactive gaussians in current view
            active_mask = self.gaussians.get_active_mask.squeeze(1)
            non_active_ids = self.gaussians.get_ids[~active_mask].squeeze(1)

            unique_ids, inverse_indices = torch.unique(non_active_ids, return_inverse=True)
            contribution_sums = torch.zeros_like(unique_ids, device=contributions.device, dtype=contributions.dtype)
            contribution_sums.scatter_add_(0, inverse_indices, contributions)
            max_index = torch.argmax(contribution_sums)
            loop_id = unique_ids[max_index].item()
            return loop_id
        
        return -1

    def is_this_loop_necessary(self, new_loop_id):
        if (len(self.key_frame_ids) - self.last_loop_at_len_kf) >= self.old_than_N_keyframe:
            return True
        else:
            new_loop_idx = self.key_frame_ids.index(new_loop_id)
            last_loop_idx = self.key_frame_ids.index(self.last_loop_id)
            if (last_loop_idx - new_loop_idx) >= self.old_than_N_keyframe:
                return True
        
        return False

    def try_loop_closure(self, query_resized_img, query_depth, query_camera):
        if self.enable_revisit_loop:
            loop_id = self.detect_loop_by_revisit(query_depth, query_camera)
            if loop_id > 0 and self.is_this_loop_necessary(loop_id):
                if self.verbose:
                    Log(f"revisit loop: {loop_id}", tag="Loop")
                if self.reloc_method == 'mast3r':
                    loop_cam = self.reloc_with_mast3r(query_camera, query_resized_img, loop_id, loop_type='revisit')
                else:
                    loop_cam = self.reloc_with_icp(query_camera, loop_id, self.reloc_method)
                if loop_cam is not None:
                    return loop_cam
        
        loop_id = self.detect_loop_by_featquery(query_resized_img, query_camera)
        if loop_id > 0 and self.is_this_loop_necessary(loop_id):
            if self.reloc_method == 'mast3r':
                loop_cam = self.reloc_with_mast3r(query_camera, query_resized_img, loop_id, loop_type='featquery')
            else:
                loop_cam = self.reloc_with_icp(query_camera, loop_id, self.reloc_method)
            if loop_cam is not None:
                return loop_cam
        
        return None
    
    def overlap_mask_for_depth_pair(self, depth_0, pose_0, depth_1, pose_1, K_0, K_1, depth_threshold=0.05):
        # return the overlap mask of depth_1
        K_0 = torch.as_tensor(K_0, dtype=self.dtype, device=depth_0.device)
        K_1 = torch.as_tensor(K_1, dtype=self.dtype, device=depth_0.device)

        R0 = pose_0[:3, :3]
        t0 = pose_0[:3, 3]
        R1 = pose_1[:3, :3]
        t1 = pose_1[:3, 3]

        H, W = depth_1.shape
        u1, v1 = torch.meshgrid(torch.arange(W, device=depth_0.device), 
                                torch.arange(H, device=depth_0.device), 
                                indexing='xy')
        
        u1 = u1.float()
        v1 = v1.float()

        uv1 = torch.stack([u1, v1, torch.ones_like(u1)], dim=-1)

        K_0_inv = torch.inverse(K_0)
        P_cam1 = torch.einsum('ij,hwj->hwi', K_0_inv, uv1)  
        P_cam1 = P_cam1 * depth_1.unsqueeze(-1)

        R1_inv = torch.inverse(R1)
        P_world = torch.einsum('ij,hwj->hwi', R1_inv, P_cam1 - t1)

        P_cam0 = torch.einsum('ij,hwj->hwi', R0, P_world) + t0  

        p0 = torch.einsum('ij,hwj->hwi', K_1, P_cam0)
        u0 = p0[..., 0] / p0[..., 2]
        v0 = p0[..., 1] / p0[..., 2]

        valid_u = (u0 >= 0) & (u0 < W)
        valid_v = (v0 >= 0) & (v0 < H)
        valid_mask = valid_u & valid_v

        u0 = u0[valid_mask]
        v0 = v0[valid_mask]

        input_depth = depth_0[v0.long(), u0.long()]
        projected_depth = p0[..., 2][valid_mask]

        valid_depth = input_depth > 0
        depth_consistent = torch.abs(projected_depth - input_depth) < depth_threshold

        overlap_mask = torch.zeros((H, W), device=depth_0.device)
        overlap_mask[valid_mask] = (valid_depth & depth_consistent).float()

        return overlap_mask

    
    def preprocess_point_cloud(self, pcd, voxel_size, camera_location):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                                max_nn=30))

        pcd_down.orient_normals_towards_camera_location(
            camera_location=camera_location)

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                                max_nn=100))
        return (pcd_down, pcd_fpfh)


    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, global_iter, conf):
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(global_iter, conf))
        return result


    def reloc_with_icp(self, cur_cam, loop_id ,icp_method): # icp_method can be "robust_icp" or "icp"

        max_correspondence_distance_coarse = 0.3
        max_correspondence_distance_fine = 0.03
        
        loop_cam = clone_obj(self.cameras[loop_id])
        loop_img, _, loop_depth, _ = self.dataset[loop_id]

        min_depth = 0.1
        max_depth = 10.0
        mask = torch.logical_and(loop_depth > min_depth, loop_depth < max_depth)
        
        fx, fy, cx, cy = loop_cam.fx, loop_cam.fy, loop_cam.cx, loop_cam.cy
        scaled_depth = loop_depth * loop_cam.scale + loop_cam.shift
        
        i, j = torch.nonzero(mask, as_tuple=True)
        x_cam = (j - cx) / fx
        y_cam = (i - cy) / fy
        z = torch.ones_like(x_cam, dtype=loop_depth.dtype, device=loop_depth.device)
        rays_cam = torch.stack((x_cam, y_cam, z), dim=-1)
        
        valid_depths = scaled_depth[mask].unsqueeze(1)
        
        with torch.no_grad():
            iR, iT = loop_cam.get_inv_RT
            new_ray_dir = torch.mm(iR, rays_cam.t()).t()
        
        source_points = new_ray_dir * valid_depths + iT
        source_points = source_points.detach().cpu().numpy()
        source_t = loop_cam.T[:3, 3].detach().cpu().numpy()


        cloud_source = o3d.geometry.PointCloud()
        cloud_source.points = o3d.utility.Vector3dVector(np.array(source_points))
        cloud_source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        cloud_source.orient_normals_towards_camera_location(camera_location=source_t)

        active_mask = self.gaussians.get_active_mask.squeeze(1)
        target_points = self.gaussians.get_xyz[active_mask].detach().cpu().numpy()
        target_t = cur_cam.T[:3, 3].detach().cpu().numpy()

        cloud_target = o3d.geometry.PointCloud()
        cloud_target.points = o3d.utility.Vector3dVector(np.array(target_points))
        cloud_target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        cloud_target.orient_normals_towards_camera_location(camera_location=target_t)

        if icp_method == "icp":
            icp_coarse = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_coarse, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            delta = icp_fine.transformation
            delta_torch = torch.from_numpy(np.array(delta)).to(device=loop_cam.device, dtype=loop_cam.dtype)
        
        elif icp_method == "robust_icp":
            voxel_size = 0.04
            sigma = 0.01
            global_iter = 10000000
            conf = 0.99999

            source_down, source_fpfh = self.preprocess_point_cloud(
                cloud_source, voxel_size, source_t)
            target_down, target_fpfh = self.preprocess_point_cloud(
                cloud_target, voxel_size, target_t)
            
            result_ransac = self.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size, global_iter, conf)

            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
            
            delta = icp_fine.transformation
            delta_torch = torch.from_numpy(np.array(delta)).to(device=loop_cam.device, dtype=loop_cam.dtype)
        
        inv_delta = torch.linalg.inv(delta_torch)
        loop_cam.T = loop_cam.T @ inv_delta

        return loop_cam

    
    def reloc_with_mast3r(self, cur_cam, cur_resized_img, loop_id, loop_type):
        
        loop_cam = clone_obj(self.cameras[loop_id])
        loop_frame = self.key_frames[loop_id]

        if self.enable_mast3r_reloc:
            _, loop_pil_img, _, _ = self.dataset[loop_id]
            loop_resized_img = self.dust3r.preprocess(loop_pil_img)
            poses, depths, confidences, Ks = self.dust3r.predict_2view([cur_resized_img, loop_resized_img])
            if confidences[0].mean() < 3.0:
                return None
            
            loop_tracking_mask = self.overlap_mask_for_depth_pair(depths[0], poses[0], depths[1], poses[1], Ks[0], Ks[1], depth_threshold=0.05)
            raw_w, raw_h = self.dataset.width, self.dataset.height
            loop_tracking_mask = F.interpolate(loop_tracking_mask.unsqueeze(0).unsqueeze(0), size=(raw_h, raw_w), mode='nearest')
            loop_tracking_mask = loop_tracking_mask.squeeze(0).squeeze(0)

            overlap_ratio = loop_tracking_mask.sum()/(raw_h * raw_w)
            if self.verbose:
                Log(f"overlap_ratio: {overlap_ratio}", tag="Loop")
            if overlap_ratio < self.loop_overlap_ratio:
                return None
            
            cur_cam_resized = Camera(cur_cam.uid, cur_cam.T, cur_cam.gt_pose, Ks[0], self.resized_h, self.resized_w)
            if self.enable_scale_optimization:
                init_scale = self.optimize_scale(cur_cam_resized, depths[0], confidences[0], 'active')
            else:
                init_scale = 1.0

            relative_pose = poses[0].inverse() @ poses[1]
            relative_pose[:-1,-1] *= init_scale

            init_pose = relative_pose @ cur_cam.T

            loop_cam.T = init_pose
        else:
            loop_tracking_mask = None  # use default tracking_mask when mast3r is disabled

        grad_mask, _ = self.compute_grad_mask(loop_frame.rgb)

        _, depth_avg_error = self.tracking(loop_cam, loop_frame.rgb, loop_frame.depth, 
                                           grad_mask, loop_tracking_mask, render_option='active')

        if self.verbose:
            Log(f"depth avg error: {depth_avg_error}", tag="Loop")
        if depth_avg_error > self.depth_error_threshold:
            return None
        
        return loop_cam
    

    def save_poses_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for i in self.key_frame_ids:
                cam = self.cameras[i]
                pose = cam.T.reshape(-1).cpu().numpy()
                pose_str = ' '.join(map(str, pose))
                f.write(f"{i} {pose_str}\n")

    def save_full_trajectory_tum(self, file_path):
        with open(file_path, 'w') as f:
            for i in sorted(self.cameras.keys()):
                cam = self.cameras[i]
                w2c = cam.T.detach().cpu().numpy()
                c2w = np.linalg.inv(w2c)
                t = c2w[:3, 3]
                quat = R.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
                f.write(f"{i} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")

    def save_trajectory_ply(self, file_path):
        import imgviz
        ids = sorted(self.cameras.keys())
        points = []
        for i in ids:
            cam = self.cameras[i]
            w2c = cam.T.detach().cpu().numpy()
            c2w = np.linalg.inv(w2c)
            points.append(c2w[:3, 3])
        points = np.array(points)
        n = len(points)
        norm_ids = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        colors_rgb = imgviz.depth2rgb(
            norm_ids.reshape(-1, 1), min_value=0.0, max_value=1.0, colormap="jet"
        ).reshape(-1, 3).astype(np.float64) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        o3d.io.write_point_cloud(file_path, pcd)


    def _export_spark_traj_json(self):
        """Trajectory polyline + intrinsics + first/latest camera poses for Spark viewer."""
        K, h = self.dataset.K, float(self.dataset.height)
        w = float(self.dataset.width)
        fy = float(
            K[1, 1].detach().cpu() if isinstance(K, torch.Tensor) else np.asarray(K)[1, 1]
        )
        viewer_intrinsics = {
            "fovy_deg": float(2 * np.rad2deg(np.arctan(h / (2 * fy)))),
            "aspect": w / h,
            "near": float(self.min_depth),
            "far": float(self.max_depth),
        }

        def spark_view(fid):
            cR, ct = self.cameras[fid].get_RT
            c2w = np.linalg.inv(getWorld2View2(cR, ct).detach().cpu().numpy())
            return {"frame_id": int(fid), "c2w_rows": c2w.astype(np.float64).tolist()}

        keys = self._sorted_frame_indices()
        first_view = spark_view(keys[0]) if keys else None
        current_view = spark_view(keys[-1]) if keys else None
        keyframe_views = [
            spark_view(fid) for fid in self.key_frame_ids if fid in self.cameras
        ]

        poses, ids = self._full_est_traj_arrays()
        positions, frame_ids, loop_edges = [], [], []
        if poses is not None:
            positions = poses[:, :3, 3].astype(np.float64).tolist()
            frame_ids = ids.astype(int).tolist()
            lp = self._vertex_loop_edges_indices(ids)
            if lp is not None and lp.size:
                loop_edges = lp.astype(int).tolist()

        tmp = self._spark_live_traj_path + ".partial"
        payload = {
            "version": 3,
            "poll_interval_ms": int(self._spark_live_interval * 1000),
            "positions": positions,
            "frame_ids": frame_ids,
            "loop_edges": loop_edges,
            "viewer_intrinsics": viewer_intrinsics,
            "first_view": first_view,
            "current_view": current_view,
            "keyframe_views": keyframe_views,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, self._spark_live_traj_path)

    def _tick_spark_live_export(self):
        if not self._spark_live_enabled:
            return
        if self.gaussians is None:
            return
        now = time.time()
        if now - self._last_spark_export_time < self._spark_live_interval:
            return
        self._last_spark_export_time = now
        try:
            os.makedirs(self._spark_live_dir, exist_ok=True)
            self._export_spark_traj_json()
            tmp = self._spark_live_ply_path + ".partial"
            floor = self._spark_live_min_linear_scale
            self.gaussians.save_ply(
                tmp,
                min_linear_scale_smallest_axis=floor if floor > 0 else None,
                pad_scale_dims_to_three=True,
            )
            os.replace(tmp, self._spark_live_ply_path)
        except Exception as e:
            Log("Spark live export failed: ", str(e), tag="Spark")

    def add_keyframe(self, uid, frame, resized_img):
        self.key_frames[uid] = frame
        self.key_frame_ids.append(uid)
        self.dust3r.add_img_to_retriever(resized_img, uid)

    def run(self):
        if self._spark_live_enabled:
            try:
                from utils.spark_live_server import start_spark_live_server

                self._shutdown_spark_live = start_spark_live_server(
                    self._spark_live_dir, self._spark_live_port
                )
            except OSError as e:
                Log("Spark live server failed to start (port in use?): ", str(e), tag="Spark")
                self._spark_live_enabled = False

        vis_copy_count = 0
        cur_frame_idx = self.frame_begin
        n_ds = len(self.dataset)
        end_exc = min(
            n_ds,
            n_ds if self.frame_end_exclusive is None else self.frame_end_exclusive,
        )
        step = self.frame_step
        track_from = cur_frame_idx + step
        total_track = max(0, len(range(track_from, end_exc, step)))
        pbar = tqdm(
            total=total_track,
            desc="SLAM",
            unit="frame",
            dynamic_ncols=True,
        )
        t_slam0 = time.perf_counter()
        try:
            while True:
                self._tick_spark_live_export()
                if self.q_vis2main.empty():
                    if self.pause:
                        continue
                else:
                    data_vis2main = self.q_vis2main.get()
                    self.pause = data_vis2main.flag_pause
                    if self.pause:
                        self.backend_queue.put(["pause"])
                        continue
                    else:
                        self.backend_queue.put(["unpause"])
    
                if self.frontend_queue.empty():
                    if self.requested_refine:
                        time.sleep(0.01)
                        continue
    
                    if cur_frame_idx >= end_exc:
                        ate_kf = self.eval_pose_keyframes_final()
                        ate_all = self.eval_pose_all_tracked()
                        rend_out = self.eval_rendering()
                        nt = self.name_tag
                        file_name = os.path.join(self.save_dir, f"gsmap_{nt}.ply")
                        self.gaussians.save_ply(file_name)
                        Log("save the gs map to:", file_name)
                        mesh_file_name = os.path.join(self.save_dir, f"mesh_{nt}.ply")
                        self.get_mesh(mesh_file_name)
                        Log("save the mesh to:", mesh_file_name)
                        pose_file_name = os.path.join(self.save_dir, f"key_pose_{nt}.txt")
                        self.save_poses_to_file(pose_file_name)
                        traj_tum_file = os.path.join(self.save_dir, f"traj_tum_{nt}.txt")
                        self.save_full_trajectory_tum(traj_tum_file)
                        Log("save full trajectory (TUM format) to:", traj_tum_file)
                        traj_ply_file = os.path.join(self.save_dir, f"traj_{nt}.ply")
                        self.save_trajectory_ply(traj_ply_file)
                        Log("save trajectory point cloud to:", traj_ply_file)
                        if self.verbose:
                            Log(f"To offline inspect the map, use: python viser.py --ply_path {file_name} --pose_path {traj_tum_file} --mesh_path {mesh_file_name}")
                        dt = time.perf_counter() - t_slam0
                        nf = len(self.cameras)
                        # Log(f"Total time \\[s\\] {dt:.4f}", tag="Eval")
                        Log(f"SLAM FPS [Hz] {nf / dt:.4f}", tag="Eval")
                        if self.map_refine is True:
                            self.request_map_refine()
                            self.requested_refine = True
                            continue
                        else:
                            self._save_run_metrics_csv(ate_kf, ate_all, rend_out)
                            break
    
                    if self.requested_init:
                        time.sleep(0.01)
                        continue
    
                    if not self.initialized :
                        self.initialize(cur_frame_idx)
                        cur_frame_idx += step
                        continue
    
                    if self.requested_keyframe > 0:
                        time.sleep(0.01)
                        continue
    
                    if self.requested_pgo > 0:
                        time.sleep(0.01)
                        continue
                    
                    original_img, pil_img, depth, gt_pose = self.dataset[cur_frame_idx]
                    
                    last_idx = cur_frame_idx - step
                    init_pose = self.cameras[last_idx].T
    
                    cur_cam = Camera(cur_frame_idx, init_pose, gt_pose, self.dataset.K, self.dataset.height, self.dataset.width)
                    grad_mask, mapping_mask = self.compute_grad_mask(original_img)
                    depth_mask = torch.logical_and(depth > self.min_depth, depth < self.max_depth)
                    mapping_mask = torch.logical_and(depth_mask, mapping_mask)
                    cur_frame = Frame(camera_id=cur_frame_idx, rgb=original_img, depth=depth, mapping_mask=mapping_mask)
    
                    render_pkg, _ = self.tracking(cur_cam, original_img, depth, grad_mask=grad_mask)
    
                    self.cameras[cur_frame_idx] = cur_cam
    
                    resized_img = self.dust3r.preprocess(pil_img)
                    loop_cam = self.try_loop_closure(resized_img, depth, cur_cam)
    
                    if loop_cam is not None :
                        self.last_loop_at_len_kf = len(self.key_frame_ids)
                        self.last_loop_id = loop_cam.uid
                        if self.verbose:
                            Log("Loop detected at frame: ", loop_cam.uid)
                        # see current frame as a key frame
    
                        self.add_keyframe(cur_frame_idx, cur_frame, resized_img)
    
                        loop_idx = self.key_frame_ids.index(loop_cam.uid)
                        left_idx = max(loop_idx - 5, 0)
                        right_idx = min(loop_idx + 5, len(self.key_frame_ids))
    
                        self.loop_frame_ids = self.key_frame_ids[left_idx : right_idx]
    
                        self.request_pgo(cur_cam, cur_frame, loop_cam)
    
                        self.requested_pgo += 1
    
                        self.eval_pose(cur_frame_idx, quiet=not self.verbose)
    
                        cur_frame_idx += step
                        pbar.update(1)
                        pbar.set_postfix(kf=len(self.key_frame_ids), frame=cur_frame_idx)
                        continue


                    last_key_cam = self.cameras[self.key_frame_ids[-1]]
                    ratio = self.caculate_overlap(render_pkg["contributions"], last_key_cam)
                    t_dis = torch.norm(cur_cam.T - last_key_cam.T)
    
                    t_check = t_dis > 0.15
                    overlap_check = ratio < self.kf_overlap
                    t_check_max = t_dis > self.kf_max_translation
                    overlap_max = ratio < self.kf_min_overlap
    
                    if (t_check and overlap_check) or t_check_max or overlap_max :
    
                        self.add_keyframe(cur_frame_idx, cur_frame, resized_img)
    
                        if (len(self.key_frame_ids) - self.last_loop_at_len_kf) >= self.old_than_N_keyframe:
                            self.loop_frame_ids = []
    
                        self.request_key_frame(cur_cam, cur_frame, self.loop_frame_ids)
                    
                        self.requested_keyframe += 1
    
                        if len(self.key_frame_ids) % self.save_trj_kf_intv == 0:
                            self.eval_pose(cur_frame_idx, quiet=not self.verbose)
    
                    cur_frame_idx += step
                    pbar.update(1)
                    pbar.set_postfix(kf=len(self.key_frame_ids), frame=cur_frame_idx)

                    if self.use_gui:

                        # For visualization
                        render_img = render_pkg["render"]
                        rendered_opacity = render_pkg["rend_alpha"]
    
                        image_ab = torch.exp(cur_cam.exposure_a) * render_img + cur_cam.exposure_b
    
                        opacity_mask = rendered_opacity > 0.95
    
                        keyframes = [clone_obj(self.cameras[kf_idx]) for kf_idx in self.key_frame_ids]
    
                        gtkeyframes = []
                        for kf_idx in self.key_frame_ids:
                            gt_cam = clone_obj(self.cameras[kf_idx])
                            gt_cam.T = gt_cam.gt_pose
                            gtkeyframes.append(gt_cam)
    
                        if (vis_copy_count%10) == 0:
                            vis_gaussians = clone_obj(self.gaussians)
                        else:
                            vis_gaussians = None
    
                        vis_copy_count += 1
    
                        est_traj, traj_frame_ids = self._full_est_traj_arrays()
                        gt_traj = self._full_gt_traj_arrays()
                        loop_edges_ij = None
                        if est_traj is not None:
                            loop_edges_ij = self._vertex_loop_edges_indices(traj_frame_ids)
                        gpu_gb = (
                            torch.cuda.memory_allocated(self.device) / (1024**3)
                            if torch.cuda.is_available()
                            else None
                        )
    
                        n_g = self.gaussians.get_xyz.shape[0]
                        act_m = self.gaussians.get_active_mask.squeeze(1)
                        n_g_act = int(act_m.sum().item())
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=vis_gaussians,
                                current_frame=cur_cam,
                                keyframes=keyframes,
                                gtframes=None,
                                gtcolor=original_img,
                                gtdepth=depth.cpu().numpy(),
                                est_traj=est_traj,
                                gt_traj=gt_traj,
                                traj_frame_ids=traj_frame_ids,
                                loop_edges=loop_edges_ij,
                                gpu_mem_usage_gb=gpu_gb,
                                num_keyframes=len(self.key_frame_ids),
                                num_loop_closures=len(self.loop_uid_pairs),
                                num_gaussians_total=n_g,
                                num_gaussians_active=n_g_act,
                            )
                        )
    
                    if cur_frame_idx % 10 == 0:
                        torch.cuda.empty_cache()
                else:
                    data = self.frontend_queue.get()
                    if data[0] == "sync_backend":
                        self.sync_backend(data)
    
                    elif data[0] == "keyframe":
                        self.sync_backend(data)
                        self.requested_keyframe -= 1
    
                    elif data[0] == "pgo":
                        self.sync_backend(data)
                        self.eval_pose(cur_frame_idx, quiet=not self.verbose)
                        self.requested_pgo -= 1
    
                    elif data[0] == "init":
                        self.sync_backend(data)
                        self.requested_init = False
    
                    elif data[0] == 'refine':
                        self.sync_backend(data)
                        ate_kf = self.eval_pose_keyframes_final()
                        ate_all = self.eval_pose_all_tracked()
                        rend_out = self.eval_rendering()
                        nt = self.name_tag
                        file_name = os.path.join(self.save_dir, f"refined_gsmap_{nt}.ply")
                        self.gaussians.save_ply(file_name)
                        Log("save the refined gs map to:", file_name)
                        mesh_file_name = os.path.join(self.save_dir, f"refined_mesh_{nt}.ply")
                        self.get_mesh(mesh_file_name)
                        Log("save the mesh to:", mesh_file_name)
                        if self.verbose:
                            pose_file_name = os.path.join(self.save_dir, f"key_pose_{nt}.txt")
                            Log(f"To offline inspect the map, use: python viser.py --ply_path {file_name} --pose_path {pose_file_name} --mesh_path {mesh_file_name}")
                        self._save_run_metrics_csv(ate_kf, ate_all, rend_out)
                        break
    
                    elif data[0] == "stop":
                        Log("Frontend Stopped.")
                        break
        finally:
            pbar.close()

