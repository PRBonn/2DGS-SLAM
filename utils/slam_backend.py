import os
import sys
import time
import random
from collections import deque

from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import numpy as np
import open3d as o3d
from einops import rearrange

from fused_ssim import fused_ssim
from simple_knn._C import distCUDA2

from utils.io_utils import clone_obj
from utils.logging_utils import Log
from utils.pgo import PoseGraphManager
from utils.voxel_utils import VoxelHash

sys.path.append('gaussian_splatting')
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from utils.general_utils import depth2normal


class BackEnd(mp.Process):
    """Gaussian map optimization; runs in a dedicated process and talks to ``FrontEnd`` via queues."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        
        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.iteration_count = 0
        self.initialized = False

        self.sliding_window_size = self.config["Training"]["old_than_N_keyframe"]
        self.cam_sliding_window = deque(maxlen=self.sliding_window_size)
        self.active_cam_id_list = []
        self.inactive_cam_id_list = []
        self.cam_id_list = []

        self.voxel_size = config["Training"]["voxel_size"]
        self.voxels = VoxelHash()
        
        self.pgo = PoseGraphManager(config)
        self.last_node_id = 0
        
        self.key_frames = dict()
        self.key_cameras = dict()

        self.accepted_loop_pairs = []  # (cur_uid, loop_uid), for GUI loop edges
        
        bg_color = self.config["Training"]["background_color"]
        if bg_color == 'white':
            self.bg_color = torch.tensor([1.0,1.0,1.0], device=self.device, dtype=self.dtype)
        else :
            self.bg_color = torch.tensor([0.0,0.0,0.0], device=self.device, dtype=self.dtype)

    def set_params(self):
        self.downsample_ratio = self.config["Dataset"]["pcd_downsample"]
        self.init_iter_num = self.config["Training"]["init_itr_num"]
        
        depth_type = self.config["Training"]["depth_type"]
        if depth_type == 'expected':
            self.depth_type = 'rend_depth_expected'
        else:
            self.depth_type = 'rend_depth_median'

        self.key_frame_iter_num = self.config["Training"]["key_frame_iter_num"]

        self.active_train_num = self.config["Training"]["active_train_num"]
        self.inactive_train_num = self.config["Training"]["inactive_train_num"]

        self.lambda_dssim = self.config["Training"]["lambda_ssim"]
        self.lambda_normal = self.config["Training"]["lambda_normal"]
        self.lambda_depth = self.config["Training"]["lambda_depth"]
        self.lambda_dist = self.config["Training"]["lambda_dist"]

        self.prune_threshold = self.config["Training"]["prune_threshold"]
        self.prune_size_threshold = self.config["Training"]["prune_size_threshold"]

        self.min_depth = self.config["Training"]["depth_min_threshold"]
        self.max_depth = self.config["Training"]["depth_max_threshold"]

        self.old_than_N_keyframe = self.config["Training"]["old_than_N_keyframe"]
        self.count_from_last_loop = 1000000

        rs = self.config.get("Results", {})
        self.map_refine_iterations = max(0, int(rs.get("map_refine_iterations", 26000)))
        self._spark_live_interval = float(rs.get("spark_live_interval_sec", 5.0))

    def reset(self):
        self.accepted_loop_pairs = []
        self.viewpoints = {}
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        while not self.backend_queue.empty():
            self.backend_queue.get()


    def mapping_iteration(self, camera, frame, option):
        train_img = frame.rgb
        train_depth = frame.depth
        mapping_mask = frame.mapping_mask

        render_pkg = render(camera, pc = self.gaussians, bg_color=self.bg_color, render_option=option)
        render_alpha = render_pkg["rend_alpha"]
        rendered_img = render_pkg["render"]
        rendered_normal = render_pkg["rend_normal"] # (3, h, w)
        rendered_depth = render_pkg[self.depth_type]
        rend_dist = render_pkg["rend_dist"]

        mask = (render_alpha > 0.95).squeeze(0)

        mask = torch.logical_and(mask, mapping_mask.squeeze(0))

        img_L1_error = torch.abs(rendered_img - train_img).mean(0)
        ssim_error = fused_ssim(rendered_img.unsqueeze(0), train_img.unsqueeze(0))[0].mean(0)
        img_error = (1.0 - self.lambda_dssim)*img_L1_error + self.lambda_dssim*(1.0 - ssim_error)

        depth_normal = depth2normal(rendered_depth, camera.fx, camera.fy, camera.cx, camera.fy) # (h, w, 3)
        depth_normal = rearrange(depth_normal, 'h w c -> c h w')
        depth_normal = depth_normal*render_alpha.detach()

        normal_error = (1 - (rendered_normal * depth_normal).sum(dim=0))[None]
        normal_error = self.lambda_normal*(normal_error.squeeze(0))

        if train_depth is not None:
            train_depth = camera.scale*train_depth + camera.shift
            depth_difference = rendered_depth - train_depth
            depth_error = self.lambda_depth*torch.abs(depth_difference).mean(0)
        else :
            depth_error = 0.0

        error_map = img_error + depth_error + normal_error

        dist_loss = self.lambda_dist * (rend_dist).mean()

        error_map[~mask] = 0.0

        loss = error_map.mean() + dist_loss
        loss.backward()
    
    def add_gaussians_from_frame(self, camera, frame):
        rgb = torch.clamp(frame.rgb, 0.0, 1.0)
        rgb = rgb.permute(1, 2, 0)
        depth = frame.depth

        mask = torch.logical_and(depth > self.min_depth, depth < self.max_depth)

        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        scaled_depth = depth*camera.scale + camera.shift
        normal = depth2normal(scaled_depth.unsqueeze(0), fx, fy, cx, cy)

        random_nums = torch.rand_like(depth, device=self.device, dtype=self.dtype)
        random_mask = random_nums < self.downsample_ratio
        final_mask = torch.logical_and(random_mask, mask)

        i, j = torch.nonzero(final_mask, as_tuple=True)
        # rays in camera coordinate 
        x_cam = (j - cx) / fx
        y_cam = (i - cy) / fy
        z = torch.ones_like(x_cam, dtype=self.dtype, device=self.device)
        rays_cam = torch.stack((x_cam, y_cam, z), dim=-1)

        new_colors = rgb[final_mask]
        new_normals = normal[final_mask]
        with torch.no_grad():
            new_depth = scaled_depth[final_mask].unsqueeze(1)
            iR, iT = camera.get_inv_RT
            new_ray_dir = torch.mm(iR, rays_cam.t()).t()
            new_normals = torch.mm(iR, new_normals.T).t()

        new_xyz = new_ray_dir*new_depth + iT

        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        dist = torch.sqrt(dist2)
        dist_mean = dist.mean()
        dist = torch.clamp_max(dist, 3*dist_mean)

        if self.voxels.resolution == None:
            self.voxels.set_resolution(self.voxel_size)
            self.voxels.update(new_xyz)
        elif new_xyz.shape[0] > 2:
            occ_mask = self.voxels.get_valid_mask(new_xyz)
            new_xyz = new_xyz[occ_mask]
            new_depth = new_depth[occ_mask]
            new_colors = new_colors[occ_mask]
            new_normals = new_normals[occ_mask]
            dist = dist[occ_mask]
            if new_xyz.shape[0] != 0:
                self.voxels.update(new_xyz)
            else:
                return
            
        self.gaussians.add_gaussians(new_depth, new_xyz, dist, new_normals, new_colors, camera.uid)

    
    def cam_ids_from_gaussian(self, query_cam, N, state):
        if state == 'active':
            mask = self.gaussians.get_active_mask.squeeze(1)
            gaussian_ids = self.gaussians.get_ids[mask].squeeze(1)
        elif state == 'inactive':
            mask = ~self.gaussians.get_active_mask.squeeze(1)
            gaussian_ids = self.gaussians.get_ids[mask].squeeze(1)
        else:
            gaussian_ids = self.gaussians.get_ids.squeeze(1)

        render_pkg = render(query_cam, pc=self.gaussians, render_option=state, bg_color=self.bg_color)
        contributions = render_pkg["contributions"]

        unique_ids, inverse_indices = torch.unique(gaussian_ids, return_inverse=True)
        contribution_sums = torch.zeros_like(unique_ids, device=contributions.device, dtype=contributions.dtype)
        contribution_sums.scatter_add_(0, inverse_indices, contributions)

        topN = min(N, unique_ids.shape[0])
        _, top_indices = torch.topk(contribution_sums, topN)
        top_ids = unique_ids[top_indices].tolist()

        return top_ids


    def update_map(self, camera, frame, iter_num, render_option='active'):
        if frame.depth is not None:
            self.add_gaussians_from_frame(camera, frame)
        for _ in range(iter_num):
            self.mapping_iteration(camera, frame, render_option)
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

    
    def prune_list(self, id_list, prune_threshold, render_option='active'):
        with torch.no_grad():
            gaussian_ids = self.gaussians.get_ids
            if render_option == 'active':
                active_state_mask = self.gaussians.get_active_mask.squeeze(1)
            elif render_option == 'inactive':
                active_state_mask = ~self.gaussians.get_active_mask.squeeze(1)
            else:
                active_state_mask = torch.ones_like(self.gaussians.get_active_mask, dtype=bool, device=self.device).squeeze(1)
            
            prune_mask = torch.zeros_like(gaussian_ids, dtype=torch.bool, device=self.device).squeeze(1)
            
            for cam_id in id_list:
                cam = self.key_cameras[cam_id]
                render_pkg = render(cam, pc = self.gaussians, bg_color=self.bg_color, render_option=render_option)
                contributions = torch.zeros_like(gaussian_ids, dtype=self.dtype, device=self.device).squeeze(1)
                contributions[active_state_mask] = render_pkg["contributions"]
                id_mask = gaussian_ids == cam_id
                contri_mask = contributions < prune_threshold

                this_prune_mask = torch.logical_and(id_mask.squeeze(1), contri_mask)
                prune_mask[this_prune_mask] = True

            final_prune_mask = torch.logical_and(prune_mask, active_state_mask)
            self.gaussians.prune_points(final_prune_mask)
    

    def prune_outlier(self):
        inactive_mask = ~self.gaussians.get_active_mask.squeeze(1)
        inactive_means = self.gaussians.get_xyz[inactive_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inactive_means.detach().cpu().numpy())
        _, ind = pcd.remove_radius_outlier(nb_points=5, radius=3*self.voxel_size)

        outlier_mask = torch.ones(inactive_means.shape[0], dtype=torch.bool)
        outlier_mask[ind] = False

        prune_mask = torch.zeros_like(self.gaussians.get_ids, dtype=torch.bool, device=self.device).squeeze(1)
        prune_mask[inactive_mask] = outlier_mask.to(self.device)

        self.gaussians.prune_points(prune_mask)


    def densify(self):
        with torch.no_grad():

            k = random.choice(list(self.active_cam_id_list))
            camera = self.key_cameras[k]
            frame = self.key_frames[camera.uid]
            img, depth = frame.rgb, frame.depth

            render_pkg = render(camera, pc = self.gaussians, render_option='active', bg_color=self.bg_color)
            rendered_img = render_pkg["render"]
            rendered_depth = render_pkg[self.depth_type]

            ab_img = (torch.exp(camera.exposure_a)) * rendered_img + camera.exposure_b

            img_L1_error = torch.abs(ab_img - img).mean(0)
            ssim_error = fused_ssim(rendered_img.unsqueeze(0), img.unsqueeze(0))[0].mean(0)
            img_error = (1.0 - self.lambda_dssim)*img_L1_error + self.lambda_dssim*(1.0 - ssim_error)
            
            depth_error_relative = (torch.abs(rendered_depth - depth)/depth).mean(0)

            img_error_contri = render(camera, pc = self.gaussians, render_option='active', bg_color=self.bg_color,
                                      error_img=img_error)["contributions"]
            
            depth_error_contri = render(camera, pc = self.gaussians, render_option='active', bg_color=self.bg_color, 
                                      error_img=depth_error_relative)["contributions"]

            active_mask = self.gaussians.get_active_mask.squeeze(1)
            img_all_gaussian_tensor = -100*torch.ones_like(self.gaussians.get_active_mask.squeeze(1)).to(self.dtype)
            img_all_gaussian_tensor[active_mask] = img_error_contri

            depth_all_gaussian_tensor = 100*torch.ones_like(self.gaussians.get_active_mask.squeeze(1)).to(self.dtype)
            depth_all_gaussian_tensor[active_mask] = depth_error_contri

            img_error_mask = (img_all_gaussian_tensor > 1.0)
            depth_error_mask = (depth_all_gaussian_tensor < 0.15)

            clone_mask = torch.logical_and(img_error_mask, depth_error_mask)

            self.gaussians.densify(clone_mask)

    def refresh_voxel(self):
        self.voxels.clear()
        activate_mask = self.gaussians.get_active_mask.squeeze(1)
        activate_xyzs = self.gaussians.get_xyz[activate_mask]
        self.voxels.update(activate_xyzs)
    
    def observed_gaussian_mask(self, cam, option):
        render_pkg = render(cam, pc=self.gaussians, render_option=option, bg_color=self.bg_color)
        rendered_normal = render_pkg["rend_normal"]
        rendered_depth = render_pkg[self.depth_type]

        ray_vectors = cam.camera_ray_vectors()
        normal_weights = -(ray_vectors*rendered_normal).sum(0)
        normal_mask = normal_weights > 0
        depth_mask = rendered_depth < self.config["Training"]["depth_max_threshold"]

        valid_mask = torch.logical_and(normal_mask, depth_mask)

        valid_contirbutions = render(cam, pc = self.gaussians, render_option=option, bg_color=self.bg_color,
                                      error_img=valid_mask.to(self.dtype))["contributions"]

        active_mask = self.gaussians.get_active_mask.squeeze(1)

        if option == "active":
            mask = active_mask
        elif option == "inactive":
            mask = ~active_mask

        all_contributions = torch.zeros_like(self.gaussians.get_ids, dtype=self.dtype, device=self.device).squeeze(1)
        all_contributions[mask] = valid_contirbutions
        observed_mask = all_contributions > 0.1

        return observed_mask

    def update_state(self, cam, loop_cam=None):
        # if loop detected not long ago, mark all observed inactive gaussians from current camera as active
        if loop_cam is not None: 
            self.count_from_last_loop = 0
            self.last_loop_id = loop_cam.uid

        if self.count_from_last_loop < self.old_than_N_keyframe:
            observed_mask = self.observed_gaussian_mask(cam, 'inactive')
            id_mask = (self.gaussians.get_last_observe_ids > self.last_loop_id).squeeze(1)
            loop_mask = torch.logical_and(observed_mask, id_mask)

            self.gaussians.get_active_mask[loop_mask] = True
            self.gaussians.get_last_observe_ids[loop_mask] = cam.uid
            self.count_from_last_loop += 1
        update_mask = self.observed_gaussian_mask(cam, 'active')
        self.gaussians.get_last_observe_ids[update_mask] = cam.uid

        # update gaussian's nearest distance and the corrsponding cam id
        _, iT = cam.get_inv_RT
        ray_vectors = self.gaussians.get_xyz[update_mask].detach() - iT
        depth = torch.norm(ray_vectors, dim=1).unsqueeze(-1)

        min_depth = self.gaussians.min_observed_depth[update_mask]
        depth_update_mask = min_depth > depth

        min_depth[depth_update_mask] = depth[depth_update_mask]
        self.gaussians.min_observed_depth[update_mask] = min_depth
        
        ids = self.gaussians.get_ids[update_mask]
        ids[depth_update_mask] = cam.uid
        self.gaussians.get_ids[update_mask] = ids
        
        # gaussian who is not observed for a long time and distance larger than a threshold will become inactive
        inactive_mask = (self.gaussians.get_last_observe_ids < self.cam_sliding_window[0]).squeeze(1)

        self.gaussians.get_active_mask[inactive_mask] = False

        if (len(self.cam_id_list) % 10 == 0) or (loop_cam is not None):
            self.refresh_voxel()
            torch.cuda.empty_cache()

    
    def add_odom_node_to_graph(self, node_cam):
        node_pose_np = node_cam.T.inverse().cpu().numpy()
        self.pgo.add_frame_node(frame_id=node_cam.uid, init_pose=node_pose_np)

        last_key_pose = self.key_cameras[self.last_node_id].T.inverse().cpu().numpy()
        odom_transform = np.linalg.inv(last_key_pose) @ node_pose_np

        self.pgo.add_odometry_factor(cur_id=node_cam.uid, last_id=self.last_node_id, 
                                     odom_transform=odom_transform)
        
        self.last_node_id = node_cam.uid


    def push_to_frontend(self, tag=None):
        cameras = []

        if tag is None:
            tag = "sync_backend"

        gaussian_copy = clone_obj(self.gaussians)
        msg = [tag, gaussian_copy, cameras, list(self.accepted_loop_pairs)]
        self.frontend_queue.put(msg)


    def push_to_frontend_after_pgo(self):
        cameras = []
        for cam_id in self.cam_id_list: 
            cameras.append(clone_obj(self.key_cameras[cam_id]))

        gaussian_copy = clone_obj(self.gaussians)
        msg = ["pgo", gaussian_copy, cameras, list(self.accepted_loop_pairs)]
        self.frontend_queue.put(msg)


    def _spark_live_export_ply(self, spark_live_dir):
        try:
            ply_path = os.path.join(spark_live_dir, "live.ply")
            tmp = ply_path + ".partial"
            self.gaussians.save_ply(
                tmp,
                min_linear_scale_smallest_axis=1e-4,
                pad_scale_dims_to_three=True,
            )
            os.replace(tmp, ply_path)
        except Exception as e:
            Log("Spark live export (refinement) failed:", str(e), tag="Spark")

    def map_refinement(self, spark_live_dir=None):
        Log("Starting map refinement")

        iteration_total = self.map_refine_iterations
        keys_list = list(self.key_cameras.keys())
        last_export_time = 0.0
        for iteration in tqdm(
            range(1, iteration_total + 1),
            desc="Map refinement",
            unit="it",
            dynamic_ncols=True,
        ):
            random_id = random.choice(keys_list)
            cam = self.key_cameras[random_id]
            frame = self.key_frames[cam.uid]
            self.mapping_iteration(cam, frame, option='all')

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 10 == 0:
                self.prune_list([random_id], self.prune_threshold, render_option='all')

            if iteration % 50 == 0:
                scale_mask = self.gaussians.get_scaling.max(dim=1).values > self.prune_size_threshold
                opacity_mask = (self.gaussians.get_opacity < 0.1).squeeze()
                prune_mask = torch.logical_or(scale_mask, opacity_mask)
                self.gaussians.prune_points(prune_mask)
                torch.cuda.empty_cache()

            if spark_live_dir is not None:
                now = time.time()
                if now - last_export_time >= self._spark_live_interval:
                    last_export_time = now
                    self._spark_live_export_ply(spark_live_dir)

        if spark_live_dir is not None:
            self._spark_live_export_ply(spark_live_dir)

        Log("Map refinement done")


    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue

                if len(self.active_cam_id_list) == 0:
                    time.sleep(0.01)
                    continue

                self.iteration_count += 1

                # select from historical frames to refine inactive map
                inactive_train_ids = []
                if len(self.inactive_cam_id_list) > 0:
                    random_inactive_id = random.choice(self.inactive_cam_id_list)
                    query_cam = self.key_cameras[random_inactive_id]
                    select_ids = self.cam_ids_from_gaussian(query_cam, self.inactive_train_num-1, 'inactive')
                    select_ids.append(random_inactive_id)
                    inactive_train_ids = list(set(select_ids))

                for k in inactive_train_ids:
                    cam = self.key_cameras[k]
                    frame = self.key_frames[cam.uid]
                    self.mapping_iteration(cam, frame, option='inactive')

                # select frames in the active window to refine active map
                map_select_num = min(len(self.active_cam_id_list), 10)
                map_random_ids = random.sample(self.active_cam_id_list, map_select_num)    

                active_select_list = list(self.cam_sliding_window) + map_random_ids
                select_num = min(len(active_select_list), self.active_train_num)
                random_ids = random.sample(active_select_list, select_num)
                for k in random_ids:
                    cam = self.key_cameras[k]
                    frame = self.key_frames[cam.uid]
                    self.mapping_iteration(cam, frame, option='active')

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                self.prune_list(inactive_train_ids, self.prune_threshold, render_option='inactive')

                if random.random() < 0.2:
                    self.prune_list(self.cam_sliding_window, self.prune_threshold, render_option='active')
                    scale_mask = self.gaussians.get_scaling.max(dim=1).values > self.prune_size_threshold
                    opacity_mask = (self.gaussians.get_opacity < 0.1).squeeze()
                    prune_mask = torch.logical_or(scale_mask, opacity_mask)
                    self.gaussians.prune_points(prune_mask)

                if self.iteration_count % 60 == 0:
                    self.densify()
                    self.iteration_count = 0
                    continue

                if self.iteration_count % 15 == 0:
                    self.prune_outlier()
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                if data[0] == "refine":
                    spark_dir = data[1] if len(data) > 1 else None
                    self.map_refinement(spark_live_dir=spark_dir)
                    self.push_to_frontend(tag='refine')

                elif data[0] == "pause":
                    self.pause = True

                elif data[0] == "unpause":
                    self.pause = False

                elif data[0] == "init":
                    init_camera, init_frame= data[1], data[2]
                    self.key_frames[init_frame.camera_id] = init_frame
                    self.key_cameras[init_camera.uid] = init_camera
                    self.cam_sliding_window.append(init_camera.uid)
                    self.cam_id_list.append(init_camera.uid)

                    Log("init the system")
                    self.reset()
                    self.viewpoints[init_camera.uid] = init_camera
                    self.update_map(init_camera, init_frame, self.init_iter_num)
                    init_pose_np = init_camera.T.inverse().cpu().numpy()
                    self.pgo.add_frame_node(frame_id=init_camera.uid, init_pose=init_pose_np)
                    self.pgo.add_pose_prior(frame_id=init_camera.uid, prior_pose=init_pose_np, fixed=True)

                    self.last_loop_id = init_camera.uid
                    self.last_node_id = init_camera.uid
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    key_camera, key_frame, window = data[1], data[2], data[3]
                    self.key_frames[key_frame.camera_id] = key_frame
                    self.key_cameras[key_camera.uid] = key_camera
                    
                    self.cam_id_list.append(key_camera.uid)
                    self.cam_sliding_window.append(key_camera.uid)

                    self.add_odom_node_to_graph(key_camera)

                    self.update_state(key_camera)
                    self.update_map(key_camera, key_frame, self.key_frame_iter_num)

                    active_mask = self.gaussians.get_active_mask.squeeze(1)
                    active_uids = self.gaussians.get_ids[active_mask]
                    active_ids = torch.unique(active_uids, sorted=False).cpu().tolist()

                    active_set = set(active_ids)
                    self.active_cam_id_list = list(active_set)
                    self.inactive_cam_id_list = [cam for cam in self.cam_id_list if cam not in active_set]

                    self.push_to_frontend(tag='keyframe')

                elif data[0] == "pgo":
                    cur_camera, cur_frame, loop_camera = data[1], data[2], data[3]

                    # put current frame into keyframe dict
                    self.key_frames[cur_frame.camera_id] = cur_frame
                    self.key_cameras[cur_camera.uid] = cur_camera

                    self.cam_id_list.append(cur_camera.uid)
                    self.cam_sliding_window.append(cur_camera.uid)

                    self.add_odom_node_to_graph(cur_camera)

                    cur_pose = cur_camera.T.inverse().cpu().numpy()
                    loop_pose = loop_camera.T.inverse().cpu().numpy()
                                   
                    loop_transform = np.linalg.inv(loop_pose) @ cur_pose
                    loop_success = self.pgo.add_loop_factor(cur_id = cur_camera.uid,
                                             loop_id = loop_camera.uid,
                                             loop_transform= loop_transform)
                    
                    if loop_success is True:
                        if self.pgo.optimize_pose_graph() is True:
                            pair = (
                                int(cur_camera.uid),
                                int(loop_camera.uid),
                            )
                            if pair not in self.accepted_loop_pairs:
                                self.accepted_loop_pairs.append(pair)
                            N = self.gaussians.get_xyz.shape[0]
                            N_pose_updates = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(N, 1, 1)
                            for kf_id in self.cam_id_list:
                                optimized_pose = torch.from_numpy(self.pgo.get_optimized_node_pose(kf_id)).to(self.device).to(self.dtype).inverse()
                                pose_update = optimized_pose.inverse() @ self.key_cameras[kf_id].T
                                mask = (self.gaussians.get_ids == kf_id).squeeze(1)
                                N_pose_updates[mask] = pose_update.unsqueeze(0).repeat(mask.sum(), 1, 1)
                                self.key_cameras[kf_id].T = optimized_pose
                            self.gaussians.update_after_pgo(N_pose_updates)
                        
                        self.update_state(cur_camera, loop_cam=loop_camera)

                    self.push_to_frontend_after_pgo()
                
                else:
                    raise Exception("Unprocessed data", data)
                
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        if ipc is not None and torch.cuda.is_available():
            ipc()
        return
