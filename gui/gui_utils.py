import queue

import cv2
import numpy as np
import open3d as o3d
import torch

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.05):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


class GaussianPacket:
    def __init__(
        self,
        gaussians=None,
        keyframe=None,
        current_frame=None,
        reference_cam_id=None,
        init_frame=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        keyframes=None,
        gtframes=None,
        finish=False,
        kf_window=None,
        est_traj=None,
        gt_traj=None,
        traj_frame_ids=None,
        loop_edges=None,
        gpu_mem_usage_gb=None,
        num_keyframes=None,
        num_loop_closures=None,
        num_gaussians_total=None,
        num_gaussians_active=None,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            self.get_xyz = gaussians.get_xyz.detach().clone()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone()
            self.get_scaling = gaussians.get_scaling.detach().clone()
            self.get_rotation = gaussians.get_rotation.detach().clone()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone()
            self.get_active_mask = gaussians.get_active_mask.detach().clone()

            self._rotation = gaussians._rotation.detach().clone()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            self.n_obs = gaussians.n_obs.clone()

        self.keyframe = keyframe
        self.current_frame = current_frame
        self.reference_cam_id = reference_cam_id
        self.init_frame = init_frame
        self.gtcolor = self.resize_img(gtcolor, 320)
        self.gtdepth = self.resize_img(gtdepth, 320)
        self.gtnormal = self.resize_img(gtnormal, 320)
        self.keyframes = keyframes
        self.gtframes = gtframes
        self.finish = finish
        self.kf_window = kf_window
        self.est_traj = est_traj
        self.gt_traj = gt_traj
        self.traj_frame_ids = traj_frame_ids
        self.loop_edges = (
            None
            if loop_edges is None
            else np.asarray(loop_edges, dtype=np.int32).copy()
        )
        self.gpu_mem_usage_gb = gpu_mem_usage_gb
        self.num_keyframes = num_keyframes
        self.num_loop_closures = num_loop_closures
        self.num_gaussians_total = num_gaussians_total
        self.num_gaussians_active = num_gaussians_active

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


class ParamsGUI:
    def __init__(
        self,
        pipe=None,
        background=None,
        gaussians=None,
        q_main2vis=None,
        q_vis2main=None,
        depth_vis_min=0.1,
        depth_vis_max=4.0,
        mesh_path=None,
    ):
        self.pipe = pipe
        self.background = background
        self.gaussians = gaussians
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.depth_vis_min = depth_vis_min
        self.depth_vis_max = depth_vis_max
        self.mesh_path = mesh_path
