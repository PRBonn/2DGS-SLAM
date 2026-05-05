# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dummy optimizer for visualizing pairs
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import cv2

from .base_opt import BasePCOptimizer
from ..utils.geometry import inv, geotrf
from .commons import edge_str
from ..post_process import estimate_focal_knowing_depth


class PairAligner (BasePCOptimizer):
    """
    Modifed from 'PairViewer'
    Caculating depth images and corresponding poses given intrisics K.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_im_poses = True
        self.verbose = False

    
    def align_points_bidirection(self):
        assert self.is_symmetrized and self.n_edges == 2

        self.Ks = []
        rel_poses = []
        confs = []
        for i in range(self.n_imgs):
            conf = float(self.conf_i[edge_str(i, 1-i)].mean() * self.conf_j[edge_str(i, 1-i)].mean())
            if self.verbose:
                print(f'  - {conf=:.3} for edge {i}-{1-i}')
            confs.append(conf)

            H, W = self.imshapes[i]
            pts3d = self.pred_i[edge_str(i, 1-i)]
            pp = torch.tensor((W/2, H/2))
            focal = float(estimate_focal_knowing_depth(pts3d[None], pp.to('cuda'), focal_mode='weiszfeld'))
            K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])
            self.Ks.append(K)
           
            # estimate the pose of pts1 in image 2
            pixels = np.mgrid[:W, :H].T.astype(np.float32)
            pts3d = self.pred_j[edge_str(1-i, i)].cpu().numpy()
            assert pts3d.shape[:2] == (H, W)
            msk = self.get_masks()[i].cpu().numpy()

            try:
                res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                         iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
                success, R, T, inliers = res
                assert success

                R = cv2.Rodrigues(R)[0] 
                pose = np.r_[np.c_[R, T], [(0, 0, 0, 1)]]  
            except:
                pose = np.eye(4)
            rel_poses.append(torch.from_numpy(pose.astype(np.float32)).to('cuda'))

        # let's use the pair with the most confidence
        if confs[0] > confs[1]:
            # ptcloud is expressed in camera1
            self.im_poses = [torch.eye(4).to('cuda'), rel_poses[1]]  # I, cam2-to-cam1
            self.depth = [self.pred_i['0_1'][..., 2], geotrf(rel_poses[1], self.pred_j['0_1'])[..., 2]]
        else:
            # ptcloud is expressed in camera2
            self.im_poses = [rel_poses[0], torch.eye(4).to('cuda')]  # I, cam1-to-cam2
            self.depth = [geotrf(rel_poses[0], self.pred_j['1_0'])[..., 2], self.pred_i['1_0'][..., 2]]
        

    def align_points(self, K_cam2):
        H, W = self.imshapes[0]
        pixels = np.mgrid[:W, :H].T.astype(np.float32)

        pts3d = self.pred_i[edge_str(0, 1)]
        pp = torch.tensor((W/2, H/2))
        focal = float(estimate_focal_knowing_depth(pts3d[None], pp.to('cuda'), focal_mode='weiszfeld'))
        K_cam1 = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])
        
        pts3d = self.pred_j[edge_str(0, 1)].cpu().numpy()
        assert pts3d.shape[:2] == (H, W)
        msk = self.get_masks()[0].cpu().numpy()

        try:
            res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K_cam2, None,
                                        iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
            success, R, T, inliers = res
            assert success

            R = cv2.Rodrigues(R)[0]  # world to cam
            pose = np.r_[np.c_[R, T], [(0, 0, 0, 1)]]  # cam to world
        except:
            pose = np.eye(4)

        pose = torch.from_numpy(pose.astype(np.float32)).to('cuda')

        # ptcloud is expressed in camera1
        self.im_poses = [torch.eye(4).to('cuda'), pose]  # I, cam2-to-cam1
        self.depth = [self.pred_i['0_1'][..., 2], geotrf(pose, self.pred_j['0_1'])[..., 2]]
        self.Ks = [K_cam1, K_cam2]


    def get_intrinsics(self):
        return self.Ks

    def get_origin_points(self):
        return [self.pred_i[edge_str(0, 1)], self.pred_j[edge_str(0, 1)]]

    def get_depthmaps(self, raw=False):
        depth = [d.to(self.device) for d in self.depth]
        return depth

    def get_im_poses(self):
        im_poses = torch.stack(self.im_poses, dim=0)
        return im_poses
    
    def get_confidences(self):
        return self.im_conf

    def forward(self):
        return float('nan')
