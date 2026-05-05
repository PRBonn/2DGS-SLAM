#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    normal2rotation,
    get_expon_lr_func,
    inverse_sigmoid,
)

from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.general_utils import rotmat2quaternion


class GaussianModel:
    def __init__(self, sh_degree=0, initial_opacity=0.99):
        self.device = 'cuda'
        self.dtype = torch.float32
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.init_opacity = initial_opacity

        self._xyz = torch.empty(0, device=self.device)
        self._features_dc = torch.empty(0, device=self.device)
        self._features_rest = torch.empty(0, device=self.device)
        self._scaling = torch.empty(0, device=self.device)
        self._rotation = torch.empty(0, device=self.device)
        self._opacity = torch.empty(0, device=self.device)

        self.min_observed_depth = torch.empty(0, device=self.device)

        self.n_obs = torch.empty(0).int()

        self.unique_kfIDs = torch.empty(0, device=self.device).int()
        self.last_observe_ids = torch.empty(0, device=self.device).int()
        self.active_mask = torch.empty(0, device=self.device).bool()

        self.optimizer = None
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.spatial_lr_scale = 6.0

        self.ply_input = None

        self.isotropic = False

    @property
    def get_ids(self):
        return self.unique_kfIDs
    
    @property
    def get_last_observe_ids(self):
        return self.last_observe_ids
    
    @property
    def get_active_mask(self):
        return self.active_mask

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def add_gaussians(self, new_depth, xyz, dist, normals, colors, cam_id):

        fused_color = RGB2SH(colors)
        color_sh = torch.zeros([fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2], dtype=self.dtype, device=self.device)
        color_sh[:, :3, 0] = fused_color
        color_sh[:, 3:, 1:] = 0.0

        new_features_dc = color_sh[:, :, 0:1].transpose(1, 2)
        new_features_rest = color_sh[:, :, 1:].transpose(1, 2)

        new_xyz = nn.Parameter(xyz.requires_grad_(True))
        new_features_dc = nn.Parameter(new_features_dc.contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(new_features_rest.contiguous().requires_grad_(True))

        new_scales = (torch.log(dist)[..., None]).repeat(1, 2)
        new_scales = nn.Parameter(new_scales.requires_grad_(True))
        new_rotation = nn.Parameter(normal2rotation(normals).requires_grad_(True))

        ones_vector = torch.ones((new_xyz.shape[0],1), dtype=self.dtype, device=self.device)
        new_opacity = nn.Parameter((ones_vector*self.init_opacity).requires_grad_(True))
        new_kf_id = ones_vector*cam_id
        new_lo_id = new_kf_id
        new_active_mask = ones_vector.to(torch.bool)

        self.densification_postfix(new_xyz,
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_opacity,
                                   new_depth,
                                   new_scales, 
                                   new_rotation, 
                                   new_kf_id,
                                   new_lo_id,
                                   new_active_mask)
        
    
    def update_after_pgo(self, pose_updates):
        R_updates = pose_updates[:, :3, :3]
        t_updates = pose_updates[:, :3, 3]

        new_xyz = torch.bmm(R_updates, self._xyz.unsqueeze(-1)).squeeze(-1) + t_updates
        q_updates = rotmat2quaternion(R_updates)
        
        q_cur = self.get_rotation
        w1, x1, y1, z1 = q_updates.unbind(dim=-1)
        w2, x2, y2, z2 = q_cur.unbind(dim=-1)

        w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        q_new = torch.stack([w_new, x_new, y_new, z_new], dim=-1)
        new_rotation = torch.nn.functional.normalize(q_new, dim=-1)

        d = {"xyz": new_xyz,
            "rotation" : new_rotation}
        
        optimizable_tensors = self.replace_tensors_in_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._rotation = optimizable_tensors["rotation"]

    
    def replace_tensors_in_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if tensors_dict.get(group["name"]) is None:
                continue
            replace_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] =  torch.zeros_like(replace_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(replace_tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(replace_tensor).requires_grad_(True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(replace_tensor).requires_grad_(True)
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    

    def densification_postfix(self, 
                              new_xyz, 
                              new_features_dc, 
                              new_features_rest, 
                              new_opacities,
                              new_depth,
                              new_scaling, 
                              new_rotation,
                              new_kfids,
                              new_loids,
                              new_active_mask):
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.min_observed_depth = torch.cat([self.min_observed_depth, new_depth], dim=0)
        self.unique_kfIDs = torch.cat([self.unique_kfIDs, new_kfids], dim=0).int()
        self.last_observe_ids = torch.cat([self.last_observe_ids, new_loids], dim=0).int()
        self.active_mask = torch.cat([self.active_mask, new_active_mask], dim=0).bool()
    

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask]
        self.min_observed_depth = self.min_observed_depth[valid_points_mask]
        self.last_observe_ids = self.last_observe_ids[valid_points_mask]
        self.active_mask = self.active_mask[valid_points_mask]

    def densify_and_clone(self, mask):
        new_xyz = self._xyz[mask]
        new_features_dc = self._features_dc[mask]
        new_features_rest = self._features_rest[mask]
        new_opacities = self._opacity[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        new_kf_id = self.unique_kfIDs[mask]
        new_min_observed_depth = self.min_observed_depth[mask]
        new_lo_id = self.last_observe_ids[mask]
        new_active_mask = self.active_mask[mask]

        self.densification_postfix(new_xyz, 
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_opacities,
                                   new_min_observed_depth, 
                                   new_scaling,
                                   new_rotation,
                                   new_kf_id,
                                   new_lo_id,
                                   new_active_mask)
        
    
    def densify_and_split(self, selected_pts_mask, N=2):
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        old_kf_id = self.unique_kfIDs[selected_pts_mask]
        new_kf_id = old_kf_id.repeat(N,1)

        old_lo_id = self.last_observe_ids[selected_pts_mask]
        new_lo_id = old_lo_id.repeat(N,1)

        old_min_observed_depth = self.min_observed_depth[selected_pts_mask]
        new_min_observed_depth = old_min_observed_depth.repeat(N,1)

        old_active_mask = self.active_mask[selected_pts_mask]
        new_active_mask = old_active_mask.repeat(N,1)

        self.densification_postfix(new_xyz, 
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_opacity,
                                   new_min_observed_depth, 
                                   new_scaling,
                                   new_rotation, 
                                   new_kf_id,
                                   new_lo_id,
                                   new_active_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    
    def densify(self, densify_mask):

        clone_mask = torch.logical_and(densify_mask,
                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*0.25)
        self.densify_and_clone(clone_mask)

        densify_mask = torch.cat((densify_mask, 
                                 torch.zeros(self.get_scaling.shape[0]-densify_mask.shape[0], device="cuda", dtype=bool)))
        
        split_mask = torch.logical_and(densify_mask,
                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*0.25)
        self.densify_and_split(split_mask)

        torch.cuda.empty_cache()

    def set_opacity(self, value):
        opacities = self._opacity.detach()
        opacities[opacities < value] = value
        self._opacity = nn.Parameter(opacities.requires_grad_(True))


    def save_ply(self, path, min_linear_scale_smallest_axis=None, pad_scale_dims_to_three=False):
        """
        min_linear_scale_smallest_axis: if set (e.g. 1e-4), each splat's smallest scale
        axis (in linear space after exp) is raised to at least this value before writing
        log-scales to PLY. Useful for 2DGS (near-zero thickness) in external viewers.

        pad_scale_dims_to_three: 2DGS stores only scale_0/scale_1; Spark etc. expect
        scale_2. When True, append ln(thickness) columns so shape is (N,3). Thickness
        uses min_linear_scale_smallest_axis if > 0, else 1e-4.
        """
        with torch.no_grad():
            xyz = self.get_xyz.cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.flatten(start_dim=1).cpu().numpy()
            f_rest = self._features_rest.flatten(start_dim=1).cpu().numpy()
            opacities = self._opacity.cpu().numpy()
            if min_linear_scale_smallest_axis is not None and float(min_linear_scale_smallest_axis) > 0:
                eps = float(min_linear_scale_smallest_axis)
                s = self.get_scaling.clone()
                min_v, min_j = s.min(dim=1)
                mask = min_v < eps
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    j = min_j[mask]
                    s[idx, j] = torch.maximum(
                        min_v[mask],
                        torch.tensor(eps, device=s.device, dtype=s.dtype),
                    )
                scale = self.scaling_inverse_activation(s).cpu().numpy()
            else:
                scale = self._scaling.cpu().numpy()

            if pad_scale_dims_to_three and scale.shape[1] < 3:
                t = (
                    float(min_linear_scale_smallest_axis)
                    if (
                        min_linear_scale_smallest_axis is not None
                        and float(min_linear_scale_smallest_axis) > 0
                    )
                    else 1e-4
                )
                log_t = np.log(np.float64(t)).astype(scale.dtype)
                pad_cols = 3 - int(scale.shape[1])
                scale = np.concatenate(
                    [
                        scale,
                        np.full((scale.shape[0], pad_cols), log_t, dtype=scale.dtype),
                    ],
                    axis=1,
                )

            rotation = self._rotation.cpu().numpy()
            active_state = self.active_mask.cpu().numpy()

        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("active_state")
        
        dtype_full = [(attribute, "f4") for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, active_state), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        active_state = np.asarray(plydata.elements[0]["active_state"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_mask = torch.tensor(active_state, dtype=torch.bool, device="cuda")
        self.active_sh_degree = self.max_sh_degree

