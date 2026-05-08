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

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.scene.gaussian_model import GaussianModel


def render(viewpoint_camera,
           pc : GaussianModel, 
           bg_color : torch.Tensor,
           render_option = 'all',
           scaling_modifier = 1.0,
           error_img = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    If 'render_option' is 'all':
        use all gaussians to render the image
    elif 'render_option' is 'active':
        only use active gaussians to render the image
    elif 'render_option' is 'inactive':
        only use inactive gaussians to render the image

    If 'the error_img' is not None :
        render_pkg[contributions] = every Gaussian's contribution to the error map
    or :
        render_pkg[contributions] = every Gaussian's contribution to the rendering result.
        
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if error_img is None:
        error_img = torch.ones((int(viewpoint_camera.image_height),
                                int(viewpoint_camera.image_width)), dtype=torch.float32, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        error_img =error_img,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if render_option == 'all':
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
    elif render_option == 'active':
        mask = pc.get_active_mask.squeeze(1)
        means3D = pc.get_xyz[mask]
        means2D = screenspace_points[mask]
        opacity = pc.get_opacity[mask]
        scales = pc.get_scaling[mask]
        rotations = pc.get_rotation[mask]
        shs = pc.get_features[mask]
    elif render_option == 'inactive':
        nonmask = ~pc.get_active_mask.squeeze(1)
        means3D = pc.get_xyz[nonmask]
        means2D = screenspace_points[nonmask]
        opacity = pc.get_opacity[nonmask]
        scales = pc.get_scaling[nonmask]
        rotations = pc.get_rotation[nonmask]
        shs = pc.get_features[nonmask]
    elif render_option == 'active_highlight':
        mask = pc.get_active_mask.squeeze(1)
        inactive = ~mask
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity.clone()
        opacity[inactive] = opacity[inactive] * 0.5
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features.clone()
        lum_weights = torch.tensor([0.299, 0.587, 0.114], device=shs.device, dtype=shs.dtype)
        dc = shs[inactive, 0:1, :]  # (M, 1, 3)
        lum = (dc * lum_weights).sum(dim=-1, keepdim=True).expand_as(dc)
        blend = 0.15
        shs[inactive, 0:1, :] = (blend * dc + (1.0 - blend) * lum) * 0.5
        if shs.shape[1] > 1:
            shs[inactive, 1:, :] = shs[inactive, 1:, :] * blend

    colors_precomp = None
    cov3D_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap, contributions = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        theta=viewpoint_camera.cam_rot_delta,
        rho=viewpoint_camera.cam_trans_delta,
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
             "viewspace_points": means2D,
             "visibility_filter" : radii > 0,
             "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'rend_depth_median': render_depth_median,
            'rend_depth_expected': render_depth_expected,
            'contributions': contributions,
    })

    return rets


def render_for_tracking(viewpoint_camera,
                        g_xyz,
                        g_opacity,
                        g_scaling,
                        g_rotation,
                        g_feature,
                        active_sh_degree,
                        bg_color,
                        scaling_modifier = 1.0,
                        error_img = None):
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(g_xyz, dtype=g_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if error_img is None:
        error_img = torch.ones((int(viewpoint_camera.image_height),
                                int(viewpoint_camera.image_width)), dtype=torch.float32, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        error_img=error_img,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = g_xyz
    means2D = screenspace_points
    opacity = g_opacity
    scales = g_scaling
    rotations = g_rotation
    shs = g_feature

    colors_precomp = None
    cov3D_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap, contributions = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        theta=viewpoint_camera.cam_rot_delta,
        rho=viewpoint_camera.cam_trans_delta,
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
             "viewspace_points": means2D,
             "visibility_filter" : radii > 0,
             "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'rend_depth_median': render_depth_median,
            'rend_depth_expected': render_depth_expected,
            'contributions': contributions,
    })

    return rets

