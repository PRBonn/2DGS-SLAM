import sys
import time
import os
import random
from tqdm import tqdm

import yaml
from munch import munchify
import numpy as np
import torch
import torch.multiprocessing as mp
import open3d as o3d

from einops import rearrange

from argparse import ArgumentParser
from utils.config_utils import load_config
from utils.logging_utils import Log
from utils.dataset import load_dataset

sys.path.append('gaussian_splatting')
from utils.camera_utils import Camera
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel

import trimesh


def clean_mesh(mesh, verbose=False):
    mesh_tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
    )
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 200
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {
            old_idx: vertex_count + new_idx for new_idx, old_idx in enumerate(component)
        }
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[
            np.any(np.isin(mesh_tri.faces, component), axis=1)
        ]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.nondegenerate_faces())
    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.unique_faces())
    if verbose:
        print(
            f"Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}"
        )

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces),
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64)
    )

    return cleaned_mesh


def tsdf_fusion(cameras, gaussians, depth_type, kf_indices, file_path, verbose_mesh_clean=False):
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=5.0 * scale / 512.0,
    sdf_trunc=0.04 * scale,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    compensate_vector = (-0.0 * scale / 512.0, 2.5 * scale / 512.0, -2.5 * scale / 512.0)
    background = torch.tensor([1.0,1.0,1.0], device="cuda", dtype=torch.float32)

    for idx in kf_indices:
        key_cam = cameras[idx]

        render_pkg = render(key_cam, pc=gaussians, render_option='all', bg_color=background)
        rendered_image = render_pkg["render"]
        rendered_depth = render_pkg[depth_type]

        rendered_image = rearrange(rendered_image, 'c h w -> h w c')
        rendered_depth = rearrange(rendered_depth, 'c h w -> h w c')

        depth = rendered_depth.detach().cpu().numpy()

        color = rendered_image.detach().cpu().numpy()

        w2c = key_cam.T.cpu().numpy()

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = np.clip(color, 0.0, 1.0)
        color = (color * 255.0).astype(np.uint8)
        color = np.ascontiguousarray(color)
        color = o3d.geometry.Image(color)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(key_cam.image_width, key_cam.image_height, 
                                                      key_cam.fx, key_cam.fy, 
                                                      key_cam.cx, key_cam.cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,
                                                                    depth,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=30,
                                                                    convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

    o3d_mesh = volume.extract_triangle_mesh()
    o3d_mesh = clean_mesh(o3d_mesh, verbose=verbose_mesh_clean)
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    o3d.io.write_triangle_mesh(file_path, o3d_mesh)



