import os
import pathlib
import threading
import time
from datetime import datetime
from functools import partial

import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from utils.torch_cpp_log import ensure_before_torch_import

ensure_before_torch_import()
import torch
import torch.nn.functional as F
from OpenGL import GL as gl

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gui.gl_render import util, util_gau
from gui.gl_render.render_ogl import OpenGLRenderer
from gui.gui_utils import (
    GaussianPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
)

# Open3D camera model-matrix uses GL-style axes; matches PINGS / gui_utils.cv_gl for setup_camera().
FromGLCamera = np.linalg.inv(cv_gl)


def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(np.asarray(model_matrix, dtype=np.float64) @ FromGLCamera)


def create_camera_intrinsic_from_size(width, height, hfov=60.0, vfov=60.0):
    fy = (height / 2.0) / np.tan(np.radians(vfov) / 2)
    fx = (width / 2.0) / np.tan(np.radians(hfov) / 2)
    fx = fy  # Same convention as PINGS (limit fx by fy for stability)
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]])


def _joystick_axes_as_list(joy_id=0):
    """Normalize glfw.get_joystick_axes across bindings (ctypes pair vs ndarray)."""
    axes_result = glfw.get_joystick_axes(joy_id)
    if axes_result is None:
        return None
    if isinstance(axes_result, tuple) and len(axes_result) == 2:
        axes_ptr, axes_count = axes_result
        if axes_count < 6 or axes_ptr is None:
            return None
        return [float(axes_ptr[i]) for i in range(int(axes_count))]
    arr = np.asarray(axes_result, dtype=np.float64).ravel()
    if arr.size < 6:
        return None
    return arr.tolist()
from gui.colorize_depth import colorize_depth_maps
from utils.camera_utils import Camera
from utils.logging_utils import Log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def _estimate_gaussian_packet_memory_mb(gp):
    if gp is None or not getattr(gp, "has_gaussians", False):
        return None
    nbytes = 0
    for name in (
        "get_xyz",
        "get_scaling",
        "get_rotation",
        "get_opacity",
        "get_features",
        "get_active_mask",
        "_rotation",
        "unique_kfIDs",
        "n_obs",
    ):
        t = getattr(gp, name, None)
        if t is not None and hasattr(t, "numel"):
            nbytes += t.numel() * t.element_size()
    return nbytes / (1024**2)


LOOP_GREEN = np.array([0, 128, 0], dtype=np.float64) / 255.0


def normal2rgb(normal):
    norm = torch.norm(normal, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    normal_norm = normal / norm
    normal_rgb = (255 - (normal_norm + 1) * 0.5 * 255).to(torch.uint8)
    
    return normal_rgb

class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.joystick_connected = False
        self.joystick_enabled = False

        self.frustum_dict = {}
        self.model_dict = {}

        self.frustum_size = 0.03
        self.trajectory_line_alpha = 0.75

        self.init_widget()

        self.q_main2vis = None
        self.gaussian_cur = None
        self.background = None

        self.init = False
        self.reference_cam_id = None
        self.render_img = None

        self.depth_vis_min = 0.1
        self.depth_vis_max = 4.0

        self.mesh_name = "colorized_mesh"
        self.mesh_o3d = None
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"
        self.mesh_material.base_roughness = 1.0
        self.mesh_material.base_metallic = 0.0
        self.mesh_material.base_reflectance = 0.5

        if params_gui is not None:
            self.background = params_gui.background
            self.gaussian_cur = params_gui.gaussians
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.depth_vis_min = getattr(params_gui, "depth_vis_min", 0.1)
            self.depth_vis_max = getattr(params_gui, "depth_vis_max", 4.0)
            self._load_mesh(getattr(params_gui, "mesh_path", None))

        self.gaussian_nums = []
        self._raster_calls = 0
        self._last_render_fps = None

        self.est_traj_name = "est_trajectory"
        self.gt_traj_name = "gt_trajectory"
        self.est_traj_lineset = o3d.geometry.LineSet()
        self.gt_traj_lineset = o3d.geometry.LineSet()

        self.loop_edges_name = "loop_edges"
        self.loop_edges_lineset = o3d.geometry.LineSet()
        self._last_est_traj_for_loops = None
        self._last_loop_edges_ij = None

        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self._init_joystick()

        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    def init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "2DGS-SLAM Viewer", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = self._unlit_line_material(max(4.0, 6 * self.window.scaling))

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.button = gui.ToggleSwitch("Resume/Pause")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)

        self.fly_chbox = gui.Checkbox("Fly")
        self.fly_chbox.checked = False
        self.fly_chbox.set_on_checked(self._set_fly_mouse_mode)
        chbox_tile.add_child(self.fly_chbox)

        vp_subtile1.add_child(chbox_tile)

        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(gui.Label("Viewpoint list"))
        combo_tile.add_child(self.combo_kf)
        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.keyframes_chbox = gui.Checkbox("Keyframes")
        self.keyframes_chbox.checked = False
        self.keyframes_chbox.set_on_checked(self._on_keyframes_chbox)
        chbox_tile_3dobj.add_child(self.keyframes_chbox)

        self.mesh_chbox = gui.Checkbox("Mesh")
        self.mesh_chbox.checked = False
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)
        self.panel.add_child(chbox_tile_3dobj)

        chbox_tile_traj = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.est_traj_chbox = gui.Checkbox("Estimated trajectory")
        self.est_traj_chbox.checked = True
        self.est_traj_chbox.set_on_checked(self._on_est_traj_chbox)
        chbox_tile_traj.add_child(self.est_traj_chbox)
        self.gt_traj_chbox = gui.Checkbox("GT trajectory")
        self.gt_traj_chbox.checked = False
        self.gt_traj_chbox.set_on_checked(self._on_gt_traj_chbox)
        chbox_tile_traj.add_child(self.gt_traj_chbox)
        self.loop_edges_chbox = gui.Checkbox("Loop edges")
        self.loop_edges_chbox.checked = True
        self.loop_edges_chbox.set_on_checked(self._on_loop_edges_chbox)
        chbox_tile_traj.add_child(self.loop_edges_chbox)
        self.panel.add_child(chbox_tile_traj)

        self.panel.add_child(gui.Label("Rendering options"))
        chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        chbox_tile_geometry.add_child(self.depth_chbox)

        self.normal_chbox = gui.Checkbox("Normal")
        self.normal_chbox.checked = False
        chbox_tile_geometry.add_child(self.normal_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        chbox_tile_geometry.add_child(self.opacity_chbox)

        self.time_shader_chbox = gui.Checkbox("Time")
        self.time_shader_chbox.checked = False
        chbox_tile_geometry.add_child(self.time_shader_chbox)

        self.active_chbox = gui.Checkbox("Active")
        self.active_chbox.checked = False
        chbox_tile_geometry.add_child(self.active_chbox)

        self._render_mode_boxes = (
            self.depth_chbox,
            self.normal_chbox,
            self.opacity_chbox,
            self.time_shader_chbox,
        )
        for cb in self._render_mode_boxes:
            cb.set_on_checked(partial(self._on_render_mode_exclusive, cb))

        self.panel.add_child(chbox_tile_geometry)

        slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Scale (0-1)")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.scaling_slider)
        self.panel.add_child(slider_tile)

        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)
        self.output_frame_id = gui.Label("Current Frame: ")
        self.keyframes_info = gui.Label("# Keyframes: —")
        self.loop_closures_info = gui.Label("# Loop Closures: —")
        self.gaussians_info = gui.Label("# Gaussians: —")
        self.freq_info = gui.Label("Render FPS: —")
        self.map_mem_info = gui.Label("Map Memory: —")
        self.gpu_mem_info = gui.Label("GPU Memory: —")
        tab_info.add_child(self.output_frame_id)
        tab_info.add_child(self.keyframes_info)
        tab_info.add_child(self.loop_closures_info)
        tab_info.add_child(self.gaussians_info)
        tab_info.add_child(self.freq_info)
        tab_info.add_child(self.map_mem_info)
        tab_info.add_child(self.gpu_mem_info)

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Observed Color/Depth Image"))
        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def _unlit_line_material(self, line_width):
        """Same alpha as trajectory / loop-edge linesets; RGB from geometry vertex colors."""
        m = rendering.MaterialRecord()
        m.shader = "unlitLine"
        m.line_width = line_width
        a = float(self.trajectory_line_alpha)
        m.base_color = [1.0, 1.0, 1.0, a]
        m.has_alpha = True
        return m

    def _traj_line_material(self):
        return self._unlit_line_material(max(3.0, 4 * self.window.scaling))

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    def _init_joystick(self):
        try:
            self.joystick_connected = bool(glfw.joystick_present(0))
        except Exception:
            self.joystick_connected = False
            return
        if self.joystick_connected:
            try:
                name = glfw.get_joystick_name(0)
                if isinstance(name, bytes):
                    name = name.decode()
                Log("Joystick connected at 0: {}".format(name), tag="GUI")
            except Exception as e:
                Log("Could not read joystick name: {}".format(e), tag="GUI")
                self.joystick_connected = False

    def _set_fly_mouse_mode(self, is_on):
        fly_ctrl = getattr(gui.SceneWidget.Controls, "FLY", None)
        if is_on:
            if fly_ctrl is None:
                Log("Fly mode requires Open3D with SceneWidget.Controls.FLY", tag="GUI")
                self.fly_chbox.checked = False
                return
            self.widget3d.set_view_controls(fly_ctrl)
            if self.joystick_connected:
                self.joystick_enabled = True
                Log("Joystick control enabled for Fly mode", tag="GUI")
            else:
                self.joystick_enabled = False
                Log("Fly mode on (no joystick); use Open3D FLY keys", tag="GUI")
            Log(
                "Fly: W/A/S/D move, Q/Z up/down, E/R yaw adjust",
                tag="GUI",
            )
        else:
            self.widget3d.set_view_controls(
                gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE
            )
            self.joystick_enabled = False

    def _update_joystick_fly_control(self):
        if not (self.joystick_connected and self.joystick_enabled):
            return
        axes = _joystick_axes_as_list(0)
        if axes is None:
            return

        move_speed = 0.3
        rotation_speed = 0.03
        dead_zone_move = 0.05
        dead_zone_rotation = 0.25

        left_x = axes[0]
        left_y = -axes[1]
        right_x = -axes[3]
        right_y = -axes[4]

        if abs(left_x) < dead_zone_move:
            left_x = 0.0
        if abs(left_y) < dead_zone_move:
            left_y = 0.0
        if abs(right_x) < dead_zone_rotation:
            right_x = 0.0
        if abs(right_y) < dead_zone_rotation:
            right_y = 0.0

        forward = left_y * move_speed
        right = left_x * move_speed
        up = 0.0
        yaw = right_x * rotation_speed
        pitch = right_y * rotation_speed

        if (
            abs(forward) > 0
            or abs(right) > 0
            or abs(up) > 0
            or abs(yaw) > 0
            or abs(pitch) > 0
        ):
            try:
                model_matrix = np.array(
                    self.widget3d.scene.camera.get_model_matrix(), dtype=np.float64
                )
                position = model_matrix[:3, 3].copy()
                rotation_matrix = model_matrix[:3, :3].copy()

                if abs(forward) > 0 or abs(right) > 0 or abs(up) > 0:
                    move_vector = np.array([right, up, -forward], dtype=np.float64)
                    world_move = rotation_matrix @ move_vector
                    position += world_move

                if abs(yaw) > 0:
                    yaw_matrix = np.array(
                        [
                            [np.cos(yaw), 0, np.sin(yaw)],
                            [0, 1, 0],
                            [-np.sin(yaw), 0, np.cos(yaw)],
                        ],
                        dtype=np.float64,
                    )
                    rotation_matrix = rotation_matrix @ yaw_matrix

                if abs(pitch) > 0:
                    pitch_matrix = np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)],
                        ],
                        dtype=np.float64,
                    )
                    rotation_matrix = rotation_matrix @ pitch_matrix

                new_model_matrix = np.eye(4, dtype=np.float64)
                new_model_matrix[:3, :3] = rotation_matrix
                new_model_matrix[:3, 3] = position

                extrinsic = model_matrix_to_extrinsic_matrix(new_model_matrix)
                height = int(self.window.size.height)
                width = int(self.widget3d_width)
                intrinsic = create_camera_intrinsic_from_size(width, height)

                self.widget3d.setup_camera(
                    intrinsic,
                    extrinsic,
                    width,
                    height,
                    self.widget3d.scene.bounding_box,
                )
            except Exception as e:
                Log("Joystick camera update failed: {}".format(e), tag="GUI")

    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(-4)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], gt=False, size=None):
        cR, ct = camera.get_RT
        W2C = getWorld2View2(cR, ct)
        W2C = W2C.detach().cpu().numpy()
        C2W = np.linalg.inv(W2C)
        sz = self.frustum_size if size is None else size
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color, size=sz)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self._camera_frustum_visible(name))
        return frustum

    def _camera_frustum_visible(self, geom_name):
        """Frusta require Cameras + (for mapped keyframes) Keyframes checkbox."""
        if not self.cameras_chbox.checked:
            return False
        if geom_name.startswith("keyframe_") and not geom_name.startswith(
            "keyframe_gt_"
        ):
            return self.keyframes_chbox.checked
        return True

    def _refresh_all_camera_frustum_visibility(self):
        for name in self.frustum_dict.keys():
            self.widget3d.scene.show_geometry(
                name, self._camera_frustum_visible(name)
            )

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.7
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _load_mesh(self, mesh_path):
        if mesh_path is None or not os.path.isfile(mesh_path):
            return
        self.mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh_o3d.compute_vertex_normals()
        if not self.mesh_o3d.has_vertex_colors():
            self.mesh_o3d.paint_uniform_color([0.75, 0.75, 0.75])
        Log("Loaded mesh from: {}".format(mesh_path), tag="GUI")

    def _on_mesh_chbox(self, is_checked):
        if self.mesh_o3d is None:
            self.mesh_chbox.checked = False
            return
        if is_checked:
            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(
                self.mesh_name, self.mesh_o3d, self.mesh_material
            )
        else:
            self.widget3d.scene.remove_geometry(self.mesh_name)

    def _on_combo_model(self, new_val, new_idx):
        model_idx = self.model_dict[new_val]
        self.global_map.active_map_idx = model_idx

    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_cameras_chbox(self, is_checked):
        self._refresh_all_camera_frustum_visibility()

    def _on_keyframes_chbox(self, is_checked):
        self._refresh_all_camera_frustum_visibility()

    def _on_est_traj_chbox(self, checked):
        self._show_traj_geometry(self.est_traj_name, self.est_traj_lineset, checked)

    def _on_gt_traj_chbox(self, checked):
        self._show_traj_geometry(self.gt_traj_name, self.gt_traj_lineset, checked)

    def _on_loop_edges_chbox(self, checked):
        self._refresh_loop_edges_geometry()

    def _show_traj_geometry(self, name, line_set, visible):
        self.widget3d.scene.remove_geometry(name)
        if not visible:
            return
        ls = np.asarray(line_set.lines)
        if ls.size == 0:
            return
        self.widget3d.scene.add_geometry(name, line_set, self._traj_line_material())

    def _refresh_loop_edges_geometry(self):
        self.widget3d.scene.remove_geometry(self.loop_edges_name)
        if not self.loop_edges_chbox.checked:
            return
        et = self._last_est_traj_for_loops
        le = self._last_loop_edges_ij
        if et is None or le is None:
            return
        le = np.asarray(le, dtype=np.int32)
        if le.size == 0:
            return
        pts = np.asarray(et[:, :3, 3], dtype=np.float64)
        n = pts.shape[0]
        if le.ndim != 2 or le.shape[1] != 2:
            return
        if np.any(le < 0) or np.any(le >= n):
            return
        ls = self.loop_edges_lineset
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(le)
        nseg = le.shape[0]
        cols = np.tile(LOOP_GREEN, (nseg, 1))
        ls.colors = o3d.utility.Vector3dVector(cols)
        self.widget3d.scene.add_geometry(
            self.loop_edges_name, ls, self._traj_line_material()
        )

    def _segment_colors_jet(self, ids):
        v = np.asarray(ids, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return np.zeros((0, 3))
        vmin, vmax = float(np.min(v)), float(np.max(v))
        denom = max(vmax - vmin, 1e-9)
        u = np.clip((v - vmin) / denom, 0.0, 1.0).astype(np.float32)
        rgb = imgviz.depth2rgb(
            u.reshape(-1, 1),
            min_value=0.0,
            max_value=1.0,
            colormap="jet",
        )
        rgb = np.asarray(rgb).reshape(-1, 3).astype(np.float64) / 255.0
        return np.clip(rgb, 0.0, 1.0)

    def _fill_traj_lineset(
        self, line_set, poses_c2w, frame_ids=None, color_scheme="jet"
    ):
        poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
        empty_pts = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        empty_lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        empty_colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        if (
            poses_c2w.ndim != 3
            or poses_c2w.shape[1:] != (4, 4)
            or poses_c2w.shape[0] < 2
        ):
            line_set.points = empty_pts
            line_set.lines = empty_lines
            line_set.colors = empty_colors
            return False
        pts = poses_c2w[:, :3, 3]
        n = pts.shape[0]
        edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
        if frame_ids is None or np.asarray(frame_ids).size != n:
            fid = np.arange(n, dtype=np.int64)
        else:
            fid = np.asarray(frame_ids).reshape(-1)
        fid_seg = fid[:-1]
        if color_scheme == "black":
            cols = np.zeros((len(fid_seg), 3), dtype=np.float64)
        else:
            cols = self._segment_colors_jet(fid_seg)
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        return True

    def _maybe_update_trajectories(self, pkt):
        if pkt.est_traj is not None:
            self._last_est_traj_for_loops = pkt.est_traj
            self._fill_traj_lineset(
                self.est_traj_lineset,
                pkt.est_traj,
                pkt.traj_frame_ids,
                color_scheme="jet",
            )
            self._show_traj_geometry(
                self.est_traj_name, self.est_traj_lineset, self.est_traj_chbox.checked
            )
        self._last_loop_edges_ij = getattr(pkt, "loop_edges", None)
        self._refresh_loop_edges_geometry()
        if pkt.gt_traj is not None:
            self._fill_traj_lineset(
                self.gt_traj_lineset,
                pkt.gt_traj,
                pkt.traj_frame_ids,
                color_scheme="black",
            )
            self._show_traj_geometry(
                self.gt_traj_name, self.gt_traj_lineset, self.gt_traj_chbox.checked
            )

    def _on_button(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.button.is_on
        self.q_vis2main.put(packet)

    def _on_render_mode_exclusive(self, selected_chbox, is_checked):
        if is_checked:
            for cb in self._render_mode_boxes:
                if cb is not selected_chbox:
                    cb.checked = False

    def _on_slider(self, value):
        packet = self.prepare_viz2main_packet()
        self.q_vis2main.put(packet)

    def _on_render_btn(self):
        packet = Packet_vis2main()
        packet.flag_nextbatch = True
        self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui.png", img)
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)
        path_gui = (save_dir / "screenshot-gui.png").resolve()
        path_rend = (save_dir / "screenshot.png").resolve()
        Log("Screenshot saved:", path_gui, "and", path_rend, tag="GUI")

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    def add_ids(self):
        indices = (
            torch.unique(self.gaussian_cur.unique_kfIDs).cpu().numpy().astype(int)
        ).tolist()
        for idx in indices:
            if idx in self.gaussian_id_dict.keys():
                continue

            self.gaussian_id_dict[idx] = 0
            self.combo_gaussian_id.add_item(str(idx))

    def receive_data(self, q):
        if q is None:
            return

        gaussian_packet = get_latest_queue(q)
        if gaussian_packet is None:
            return

        if gaussian_packet.num_keyframes is not None:
            self.keyframes_info.text = "# Keyframes: {}".format(
                gaussian_packet.num_keyframes
            )
        if gaussian_packet.num_loop_closures is not None:
            self.loop_closures_info.text = "# Loop Closures: {}".format(
                gaussian_packet.num_loop_closures
            )

        if (
            gaussian_packet.num_gaussians_total is not None
            and gaussian_packet.num_gaussians_active is not None
        ):
            nt = gaussian_packet.num_gaussians_total
            na = gaussian_packet.num_gaussians_active
            ni = nt - na
            self.gaussians_info.text = (
                "# Gaussians: {} active, {} inactive".format(na, ni)
            )

        if gaussian_packet.has_gaussians:
            self.gaussian_cur = gaussian_packet
            mm = _estimate_gaussian_packet_memory_mb(gaussian_packet)
            if mm is not None:
                self.map_mem_info.text = "Map Memory: {:.1f} MB".format(mm)
            self.init = True
        if gaussian_packet.current_frame is not None:
            self.output_frame_id.text = "Current Frame: {}".format(
                gaussian_packet.current_frame.uid
            )
            frustum = self.add_camera(
                gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.followcam_chbox.checked and not self.fly_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        if gaussian_packet.init_frame is not None:
            frustum = self.add_camera(
                gaussian_packet.init_frame, name="init", color=[1, 0, 0]
            )
            if self.followcam_chbox.checked and not self.fly_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        if gaussian_packet.keyframe is not None:
            name = "keyframe_{}".format(gaussian_packet.keyframe.uid)
            frustum = self.add_camera(
                gaussian_packet.keyframe, name=name, color=[1, 0, 0]
            )

        if gaussian_packet.keyframes is not None:
            for keyframe in gaussian_packet.keyframes:
                name = "keyframe_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[1, 0, 0])

        if gaussian_packet.gtframes is not None:
            for keyframe in gaussian_packet.gtframes:
                name = "keyframe_gt_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[1, 0, 0])

        if gaussian_packet.gtcolor is not None:
            rgb = torch.clamp(gaussian_packet.gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if gaussian_packet.gtdepth is not None:
            self.in_depth_widget.update_image(
                self._depth_to_o3d_image_spectral(gaussian_packet.gtdepth)
            )

        if gaussian_packet.gpu_mem_usage_gb is not None:
            self.gpu_mem_info.text = "GPU Memory: {:.2f} GB".format(
                gaussian_packet.gpu_mem_usage_gb
            )

        self._maybe_update_trajectories(gaussian_packet)

        if gaussian_packet.finish:
            Log("Received terminate signal", tag="GUI")
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    def _depth_to_o3d_image_spectral(self, depth_map):
        """PINGS-style depth false-color (Spectral, min/max range)."""
        if isinstance(depth_map, torch.Tensor):
            d = depth_map.detach().cpu().numpy()
        else:
            d = np.asarray(depth_map)
        colored = colorize_depth_maps(
            d,
            self.depth_vis_min,
            self.depth_vis_max,
            cmap="Spectral",
        )
        if isinstance(colored, torch.Tensor):
            colored = colored.detach().cpu().numpy()
        dc = (colored[0] * 255.0).astype(np.uint8)
        dc = np.ascontiguousarray(np.transpose(dc, (1, 2, 0)))
        return o3d.geometry.Image(dc)

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c)
        has_nan = torch.isnan(T).any()
        if has_nan:
            return None
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        return current_cam

    def rasterise(self, current_cam):
        if self.active_chbox.checked is True:
            state = "active"
        else:
            state = "all"
        t_render0 = time.perf_counter()
        if (
            self.time_shader_chbox.checked
            and self.gaussian_cur is not None
            and type(self.gaussian_cur) == GaussianPacket
        ):
            features = self.gaussian_cur.get_features.clone()
            kf_ids = self.gaussian_cur.unique_kfIDs.float()
            rgb_kf = imgviz.depth2rgb(
                kf_ids.view(-1, 1).cpu().numpy(), colormap="jet", dtype=np.float32
            )
            alpha = 0.1
            self.gaussian_cur.get_features = alpha * features + (
                1 - alpha
            ) * torch.from_numpy(rgb_kf).to(features.device)
            rendering_data = render(
                current_cam,
                self.gaussian_cur,
                self.background,
                render_option=state,
                scaling_modifier=self.scaling_slider.double_value,
            )
            self.gaussian_cur.get_features = features
        else:
            rendering_data = render(
                current_cam,
                self.gaussian_cur,
                self.background,
                render_option=state,
                scaling_modifier=self.scaling_slider.double_value,
            )
        dt = max(time.perf_counter() - t_render0, 1e-9)
        self._last_render_fps = 1.0 / dt
        self._raster_calls += 1
        if self._raster_calls % 5 == 0 and self._last_render_fps is not None:
            self.freq_info.text = "Render FPS: {:.1f}".format(self._last_render_fps)

        return rendering_data

    def render_o3d_image(self, results, current_cam):
        if self.depth_chbox.checked:
            depth = results["rend_depth_median"]
            depth = depth[0, :, :].detach().cpu().numpy()
            render_img = self._depth_to_o3d_image_spectral(depth)

        elif self.opacity_chbox.checked:
            opacity = results["rend_alpha"]
            opacity = opacity[0, :, :].detach().cpu().numpy()
            max_opacity = np.max(opacity)
            opacity = imgviz.depth2rgb(
                opacity, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity = torch.from_numpy(opacity)
            opacity = torch.permute(opacity, (2, 0, 1)).float()
            opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(opacity)

        elif self.normal_chbox.checked:
            normal = results["rend_normal"]
            normal_numpy = normal.detach().permute(1, 2, 0)
            normal = normal2rgb(normal_numpy)
            normal = normal.contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(normal)
        else:
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)
        return render_img

    def render_gui(self):
        if not self.init:
            return
        current_cam = self.get_current_cam()
        if current_cam is None:
            return
        results = self.rasterise(current_cam)
        if results is None:
            return
        self.render_img = self.render_o3d_image(results, current_cam)
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.render_gui()

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                Log("Closing Visualization", tag="GUI")
                break

            def update():
                if self.step % 3 == 0:
                    if self.fly_chbox.checked:
                        self._update_joystick_fly_control()
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
