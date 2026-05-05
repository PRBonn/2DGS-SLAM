import sys
import time
import os

import numpy as np

from utils.torch_cpp_log import ensure_before_torch_import

ensure_before_torch_import()
import torch
import torch.multiprocessing as mp

from utils.io_utils import clone_obj
from utils.logging_utils import Log

sys.path.append("gaussian_splatting")
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getWorld2View2

from gaussian_renderer import render, render_for_tracking
from scene.gaussian_model import GaussianModel

from gui import gui_utils, slam_gui

import signal


def load_pose_from_file(file):
    if not os.path.exists(file):
        Log(f"Pose file does not exist: {file}", tag="Viser")
        return None
    
    cameras = []

    K = torch.eye(3)
    K[0, 0] = 600
    K[1, 1] = 600
    K[0, 2] = 320
    K[1, 2] = 240

    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            pose_flat = list(map(float, parts[1:]))
            pose = np.array(pose_flat).reshape(4, 4)
            pose_torch = torch.from_numpy(pose)
            cam = Camera(frame_id, pose_torch, None, K, 480, 640)

            cameras.append(cam)

    return cameras

class Viser:
    def __init__(self, ply_path, pose_path, mesh_path=None):
        self.device = 'cuda'
        self.dtype = torch.float32

        self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        signal.signal(signal.SIGINT, self.signal_handler)

        self.gaussians = GaussianModel()
        self.gaussians.load_ply(ply_path)

        self.cameras = load_pose_from_file(pose_path)

        # init gui
        mp.set_start_method("spawn")
        self.q_main2vis = mp.Queue()
        self.q_vis2main = mp.Queue()
        self.params_gui = gui_utils.ParamsGUI(
            pipe = None,
            background = self.background,
            gaussians = self.gaussians,
            q_main2vis = self.q_main2vis,
            q_vis2main = self.q_vis2main,
            mesh_path = mesh_path,
        )
        self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
        self.gui_process.start()
        
        time.sleep(3)

    def shutdown_gui(self):
        if self.gui_process and self.gui_process.is_alive():
            Log("Shutting down viewer GUI...", tag="GUI")
            self.gui_process.terminate()
            self.gui_process.join(timeout=5)
            if self.gui_process.is_alive():
                self.gui_process.kill()
            self.gui_process.close()


    def signal_handler(self, sig, frame):
        self.shutdown_gui()
        sys.exit(0)

    def run(self):

        vis_cameras = []
        for i in range(len(self.cameras)):
            if i % 25 == 0:
                vis_cameras.append(self.cameras[i])

        est_traj, traj_ids = None, None
        if len(self.cameras) >= 2:
            poses = []
            ids = []
            for cam in self.cameras:
                cR, ct = cam.get_RT
                W2C = getWorld2View2(cR, ct).detach().cpu().numpy()
                poses.append(np.linalg.inv(W2C))
                ids.append(cam.uid)
            est_traj = np.stack(poses, axis=0)
            traj_ids = np.asarray(ids, dtype=np.int64)

        while True:
            gpu_gb = (
                torch.cuda.memory_allocated(0) / (1024**3)
                if torch.cuda.is_available()
                else None
            )
            n_g = self.gaussians.get_xyz.shape[0]
            n_g_act = int(self.gaussians.get_active_mask.squeeze(1).sum().item())
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=self.gaussians,
                    current_frame=self.cameras[-1],
                    keyframes=vis_cameras,
                    gtframes=None,
                    gtcolor=None,
                    gtdepth=None,
                    est_traj=est_traj,
                    traj_frame_ids=traj_ids,
                    gpu_mem_usage_gb=gpu_gb,
                    num_gaussians_total=n_g,
                    num_gaussians_active=n_g_act,
                )
            )

            time.sleep(0.1)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Visualize Gaussian model")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the Gaussian model PLY file")
    parser.add_argument("--pose_path", type=str, required=True, help="Path to the estimated pose file")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to the mesh PLY file for visualization")
    args = parser.parse_args()

    viser = Viser(args.ply_path, args.pose_path, mesh_path=args.mesh_path)
    viser.run()