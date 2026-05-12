import csv
import glob
import os
import math
import pickle

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }
            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class BS3DParser:
    def __init__(self, input_folder, depth_dir="depth_render"):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        self.depth_paths_all = sorted(
            glob.glob(f"{self.input_folder}/{depth_dir}/*.png")
        )
        self.depth_paths = []
        self.poses = []
        self.frames = []
        self.load_poses(os.path.join(self.input_folder, "poses.txt"))

    def load_poses(self, path):
        pose_dict = {}
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            tstamp = parts[0]
            pose_vec = np.array(list(map(float, parts[1:])))
            trans = pose_vec[:3]
            quat = pose_vec[3:]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = np.linalg.inv(T)
            pose_dict[tstamp] = pose

        depth_items = []
        for p in self.depth_paths_all:
            t = float(os.path.splitext(os.path.basename(p))[0])
            depth_items.append((t, p))
        depth_items.sort(key=lambda x: x[0])
        depth_ts = np.array([x[0] for x in depth_items], dtype=np.float64)

        pose_items = []
        for ts, pose in pose_dict.items():
            pose_items.append((float(ts), pose))
        pose_items.sort(key=lambda x: x[0])
        pose_ts = np.array([x[0] for x in pose_items], dtype=np.float64)

        matched_colors = []
        max_dt = 0.02
        for color_path in self.color_paths:
            t_color = float(os.path.splitext(os.path.basename(color_path))[0])

            d_idx = int(np.argmin(np.abs(depth_ts - t_color)))
            p_idx = int(np.argmin(np.abs(pose_ts - t_color)))
            d_dt = abs(depth_ts[d_idx] - t_color)
            p_dt = abs(pose_ts[p_idx] - t_color)
            if d_dt > max_dt or p_dt > max_dt:
                continue

            depth_path = depth_items[d_idx][1]
            pose = pose_items[p_idx][1]
            matched_colors.append(color_path)
            self.depth_paths.append(depth_path)
            self.poses.append(pose)
            frame = {
                "file_path": color_path,
                "depth_path": depth_path,
                "transform_matrix": pose.tolist(),
            }
            self.frames.append(frame)

        self.color_paths = matched_colors
        self.n_img = len(self.frames)


class ScanNetParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/frame_*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/frame_*.TIFF"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt_pose.txt")

    def load_poses(self, path):
        self.poses = []
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = np.linalg.inv(T)
            self.poses.append(pose)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera parameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.crop_edge = calibration["crop_edge"] if 'crop_edge' in calibration else 0

        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        
        pil_image = Image.open(color_path)
        image = np.array(pil_image)
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale  
        
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        if self.crop_edge > 0:
            edge = self.crop_edge
            image = image[:, edge:-edge, edge:-edge]
            depth = depth[edge:-edge, edge:-edge]

        pose = torch.from_numpy(pose).to(device=self.device).to(self.dtype)
        depth = torch.from_numpy(depth).to(device=self.device).to(self.dtype)
        
        return image, pil_image, depth, pose
    

class ScanNetDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ScanNetParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

class BS3DDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        depth_dir = config["Dataset"].get("depth_dir", "depth_render")
        parser = BS3DParser(dataset_path, depth_dir=depth_dir)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class StrayScannerDataset(MonocularDataset):
    """StrayScanner (iPhone RGB-D) dataset. Expects preprocessed output from scripts/stray_preprocess.py."""
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ScanNetParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280

        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True
        else:
            self.has_depth = False

        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(rs_config)

        if self.has_depth:
            self.align = rs.align(rs.stream.color)

        rgb_sensor = self.profile.get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        rgb_sensor.set_option(rs.option.exposure, 200)

        rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        rgb_intrinsics = rgb_profile.get_intrinsics()

        self.fx = rgb_intrinsics.fx
        self.fy = rgb_intrinsics.fy
        self.cx = rgb_intrinsics.ppx
        self.cy = rgb_intrinsics.ppy
        self.width = rgb_intrinsics.width
        self.height = rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K,
            (self.w, self.h), cv2.CV_32FC1,
        )

        if self.has_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

        self._frame_cache = {}

    def __getitem__(self, idx):
        if idx in self._frame_cache:
            return self._frame_cache[idx]

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data()).astype(np.float64) * self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()
            depth = None

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        pil_image = Image.fromarray(image)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        pose = torch.eye(4, device=self.device, dtype=self.dtype)

        if depth is not None:
            depth = torch.from_numpy(depth).to(device=self.device, dtype=self.dtype)

        result = (image, pil_image, depth, pose)
        self._frame_cache[idx] = result
        return result


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "scannet":
        return ScanNetDataset(args, path, config)
    elif config["Dataset"]["type"] == "bs3d":
        return BS3DDataset(args, path, config)
    elif config["Dataset"]["type"] == "stray":
        return StrayScannerDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
