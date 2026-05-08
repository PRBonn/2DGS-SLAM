#!/usr/bin/env python3
# @file      pgo.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from rich import print



class PoseGraphManager:
    def __init__(self, config):

        self.config = config
        self.silence = not bool(config.get("Results", {}).get("verbose", False))

        self.fixed_cov = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
        )  # fixed

        self.pgo_max_iter = self.config["Training"]["pgo_max_iter"]
        self.pgo_error_thre_frame = self.config["Training"]["pgo_error_thre_frame"]

        tran_std = self.config["Training"]["pgo_tran_std"]  # m
        rot_std = self.config["Training"]["pgo_rot_std"]  # degree # better to be small
        self.const_cov = np.array(
            [
                np.radians(rot_std),
                np.radians(rot_std),
                np.radians(rot_std),
                tran_std,
                tran_std,
                tran_std,
            ]
        )  # first rotation, then translation
        
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)

        self.graph_factors = gtsam.NonlinearFactorGraph() # edges # with pose and pose covariance
        self.graph_initials = gtsam.Values()  # initial guess of the node

        self.cur_pose = None
        self.curr_node_idx = None
        self.graph_optimized = None
        self.init_poses = None # as np.array
        self.pgo_poses = None # as np.array

        self.loop_edges_vis = []
        self.loop_edges = []
        self.loop_trans = []

        self.last_loop_idx = 0
        self.drift_radius = 0.0  # m
        self.pgo_count = 0
        self.last_error = 0.0

    def add_frame_node(self, frame_id, init_pose):
        """create frame pose node and set pose initial guess
        Args:
            frame_id: int
            init_pose (np.array): 4x4, as T_world<-cur
        """
        self.curr_node_idx = frame_id  # make start with 0
        if not self.graph_initials.exists(
            gtsam.symbol("x", frame_id)
        ):  # create if not yet exists
            self.graph_initials.insert(
                gtsam.symbol("x", frame_id), gtsam.Pose3(init_pose)
            )

    def add_pose_prior(
        self, frame_id: int, prior_pose: np.ndarray, fixed: bool = False
    ):
        """add pose prior unary factor
        Args:
            frame_id: int
            prior_pose (np.array): 4x4, as T_world<-cur
            dist_ratio: float , use to determine the covariance, the std is porpotional to this dist_ratio
            fixed: bool, if True, this frame is fixed with very low covariance
        """

        if fixed:
            cov_model = self.fixed_cov
        else:
            tran_sigma = self.drift_radius + 1e-4  # avoid divide by 0
            rot_sigma = self.drift_radius * np.radians(10.0)
            cov_model = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [
                        rot_sigma,
                        rot_sigma,
                        rot_sigma,
                        tran_sigma,
                        tran_sigma,
                        tran_sigma,
                    ]
                )
            )

        self.graph_factors.add(
            gtsam.PriorFactorPose3(
                gtsam.symbol("x", frame_id), gtsam.Pose3(prior_pose), cov_model
            )
        )

    def add_odometry_factor(
        self, cur_id: int, last_id: int, odom_transform: np.ndarray, cov=None
    ):
        """add a odometry factor between two adjacent pose nodes
        Args:
            cur_id: int
            last_id: int
            odom_transform (np.array): 4x4 , as T_prev<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """

        if cov is None:
            cov_model = self.odom_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", last_id),  # t-1
                gtsam.symbol("x", cur_id),  # t
                gtsam.Pose3(odom_transform),  # T_prev<-cur
                cov_model,
            )
        )

    def add_loop_factor(
        self,
        cur_id: int,
        loop_id: int,
        loop_transform: np.ndarray,
        cov=None,
        reject_outlier=True,
        loop_error_threshold = 50
    ):
        """add a loop closure factor between two pose nodes
        Args:
            cur_id: int
            loop_id: int
            loop_transform (np.array): 4x4 , as T_loop<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """

        if cov is None:
            cov_model = self.loop_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol("x", loop_id),  # l
                gtsam.symbol("x", cur_id),  # t
                gtsam.Pose3(loop_transform),  # T_loop<-cur
                cov_model,
            )
        )

        cur_error = self.graph_factors.error(self.graph_initials)
        if reject_outlier:
            valid_error_thre = (
                self.last_error
                + (cur_id - self.last_loop_idx) * self.pgo_error_thre_frame
            )
            if reject_outlier and cur_error > valid_error_thre:
                if not self.silence:
                    print(
                        "[bold yellow]A loop edge rejected due to too large error[/bold yellow]"
                    )
                self.graph_factors.remove(self.graph_factors.size() - 1)
                return False
            
        if cur_error < loop_error_threshold:
            self.graph_factors.remove(self.graph_factors.size() - 1)
        
            
        return True

    def optimize_pose_graph(self):
        
        error_before = self.graph_factors.error(self.graph_initials)
        if error_before < 0.0001:
            return False

        opt_param = gtsam.LevenbergMarquardtParams()
        opt_param.setMaxIterations(self.pgo_max_iter)
        opt = gtsam.LevenbergMarquardtOptimizer(
            self.graph_factors, self.graph_initials, opt_param
        )

        self.graph_optimized = opt.optimizeSafely()

        error_after = self.graph_factors.error(self.graph_optimized)
        self.last_error = error_after
        if not self.silence:
            print("[bold red]PGO done[/bold red]")
            print("error %f --> %f:" % (error_before, error_after))

        self.graph_initials = self.graph_optimized

        self.pgo_count += 1
        return True

    
    def get_optimized_node_pose(self, idx):
        pose = self.graph_optimized.atPose3(gtsam.symbol("x", idx))
        pose_se3 = np.eye(4)
        pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
        pose_se3[:3, :3] = pose.rotation().matrix()

        return pose_se3


def get_node_pose(self, graph, idx):
    pose = graph.atPose3(gtsam.symbol("x", idx))
    pose_se3 = np.eye(4)
    pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
    pose_se3[:3, :3] = pose.rotation().matrix()

    return pose_se3


def get_node_cov(marginals, idx):

    cov = marginals.marginalCovariance(gtsam.symbol("x", idx))
    return cov