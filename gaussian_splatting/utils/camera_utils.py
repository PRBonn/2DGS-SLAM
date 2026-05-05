import torch
from torch import nn
import math
import lietorch

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        init_pose,
        gt_pose,
        K,
        image_height,
        image_width,
        device = "cuda:0",
    ):
        super(Camera, self).__init__()
        self.device = device
        self.dtype = torch.float32
        self.uid = int(uid)

        if init_pose is not None:
            self.T = init_pose.to(device=self.device).to(self.dtype)
        else:
            self.T = torch.eye(4, device=device).to(self.dtype)

        self.cam_rot_delta = nn.Parameter(torch.zeros(3, dtype=self.dtype, device=device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, dtype=self.dtype, device=device))

        self.scale = nn.Parameter(torch.tensor([1.0], dtype=self.dtype, device=self.device))
        self.shift = nn.Parameter(torch.tensor([0.0], dtype=self.dtype, device=self.device))

        self.exposure_a = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))
        self.exposure_b = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))

        self.pred_K = torch.eye(3, dtype=self.dtype, device=self.device)

        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        self.gt_pose = gt_pose

        self.image_width = image_width
        self.image_height = image_height

        self.FoVx = self.focal2fov(self.fx, self.image_width)
        self.FoVy = self.focal2fov(self.fy, self.image_height)

        self.prcppoint = torch.tensor([self.cx / image_width, self.cy / image_height], device=device).to(float)

        self.projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, W=image_width, H=image_height
        ).transpose(0, 1).to(device=device).to(device=self.device)

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        K = torch.eye(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return Camera(uid, T, None, K, H, W)
    
    def set_pred_K(self, K):
        self.pred_K = K
    
    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    
    @property
    def get_inv_RT(self):
        '''
        use to transform points.
        '''
        r = self.T[:3, :3]
        t = self.T[:3, 3]
        inv_r = r.t()
        inv_t = -inv_r @ t 
        return inv_r, inv_t
    
    @property
    def get_RT(self):

        r = self.T[:3, :3]
        t = self.T[:3, 3]
        return r, t
    
    @property
    def depth_scale(self):
        return self.scale
    
    def set_scale(self, depth_scale):
        self.scale = nn.Parameter(torch.tensor([depth_scale], dtype=self.dtype, device=self.device))

    def focal2fov(self, focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))
    
    
    def camera_ray_vectors(self):
        u = torch.arange(self.image_width, dtype=self.dtype, device=self.device)
        v = torch.arange(self.image_height, dtype=self.dtype, device=self.device)
        u, v = torch.meshgrid(u, v, indexing="xy")

        x = (u-self.cx)/self.fx
        y = (v-self.cy)/self.fy
        z = torch.ones_like(x, dtype=self.dtype, device=self.device)
        rays = torch.stack((x, y, z), dim=-1).permute([2, 0, 1])

        ray_vectors = torch.nn.functional.normalize(rays, dim=0)

        return ray_vectors


    def update_pose(self, converged_threshold=1e-4):
        tau = torch.cat([self.cam_trans_delta,
                        self.cam_rot_delta], axis=0)
        new_w2c = lietorch.SE3.exp(tau).matrix() @ self.T
        converged = (tau**2).sum() < (converged_threshold**2)
        self.T = new_w2c
        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)

        return converged
    
