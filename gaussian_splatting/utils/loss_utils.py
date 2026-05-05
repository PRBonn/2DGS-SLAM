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

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt, mask=None):
    loss = torch.abs(network_output - gt)
    if mask is not None:
        loss = (loss.mean(0))[mask]
    return loss.mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def tukey_loss(network_output, gt, mask=None, c=0.3):
    residuals = network_output - gt
    abs_residuals = torch.abs(residuals).sum(0)
    loss = torch.zeros_like(abs_residuals, dtype=residuals.dtype, device=residuals.device)
    res_mask = abs_residuals <= c
    loss[res_mask] = (c ** 2 / 6) * (1 - (1 - (abs_residuals[res_mask] / c) ** 2) ** 3)
    loss[~res_mask] = (c ** 2) / 6

    if mask is not None:
        loss = loss[mask]

    return loss.mean()



def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()
