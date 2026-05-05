import torch
import numpy as np
from PIL import Image

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)



def tukey_loss(prediction, gt, c=0.3):
    residuals = prediction - gt 
    abs_residuals = torch.abs(residuals).sum(0)
    loss = torch.zeros_like(abs_residuals, dtype=residuals.dtype, device=residuals.device)
    res_mask = abs_residuals <= c
    loss[res_mask] = (c ** 2 / 6) * (1 - (1 - (abs_residuals[res_mask] / c) ** 2) ** 3)
    loss[~res_mask] = (c ** 2) / 6

    return loss


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def resize_img_K(o_pil_image, o_K, o_h, o_w, downsample_ratio=2.0):
    
    new_h = int(round(o_h/downsample_ratio))
    new_w = int(round(o_w/downsample_ratio))
    new_size = tuple([new_w, new_h])

    new_K = np.eye(3)
    new_K[0,:] = o_K[0,:]*float(new_size[0])/o_w
    new_K[1,:] = o_K[1,:]*float(new_size[1])/o_h
    
    resized_img = o_pil_image.resize(new_size, resample=Image.BILINEAR)
    
    resized_torch_img = (
            torch.from_numpy(np.array(resized_img) / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device='cuda', dtype=torch.float32)
    )

    return resized_torch_img, new_K
