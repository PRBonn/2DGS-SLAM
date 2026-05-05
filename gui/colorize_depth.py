# Depth colorization adapted from PINGS utils/tools.py (Marigold-style; default cmap Spectral).

import numpy as np
import torch


def colorize_depth_maps(
    depth_map,
    min_depth,
    max_depth,
    cmap="Spectral",
    valid_mask=None,
    use_valid_depth_mask=True,
):
    """Colorize depth maps in numpy. Output (B, 3, H, W) float in [0, 1]."""
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    try:
        cm = __import__("matplotlib", fromlist=["colormaps"]).colormaps[cmap]
    except (AttributeError, KeyError, TypeError):
        import matplotlib.pyplot as plt

        cm = plt.get_cmap(cmap)

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().cpu().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    else:
        raise TypeError(type(depth_map))

    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    if valid_mask is None and use_valid_depth_mask:
        valid_mask = depth > 0

    depth_norm = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth_norm, bytes=False)[:, :, :, 0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            vm = valid_mask.detach().cpu().numpy()
        else:
            vm = np.asarray(valid_mask)
        vm = vm.squeeze()
        if vm.ndim < 3:
            vm = vm[np.newaxis, np.newaxis, :, :]
        else:
            vm = vm[:, np.newaxis, :, :]
        vm = np.repeat(vm, 3, axis=1)
        img_colored_np = img_colored_np.copy()
        img_colored_np[~vm] = 0

    if isinstance(depth_map, torch.Tensor):
        return torch.from_numpy(img_colored_np).float()
    return img_colored_np
