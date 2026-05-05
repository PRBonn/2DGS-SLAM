import csv
import os

import cv2
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from fused_ssim import fused_ssim
from utils.logging_utils import Log

def evaluate_evo(poses_gt, poses_est, label, monocular=False, quiet=False):
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_metric.get_all_statistics()
    if not quiet:
        Log(
            f"RMSE ATE \\[m] ({label}): {float(ape_stat):.4f}",
            tag="Eval",
        )

    return ape_stat


def eval_ate_for_indices(
    frames, frame_ids, label_evo, monocular=False, quiet=False
):
    """ATE on an ordered sequence of dataset frame ids present in frames (uids)."""
    frame_ids_sorted = sorted(frame_ids)
    trj_est_np, trj_gt_np = [], []

    for fid in frame_ids_sorted:
        kf = frames[fid]
        pose_est = np.linalg.inv(kf.T.cpu().numpy())
        pose_gt = np.linalg.inv(kf.gt_pose.cpu().numpy())
        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    return evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        label=label_evo,
        monocular=monocular,
        quiet=quiet,
    )


def eval_ate(frames, kf_ids, iterations, final=False, monocular=False, quiet=False):
    label_evo = "final" if final else "{:04}".format(iterations)
    return eval_ate_for_indices(
        frames, kf_ids, label_evo, monocular=monocular, quiet=quiet
    )


def eval_rendering(
    cameras,
    gaussians,
    dataset,
    background,
    kf_indices,
):
    if not kf_indices:
        return None
    img_pred, img_gt, saved_frame_idx = [], [], []
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    for idx in kf_indices:
        saved_frame_idx.append(idx)
        eval_cam = cameras[idx]
        gt_image, _, _, _ = dataset[eval_cam.uid]

        render_img = render(eval_cam, pc = gaussians, render_option='all', bg_color=background)["render"]
        image = torch.clamp(render_img, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        gt_black_mask = gt_image > 0
        rend_black_mask = image > 0
        mask = torch.logical_and(gt_black_mask, rend_black_mask)

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = fused_ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))[0].mean()
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]:.4f}, ssim: {output["mean_ssim"]:.4f}, lpips: {output["mean_lpips"]:.4f}',
        tag="Eval",
    )
    return output


def write_run_metrics_csv(
    csv_path,
    run_name,
    *,
    ate_rmse_keyframes_m=None,
    ate_rmse_all_tracked_m=None,
    mean_psnr=None,
    mean_ssim=None,
    mean_lpips=None,
):
    """One header row + one data row (per run)."""
    fieldnames = [
        "run_name",
        "ate_rmse_keyframes_m",
        "ate_rmse_all_tracked_m",
        "mean_psnr",
        "mean_ssim",
        "mean_lpips",
    ]
    row = {
        "run_name": run_name,
        "ate_rmse_keyframes_m": "" if ate_rmse_keyframes_m is None else float(ate_rmse_keyframes_m),
        "ate_rmse_all_tracked_m": ""
        if ate_rmse_all_tracked_m is None
        else float(ate_rmse_all_tracked_m),
        "mean_psnr": "" if mean_psnr is None else float(mean_psnr),
        "mean_ssim": "" if mean_ssim is None else float(mean_ssim),
        "mean_lpips": "" if mean_lpips is None else float(mean_lpips),
    }
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
