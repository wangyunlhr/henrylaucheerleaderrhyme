"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""
import multiprocessing
from functools import partial

import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from . import OUTPUT_TEMPLATE
from .metric_utils import compute_logits, compute_pairwise_cd, \
    compute_pairwise_emd, pcd2bev_sum, compute_pairwise_cd_batch, pcd2bev_bin
from .fid_score import calculate_frechet_distance
import lpips
from skimage.metrics import structural_similarity
import torch
#! add depth 指标

lpips_fn = lpips.LPIPS(net="alex").eval()
def compute_depth_metrics(gt, pred, min_depth=1e-6, max_depth=80):
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth
    gt[gt < min_depth] = min_depth
    gt[gt > max_depth] = max_depth

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    mae = np.mean(np.abs(gt - pred))
    medae = np.median(np.abs(gt - pred))

    psnr_loss = 10 * np.log10(max_depth**2 / np.mean((pred - gt) ** 2))

    ssim_loss = structural_similarity(
        pred.squeeze(-1), gt.squeeze(-1), data_range=np.max(gt) - np.min(gt)
    )

    lpips_loss = lpips_fn(
        torch.from_numpy(pred).permute(2, 0, 1),
        torch.from_numpy(gt).permute(2, 0, 1),
        normalize=True,
    ).item()

    return [rmse, mae, medae, lpips_loss, ssim_loss, psnr_loss]


def evaluate_range_image(gt_range_maps_list, sample_range_images_list, sample_gt_masks_list):
    print('Evaluating Depth Metrics ...')
    rmse_list, mae_list, medae_list = [], [], []
    lpips_list, ssim_list, psnr_list = [], [], []
    for gt_range_map, sample_range_image, sample_gt_mask in tqdm(zip(gt_range_maps_list, sample_range_images_list, sample_gt_masks_list), total=len(gt_range_maps_list)):
        # apply mask
        gt_depth = (gt_range_map * sample_gt_mask).transpose(1, 2, 0)
        pred_depth = (sample_range_image[None] * sample_gt_mask).transpose(1, 2, 0)

        rmse, mae, medae, lpips_loss, ssim_loss, psnr_loss = compute_depth_metrics(
            gt_depth, pred_depth
        )
        rmse_list.append(rmse)
        mae_list.append(mae)
        medae_list.append(medae)
        lpips_list.append(lpips_loss)
        ssim_list.append(ssim_loss)
        psnr_list.append(psnr_loss)

    print(OUTPUT_TEMPLATE.format('RMSE', np.mean(rmse_list)))
    print(OUTPUT_TEMPLATE.format('MAE ', np.mean(mae_list)))
    print(OUTPUT_TEMPLATE.format('MedAE', np.mean(medae_list)))
    print(OUTPUT_TEMPLATE.format('LPIPS', np.mean(lpips_list)))
    print(OUTPUT_TEMPLATE.format('SSIM ', np.mean(ssim_list)))
    print(OUTPUT_TEMPLATE.format('PSNR ', np.mean(psnr_list)))


def evaluate(reference, samples, metrics, data):
    # perceptual
    if 'frid' in metrics:
        compute_frid(reference, samples, data)
    if 'fsvd' in metrics:
        compute_fsvd(reference, samples, data)
    if 'fpvd' in metrics:
        compute_fpvd(reference, samples, data)

    # reconstruction
    if 'cd' in metrics:
        compute_cd(reference, samples)
    if 'emd' in metrics:
        compute_emd(reference, samples)

    # statistical
    if 'jsd' in metrics:
        compute_jsd(reference, samples, data)
    if 'mmd' in metrics:
        compute_mmd(reference, samples, data)





def compute_cd(reference, samples):
    """
    Calculate score of Chamfer Distance (CD)

    """
    print('Evaluating (CD) ...')
    results = []
    for x, y in zip(reference, samples):
        d = compute_pairwise_cd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('CD  ', score))


def compute_emd(reference, samples):
    """
    Calculate score of Earth Mover's Distance (EMD)

    """
    print('Evaluating (EMD) ...')
    results = []
    for x, y in zip(reference, samples):
        d = compute_pairwise_emd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('EMD ', score))


def compute_mmd(reference, samples, data, dist='cd', verbose=True):
    """
    Calculate the score of Minimum Matching Distance (MMD)

    """
    print('Evaluating (MMD) ...')
    assert dist in ['cd', 'emd']
    reference, samples = pcd2bev_bin(data, reference, samples)
    compute_dist_func = compute_pairwise_cd_batch if dist == 'cd' else compute_pairwise_emd
    results = []
    for r in tqdm(reference, disable=not verbose):
        dists = compute_dist_func(r, samples)
        results.append(min(dists))
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('MMD ', score))


def compute_jsd(reference, samples, data):
    """
    Calculate the score of Jensen-Shannon Divergence (JSD)

    """
    print('Evaluating (JSD) ...')
    reference, samples = pcd2bev_sum(data, reference, samples)
    reference = (reference / np.sum(reference)).flatten()
    samples = (samples / np.sum(samples)).flatten()
    score = jensenshannon(reference, samples)
    print(OUTPUT_TEMPLATE.format('JSD ', score))


def compute_fd(reference, samples):
    mu1, mu2 = np.mean(reference, axis=0), np.mean(samples, axis=0)
    sigma1, sigma2 = np.cov(reference, rowvar=False), np.cov(samples, rowvar=False)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return distance


def compute_frid(reference, samples, data):
    """
    Calculate the score of Fréchet Range Image Distance (FRID)

    """
    print('Evaluating (FRID) ...')
    gt_logits, samples_logits = compute_logits(data, 'range', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FRID', score))


def compute_fsvd(reference, samples, data):
    """
    Calculate the score of Fréchet Sparse Volume Distance (FSVD)

    """
    print('Evaluating (FSVD) ...')
    gt_logits, samples_logits = compute_logits(data, 'voxel', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FSVD', score))


def compute_fpvd(reference, samples, data):
    """
    Calculate the score of Fréchet Point-based Volume Distance (FPVD)

    """
    print('Evaluating (FPVD) ...')
    gt_logits, samples_logits = compute_logits(data, 'point_voxel', reference, samples)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FPVD', score))

