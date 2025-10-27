"""
Metrics for evaluation: FID, IS, PSNR, SSIM (masked and global).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm


class FIDScore:
    """
    Fréchet Inception Distance calculator.
    """
    def __init__(self, device='cuda', batch_size=128):
        self.device = device
        self.batch_size = batch_size

        # Load Inception v3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove classification head
        self.inception.eval()
        self.inception.to(device)

    @torch.no_grad()
    def compute_statistics(self, dataloader_or_sampler, num_samples, is_dataloader=True):
        """
        Compute mean and covariance of Inception features.

        Args:
            dataloader_or_sampler: DataLoader or sampling function
            num_samples: Number of samples to evaluate
            is_dataloader: True if input is DataLoader, False if sampling function

        Returns:
            mu: Mean of features
            sigma: Covariance of features
        """
        features = []
        n_collected = 0

        if is_dataloader:
            # Real data path
            for batch in tqdm(dataloader_or_sampler, desc='Computing FID statistics'):
                if isinstance(batch, (tuple, list)):
                    imgs = batch[0]
                else:
                    imgs = batch

                imgs = imgs.to(self.device)

                # Resize to 299x299 for Inception
                if imgs.shape[-1] != 299:
                    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)

                # Map from [-1, 1] to [0, 1] for Inception
                imgs = (imgs + 1) / 2
                imgs = imgs.clamp(0, 1)

                feat = self.inception(imgs)
                features.append(feat.cpu())

                n_collected += imgs.shape[0]
                if n_collected >= num_samples:
                    break
        else:
            # Generated data path
            pbar = tqdm(total=num_samples, desc='Generating samples for FID')
            while n_collected < num_samples:
                n = min(self.batch_size, num_samples - n_collected)
                imgs = dataloader_or_sampler(batch_size=n)

                imgs = imgs.to(self.device)

                # Resize to 299x299
                if imgs.shape[-1] != 299:
                    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)

                # Map to [0, 1]
                imgs = (imgs + 1) / 2
                imgs = imgs.clamp(0, 1)

                feat = self.inception(imgs)
                features.append(feat.cpu())

                n_collected += imgs.shape[0]
                pbar.update(n)
            pbar.close()

        features = torch.cat(features, dim=0)[:num_samples]
        features = features.numpy()

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate FID between two sets of statistics."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def compute_fid(self, real_dataloader, sampler_fn, num_real, num_gen):
        """
        Compute FID score.

        Args:
            real_dataloader: DataLoader for real images
            sampler_fn: Function that takes batch_size and returns generated images
            num_real: Number of real samples
            num_gen: Number of generated samples

        Returns:
            FID score
        """
        print("Computing statistics for real images...")
        mu_real, sigma_real = self.compute_statistics(real_dataloader, num_real, is_dataloader=True)

        print("Computing statistics for generated images...")
        mu_gen, sigma_gen = self.compute_statistics(sampler_fn, num_gen, is_dataloader=False)

        fid = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid


def compute_psnr(img1, img2, mask=None, data_range=2.0):
    """
    Compute PSNR between two images.

    Args:
        img1: (B, C, H, W) in [-1, 1]
        img2: (B, C, H, W) in [-1, 1]
        mask: Optional (B, C, H, W) binary mask (1 = compute here, 0 = ignore)
        data_range: Range of pixel values (2.0 for [-1, 1])

    Returns:
        PSNR value
    """
    if mask is not None:
        mse = ((img1 - img2) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    else:
        mse = F.mse_loss(img1, img2)

    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(img1, img2, mask=None, window_size=11, data_range=2.0):
    """
    Compute SSIM between two images (simplified version).

    Args:
        img1: (B, C, H, W) in [-1, 1]
        img2: (B, C, H, W) in [-1, 1]
        mask: Optional (B, C, H, W) binary mask
        window_size: Size of Gaussian window
        data_range: Range of pixel values

    Returns:
        SSIM value
    """
    # Simplified SSIM (for full implementation, use pytorch-msssim or similar)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if mask is not None:
        # Masked means
        mu1 = (img1 * mask).sum() / (mask.sum() + 1e-8)
        mu2 = (img2 * mask).sum() / (mask.sum() + 1e-8)

        # Masked variances and covariance
        var1 = ((img1 - mu1) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        var2 = ((img2 - mu2) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        cov = ((img1 - mu1) * (img2 - mu2) * mask).sum() / (mask.sum() + 1e-8)
    else:
        mu1 = img1.mean()
        mu2 = img2.mean()
        var1 = img1.var()
        var2 = img2.var()
        cov = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))

    return ssim.item()


def compute_masked_metrics(pred, target, mask):
    """
    Compute PSNR and SSIM on masked region only.

    Args:
        pred: (B, C, H, W) predicted images
        target: (B, C, H, W) ground truth images
        mask: (B, C, H, W) binary mask (1 = evaluate here)

    Returns:
        Dictionary with psnr and ssim values
    """
    return {
        'psnr': compute_psnr(pred, target, mask=mask),
        'ssim': compute_ssim(pred, target, mask=mask)
    }
