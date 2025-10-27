"""
Visualization utilities for training monitoring.
"""
import torch
from torchvision.utils import make_grid, save_image
import os


def save_image_grid(images, save_path, nrow=5, normalize=True, value_range=(-1, 1)):
    """
    Save a grid of images.

    Args:
        images: (B, C, H, W) tensor
        save_path: Path to save image
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
        value_range: Range of input values for normalization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if normalize:
        # Map from value_range to [0, 1]
        images = (images - value_range[0]) / (value_range[1] - value_range[0])
        images = images.clamp(0, 1)

    save_image(images, save_path, nrow=nrow)


def create_comparison_grid(gt, sparse, mask, output, target_mask=None, nrow=8):
    """
    Create a comparison grid showing: GT | Sparse | Mask | Output | Target Mask.

    Args:
        gt: (B, C, H, W) ground truth images in [-1, 1]
        sparse: (B, C, H, W) sparse input in [-1, 1]
        mask: (B, C, H, W) conditioning mask
        output: (B, C, H, W) model output in [-1, 1]
        target_mask: Optional (B, C, H, W) target supervision mask
        nrow: Number of images per row

    Returns:
        Grid tensor ready for saving
    """
    # Normalize to [0, 1] for visualization
    gt_vis = (gt + 1) / 2
    sparse_vis = (sparse + 1) / 2
    output_vis = (output + 1) / 2

    # Ensure masks are in [0, 1]
    mask_vis = mask.float()
    if mask_vis.max() > 1:
        mask_vis = mask_vis / mask_vis.max()

    # Stack all components
    if target_mask is not None:
        target_vis = target_mask.float()
        if target_vis.max() > 1:
            target_vis = target_vis / target_vis.max()
        composite = torch.cat([gt_vis, sparse_vis, mask_vis, output_vis, target_vis], dim=0)
    else:
        composite = torch.cat([gt_vis, sparse_vis, mask_vis, output_vis], dim=0)

    grid = make_grid(composite, nrow=nrow)
    return grid


def save_comparison_grid(gt, sparse, mask, output, target_mask, save_path, nrow=8):
    """
    Create and save a comparison grid.

    Args:
        gt, sparse, mask, output, target_mask: As in create_comparison_grid
        save_path: Path to save grid image
        nrow: Number of images per row
    """
    grid = create_comparison_grid(gt, sparse, mask, output, target_mask, nrow=nrow)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(grid, save_path)


def visualize_training_batch(batch_dict, save_dir, step, nrow=8):
    """
    Visualize a training batch with all components.

    Args:
        batch_dict: Dictionary containing 'gt', 'sparse', 'cond_mask', 'output', 'target_mask'
        save_dir: Directory to save visualizations
        step: Training step number
        nrow: Images per row
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save comparison grid
    save_comparison_grid(
        batch_dict['gt'],
        batch_dict['sparse'],
        batch_dict['cond_mask'],
        batch_dict['output'],
        batch_dict['target_mask'],
        os.path.join(save_dir, f'comparison_step{step}.png'),
        nrow=nrow
    )

    # Save individual components
    save_image_grid(
        batch_dict['output'],
        os.path.join(save_dir, f'samples_step{step}.png'),
        nrow=nrow
    )
