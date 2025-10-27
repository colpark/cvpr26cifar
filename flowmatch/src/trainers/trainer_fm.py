"""
Flow Matching trainer with masked conditioning.
"""
import torch
import os
from .trainer_base import BaseTrainer
from ..utils.viz import save_image_grid, save_comparison_grid
from ..utils.metrics import compute_masked_metrics


class FMTrainer(BaseTrainer):
    """
    Trainer for Flow Matching (Rectified Flow) with masked conditioning.

    Args:
        flow: RectifiedFlow model
        train_loader: Training DataLoader
        val_loader: Optional validation DataLoader
        optimizer_config: Optimizer configuration
        device: Training device
        save_dir: Save directory
        sparsity_controller: SparsityController instance
        sampling_steps: Number of ODE integration steps for sampling
        clip_sampling: Whether to clip during sampling
        max_grad_norm: Maximum gradient norm for clipping
        target_size: Optional (H, W) for zero-shot super-resolution
    """
    def __init__(self, flow, train_loader, val_loader=None,
                 optimizer_config=None, device='cuda', save_dir='./results',
                 sparsity_controller=None, sampling_steps=50, clip_sampling=True,
                 max_grad_norm=1.0, target_size=None):
        super().__init__(
            flow, train_loader, val_loader,
            optimizer_config, device, save_dir, sparsity_controller
        )
        self.sampling_steps = sampling_steps
        self.clip_sampling = clip_sampling
        self.max_grad_norm = max_grad_norm
        self.target_size = target_size

    def train_step(self, batch):
        """
        Single Flow Matching training step.

        Args:
            batch: Tuple of (images, indices)

        Returns:
            Loss value
        """
        images, indices = batch
        images = images.to(self.device)
        B, C = images.shape[:2]

        # Generate masks
        cond_masks, target_masks = self.sparsity_controller.get_masks(
            B, C, sample_ids=indices.tolist()
        )
        cond_mask = torch.stack(cond_masks).to(self.device)
        target_mask = torch.stack(target_masks).to(self.device)

        # Create sparse input
        sparse_input = images * cond_mask

        # Forward pass
        self.optimizer.zero_grad()
        loss = self.model(
            images,
            sparse_input=sparse_input,
            mask=cond_mask,
            loss_mask=target_mask
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def sample(self, batch_size, sparse_input, mask):
        """
        Generate samples using Flow Matching (ODE integration).

        Args:
            batch_size: Number of samples
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask

        Returns:
            (B, C, H, W) generated images
        """
        return self.model.sample(
            batch_size=batch_size,
            sparse_input=sparse_input,
            mask=mask,
            steps=self.sampling_steps,
            target_size=self.target_size,
            clip=self.clip_sampling,
            device=self.device
        )

    def generate_samples(self):
        """
        Generate and save sample images with comprehensive visualizations.

        For Flow Matching, this includes super-resolution at 64x64 and 96x96.
        """
        self.model.eval()

        with torch.no_grad():
            # Get a batch from training data
            images, indices = next(iter(self.train_loader))
            images = images[:16].to(self.device)  # Take first 16
            indices = indices[:16]

            # Generate masks
            cond_masks, target_masks = self.sparsity_controller.get_masks(
                images.shape[0], images.shape[1], sample_ids=indices.tolist()
            )
            cond_mask = torch.stack(cond_masks).to(self.device)
            target_mask = torch.stack(target_masks).to(self.device)

            sparse_input = images * cond_mask

            # 1. Conditional generation at native resolution (32x32)
            samples_cond = self.model.sample(
                batch_size=images.shape[0],
                sparse_input=sparse_input,
                mask=cond_mask,
                steps=self.sampling_steps,
                target_size=None,  # Native resolution
                clip=self.clip_sampling,
                device=self.device
            )

            # 2. Unconditional generation (field prediction 100%)
            samples_uncond = self.model.sample(
                batch_size=images.shape[0],
                sparse_input=None,
                mask=None,
                steps=self.sampling_steps,
                target_size=None,
                clip=self.clip_sampling,
                device=self.device
            )

            # 3. Super-resolution at 64x64
            samples_sr_64 = self.model.sample(
                batch_size=images.shape[0],
                sparse_input=sparse_input,
                mask=cond_mask,
                steps=self.sampling_steps,
                target_size=(64, 64),
                clip=self.clip_sampling,
                device=self.device
            )

            # 4. Super-resolution at 96x96
            samples_sr_96 = self.model.sample(
                batch_size=images.shape[0],
                sparse_input=sparse_input,
                mask=cond_mask,
                steps=self.sampling_steps,
                target_size=(96, 96),
                clip=self.clip_sampling,
                device=self.device
            )

            # Save comparison grid (conditional at native resolution)
            save_comparison_grid(
                images, sparse_input, cond_mask, samples_cond, target_mask,
                os.path.join(self.save_dir, 'grids', f'step_{self.global_step}.png'),
                nrow=4
            )

            # Save individual components
            os.makedirs(os.path.join(self.save_dir, 'components'), exist_ok=True)

            save_image_grid(
                images,
                os.path.join(self.save_dir, 'components', f'gt_step_{self.global_step}.png'),
                nrow=4
            )

            save_image_grid(
                sparse_input,
                os.path.join(self.save_dir, 'components', f'sparse_step_{self.global_step}.png'),
                nrow=4
            )

            save_image_grid(
                samples_cond,
                os.path.join(self.save_dir, 'components', f'output_step_{self.global_step}.png'),
                nrow=4
            )

            save_image_grid(
                samples_uncond,
                os.path.join(self.save_dir, 'components', f'uncond_step_{self.global_step}.png'),
                nrow=4
            )

            # Save super-resolution outputs
            os.makedirs(os.path.join(self.save_dir, 'superres'), exist_ok=True)

            save_image_grid(
                samples_sr_64,
                os.path.join(self.save_dir, 'superres', f'sr64_step_{self.global_step}.png'),
                nrow=4
            )

            save_image_grid(
                samples_sr_96,
                os.path.join(self.save_dir, 'superres', f'sr96_step_{self.global_step}.png'),
                nrow=4
            )

            # Compute metrics
            metrics = compute_masked_metrics(samples_cond, images, target_mask)
            print(f"Step {self.global_step} | Conditional (32x32) - PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
            print(f"Step {self.global_step} | Super-resolution saved: 64x64, 96x96")

        self.model.train()
