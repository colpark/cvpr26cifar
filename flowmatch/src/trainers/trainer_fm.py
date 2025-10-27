"""
Flow Matching trainer with masked conditioning.
"""
import torch
from .trainer_base import BaseTrainer


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
