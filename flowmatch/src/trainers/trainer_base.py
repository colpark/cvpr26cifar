"""
Base trainer with common functionality for DDPM and Flow Matching.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm
import os
from ..utils.viz import save_comparison_grid, save_image_grid
from ..utils.metrics import compute_masked_metrics


class BaseTrainer:
    """
    Base trainer class with common training loop and logging.

    Args:
        model: Diffusion or Flow model
        train_loader: Training DataLoader
        val_loader: Optional validation DataLoader
        optimizer_config: Dict with optimizer params
        device: Training device
        save_dir: Directory for checkpoints and samples
        sparsity_controller: SparsityController instance
    """
    def __init__(self, model, train_loader, val_loader=None,
                 optimizer_config=None, device='cuda', save_dir='./results',
                 sparsity_controller=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.sparsity_controller = sparsity_controller

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'grids'), exist_ok=True)

        # Optimizer
        opt_config = optimizer_config or {}
        lr = opt_config.get('lr', 2e-4)
        weight_decay = opt_config.get('weight_decay', 0.0)
        betas = opt_config.get('betas', (0.9, 0.999))

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def train_step(self, batch):
        """
        Single training step. Override in subclasses.

        Args:
            batch: Tuple of (images, indices)

        Returns:
            Loss value
        """
        raise NotImplementedError

    def sample(self, batch_size, sparse_input, mask):
        """
        Generate samples. Override in subclasses.

        Args:
            batch_size: Number of samples
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask

        Returns:
            (B, C, H, W) generated images
        """
        raise NotImplementedError

    def train(self, num_steps, log_every=100, sample_every=1000, save_every=5000):
        """
        Main training loop.

        Args:
            num_steps: Total training steps
            log_every: Log interval
            sample_every: Sampling interval
            save_every: Checkpoint save interval
        """
        self.model.train()
        pbar = tqdm(total=num_steps, desc='Training')

        train_iter = iter(self.train_loader)
        running_loss = 0.0
        running_count = 0

        while self.global_step < num_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1

            # Training step
            loss = self.train_step(batch)

            # Logging
            running_loss += loss
            running_count += 1

            if self.global_step % log_every == 0 and running_count > 0:
                avg_loss = running_loss / running_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': self.global_step})
                running_loss = 0.0
                running_count = 0

            # Sampling (include step 0 for debugging)
            if self.global_step % sample_every == 0:
                self.generate_samples()

            # Checkpointing
            if self.global_step % save_every == 0 and self.global_step > 0:
                self.save_checkpoint('latest.pt')

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint('best.pt')

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print(f"Training completed. Best loss: {self.best_loss:.4f}")

    def generate_samples(self):
        """Generate and save sample images with comprehensive visualizations."""
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

            # 1. Conditional generation (with sparse conditioning)
            samples_cond = self.sample(images.shape[0], sparse_input, cond_mask)

            # 2. Unconditional generation (field prediction 100% - no conditioning)
            samples_uncond = self.sample(images.shape[0], None, None)

            # Save comparison grid (conditional)
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

            # Compute metrics
            metrics = compute_masked_metrics(samples_cond, images, target_mask)
            print(f"Step {self.global_step} | Conditional - PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")

        self.model.train()

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        save_path = os.path.join(self.save_dir, 'checkpoints', filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        load_path = os.path.join(self.save_dir, 'checkpoints', filename)
        if not os.path.exists(load_path):
            print(f"Checkpoint not found: {load_path}")
            return

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        print(f"Checkpoint loaded: {load_path} (step {self.global_step})")
