"""
Trainer for coordinate-based Flow Matching models.

Key features:
- Trains on coordinate-value pairs
- Dense supervision on all pixels by default
- Multi-resolution sampling for evaluation
- Scale-invariant generation
"""
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid

from .trainer_base import BaseTrainer


class CoordinateFMTrainer(BaseTrainer):
    """
    Trainer for coordinate-based Flow Matching.

    Args:
        flow: RectifiedFlow instance with coordinate-based model
        train_loader: DataLoader returning coordinate-value pairs
        optimizer_config: Optimizer configuration
        device: Device to train on
        save_dir: Directory to save checkpoints and samples
        sampling_steps: Number of ODE integration steps
        clip_sampling: Whether to clip samples to [0, 1]
        max_grad_norm: Maximum gradient norm for clipping
        eval_resolutions: List of resolutions for multi-scale evaluation
    """
    def __init__(self, flow, train_loader, optimizer_config, device='cuda',
                 save_dir='./results', sampling_steps=50, clip_sampling=True,
                 max_grad_norm=1.0, eval_resolutions=[32, 64, 96]):
        super().__init__(
            flow, train_loader, optimizer_config, device, save_dir,
            sparsity_controller=None  # Not used in coordinate-based
        )
        self.sampling_steps = sampling_steps
        self.clip_sampling = clip_sampling
        self.max_grad_norm = max_grad_norm
        self.eval_resolutions = eval_resolutions

        # Create coordinate grids for each evaluation resolution
        self.coord_grids = {}
        for res in eval_resolutions:
            self.coord_grids[res] = self._create_coord_grid(res)

    def _create_coord_grid(self, resolution):
        """Create uniform coordinate grid for given resolution."""
        y = torch.linspace(0, 1, resolution)
        x = torch.linspace(0, 1, resolution)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (res², 2)
        return coords.to(self.device)

    def train_step(self, batch):
        """
        Single training step with coordinate-based data.

        Args:
            batch: Dict with 'input_coords', 'input_values',
                               'target_coords', 'target_values'

        Returns:
            Loss value
        """
        input_coords = batch['input_coords'].to(self.device)
        input_values = batch['input_values'].to(self.device)
        target_coords = batch['target_coords'].to(self.device)
        target_values = batch['target_values'].to(self.device)

        B = input_coords.shape[0]

        # Sample random time
        t = torch.rand(B, device=self.device)

        # Sample noise
        z = torch.randn_like(target_values)

        # Flow matching: x_t = (1-t)·x_0 + t·z
        t_expanded = t.view(B, 1, 1)
        x_t = (1 - t_expanded) * target_values + t_expanded * z

        # Target velocity: v* = z - x_0
        v_target = z - target_values

        # Predict velocity
        # Note: We pass x_t (noisy target values) along with target_coords
        # The model will learn to denoise based on input_coords/values
        v_pred = self.model.model(target_coords, t, input_coords, input_values)

        # However, we need to condition v_pred on the noisy state x_t
        # Let's create a modified forward that takes noisy values at query coords

        # Actually, for proper flow matching, we need to concatenate
        # noisy values with query coordinates as input
        # Let me fix this in the loss computation

        # For now, let's use a simpler approach:
        # Concatenate input coords/values with noisy target coords/values
        # and predict velocity

        # Combine inputs and noisy targets
        combined_coords = torch.cat([input_coords, target_coords], dim=1)
        combined_values = torch.cat([input_values, x_t], dim=1)

        # Predict velocity for target positions
        v_pred = self.model.model(target_coords, t, combined_coords, combined_values)

        # Compute loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample_at_resolution(self, input_coords, input_values, resolution, steps=50):
        """
        Generate samples at specified resolution.

        Args:
            input_coords: (B, N_input, 2) sparse input positions
            input_values: (B, N_input, 3) RGB values at inputs
            resolution: Target resolution (e.g., 32, 64, 96)
            steps: Number of ODE integration steps

        Returns:
            (B, 3, resolution, resolution) generated images
        """
        B = input_coords.shape[0]

        # Get coordinate grid for this resolution
        query_coords = self.coord_grids[resolution]  # (res², 2)
        query_coords = query_coords.unsqueeze(0).expand(B, -1, -1)  # (B, res², 2)

        # Initialize from noise
        x = torch.randn(B, query_coords.shape[1], 3, device=self.device)

        # ODE integration (Euler method)
        dt = 1.0 / steps
        for i in range(steps):
            t_current = 1.0 - i / steps
            t_batch = torch.full((B,), t_current, device=self.device)

            # Combine inputs with current state for conditioning
            combined_coords = torch.cat([input_coords, query_coords], dim=1)
            combined_values = torch.cat([input_values, x], dim=1)

            # Predict velocity
            v = self.model.model(query_coords, t_batch, combined_coords, combined_values)

            # Euler step
            x = x - dt * v

        # Clip if requested
        if self.clip_sampling:
            x = x.clamp(0.0, 1.0)

        # Reshape to image
        images = x.view(B, resolution, resolution, 3).permute(0, 3, 1, 2)

        return images

    def generate_samples(self):
        """Generate samples at multiple resolutions."""
        self.model.eval()

        # Get a batch from training data
        batch = next(iter(self.train_loader))
        input_coords = batch['input_coords'][:4].to(self.device)
        input_values = batch['input_values'][:4].to(self.device)
        full_images = batch['full_image'][:4].to(self.device)

        samples = {}

        # Generate at each resolution
        for res in self.eval_resolutions:
            samples[res] = self.sample_at_resolution(
                input_coords, input_values, res, self.sampling_steps
            )

        # Save visualizations
        self._save_multi_resolution_samples(samples, full_images, input_coords, input_values)

        self.model.train()

    def _save_multi_resolution_samples(self, samples, full_images, input_coords, input_values):
        """Save multi-resolution comparison."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, len(self.eval_resolutions) + 2,
                                figsize=(4 * (len(self.eval_resolutions) + 2), 16))

        for i in range(4):
            # Ground truth
            axes[i, 0].imshow(full_images[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title('Ground Truth (32×32)' if i == 0 else '', fontsize=10)
            axes[i, 0].axis('off')

            # Sparse input visualization
            sparse_img = torch.zeros(3, 32, 32, device=self.device)
            input_coords_rescaled = (input_coords[i] * 31).long()  # [0,1] → [0,31]
            for j in range(input_coords_rescaled.shape[0]):
                y, x = input_coords_rescaled[j, 1], input_coords_rescaled[j, 0]
                sparse_img[:, y, x] = input_values[i, j]
            axes[i, 1].imshow(sparse_img.permute(1, 2, 0).cpu().numpy())
            axes[i, 1].set_title('Sparse Input (20%)' if i == 0 else '', fontsize=10)
            axes[i, 1].axis('off')

            # Generated samples at each resolution
            for col_idx, res in enumerate(self.eval_resolutions):
                img = samples[res][i].permute(1, 2, 0).cpu().numpy()
                axes[i, col_idx + 2].imshow(np.clip(img, 0, 1))
                title = f'Generated {res}×{res}'
                if res == 32:
                    title += '\n(Dense Training)'
                elif res > 32:
                    title += f'\n({res//32}× SR)'
                axes[i, col_idx + 2].set_title(title if i == 0 else '', fontsize=10)
                axes[i, col_idx + 2].axis('off')

        plt.suptitle(
            f'Coordinate-Based Flow Matching - Step {self.global_step}\n'
            f'Dense supervision on all pixels, scale-invariant generation',
            fontsize=14, y=0.995
        )
        plt.tight_layout()

        # Save
        save_path = os.path.join(self.samples_dir, f'multiscale_step_{self.global_step:06d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save grids for each resolution
        for res in self.eval_resolutions:
            grid = make_grid(samples[res], nrow=2, normalize=False, value_range=(0, 1))
            save_path = os.path.join(
                self.samples_dir,
                f'step_{self.global_step:06d}_res_{res}.png'
            )
            save_image(grid, save_path)

    def train(self, num_steps, log_every=100, sample_every=2500, save_every=10000):
        """Training loop."""
        print(f"Starting coordinate-based Flow Matching training for {num_steps} steps")
        print(f"Dense supervision: {self.train_loader.dataset.target_ratio * 100:.0f}% of pixels")
        print(f"Evaluation resolutions: {self.eval_resolutions}")

        self.model.train()
        pbar = tqdm(total=num_steps, initial=self.global_step)

        while self.global_step < num_steps:
            for batch in self.train_loader:
                if self.global_step >= num_steps:
                    break

                # Training step
                loss = self.train_step(batch)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                self.optimizer.step()

                # Logging
                self.loss_history.append(loss.item())
                if self.global_step % log_every == 0:
                    avg_loss = np.mean(self.loss_history[-log_every:])
                    pbar.set_description(f"Loss: {avg_loss:.6f}")

                # Sampling
                if self.global_step % sample_every == 0:
                    print(f"\nGenerating samples at step {self.global_step}")
                    self.generate_samples()

                # Checkpointing
                if self.global_step % save_every == 0 and self.global_step > 0:
                    self.save_checkpoint()

                self.global_step += 1
                pbar.update(1)

        pbar.close()
        print(f"\nTraining completed! Final step: {self.global_step}")
        self.save_checkpoint()
