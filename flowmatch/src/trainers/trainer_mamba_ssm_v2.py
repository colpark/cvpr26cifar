"""
Trainer for Mamba SSM Flow Matching V2 - Reference Implementation

Matches ref/mamba_diffusion.ipynb exactly:
- Correct forward signature: forward(noisy_values, query_coords, t, ...)
- Dataset format: output_coords/output_values
- Flow matching training loop
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from .trainer_base import BaseTrainer


class MambaSSMTrainerV2(BaseTrainer):
    """
    Trainer for Mamba SSM Flow Matching following reference implementation

    Key differences from standard trainer:
    - Forward signature: forward(noisy_values, query_coords, t, input_coords, input_values)
    - Uses 'output_coords'/'output_values' (not 'target_*')
    - Flow matching: x_t = (1-t)*x_0 + t*x_1, predict velocity v = x_1 - x_0
    """

    def __init__(
        self,
        flow,
        train_loader,
        optimizer_config,
        device,
        save_dir,
        sampling_steps=50,
        clip_sampling=True,
        max_grad_norm=1.0,
        eval_resolutions=[32, 64, 96]
    ):
        super().__init__(
            model=flow,
            train_loader=train_loader,
            val_loader=None,  # Not used for coordinate-based
            optimizer_config=optimizer_config,
            device=device,
            save_dir=save_dir
        )

        self.flow = flow
        self.sampling_steps = sampling_steps
        self.clip_sampling = clip_sampling
        self.max_grad_norm = max_grad_norm
        self.eval_resolutions = eval_resolutions

        # Loss tracking
        self.loss_history = []
        self.global_step = 0

        # Directories
        self.samples_dir = os.path.join(save_dir, 'samples')
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(self, batch):
        """
        Single training step with flow matching

        Args:
            batch: Dict with keys:
                - 'input_coords': (B, N_in, 2)
                - 'input_values': (B, N_in, 3)
                - 'output_coords': (B, N_out, 2)  # CRITICAL: 'output_coords', not 'target_coords'!
                - 'output_values': (B, N_out, 3)   # CRITICAL: 'output_values', not 'target_values'!
        """
        self.flow.train()

        # Extract data (using correct field names)
        input_coords = batch['input_coords'].to(self.device)
        input_values = batch['input_values'].to(self.device)
        output_coords = batch['output_coords'].to(self.device)  # Correct field name!
        output_values = batch['output_values'].to(self.device)  # Correct field name!

        B = input_coords.shape[0]

        # Sample random timestep t ~ U[0, 1]
        t = torch.rand(B, device=self.device)

        # Flow matching
        x_0 = torch.randn_like(output_values)  # Noise
        x_1 = output_values  # Data

        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_broadcast = t.view(B, 1, 1)
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1

        # Target velocity: v = x_1 - x_0
        target_velocity = x_1 - x_0

        # Predict velocity with CORRECT forward signature
        # CRITICAL: forward(noisy_values, query_coords, t, input_coords, input_values)
        pred_velocity = self.flow(x_t, output_coords, t, input_coords, input_values)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def sample_ode(self, output_coords, input_coords, input_values, num_steps=None):
        """
        Sample using Heun ODE solver (2nd order Runge-Kutta)

        Args:
            output_coords: (B, N_out, 2) - query coordinates
            input_coords: (B, N_in, 2) - input coordinates
            input_values: (B, N_in, 3) - input RGB values
            num_steps: Number of ODE solver steps

        Returns:
            x_t: (B, N_out, 3) - generated RGB values
        """
        self.flow.eval()

        if num_steps is None:
            num_steps = self.sampling_steps

        B, N_out = output_coords.shape[0], output_coords.shape[1]

        # Start from noise
        x_t = torch.randn(B, N_out, 3, device=self.device)

        # Heun solver
        dt = 1.0 / num_steps
        ts = torch.linspace(0, 1 - dt, num_steps)

        for t_val in ts:
            t = torch.full((B,), t_val.item(), device=self.device)
            t_next = torch.full((B,), t_val.item() + dt, device=self.device)

            # First velocity estimate
            v1 = self.flow(x_t, output_coords, t, input_coords, input_values)

            # Predict next state
            x_next_pred = x_t + dt * v1

            # Second velocity estimate
            v2 = self.flow(x_next_pred, output_coords, t_next, input_coords, input_values)

            # Average velocities (Heun's method)
            x_t = x_t + dt * 0.5 * (v1 + v2)

        if self.clip_sampling:
            x_t = torch.clamp(x_t, 0, 1)

        return x_t

    @torch.no_grad()
    def visualize_samples(self, step):
        """Generate visualization samples"""
        self.flow.eval()

        # Get a batch from training data
        batch = next(iter(self.train_loader))
        input_coords = batch['input_coords'][:4].to(self.device)
        input_values = batch['input_values'][:4].to(self.device)
        output_coords = batch['output_coords'][:4].to(self.device)
        output_values = batch['output_values'][:4].to(self.device)
        full_images = batch['full_image'][:4].to(self.device)

        # Sample
        pred_values = self.sample_ode(output_coords, input_coords, input_values)

        # Visualize
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))

        for i in range(4):
            # Ground truth
            axes[i, 0].imshow(full_images[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title('Ground Truth' if i == 0 else '', fontsize=10)
            axes[i, 0].axis('off')

            # Sparse input
            input_img = torch.zeros(3, 32, 32, device=self.device)
            if 'input_indices' in batch:
                input_idx = batch['input_indices'][i]
                input_img.view(3, -1)[:, input_idx] = input_values[i].T
            axes[i, 1].imshow(input_img.permute(1, 2, 0).cpu().numpy())
            axes[i, 1].set_title('Sparse Input (20%)' if i == 0 else '', fontsize=10)
            axes[i, 1].axis('off')

            # Sparse target
            target_img = torch.zeros(3, 32, 32, device=self.device)
            if 'output_indices' in batch:
                output_idx = batch['output_indices'][i]
                target_img.view(3, -1)[:, output_idx] = output_values[i].T
            axes[i, 2].imshow(target_img.permute(1, 2, 0).cpu().numpy())
            axes[i, 2].set_title('Sparse Target (20%)' if i == 0 else '', fontsize=10)
            axes[i, 2].axis('off')

            # Sparse prediction
            pred_img = torch.zeros(3, 32, 32, device=self.device)
            if 'output_indices' in batch:
                pred_img.view(3, -1)[:, output_idx] = pred_values[i].T
            axes[i, 3].imshow(np.clip(pred_img.permute(1, 2, 0).cpu().numpy(), 0, 1))
            axes[i, 3].set_title('Sparse Prediction' if i == 0 else '', fontsize=10)
            axes[i, 3].axis('off')

        plt.suptitle(f'Mamba SSM V2 - Step {step}', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.samples_dir}/sample_step_{step:06d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, num_steps, log_every=100, sample_every=1000, save_every=5000):
        """
        Training loop

        Args:
            num_steps: Total training steps
            log_every: Log loss every N steps
            sample_every: Generate samples every N steps
            save_every: Save checkpoint every N steps
        """
        print(f"Starting Mamba SSM V2 training for {num_steps} steps")
        print(f"  Input ratio: 20% (sparse)")
        print(f"  Target ratio: 20% (sparse, non-overlapping)")
        print(f"  Sampling steps: {self.sampling_steps}")

        step = 0
        epoch = 0
        running_loss = 0.0

        while step < num_steps:
            epoch += 1
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for batch in progress_bar:
                loss = self.train_step(batch)
                running_loss += loss

                if (step + 1) % log_every == 0:
                    avg_loss = running_loss / log_every
                    self.loss_history.append(avg_loss)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.6f}',
                        'step': f'{step+1}/{num_steps}'
                    })
                    running_loss = 0.0

                if (step + 1) % sample_every == 0:
                    self.visualize_samples(step + 1)

                if (step + 1) % save_every == 0:
                    self.save_checkpoint(epoch, step + 1, avg_loss if running_loss == 0 else loss)

                step += 1
                if step >= num_steps:
                    break

        print(f"\n✓ Training complete! Total steps: {step}")
        print(f"  Final loss: {self.loss_history[-1] if self.loss_history else loss:.6f}")

        # Save final checkpoint
        self.save_checkpoint(epoch, step, self.loss_history[-1] if self.loss_history else loss)

    def save_checkpoint(self, epoch, step, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.flow.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'loss_history': self.loss_history
        }
        path = f'{self.checkpoint_dir}/checkpoint_step_{step:06d}.pth'
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.global_step = checkpoint.get('step', 0)
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.6f}")
