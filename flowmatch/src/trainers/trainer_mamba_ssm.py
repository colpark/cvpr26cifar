"""
Trainer for Mamba SSM Flow Matching

Follows ref/mamba_diffusion.ipynb training approach:
- Flow matching with linear interpolation
- Target velocity prediction
- Heun ODE solver for sampling
"""
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid

from .trainer_coordinate_fm import CoordinateFMTrainer


class MambaSSMTrainer(CoordinateFMTrainer):
    """
    Trainer for Mamba SSM models with modified forward signature.

    The Mamba SSM model needs noisy values during training,
    so we override train_step to handle this correctly.
    """

    def train_step(self, batch):
        """
        Single training step with flow matching for Mamba SSM.

        Key difference from base trainer:
        - Mamba model accepts noisy_values parameter
        - We compute x_t = (1-t)*x_0 + t*z for target coordinates
        - Model predicts velocity v = z - x_0

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

        # Predict velocity - pass noisy values to Mamba model
        v_pred = self.model.model(
            target_coords, t, input_coords, input_values, noisy_values=x_t
        )

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss
