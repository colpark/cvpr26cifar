"""
Rectified Flow (Flow Matching) with masked conditioning and zero-shot super-resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class RectifiedFlow(nn.Module):
    """
    Rectified Flow (straight-path flow matching) with masked conditioning.

    The key insight: on known pixels (mask=1), we keep them fixed in the flow.
    - z* = (1-mask)·z + mask·x0  (noise only on unknown pixels)
    - x_t = (1-t)·x0 + t·z*
    - v* = z* - x0 = (1-mask)·(z - x0)  (velocity is zero on known pixels)

    Args:
        model: Velocity prediction network (U-Net or transformer)
        image_size: Base training resolution
    """
    def __init__(self, model, image_size=32):
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(self, x0, sparse_input=None, mask=None, loss_mask=None):
        """
        Compute flow matching training loss.

        Args:
            x0: (B, C, H, W) ground truth images in [-1, 1]
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask (1 = known, 0 = unknown)
            loss_mask: (B, C, H, W) target supervision mask

        Returns:
            Loss value
        """
        B, C, H, W = x0.shape
        device = x0.device

        # Sample continuous time t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)

        # Sample noise
        z = torch.randn_like(x0)

        # Mask-aware noise: known pixels stay at x0
        if mask is not None:
            z_star = (1 - mask) * z + mask * x0
        else:
            z_star = z

        # Interpolate: x_t = (1-t)·x0 + t·z*
        t_expanded = t[:, None, None, None]
        x_t = (1 - t_expanded) * x0 + t_expanded * z_star

        # Target velocity: v* = z* - x0
        v_target = z_star - x0

        # Predict velocity
        v_pred = self.model(x_t, t, sparse_input=sparse_input, mask=mask)

        # Compute loss (MSE on velocity)
        raw_loss = F.mse_loss(v_pred, v_target, reduction='none')

        # Apply masked loss
        # Note: For FM, λ can be 0 since v* is already zero on known pixels
        if loss_mask is not None:
            # Focus supervision on unknown pixels
            loss = (raw_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        else:
            loss = raw_loss.mean()

        return loss

    @torch.no_grad()
    def sample(self, batch_size, sparse_input=None, mask=None, steps=50,
               target_size=None, clip=True, device='cuda'):
        """
        Generate samples using probability-flow ODE (Euler method).

        Args:
            batch_size: Number of samples
            sparse_input: (B, C, H, W) sparse observations (None for unconditional)
            mask: (B, C, H, W) conditioning mask (None for unconditional)
            steps: Number of ODE integration steps
            target_size: Optional (H, W) for zero-shot super-resolution
            clip: Whether to clip output to [-1, 1]
            device: Device to generate on

        Returns:
            (B, C, H, W) or (B, C, target_H, target_W) generated images
        """
        # Determine number of channels
        if sparse_input is not None:
            C = sparse_input.shape[1]
        else:
            C = 3  # RGB for CIFAR-10

        # Determine output size
        if target_size is None:
            H, W = self.image_size, self.image_size
        else:
            H, W = target_size

        # Upsample sparse and mask for SR if needed
        # For zero-shot SR: model trained at 32x32 generalizes to higher resolutions
        # via resolution-agnostic positional encoding (Fourier features)
        if sparse_input is not None and mask is not None:
            if H != sparse_input.shape[-2] or W != sparse_input.shape[-1]:
                # Upsample conditioning to target resolution (nearest neighbor preserves sparse pixels)
                sparse_input = F.interpolate(sparse_input, size=(H, W), mode='nearest')
                mask = F.interpolate(mask.float(), size=(H, W), mode='nearest')

        # Initialize from noise
        x = torch.randn(batch_size, C, H, W, device=device)

        # ODE integration: dx/dt = v(x, t)
        # Euler: x_{k+1} = x_k + (1/steps) * v(x_k, t_k)
        # t goes from 1 → 0 (reverse time)
        dt = 1.0 / steps

        for i in tqdm(range(steps), desc='Flow Matching Sampling', leave=False):
            t_current = 1.0 - i / steps
            t_batch = torch.full((batch_size,), t_current, device=device)

            # Predict velocity
            v = self.model(x, t_batch, sparse_input=sparse_input, mask=mask)

            # Euler step: move backwards (t: 1→0, so velocity is -v)
            x = x - dt * v

            # Hard projection: enforce known pixels (skip for unconditional)
            if mask is not None and sparse_input is not None:
                x = mask * sparse_input + (1 - mask) * x

        if clip:
            x = x.clamp(-1.0, 1.0)

        return x


class RectifiedFlowODESampler:
    """
    Alternative ODE sampler with different integration schemes.

    Args:
        flow: RectifiedFlow instance
        method: Integration method ('euler', 'heun', 'rk4')
    """
    def __init__(self, flow, method='euler'):
        self.flow = flow
        self.method = method

    @torch.no_grad()
    def sample(self, batch_size, sparse_input=None, mask=None, steps=50,
               target_size=None, clip=True, device='cuda'):
        """
        Sample using specified ODE integration method.
        """
        if self.method == 'euler':
            return self.flow.sample(
                batch_size, sparse_input, mask, steps, target_size, clip, device
            )
        elif self.method == 'heun':
            return self._heun_sample(
                batch_size, sparse_input, mask, steps, target_size, clip, device
            )
        elif self.method == 'rk4':
            return self._rk4_sample(
                batch_size, sparse_input, mask, steps, target_size, clip, device
            )
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

    @torch.no_grad()
    def _heun_sample(self, batch_size, sparse_input, mask, steps, target_size, clip, device):
        """Heun's method (2nd order)."""
        # Determine number of channels
        if sparse_input is not None:
            C = sparse_input.shape[1]
        else:
            C = 3  # RGB for CIFAR-10

        H, W = target_size if target_size else (self.flow.image_size, self.flow.image_size)

        # Upsample sparse and mask for SR if needed
        if sparse_input is not None and mask is not None:
            if H != sparse_input.shape[-2] or W != sparse_input.shape[-1]:
                sparse_input = F.interpolate(sparse_input, size=(H, W), mode='nearest')
                mask = F.interpolate(mask.float(), size=(H, W), mode='nearest')

        x = torch.randn(batch_size, C, H, W, device=device)
        dt = 1.0 / steps

        for i in tqdm(range(steps), desc='Heun Sampling', leave=False):
            t_current = 1.0 - i / steps
            t_batch = torch.full((batch_size,), t_current, device=device)

            # First velocity evaluation
            v1 = self.flow.model(x, t_batch, sparse_input=sparse_input, mask=mask)

            # Predict next point
            x_next = x - dt * v1
            if mask is not None and sparse_input is not None:
                x_next = mask * sparse_input + (1 - mask) * x_next

            # Second velocity evaluation
            t_next = t_current - dt
            t_next_batch = torch.full((batch_size,), t_next, device=device)
            v2 = self.flow.model(x_next, t_next_batch, sparse_input=sparse_input, mask=mask)

            # Average velocities
            v_avg = (v1 + v2) / 2
            x = x - dt * v_avg
            if mask is not None and sparse_input is not None:
                x = mask * sparse_input + (1 - mask) * x

        return x.clamp(-1.0, 1.0) if clip else x

    @torch.no_grad()
    def _rk4_sample(self, batch_size, sparse_input, mask, steps, target_size, clip, device):
        """4th-order Runge-Kutta method."""
        # Determine number of channels
        if sparse_input is not None:
            C = sparse_input.shape[1]
        else:
            C = 3  # RGB for CIFAR-10

        H, W = target_size if target_size else (self.flow.image_size, self.flow.image_size)

        # Upsample sparse and mask for SR if needed
        if sparse_input is not None and mask is not None:
            if H != sparse_input.shape[-2] or W != sparse_input.shape[-1]:
                sparse_input = F.interpolate(sparse_input, size=(H, W), mode='nearest')
                mask = F.interpolate(mask.float(), size=(H, W), mode='nearest')

        x = torch.randn(batch_size, C, H, W, device=device)
        dt = 1.0 / steps

        for i in tqdm(range(steps), desc='RK4 Sampling', leave=False):
            t_current = 1.0 - i / steps
            t_batch = torch.full((batch_size,), t_current, device=device)

            # k1
            k1 = self.flow.model(x, t_batch, sparse_input=sparse_input, mask=mask)

            # k2
            x_k2 = x - 0.5 * dt * k1
            if mask is not None and sparse_input is not None:
                x_k2 = mask * sparse_input + (1 - mask) * x_k2
            t_k2 = torch.full((batch_size,), t_current - 0.5 * dt, device=device)
            k2 = self.flow.model(x_k2, t_k2, sparse_input=sparse_input, mask=mask)

            # k3
            x_k3 = x - 0.5 * dt * k2
            if mask is not None and sparse_input is not None:
                x_k3 = mask * sparse_input + (1 - mask) * x_k3
            k3 = self.flow.model(x_k3, t_k2, sparse_input=sparse_input, mask=mask)

            # k4
            x_k4 = x - dt * k3
            if mask is not None and sparse_input is not None:
                x_k4 = mask * sparse_input + (1 - mask) * x_k4
            t_k4 = torch.full((batch_size,), t_current - dt, device=device)
            k4 = self.flow.model(x_k4, t_k4, sparse_input=sparse_input, mask=mask)

            # Combine
            x = x - (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            if mask is not None and sparse_input is not None:
                x = mask * sparse_input + (1 - mask) * x

        return x.clamp(-1.0, 1.0) if clip else x
