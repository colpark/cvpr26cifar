"""
DDPM (Denoising Diffusion Probabilistic Models) with masked conditioning.
Adapted from reference code for CIFAR-10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    """
    DDPM diffusion module with masked conditioning support.

    Args:
        model: Denoising U-Net
        image_size: Image resolution
        timesteps: Number of diffusion steps T
        loss_type: Loss function ('l1', 'l2', or 'huber')
    """
    def __init__(self, model, image_size=32, timesteps=1000, loss_type='l2'):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Linear beta schedule
        beta = self.linear_beta_schedule()
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.0)

        # Register buffers
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

        # For q(x_t | x_0)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))

        # For q(x_{t-1} | x_t, x_0)
        self.register_buffer('beta_tilde', beta * (1 - alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('mean_tilde_x0_coeff', beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer('mean_tilde_xt_coeff', torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))

        # For predicted x0
        self.register_buffer('sqrt_recip_alpha_bar', torch.sqrt(1.0 / alpha_bar))
        self.register_buffer('sqrt_recip_alpha_bar_min_1', torch.sqrt(1.0 / alpha_bar - 1))

        # For sampling
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1.0 / alpha))
        self.register_buffer('beta_over_sqrt_one_minus_alpha_bar', beta / torch.sqrt(1.0 - alpha_bar))

    def linear_beta_schedule(self):
        """Linear beta schedule as in DDPM paper."""
        scale = 1000 / self.timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)

    def q_sample(self, x0, t, noise):
        """
        Sample q(x_t | x_0) using reparameterization trick.

        Args:
            x0: (B, C, H, W) clean images
            t: (B,) timesteps
            noise: (B, C, H, W) Gaussian noise

        Returns:
            x_t: (B, C, H, W) noised images at timestep t
        """
        return (
            self.sqrt_alpha_bar[t][:, None, None, None] * x0 +
            self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise
        )

    def forward(self, img, sparse_input=None, mask=None, loss_mask=None):
        """
        Compute training loss.

        Args:
            img: (B, C, H, W) ground truth images in [-1, 1]
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask
            loss_mask: (B, C, H, W) target supervision mask (disjoint from mask)

        Returns:
            Loss value
        """
        B, C, H, W = img.shape
        device = img.device

        # Sample timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # Sample noise and create noised image
        noise = torch.randn_like(img)
        x_t = self.q_sample(img, t, noise)

        # Predict noise
        pred_noise = self.model(x_t, t, sparse_input=sparse_input, mask=mask)

        # Compute loss
        if self.loss_type == 'l1':
            raw_loss = F.l1_loss(pred_noise, noise, reduction='none')
        elif self.loss_type == 'l2':
            raw_loss = F.mse_loss(pred_noise, noise, reduction='none')
        elif self.loss_type == 'huber':
            raw_loss = F.smooth_l1_loss(pred_noise, noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply masked loss with small penalty on conditioned pixels
        if loss_mask is not None and mask is not None:
            lambda_cond = 0.05  # Small penalty on known pixels
            combined_mask = loss_mask + lambda_cond * mask
            combined_mask = combined_mask.clamp(max=1.0)
            loss = (raw_loss * combined_mask).sum() / (combined_mask.sum() + 1e-8)
        else:
            loss = raw_loss.mean()

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t, sparse_input=None, mask=None, clip=True):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            x_t: (B, C, H, W) noised images at timestep t
            t: int, current timestep
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask
            clip: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            x_{t-1}: (B, C, H, W) denoised images
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        pred_noise = self.model(x_t, t_batch, sparse_input=sparse_input, mask=mask)

        if clip:
            # Predict x_0 and clip
            x0 = self.sqrt_recip_alpha_bar[t] * x_t - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
            x0 = x0.clamp(-1.0, 1.0)
            # Recompute mean using clipped x0
            mean = self.mean_tilde_x0_coeff[t] * x0 + self.mean_tilde_xt_coeff[t] * x_t
        else:
            # Direct mean computation
            mean = self.sqrt_recip_alpha[t] * (x_t - self.beta_over_sqrt_one_minus_alpha_bar[t] * pred_noise)

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.beta_tilde[t]
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, batch_size, sparse_input=None, mask=None, clip=True, device='cuda'):
        """
        Generate samples using DDPM sampling.

        Args:
            batch_size: Number of samples to generate
            sparse_input: (B, C, H, W) sparse observations
            mask: (B, C, H, W) conditioning mask
            clip: Whether to clip during sampling
            device: Device to generate on

        Returns:
            (B, C, H, W) generated images in [-1, 1]
        """
        assert sparse_input is not None and mask is not None, "Must provide sparse_input and mask"

        C = sparse_input.shape[1]
        x_t = torch.randn(batch_size, C, self.image_size, self.image_size, device=device)

        for t in tqdm(reversed(range(self.timesteps)), desc='DDPM Sampling', total=self.timesteps, leave=False):
            x_t = self.p_sample(x_t, t, sparse_input=sparse_input, mask=mask, clip=clip)

        return x_t.clamp(-1.0, 1.0)


class DDIMSampler:
    """
    DDIM sampler for accelerated sampling.

    Args:
        diffusion: GaussianDiffusion instance
        ddim_steps: Number of sampling steps (< timesteps for acceleration)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
    """
    def __init__(self, diffusion, ddim_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        self.eta = eta

        # Create sub-sequence of timesteps
        self.tau = torch.linspace(0, diffusion.timesteps - 1, ddim_steps, dtype=torch.long)

        # Precompute coefficients
        alpha_bar = diffusion.alpha_bar[self.tau]
        alpha_bar_prev = F.pad(diffusion.alpha_bar[self.tau[:-1]], pad=(1, 0), value=1.0)

        self.sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
        self.coeff = (1 - alpha_bar_prev - self.sigma ** 2).sqrt()
        self.sqrt_alpha_bar_prev = alpha_bar_prev.sqrt()

    @torch.no_grad()
    def sample(self, batch_size, sparse_input=None, mask=None, clip=True, device='cuda'):
        """
        Generate samples using DDIM sampling.

        Returns:
            (B, C, H, W) generated images in [-1, 1]
        """
        assert sparse_input is not None and mask is not None, "Must provide sparse_input and mask"

        C = sparse_input.shape[1]
        x_t = torch.randn(batch_size, C, self.diffusion.image_size, self.diffusion.image_size, device=device)

        for i in tqdm(reversed(range(self.ddim_steps)), desc='DDIM Sampling', total=self.ddim_steps, leave=False):
            t = self.tau[i].item()
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.diffusion.model(x_t, t_batch, sparse_input=sparse_input, mask=mask)

            # Predict x_0
            x0 = (
                self.diffusion.sqrt_recip_alpha_bar[t] * x_t -
                self.diffusion.sqrt_recip_alpha_bar_min_1[t] * pred_noise
            )

            if clip:
                x0 = x0.clamp(-1.0, 1.0)
                # Recompute noise from clipped x0
                pred_noise = (
                    (self.diffusion.sqrt_recip_alpha_bar[t] * x_t - x0) /
                    self.diffusion.sqrt_recip_alpha_bar_min_1[t]
                )

            # Compute x_{t-1}
            mean = self.sqrt_alpha_bar_prev[i] * x0 + self.coeff[i] * pred_noise

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + self.sigma[i] * noise
            else:
                x_t = mean

        return x_t.clamp(-1.0, 1.0)
