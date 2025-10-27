"""
Test shape correctness for all model forward passes.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.unet_ddpm import UNetDDPM
from src.models.unet_fm import UNetFM
from src.models.dit_fm import DiTFM
from src.models.perceiver_io_fm import PerceiverIOFM
from src.diffusion.ddpm import GaussianDiffusion
from src.diffusion.rectified_flow import RectifiedFlow


def test_unet_ddpm_shapes():
    """Test UNetDDPM forward pass shapes."""
    print("Testing UNetDDPM shapes...")

    model = UNetDDPM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8
    )

    # Test with concatenated input
    B, C, H, W = 2, 9, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    out = model(x, t)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    # Test with separate inputs
    x_t = torch.randn(B, 3, H, W)
    sparse_input = torch.randn(B, 3, H, W)
    mask = torch.randn(B, 1, H, W)

    out = model(x_t, t, sparse_input=sparse_input, mask=mask)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    print("✓ UNetDDPM shapes correct")


def test_unet_fm_shapes():
    """Test UNetFM forward pass shapes."""
    print("Testing UNetFM shapes...")

    model = UNetFM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8,
        time_emb_scale=16.0
    )

    # Test with continuous time
    B, C, H, W = 2, 9, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)  # Continuous time [0, 1]

    out = model(x, t)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    # Test with separate inputs
    x_t = torch.randn(B, 3, H, W)
    sparse_input = torch.randn(B, 3, H, W)
    mask = torch.randn(B, 1, H, W)

    out = model(x_t, t, sparse_input=sparse_input, mask=mask)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    print("✓ UNetFM shapes correct")


def test_dit_fm_shapes():
    """Test DiTFM forward pass shapes."""
    print("Testing DiTFM shapes...")

    model = DiTFM(
        patch_size=4,
        dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        channel=3,
        dropout=0.0,
        window_size=8
    )

    # Test at 32x32
    B, H, W = 2, 32, 32
    x_t = torch.randn(B, 3, H, W)
    sparse_input = torch.randn(B, 3, H, W)
    mask = torch.randn(B, 1, H, W)
    t = torch.rand(B)

    out = model(x_t, t, sparse_input=sparse_input, mask=mask)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    # Test at different resolution (64x64) - resolution agnostic
    H2, W2 = 64, 64
    x_t_64 = torch.randn(B, 3, H2, W2)
    sparse_input_64 = torch.randn(B, 3, H2, W2)
    mask_64 = torch.randn(B, 1, H2, W2)

    out_64 = model(x_t_64, t, sparse_input=sparse_input_64, mask=mask_64)
    assert out_64.shape == (B, 3, H2, W2), f"Expected shape {(B, 3, H2, W2)}, got {out_64.shape}"

    print("✓ DiTFM shapes correct (including resolution-agnostic)")


def test_perceiver_io_fm_shapes():
    """Test PerceiverIOFM forward pass shapes."""
    print("Testing PerceiverIOFM shapes...")

    model = PerceiverIOFM(
        channel=3,
        latent_dim=128,
        num_latents=64,
        depth=2,
        num_heads=4,
        head_dim=32,
        input_fourier_features=32,
        query_fourier_features=32
    )

    # Test at 32x32
    B, H, W = 2, 32, 32
    x_t = torch.randn(B, 3, H, W)
    sparse_input = torch.randn(B, 3, H, W)
    mask = torch.randn(B, 1, H, W)
    t = torch.rand(B)

    out = model(x_t, t, sparse_input=sparse_input, mask=mask)
    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"

    # Test at different resolution (48x48) - resolution agnostic
    H2, W2 = 48, 48
    x_t_48 = torch.randn(B, 3, H2, W2)
    sparse_input_48 = torch.randn(B, 3, H2, W2)
    mask_48 = torch.randn(B, 1, H2, W2)

    out_48 = model(x_t_48, t, sparse_input=sparse_input_48, mask=mask_48)
    assert out_48.shape == (B, 3, H2, W2), f"Expected shape {(B, 3, H2, W2)}, got {out_48.shape}"

    print("✓ PerceiverIOFM shapes correct (including resolution-agnostic)")


def test_ddpm_forward():
    """Test GaussianDiffusion forward pass."""
    print("Testing GaussianDiffusion forward...")

    backbone = UNetDDPM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8
    )

    model = GaussianDiffusion(
        model=backbone,
        image_size=32,
        timesteps=1000,
        loss_type='l2'
    )

    B, C, H, W = 2, 3, 32, 32
    img = torch.randn(B, C, H, W)
    sparse_input = torch.randn(B, C, H, W)
    mask = torch.rand(B, 1, H, W) > 0.5
    loss_mask = 1.0 - mask.float()

    loss = model(img, sparse_input=sparse_input, mask=mask.float(), loss_mask=loss_mask)
    assert isinstance(loss.item(), float), "Loss should be a scalar"

    print("✓ GaussianDiffusion forward pass correct")


def test_rectified_flow_forward():
    """Test RectifiedFlow forward pass."""
    print("Testing RectifiedFlow forward...")

    backbone = UNetFM(
        dim=64,
        image_size=32,
        dim_multiply=(1, 2, 4),
        channel=3,
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        groups=8,
        time_emb_scale=16.0
    )

    model = RectifiedFlow(
        model=backbone,
        image_size=32
    )

    B, C, H, W = 2, 3, 32, 32
    x0 = torch.randn(B, C, H, W)
    sparse_input = torch.randn(B, C, H, W)
    mask = torch.rand(B, 1, H, W) > 0.5
    loss_mask = 1.0 - mask.float()

    loss = model(x0, sparse_input=sparse_input, mask=mask.float(), loss_mask=loss_mask)
    assert isinstance(loss.item(), float), "Loss should be a scalar"

    print("✓ RectifiedFlow forward pass correct")


if __name__ == '__main__':
    print("="*60)
    print("Running shape tests...")
    print("="*60)

    test_unet_ddpm_shapes()
    test_unet_fm_shapes()
    test_dit_fm_shapes()
    test_perceiver_io_fm_shapes()
    test_ddpm_forward()
    test_rectified_flow_forward()

    print("="*60)
    print("All shape tests passed!")
    print("="*60)
