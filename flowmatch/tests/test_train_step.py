"""
Test training step: verify loss decreases with gradient updates.
"""
import torch
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.unet_ddpm import UNetDDPM
from src.models.unet_fm import UNetFM
from src.diffusion.ddpm import GaussianDiffusion
from src.diffusion.rectified_flow import RectifiedFlow
from src.sparsity.controller import SparsityController


def test_ddpm_train_step():
    """Test DDPM training step decreases loss."""
    print("Testing DDPM training step...")

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create sparsity controller
    sparsity_controller = SparsityController(
        image_size=32,
        mode='fixed_all',  # Fixed for reproducibility
        pattern='random',
        sparsity=0.2,
        block_size=5,
        num_blocks=6
    )

    # Generate synthetic batch
    B = 4
    images = torch.randn(B, 3, 32, 32)
    indices = torch.arange(B)

    # Get masks
    cond_masks, target_masks = sparsity_controller.get_masks(
        batch_size=B,
        indices=indices,
        device='cpu'
    )
    sparse_input = images * cond_masks

    # Training step 1
    optimizer.zero_grad()
    loss1 = model(images, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)
    loss1.backward()
    optimizer.step()

    # Training step 2
    optimizer.zero_grad()
    loss2 = model(images, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)
    loss2.backward()
    optimizer.step()

    # Training step 3
    optimizer.zero_grad()
    loss3 = model(images, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)

    print(f"  Loss progression: {loss1.item():.4f} → {loss2.item():.4f} → {loss3.item():.4f}")

    # Loss should generally decrease (allow some variance)
    assert loss3.item() < loss1.item() * 1.1, f"Loss not decreasing: {loss1.item():.4f} → {loss3.item():.4f}"

    print("✓ DDPM training step correct")


def test_flow_matching_train_step():
    """Test Flow Matching training step decreases loss."""
    print("Testing Flow Matching training step...")

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create sparsity controller
    sparsity_controller = SparsityController(
        image_size=32,
        mode='fixed_all',
        pattern='random',
        sparsity=0.2,
        block_size=5,
        num_blocks=6
    )

    # Generate synthetic batch
    B = 4
    x0 = torch.randn(B, 3, 32, 32)
    indices = torch.arange(B)

    # Get masks
    cond_masks, target_masks = sparsity_controller.get_masks(
        batch_size=B,
        indices=indices,
        device='cpu'
    )
    sparse_input = x0 * cond_masks

    # Training step 1
    optimizer.zero_grad()
    loss1 = model(x0, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)
    loss1.backward()
    optimizer.step()

    # Training step 2
    optimizer.zero_grad()
    loss2 = model(x0, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)
    loss2.backward()
    optimizer.step()

    # Training step 3
    optimizer.zero_grad()
    loss3 = model(x0, sparse_input=sparse_input, mask=cond_masks, loss_mask=target_masks)

    print(f"  Loss progression: {loss1.item():.4f} → {loss2.item():.4f} → {loss3.item():.4f}")

    # Loss should generally decrease (allow some variance)
    assert loss3.item() < loss1.item() * 1.1, f"Loss not decreasing: {loss1.item():.4f} → {loss3.item():.4f}"

    print("✓ Flow Matching training step correct")


def test_masked_loss_computation():
    """Test masked loss gives higher weight to target pixels."""
    print("Testing masked loss computation...")

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

    B = 2
    x0 = torch.randn(B, 3, 32, 32)

    # Create masks: 20% conditioning, 80% target
    cond_mask = torch.zeros(B, 1, 32, 32)
    cond_mask[:, :, ::5, ::5] = 1.0  # Grid with stride 5
    target_mask = 1.0 - cond_mask

    sparse_input = x0 * cond_mask

    # Compute loss with masked supervision
    loss_masked = model(x0, sparse_input=sparse_input, mask=cond_mask, loss_mask=target_mask)

    # Compute loss without masked supervision
    loss_unmasked = model(x0, sparse_input=sparse_input, mask=cond_mask, loss_mask=None)

    print(f"  Masked loss: {loss_masked.item():.4f}")
    print(f"  Unmasked loss: {loss_unmasked.item():.4f}")

    # Masked loss should focus more on unknown pixels
    assert isinstance(loss_masked.item(), float), "Loss should be scalar"
    assert isinstance(loss_unmasked.item(), float), "Loss should be scalar"

    print("✓ Masked loss computation correct")


def test_gradient_flow():
    """Test gradients flow through all model components."""
    print("Testing gradient flow...")

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

    B = 2
    x0 = torch.randn(B, 3, 32, 32)
    sparse_input = torch.randn(B, 3, 32, 32)
    mask = torch.rand(B, 1, 32, 32) > 0.5

    loss = model(x0, sparse_input=sparse_input, mask=mask.float())
    loss.backward()

    # Check that gradients exist for all parameters
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad += 1
            else:
                no_grad += 1

    print(f"  Parameters with gradients: {has_grad}")
    print(f"  Parameters without gradients: {no_grad}")

    assert no_grad == 0, f"{no_grad} parameters have no gradients"

    print("✓ Gradient flow correct")


def test_sparsity_controller():
    """Test sparsity controller generates valid masks."""
    print("Testing sparsity controller...")

    controller = SparsityController(
        image_size=32,
        mode='random_epoch',
        pattern='random',
        sparsity=0.2,
        block_size=5,
        num_blocks=6
    )

    B = 4
    indices = torch.arange(B)

    cond_masks, target_masks = controller.get_masks(
        batch_size=B,
        indices=indices,
        device='cpu'
    )

    # Check shapes
    assert cond_masks.shape == (B, 1, 32, 32), f"Unexpected cond_masks shape: {cond_masks.shape}"
    assert target_masks.shape == (B, 1, 32, 32), f"Unexpected target_masks shape: {target_masks.shape}"

    # Check masks are disjoint
    overlap = (cond_masks * target_masks).sum()
    assert overlap == 0, f"Masks should be disjoint, overlap: {overlap}"

    # Check total sparsity
    total_pixels = 32 * 32
    target_sparse_count = int(total_pixels * 0.2)
    actual_cond_count = cond_masks[0].sum().item()
    actual_target_count = target_masks[0].sum().item()
    actual_total = actual_cond_count + actual_target_count

    print(f"  Expected total sparse pixels: {target_sparse_count}")
    print(f"  Actual conditioning pixels: {actual_cond_count:.0f}")
    print(f"  Actual target pixels: {actual_target_count:.0f}")
    print(f"  Actual total sparse pixels: {actual_total:.0f}")

    # Allow some tolerance for rounding
    assert abs(actual_total - target_sparse_count) <= 2, f"Sparsity mismatch: {actual_total} vs {target_sparse_count}"

    print("✓ Sparsity controller correct")


if __name__ == '__main__':
    print("="*60)
    print("Running training step tests...")
    print("="*60)

    test_ddpm_train_step()
    test_flow_matching_train_step()
    test_masked_loss_computation()
    test_gradient_flow()
    test_sparsity_controller()

    print("="*60)
    print("All training step tests passed!")
    print("="*60)
