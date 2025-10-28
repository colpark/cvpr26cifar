# DiT (Diffusion Transformer) Training Details

## Model Architecture

The DiT model uses a **Vision Transformer (ViT)** style architecture adapted for Flow Matching with masked conditioning:

### Architecture Configuration
```yaml
# From configs/cifar10_fm_dit.yaml
model:
  type: "dit_fm"
  image_size: 32
  patch_size: 4          # 32÷4 = 8×8 patches = 64 tokens
  dim: 512               # Model dimension (embeddings)
  depth: 12              # 12 transformer blocks
  num_heads: 8           # Multi-head attention
  mlp_ratio: 4.0         # MLP hidden dim = 512 × 4 = 2048
  channel: 3             # RGB
  dropout: 0.1
  window_size: 8         # (Not used - full attention)
```

**Model Size:** ~57M parameters

### Key Components

#### 1. Patch Embedding
```python
# Input: [x_t, sparse_input, mask] concatenated → 9 channels
self.patch_embed = nn.Conv2d(channel * 3, dim, kernel_size=4, stride=4)
# 32×32×9 → 8×8×512 (64 tokens)
```

#### 2. Positional Encoding (Resolution-Agnostic)
```python
# 2D Fourier Positional Encoding
# Normalized coordinates: [-1, 1] for any resolution
# Frequency bands: 1.0 to 10.0 (128 frequencies)
# Output: (h_patches * w_patches, dim)
```

**Why Fourier PE?**
- Normalized coordinates work at any resolution
- Enables zero-shot super-resolution
- 32×32 training → 64×64 or 96×96 inference

#### 3. Time Embedding
```python
# Continuous time t ∈ [0, 1]
self.time_embedding = ContinuousTimeEmbedding(dim // 4, dim)
# Gaussian Fourier features → MLP → 512-dim
```

#### 4. DiT Transformer Block (×12)
```python
class DiTBlock:
    # Adaptive Layer Norm (AdaLN)
    self.adaLN_modulation(time_emb) → [shift, scale, gate] × 2

    # Attention with time conditioning
    x = x + gate_attn * Attention(AdaLN(x, time_emb))

    # MLP with time conditioning
    x = x + gate_mlp * MLP(AdaLN(x, time_emb))
```

**AdaLN Innovation:**
- Time information modulates layer norms
- Scale and shift parameters from time embedding
- Gate parameters control residual strength

#### 5. Output Projection
```python
# Tokens → Patches → Image
self.output_proj = nn.Linear(dim, patch_size² × channel)
# 512 → 16 × 3 = 48 values per token
# Reshape 64 tokens → 32×32×3 image
```

## Training Process

### 1. Data Flow
```
CIFAR-10 image (32×32×3)
↓
[x_t, sparse_input, mask] → (32×32×9)
↓
Patch embedding → 64 tokens (8×8 grid)
↓
+ Fourier Positional Encoding (resolution-agnostic)
↓
12 × DiT Block (with time embedding modulation)
↓
Output projection → velocity (32×32×3)
```

### 2. Training Configuration
```yaml
training:
  num_steps: 200000
  lr: 0.0001              # Lower LR for transformer (vs 0.0002 for U-Net)
  weight_decay: 0.0       # No weight decay
  betas: [0.9, 0.999]     # Adam optimizer
  max_grad_norm: 1.0      # Gradient clipping
  batch_size: 64          # Smaller batch (vs 128 for U-Net)
```

**Why lower LR and smaller batch?**
- Transformers are more sensitive to learning rate
- Attention layers can have unstable gradients
- Smaller batch size reduces memory usage (transformer is larger)

### 3. Loss Function (Flow Matching)
```python
# Rectified Flow loss
# Sample time: t ~ Uniform(0, 1)
# Interpolation: x_t = (1-t)·x_0 + t·z*
# Target velocity: v* = z* - x_0
# Loss: MSE(v_pred, v_target) on target pixels
```

**Masked loss:**
```python
if loss_mask is not None:
    loss = (raw_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
```

### 4. Gradient Clipping
```python
# After backward pass:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

**Why gradient clipping?**
- Transformers can have exploding gradients
- Attention layers can be unstable early in training
- Prevents training divergence

### 5. Masked Conditioning
```yaml
sparsity:
  mode: "fixed_instance"   # Same mask per image across epochs
  pattern: "random"        # Random pixel sampling
  sparsity: 0.2           # 20% total (10% cond + 10% target)
```

**Split masks:**
- **Conditioning mask** (10%): Input pixels shown to model
- **Target mask** (10%): Supervision pixels (higher loss weight)
- **Unknown** (80%): Never used during training

## Sampling Process

### 1. ODE Integration (Euler Method)
```python
# Initialize from noise
x = torch.randn(B, 3, H, W)

# Integrate from t=1 to t=0
for i in range(50):  # sampling_steps
    t = 1.0 - i / 50
    v = model(x, t, sparse_input, mask)
    x = x - (1/50) * v  # Euler step backward

    # Hard projection: enforce known pixels
    x = mask * sparse_input + (1 - mask) * x
```

### 2. Zero-Shot Super-Resolution
```python
# Train at 32×32 (64 tokens, 8×8 grid)
# Sample at 64×64 (256 tokens, 16×16 grid)
# Sample at 96×96 (576 tokens, 24×24 grid)

# Fourier PE uses normalized coords [-1, 1]
# Works at any resolution due to coordinate normalization
```

## Training Dynamics

### Expected Behavior

**Early training (0-10K steps):**
- Loss decreases rapidly from ~0.3 to ~0.15
- Samples are blurry but recognizable
- SR outputs are poor (model hasn't learned spatial structure)

**Mid training (10K-50K steps):**
- Loss plateaus around ~0.14
- Conditional generation improves significantly
- SR quality starts improving
- PSNR: 15-20 dB, SSIM: 0.65-0.85

**Late training (50K-200K steps):**
- Loss slowly decreases to ~0.12-0.13
- High-quality conditional generation
- SR maintains structure and details
- PSNR: 20-25 dB, SSIM: 0.85-0.90

### Memory Usage

**Per GPU (batch_size=64):**
- Model parameters: ~230 MB (57M × 4 bytes)
- Optimizer state: ~460 MB (Adam = 2× params)
- Activations: ~2-3 GB (depends on depth)
- **Total**: ~3-4 GB per GPU

**To reduce memory:**
- Reduce batch_size: 64 → 32 or 16
- Reduce depth: 12 → 8 or 6
- Reduce dim: 512 → 384 or 256
- Use gradient checkpointing (not implemented)

## Comparison with U-Net

| Aspect | DiT (Transformer) | U-Net |
|--------|------------------|-------|
| **Parameters** | ~57M | ~97M (large) / ~35M (small) |
| **Batch size** | 64 | 128 |
| **Learning rate** | 1e-4 | 2e-4 |
| **Training speed** | ~20 it/s | ~25 it/s |
| **Memory** | ~3-4 GB | ~2-3 GB |
| **SR support** | ✅ Native (Fourier PE) | ❌ Implicit only |
| **32×32 quality** | Similar | Similar |
| **64×64 SR** | Better structure | More artifacts |
| **Convergence** | Slower (50K steps) | Faster (25K steps) |

## Best Practices

### 1. Learning Rate Schedule
```python
# Current: Fixed LR = 1e-4
# Better: Warmup + cosine decay
# Warmup: 0 → 1e-4 over 5K steps
# Decay: 1e-4 → 1e-5 over 195K steps
```

### 2. Data Augmentation
```yaml
augment_horizontal_flip: false  # Currently disabled
# Consider enabling for better generalization
```

### 3. Training Duration
- **Minimum**: 50K steps for reasonable quality
- **Recommended**: 100K steps for good quality
- **Best**: 200K steps for best SR performance

### 4. Monitoring
Key metrics to watch:
- **Loss convergence**: Should reach ~0.12-0.13
- **PSNR**: Target >20 dB at 32×32
- **SSIM**: Target >0.80 at 32×32
- **SR quality**: Check 64×64 samples after 50K steps

### 5. Debugging SR Issues
If SR quality is poor:
1. **Check positional encoding**: Assert shape matches in forward()
2. **Verify coordinate normalization**: Should be in [-1, 1]
3. **Train longer**: SR improves significantly after 50K steps
4. **Check frequency bands**: Should cover 1.0 to 10.0
5. **Visualize attention**: Are tokens attending to spatial neighbors?

## Known Issues

1. **Slow convergence**: Transformers need more steps than U-Nets
2. **Memory hungry**: 57M parameters + attention = high memory
3. **Sensitive to LR**: Too high → divergence, too low → slow training
4. **SR artifacts early**: Needs 50K+ steps to learn spatial structure

## Future Improvements

1. **FlashAttention**: Faster and more memory-efficient attention
2. **Gradient checkpointing**: Reduce memory at cost of speed
3. **Mixed precision training**: Use fp16 for 2× speedup
4. **LR scheduling**: Warmup + cosine decay
5. **Relative positional encoding**: Better than absolute Fourier PE
6. **Windowed attention**: For higher resolutions (>128×128)
