# Perceiver IO V2 Architecture and Training Details

## Why Perceiver IO?

The **Perceiver IO** architecture offers a fundamentally different approach to super-resolution compared to traditional CNNs or transformers:

**Key Philosophy:**
- **Attention-based encoding/decoding** rather than convolutional layers
- **Latent bottleneck** for computational efficiency
- **Query-based generation** allows arbitrary output resolutions
- **Resolution-agnostic by design** through Fourier positional encoding

## Architecture Overview

### V1 vs V2 Comparison

| Aspect | V1 (Original) | V2 (Improved) |
|--------|---------------|---------------|
| **Target resolution** | ❌ Always outputs same size as input | ✅ Explicit `target_size` parameter |
| **Latent structure** | Random initialization | Spatial 16×16 grid structure |
| **Positional encoding** | Only time embedding | Time + spatial latent positions |
| **SR mechanism** | Implicit (hope it generalizes) | Explicit (separate encode/decode resolutions) |
| **Model size** | ~30M params (512 dim, 256 latents, depth 6) | ~8M params (256 dim, 256 latents, depth 4) |

## Perceiver IO V2 Architecture

### Model Configuration
```yaml
# From configs/cifar10_fm_perceiver_v2.yaml
model:
  type: "perceiver_io_fm_v2"
  image_size: 32
  channel: 3
  latent_dim: 256           # Embedding dimension
  num_latents: 256          # 16×16 spatial grid
  depth: 4                  # Perceiver blocks
  num_heads: 4              # Multi-head attention
  head_dim: 64              # Dimension per head
  input_fourier_features: 32   # Frequency bands for encoding
  query_fourier_features: 32   # Frequency bands for decoding
```

**Model Size:** ~8-10M parameters (much smaller than DiT's 57M)

### Architecture Components

#### 1. Input Encoding (at training resolution)

```python
# Input: [x_t, sparse_input, mask] → 9 channels at 32×32
# Flatten: 32×32 → 1024 tokens

# Step 1: Generate coordinate grid [-1, 1]
coords_in = create_normalized_coords(32, 32)  # (1024, 2)

# Step 2: Create Fourier positional features
fourier_feats = fourier_features(coords_in, scale=32)  # (1024, 64)
# Frequency bands: 1.0, 2.0, ..., 16.0 (16 bands)
# Each band: sin(2π·f·x), cos(2π·f·x), sin(2π·f·y), cos(2π·f·y)

# Step 3: Combine pixel values + Fourier features
input_tokens = concat([pixels, fourier_feats])  # (1024, 9+64=73)
input_tokens = linear_proj(input_tokens)  # (1024, 256)
```

**Why normalized coordinates [-1, 1]?**
- 32×32 training: coords span [-1, 1]
- 64×64 testing: coords still span [-1, 1]
- **Same frequency bands work at any resolution**

#### 2. Latent Bottleneck (spatial grid structure)

```python
# V2 Innovation: Spatial awareness
latent_h = latent_w = sqrt(256) = 16  # 16×16 grid
latents = Parameter(randn(256, 256))  # Learnable latent vectors

# Spatial positional encoding (learned)
latent_pos_embed = Parameter(randn(256, 256))
latents = latents + latent_pos_embed  # Add spatial structure

# Time conditioning
time_emb = time_embedding(t)  # (B, 256)
latents = latents + time_emb  # Broadcast to all latents
```

**Why 16×16 grid?**
- Provides spatial inductive bias
- Latents correspond to image regions
- Better generalization to higher resolutions
- Each latent roughly corresponds to 2×2 input patches

#### 3. Cross-Attention Encoding

```python
# Perceiver Block (×4 layers)
for block in self.blocks:
    # Cross-attention: latents attend to input tokens
    latents = latents + cross_attn(latents, input_tokens)
    #         (B, 256, 256)      (B, 1024, 256)

    # Self-attention: latents process among themselves (×2 per block)
    latents = latents + self_attn(latents)
    latents = latents + ffn(latents)
```

**Computational efficiency:**
- Input: 1024 tokens (32×32)
- Latents: 256 vectors (constant bottleneck)
- **Attention complexity: O(N_latent × N_input) instead of O(N_input²)**
- For 64×64: Process 4096 input tokens with same 256 latents

#### 4. Query-Based Decoding (at target resolution)

```python
# V2 Critical Feature: Separate output resolution
if target_size is None:
    H_out, W_out = 32, 32  # Same as input
else:
    H_out, W_out = target_size  # e.g., (64, 64)

# Generate queries at TARGET resolution
coords_out = create_normalized_coords(H_out, W_out)  # (H_out*W_out, 2)
query_fourier = fourier_features(coords_out, scale=32)  # Same frequency bands!
queries = linear_proj(query_fourier)  # (B, H_out*W_out, 256)

# Decode: queries cross-attend to latents
output_tokens = cross_attn(queries, latents)
#                         (B, 4096, 256) for 64×64
#                              ↓
#                         (B, 256, 256) latents

# Project to velocity
output = output_proj(output_tokens)  # (B, 4096, 3)
output = reshape(output, (B, 3, 64, 64))
```

**Why this works for SR:**
- 32×32 training: 1024 queries attend to 256 latents
- 64×64 testing: 4096 queries attend to same 256 latents
- **Latents have learned spatial structure** from 16×16 grid
- **Queries use same normalized coords** [-1, 1]

#### 5. Output Projection

```python
# Simple linear projection to velocity
self.output_proj = nn.Linear(latent_dim, channel)
# 256 → 3 (RGB)
```

## Training Process

### 1. Data Flow (32×32 Training)

```
CIFAR-10 image (32×32×3)
↓
[x_t, sparse_input, mask] → (32×32×9)
↓
Flatten + Fourier PE → 1024 tokens × 256 dim
↓
Cross-attention encoding → 256 latents × 256 dim
  ├─ Latent spatial grid (16×16 structure)
  └─ Time conditioning
↓
4 × Perceiver Block (cross-attn + 2×self-attn + FFN)
↓
Query generation (same 32×32 resolution)
↓
Cross-attention decoding → 1024 output tokens
↓
Output projection → velocity (32×32×3)
```

### 2. Training Configuration

```yaml
training:
  num_steps: 200000
  lr: 0.0002              # Higher than DiT (V2 is smaller)
  weight_decay: 0.0
  betas: [0.9, 0.999]
  max_grad_norm: 1.0
  batch_size: 128         # Larger than DiT (V2 more efficient)
```

**Why higher batch size?**
- V2 uses 256 latent bottleneck → less memory than DiT's 512 tokens
- Attention is O(N_latent × N_input), not O(N_input²)
- Can fit 2× batch size of DiT in same memory

### 3. Loss Function (Flow Matching)

```python
# Same as DiT - Rectified Flow loss
# Sample time: t ~ Uniform(0, 1)
# Interpolation: x_t = (1-t)·x_0 + t·z*
# Target velocity: v* = z* - x_0
# Loss: MSE(v_pred, v_target) on target pixels
```

### 4. Sparsity Configuration

```yaml
sparsity:
  mode: "fixed_instance"  # Same mask per image across epochs
  pattern: "random"       # Random pixel sampling
  sparsity: 0.2          # 20% total (10% cond + 10% target)
```

## Super-Resolution Process

### Zero-Shot SR at 64×64

```python
# Training: Model trained at 32×32
# Testing: Generate at 64×64

# Step 1: Encode at 32×32 (upsampled conditioning)
sparse_input_64 = F.interpolate(sparse_32, size=(64, 64), mode='nearest')
mask_64 = F.interpolate(mask_32, size=(64, 64), mode='nearest')

coords_in = create_coords(32, 32)  # Still encode at training resolution
input_tokens = encode(x_32, coords_in)  # (B, 1024, 256)

# Step 2: Process latents (resolution-independent)
latents = cross_attn(latents, input_tokens)  # (B, 256, 256)
latents = self_attn(latents)

# Step 3: Decode at 64×64 (target resolution)
coords_out = create_coords(64, 64)  # Generate 4096 queries
queries = fourier_features(coords_out)  # Same frequency bands!
output_tokens = cross_attn(queries, latents)  # (B, 4096, 256)
output_64 = output_proj(output_tokens)  # (B, 3, 64, 64)
```

**Why V2 works better than V1:**

| Issue | V1 Problem | V2 Solution |
|-------|------------|-------------|
| **Resolution mismatch** | forward() doesn't accept target_size | Explicit target_size parameter |
| **Query generation** | Uses same coords as input (32×32) | Generates queries at target resolution |
| **Latent structure** | Random initialization, no spatial info | 16×16 spatial grid with learned positions |
| **Generalization** | Latents don't know where they are | Spatial structure aids multi-scale reasoning |

## Training Dynamics

### Expected Behavior

**Early training (0-10K steps):**
- Loss decreases rapidly from ~0.3 to ~0.15
- 32×32 samples are blurry but recognizable
- 64×64 SR shows some structure but many artifacts
- Latents learning spatial correspondence

**Mid training (10K-50K steps):**
- Loss plateaus around ~0.13-0.14
- 32×32 conditional generation improves significantly
- 64×64 SR quality starts to improve (better than V1)
- Spatial latent structure becoming useful

**Late training (50K-200K steps):**
- Loss slowly decreases to ~0.11-0.12
- High-quality 32×32 conditional generation
- 64×64 SR maintains structure and some details
- Better than V1, but still not perfect (zero-shot limitation)

### Memory Usage

**Per GPU (batch_size=128):**
- Model parameters: ~40 MB (10M × 4 bytes)
- Optimizer state: ~80 MB (Adam = 2× params)
- Activations: ~1-2 GB (depends on depth)
- **Total**: ~1.5-2.5 GB per GPU

**Much more efficient than DiT:**
- DiT: ~3-4 GB (57M params, 64 batch size)
- Perceiver V2: ~2 GB (10M params, 128 batch size)
- **2× throughput with half the memory**

## Comparison Table

| Aspect | Perceiver V2 | DiT | U-Net |
|--------|-------------|-----|-------|
| **Parameters** | ~10M | ~57M | ~35M |
| **Batch size** | 128 | 64 | 128 |
| **Learning rate** | 2e-4 | 1e-4 | 2e-4 |
| **Training speed** | ~30 it/s | ~20 it/s | ~25 it/s |
| **Memory** | ~2 GB | ~3-4 GB | ~2-3 GB |
| **SR mechanism** | Query-based (explicit) | Fourier PE (implicit) | Interpolation |
| **SR quality** | Good structure | Best structure | Artifacts |
| **Convergence** | Fast (30K steps) | Slow (50K steps) | Fast (25K steps) |
| **Efficiency** | Best | Worst | Good |

## Key Innovations in V2

### 1. Explicit Resolution Decoupling

**V1 Architecture:**
```python
# Encode and decode at same resolution
coords = create_coords(H, W)
inputs = encode(x, coords)
latents = process(inputs)
outputs = decode(latents, coords)  # Same coords!
```

**V2 Architecture:**
```python
# Separate encoding and decoding resolutions
coords_in = create_coords(H_in, W_in)    # 32×32
inputs = encode(x, coords_in)
latents = process(inputs)
coords_out = create_coords(H_out, W_out)  # 64×64
outputs = decode(latents, coords_out)     # Different coords!
```

### 2. Spatial Latent Grid

**Why a grid structure?**
- **Inductive bias**: Natural images have spatial locality
- **Correspondence**: Latent[i, j] roughly corresponds to image region
- **Generalization**: Spatial structure helps cross-resolution reasoning
- **Learned positions**: Each latent knows where it is in the grid

**Implementation:**
```python
# Verify num_latents is a perfect square
latent_h = latent_w = int(math.sqrt(num_latents))  # 16
assert latent_h * latent_w == num_latents, "Must be perfect square"

# Spatial positional encoding
self.latent_pos_embed = nn.Parameter(torch.randn(num_latents, latent_dim))
latents = self.latents + self.latent_pos_embed  # Add spatial awareness
```

### 3. Resolution-Normalized Fourier Features

**The key insight:**
```python
# Coordinates always normalized to [-1, 1]
# 32×32: y ∈ [-1, 1], x ∈ [-1, 1], spacing = 2/32 = 0.0625
# 64×64: y ∈ [-1, 1], x ∈ [-1, 1], spacing = 2/64 = 0.03125

# Same frequency bands work for both!
freq_bands = [1.0, 2.0, 3.0, ..., 16.0]
for freq in freq_bands:
    features.append(sin(2π · freq · normalized_coord))
    features.append(cos(2π · freq · coord))
```

**Why this enables SR:**
- Model learns features at **semantic scales**, not pixel scales
- Frequency 1.0 corresponds to one full cycle across the image
- Frequency 16.0 corresponds to 16 cycles across the image
- These semantic scales transfer across resolutions

### 4. RectifiedFlow Auto-Detection

**Automatic V2 Support:**
```python
# In rectified_flow.py (lines 132-137)
import inspect
model_params = inspect.signature(self.model.forward).parameters

if 'target_size' in model_params:
    # V2 model detected - use explicit SR
    v = self.model(x, t, sparse_input, mask, target_size=(64, 64))
else:
    # V1 model - implicit SR (hope for the best)
    v = self.model(x, t, sparse_input, mask)
```

## Why V2 Should Work Better for SR

### Theory

**1. Explicit Resolution Handling**
- V1: "Hope the model figures out different resolutions"
- V2: "Tell the model exactly what resolution we want"

**2. Spatial Latent Structure**
- V1: Random latents have no spatial meaning
- V2: 16×16 grid provides spatial correspondence

**3. Query-Based Generation**
- V1: Decoder queries at input resolution only
- V2: Decoder queries can be at any resolution

**4. Learned Spatial Positions**
- V1: Latents don't know their spatial role
- V2: Each latent has learned positional embedding

### Practice

**What we expect at 30K steps:**

**Perceiver V1:**
- 32×32: Good quality (trained directly)
- 64×64: Poor - model never learned what "higher resolution" means
- Artifacts: Blocky, blurry, incoherent structures

**Perceiver V2:**
- 32×32: Good quality (trained directly)
- 64×64: Moderate quality - explicit SR mechanism helps
- Structures: Better preservation, some fine details
- Still limitations: Zero-shot extrapolation is hard

**The Reality Check:**
Even V2 performs **zero-shot SR** - no high-res examples during training. It learns to:
1. Encode 32×32 inputs into 256 latents
2. Decode latents back to 32×32
3. Hope spatial structure generalizes to 64×64

This is fundamentally **easier** than DiT's challenge (64 tokens → 256 tokens), but still requires extrapolation.

## Best Practices

### 1. Latent Configuration

```yaml
# Recommended configurations
standard:
  num_latents: 256  # 16×16 grid
  latent_dim: 256

large:
  num_latents: 256  # Keep grid, increase capacity
  latent_dim: 512

very_large:
  num_latents: 512  # 22.6×22.6 ❌ Not a perfect square!
  # Use 441 (21×21) or 529 (23×23) instead
```

**Critical:** num_latents must be a perfect square for spatial grid.

### 2. Training Duration

- **Minimum**: 30K steps for reasonable 32×32 quality
- **Recommended**: 50K steps for decent SR
- **Best**: 100K steps for stable SR quality

### 3. Monitoring SR Quality

```python
# At each sample_every:
# 1. Check 32×32 conditional (should improve steadily)
# 2. Check 64×64 SR (should gradually get better)
# 3. Look for spatial coherence, not just PSNR

# Good SR signs:
# - Edges preserved across scales
# - No checkerboard artifacts
# - Consistent textures
# - Semantic content maintained

# Bad SR signs:
# - Blocky upsampling
# - Incoherent structures
# - Blurry everywhere
# - Checkerboard patterns
```

### 4. Debugging V2 Issues

**If 32×32 quality is poor:**
- Increase depth (4 → 6 blocks)
- Increase latent_dim (256 → 512)
- Check loss convergence
- Verify sparsity mode is "fixed_instance"

**If 64×64 SR is poor even at 50K:**
- Check target_size is being passed correctly
- Verify num_latents is perfect square
- Try larger latent_dim (more capacity)
- Consider: Zero-shot SR has fundamental limits

**If training is unstable:**
- Reduce learning rate (2e-4 → 1e-4)
- Enable gradient clipping (max_grad_norm=1.0)
- Check for NaN in Fourier features
- Verify input normalization [-1, 1]

## Limitations

### Zero-Shot SR Ceiling

Even with V2 improvements, zero-shot SR has limits:
1. **No high-res training**: Model never sees 64×64 examples
2. **Frequency gap**: High-frequency details at 64×64 are extrapolated
3. **Spatial reasoning**: Must infer 4× pixels from spatial latent grid
4. **Conditioning**: Sparse 10% at 32×32 → blocky 10% at 64×64

**V2 makes SR better, but doesn't eliminate the fundamental challenge.**

### When V2 Won't Help

- **Very high resolutions** (>128×128): Too far from training
- **Extremely sparse conditioning** (<5%): Not enough information
- **Complex fine details**: Model hasn't learned high-frequency features
- **Arbitrary upsampling factors**: 2× works better than 3× or 5×

### Realistic Expectations

**V2 is better than V1 for SR because:**
- ✅ Explicit resolution handling
- ✅ Spatial latent structure
- ✅ Query-based decoding

**But V2 is NOT magic:**
- ❌ Still zero-shot extrapolation
- ❌ Still trained only at 32×32
- ❌ Still can't generate details never seen

## Future Improvements

### Multi-Resolution Training

**The real solution for SR:**
```yaml
training:
  multi_resolution: true
  resolutions: [32, 48, 64]
  resolution_prob: [0.5, 0.3, 0.2]
```

Train on multiple resolutions so model learns what "high-res" means.

### Progressive Resolution

```yaml
training:
  0-50k: 32×32
  50k-100k: 48×48
  100k-150k: 64×64
```

Gradually increase resolution during training.

### Latent Upsampling

```python
# Instead of fixed 16×16 latent grid
# Use resolution-dependent latent count
def get_num_latents(image_size):
    latent_size = image_size // 2  # Always half resolution
    return latent_size * latent_size

# 32×32 → 256 latents (16×16)
# 64×64 → 1024 latents (32×32)
```

### Learned Upsampling

```python
# Replace nearest-neighbor sparse upsampling
sparse_upsampled = F.interpolate(sparse, mode='nearest')  # ❌ Blocky

# With learned refinement
sparse_upsampled = F.interpolate(sparse, mode='bilinear')
sparse_refined = refine_network(sparse_upsampled)  # ✅ Smooth
```

## Summary

**Perceiver IO V2 offers:**
- **Efficiency**: 10M params vs DiT's 57M
- **Explicit SR**: target_size parameter for clear resolution control
- **Spatial Structure**: 16×16 latent grid provides inductive bias
- **Query-Based Decoding**: Generate any resolution via cross-attention

**Compared to DiT:**
- ✅ Smaller, faster, more memory-efficient
- ✅ Explicit resolution handling
- ✅ Better than V1 for zero-shot SR
- ❌ Still not as good as DiT for highest SR quality
- ❌ Cross-attention can be slower than self-attention

**The Bottom Line:**
V2 architecture makes zero-shot SR **more feasible**, but true high-quality SR requires seeing high-resolution examples during training. The spatial latent grid and explicit resolution handling provide the foundation for better generalization, but cannot fully overcome the fundamental challenge of extrapolating to unseen resolutions.
