"""
Profile Mamba SSM V2 to find performance bottlenecks
"""
import torch
import time
from src.models.mamba_ssm_fm_v2 import MambaSSMFMV2, SSMBlockFast
from src.datasets.cifar10_coordinate import get_cifar10_coordinate_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create model
model = MambaSSMFMV2(
    channel=3,
    num_fourier_feats=256,
    d_model=512,
    num_layers=6,
    d_state=16,
    dropout=0.1
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dataloader
train_loader = get_cifar10_coordinate_dataloader(
    root='./data',
    train=True,
    batch_size=64,
    input_ratio=0.2,
    target_ratio=0.2,
    num_workers=4,
    download=False,
    seed=42
)

# Get a batch
batch = next(iter(train_loader))
input_coords = batch['input_coords'].to(device)
input_values = batch['input_values'].to(device)
target_coords = batch['target_coords'].to(device)
target_values = batch['target_values'].to(device)

print(f"\nBatch shapes:")
print(f"  input_coords: {input_coords.shape}")
print(f"  input_values: {input_values.shape}")
print(f"  target_coords: {target_coords.shape}")
print(f"  target_values: {target_values.shape}")

B = input_coords.shape[0]
N_in = input_coords.shape[1]
N_out = target_coords.shape[1]
total_seq = N_in + N_out

print(f"\nSequence info:")
print(f"  Batch size: {B}")
print(f"  Input tokens: {N_in}")
print(f"  Output tokens: {N_out}")
print(f"  Total sequence: {total_seq}")
print(f"  Decay matrix size: {total_seq} × {total_seq} × 16 = {total_seq * total_seq * 16:,} elements")
print(f"  Decay matrix memory: {total_seq * total_seq * 16 * 4 / 1024 / 1024:.2f} MB (per sample)")
print(f"  Decay matrix memory: {total_seq * total_seq * 16 * 4 * B / 1024 / 1024:.2f} MB (per batch)")

# Profile forward pass
model.eval()
with torch.no_grad():
    # Prepare inputs
    t = torch.rand(B, device=device)
    x_t = torch.randn_like(target_values)

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = model(x_t, target_coords, t, input_coords, input_values)

    torch.cuda.synchronize()

    # Time forward pass
    print("\nTiming forward pass (10 iterations)...")
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        output = model(x_t, target_coords, t, input_coords, input_values)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"\nAverage forward pass: {avg_time*1000:.2f} ms")
    print(f"Throughput: {B / avg_time:.2f} samples/sec")

# Profile SSMBlockFast specifically
print("\n" + "="*60)
print("Profiling SSMBlockFast directly")
print("="*60)

ssm_block = SSMBlockFast(d_model=1024, d_state=16, dropout=0.0).to(device)
test_input = torch.randn(B, total_seq, 1024, device=device)

print(f"SSMBlockFast input shape: {test_input.shape}")

ssm_block.eval()
with torch.no_grad():
    # Warmup
    for _ in range(3):
        _ = ssm_block(test_input)

    torch.cuda.synchronize()

    # Time
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        output = ssm_block(test_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"Average SSMBlockFast forward: {avg_time*1000:.2f} ms")

# Profile decay matrix creation specifically
print("\n" + "="*60)
print("Profiling decay matrix creation")
print("="*60)

N = total_seq
d_state = 16

with torch.no_grad():
    A_bar = torch.randn(d_state, device=device)
    indices = torch.arange(N, device=device)

    # Warmup
    for _ in range(3):
        decay = A_bar.unsqueeze(0).pow(
            (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0).unsqueeze(-1)
        )
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1)
        decay = decay * mask.unsqueeze(-1).float()

    torch.cuda.synchronize()

    # Time
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        decay = A_bar.unsqueeze(0).pow(
            (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0).unsqueeze(-1)
        )
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1)
        decay = decay * mask.unsqueeze(-1).float()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"Average decay matrix creation: {avg_time*1000:.2f} ms")
    print(f"Decay matrix shape: {decay.shape}")
    print(f"Decay matrix size: {decay.numel() * 4 / 1024 / 1024:.2f} MB")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("If decay matrix creation is a significant portion of SSMBlockFast time,")
print("then we need to optimize or replace the algorithm.")
