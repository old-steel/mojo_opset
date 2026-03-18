import time, torch
import sys
from mojo_opset.backends.ttx.kernels.npu.sample import top_k_sampling_impl

print("Setting up tensor...", flush=True)
x = torch.randn(20, 1519, device='npu', dtype=torch.float32)
print("Starting loop...", flush=True)
for i in range(2):
    t0=time.time()
    print(f"iter {i} starting...", flush=True)
    p,t=top_k_sampling_impl(x, top_k=10, min_tokens_to_keep=1)
    print(f"iter {i} kernel done, synchronizing...", flush=True)
    torch.npu.synchronize()
    print(f"iter {i} time: {time.time()-t0}, out: {p.shape}, {t.shape}", flush=True)
