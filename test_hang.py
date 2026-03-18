import time, torch
import sys
import threading
import traceback
from mojo_opset.backends.ttx.kernels.npu.sample import top_k_sampling_impl

def dump_stack():
    time.sleep(20)
    print("===== DUMPING STACK TRACE =====", flush=True)
    for th in threading.enumerate():
        print(th, flush=True)
        traceback.print_stack(sys._current_frames()[th.ident])
    print("===============================", flush=True)
    import os
    os._exit(1)

threading.Thread(target=dump_stack, daemon=True).start()

x = torch.randn(48, 1519, device='npu', dtype=torch.float32)
top_k_sampling_impl(x, top_k=10, min_tokens_to_keep=1)
print("Done", flush=True)
