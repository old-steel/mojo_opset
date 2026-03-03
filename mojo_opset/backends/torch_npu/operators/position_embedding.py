from typing import Optional, Tuple
import torch
import torch_npu

from mojo_opset.core import MojoRoPE
from mojo_opset.core.operators.position_embedding import generate_pos_embs



class TorchNpuRoPE(MojoRoPE):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = True,
        # rope_percentage: float = 1.0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # case T, N, D
        if cu_seqlens is not None:
            num_seqs = cu_seqlens.size(0) - 1
            if kv_lens is None:
                kv_lens = torch.zeros(num_seqs, device=q.device, dtype=torch.long)
            cos, sin = generate_pos_embs(sin, cos, kv_lens, cu_seqlens=cu_seqlens)
            return torch_npu.npu_apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        
        # case B, N, D
        if q.dim() == 3:
            bsz = q.shape[0]
            if kv_lens is None:
                kv_lens = torch.zeros(bsz, device=q.device, dtype=torch.long)
            cos, sin = generate_pos_embs(sin, cos, kv_lens)
            return torch_npu.npu_apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        # Padded prefill: [B, S, N, D] or [B, N, S, D]
        if head_first:
            return torch_npu.npu_apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        return torch_npu.npu_apply_rotary_pos_emb(q, k, cos.unsqueeze(2), sin.unsqueeze(2))