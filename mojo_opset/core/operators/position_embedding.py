from typing import Optional
from typing import Sequence
from typing import Tuple, List

import torch

from ..operator import MojoOperator


class MojoRotaryEmbedding(MojoOperator):
    def __init__(self, rope_theta, rope_dim, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device = self.tensor_factory_kwargs.get("device")) / rope_dim)
        )
        self.attention_scaling = 1.0
        self.init_max_length = init_max_length
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if init_max_length is not None:
            self._rope_init(init_max_length)

        def load_state_dict_post_hook(module, incompatible_keys) -> None:
            key2ignore = []
            for miss in incompatible_keys.missing_keys:
                if miss.split('.')[-1] in ("inv_freq", "cos", "sin"):
                    key2ignore.append(miss)
            for key in key2ignore:
                incompatible_keys.missing_keys.remove(key)
        self.register_load_state_dict_post_hook(load_state_dict_post_hook)


    def _rope_init(self, max_length: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(max_length, device = self.tensor_factory_kwargs.get("device"))
        freqs = position_ids[..., None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos/sin for Rotary Position Embedding (RoPE).
        x is necessary for the kernel to determine the output shape.

        Scenario descriptions:
        1. Varlen prefill: input [T, H], cu_seqlens_q [T+1] or position_ids [T].
        2. Padded prefill: input [B, S, H], cu_seqlens_q None, position_ids None.
        3. Decode: input [B, H], cu_seqlens_q None, position_ids [B].
        """
        assert position_ids is None or cu_seqlens_q is None, "At most one of cu_seqlens_q or position_ids should be provided"

        if cu_seqlens_q is not None:
            assert x.dim() == 2, "x must be 2D: [T, D]"
            position_ids = torch.full((x.shape[0],), -1, device = x.device, dtype = torch.int32)
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            bsz = seqlens_q.size(0)
            for i in range(bsz):
                q_len = seqlens_q[i].item()
                context_len = 0 if seqlens_kv is None else seqlens_kv[i].item() - q_len
                position_ids[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = torch.arange(
                    context_len,
                    context_len + q_len, 
                    device = cu_seqlens_q.device,
                    dtype = torch.int32,
                )
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
        else:
            position_ids = torch.arange(x.shape[1], device = x.device, dtype = torch.int32)

        if self.init_max_length is None:
            freqs = position_ids[..., None] * self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        else:
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]
        
        return cos, sin


class MojoApplyRoPE(MojoOperator):
    """Rotary Position Embedding (RoPE) operator.

    Supports three scenarios:
    1. Varlen prefill: input [T, N, D] or [N, T, D], cos/sin [T, d].
    2. Padded prefill: input [B, S, N, D] or [B, N, S, D], cos/sin [S, d].
    3. Decode: input [B, N, D] or [N, B, D], cos/sin [B, d].
    """

    def __init__(self, interleaved: bool = False):
        super().__init__()
        assert not interleaved, "interleaved impl is not supported yet."
        self.interleaved = interleaved

    def extra_repr(self) -> str:
        return f"{self.interleaved=}".replace("self.", "")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = cos.shape[-1]
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k = torch.split(k, [nope_dim, rope_dim], dim=-1)

        q_rot = (q * cos + self._rotate_half(q) * sin).to(q.dtype)
        k_rot = (k * cos + self._rotate_half(k) * sin).to(k.dtype)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embedding (RoPE).

        Scenario descriptions:
        1. Varlen prefill: q/k [T, N, D], requires cu_seqlens, sin/cos are full
        2. Padded prefill: q/k [B, S, N, D] or [B, N, S, D], sin/cos pre-sliced [B, S, d]
        3. Decode: q/k [B, N, D], requires kv_lens, sin/cos are full

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            unsqueeze_dim: Unsqueeze dimension for cos and sin for multi-heads

        Returns:
            (q_rot, k_rot) with same shape as input
        """
        if head_first:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return self._apply_rope(q, k, cos, sin)


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass


class MojoGridRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply 3D grid rotary position embeddings (RoPE) over (F, H, W) axes using
        precomputed per-sample frequency tensors.

        Args:
            x (torch.Tensor): [B, L, N, D]; D must be even (paired into complex components).
            grid_sizes (torch.Tensor): [B, 3] per-sample (F, H, W); seq_len = F*H*W.
            freqs_list (List[torch.Tensor]): length-B list; each item is a complex unit-phase tensor
                of shape [seq_len, 1, D/2], broadcastable to [seq_len, N, D/2].

        Returns:
            torch.Tensor: Same shape as `x`. Per sample, the first F*H*W tokens are rotated;
                remaining padding tokens are preserved. Output dtype matches input.
        """
        return _apply_grid_rope_impl(x, grid_sizes, freqs_list)


def _apply_grid_rope_impl(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs_list: List[torch.Tensor],
) -> torch.Tensor:
    assert x.dim() == 4, "x must be 4D: [B, L, N, D]"
    assert x.size(-1) % 2 == 0, "D must be even for complex pairing"
    assert grid_sizes.dim() == 2 and grid_sizes.size(1) == 3, "grid_sizes must be [B, 3]"
    if len(freqs_list) != x.size(0):
        raise ValueError(f"freqs_list length must equal batch size {x.size(0)}, but got {len(freqs_list)}.")

    n = x.size(2)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        if seq_len > x.size(1):
            raise ValueError(f"Valid sequence length {seq_len} exceeds input length {x.size(1)} for sample {i}.")
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = freqs_list[i]
        if freqs_i.shape != (seq_len, 1, x.size(-1) // 2):
            raise ValueError(
                f"freqs_list[{i}] must have shape {(seq_len, 1, x.size(-1) // 2)}, but got {tuple(freqs_i.shape)}."
            )
        x_i = torch.view_as_real(x_i * freqs_i.to(device=x.device)).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    y = torch.stack(output)
    return y.type_as(x)


class MojoMRoPE(MojoOperator):
    """Multimodal rotary position embedding with built-in frequency generation.

    This operator supports two modes:
    1. Provide `freqs_list` directly, matching `MojoGridRoPE`.
    2. Omit `freqs_list` and let the operator build per-sample 3D rotary
       frequencies from `grid_sizes`.
    """

    def __init__(
        self,
        head_dim: Optional[int] = None,
        theta: float = 10000.0,
        section_sizes: Optional[Sequence[int]] = None,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if head_dim is not None and head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, but got {head_dim}.")
        self.head_dim = head_dim
        self.theta = theta
        self.section_sizes = None if section_sizes is None else tuple(int(v) for v in section_sizes)
        self.max_seq_len = max_seq_len
        if max_seq_len is not None:
            if head_dim is None:
                raise ValueError("head_dim must be provided when max_seq_len is set.")
            self._build_base_freqs(max_seq_len, head_dim)

    def _resolve_head_dim(self, x: torch.Tensor) -> int:
        head_dim = x.size(-1) if self.head_dim is None else self.head_dim
        if head_dim != x.size(-1):
            raise ValueError(f"Configured head_dim {head_dim} does not match input head_dim {x.size(-1)}.")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, but got {head_dim}.")
        return head_dim

    def _resolve_section_sizes(self, pair_dim: int) -> Tuple[int, int, int]:
        if self.section_sizes is None:
            section_sizes = (pair_dim - 2 * (pair_dim // 3), pair_dim // 3, pair_dim // 3)
        else:
            if len(self.section_sizes) != 3:
                raise ValueError(f"section_sizes must contain 3 values, but got {self.section_sizes}.")
            section_sizes = tuple(self.section_sizes)
        if sum(section_sizes) != pair_dim:
            raise ValueError(f"section_sizes sum must equal head_dim // 2 ({pair_dim}), but got {section_sizes}.")
        return section_sizes

    def _build_base_freqs(self, max_seq_len: int, head_dim: int) -> torch.Tensor:
        pair_dim = head_dim // 2
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.tensor_factory_kwargs.get("device"),
                )
                / head_dim
            )
        )
        positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        base_freqs = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        if base_freqs.shape[-1] != pair_dim:
            raise RuntimeError(f"Expected base freqs last dim {pair_dim}, but got {base_freqs.shape[-1]}.")
        self.register_buffer("base_freqs", base_freqs, persistent=False)
        return self.base_freqs

    def _get_base_freqs(self, grid_sizes: torch.Tensor, head_dim: int, device: torch.device) -> torch.Tensor:
        needed_seq_len = int(grid_sizes.max().item())
        if not hasattr(self, "base_freqs") or self.base_freqs.size(0) < needed_seq_len:
            return self._build_base_freqs(needed_seq_len, head_dim).to(device=device)
        return self.base_freqs.to(device=device)

    def build_freqs_list(
        self,
        grid_sizes: torch.Tensor,
        head_dim: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if grid_sizes.dim() != 2 or grid_sizes.size(1) != 3:
            raise ValueError(f"grid_sizes must be [B, 3], but got {tuple(grid_sizes.shape)}.")
        pair_dim = head_dim // 2
        section_sizes = self._resolve_section_sizes(pair_dim)
        base_freqs = self._get_base_freqs(grid_sizes, head_dim, device)
        freqs_sections = base_freqs.split(section_sizes, dim=1)

        freqs_list = []
        for f, h, w in grid_sizes.tolist():
            freqs_i = torch.cat(
                [
                    freqs_sections[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_sections[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_sections[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(f * h * w, 1, pair_dim)
            freqs_list.append(freqs_i)
        return freqs_list

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        head_dim = self._resolve_head_dim(x)
        if freqs_list is None:
            freqs_list = self.build_freqs_list(grid_sizes, head_dim, x.device)
        return _apply_grid_rope_impl(x, grid_sizes, freqs_list)

    def extra_repr(self) -> str:
        return (
            f"{self.head_dim=}, {self.theta=}, {self.section_sizes=}, {self.max_seq_len=}"
        ).replace("self.", "")
