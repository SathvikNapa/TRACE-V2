"""
mamba_ssm_lite.py
------------------
Pure-PyTorch, bidirectional, chunk-parallel selective-scan ("Mamba-lite")
audio encoder for TRACE v2.

Why pure PyTorch instead of the `mamba_ssm` package: that package needs a
compiled CUDA kernel (`nvcc`), which isn't available on this cluster's login
node (and almost certainly not on the compute nodes either, same toolchain).

Why *chunked* (unlike a simpler from-scratch Mamba block one might write for
short sequences): TRACE dyads interleave up to ~256 windows per speaker, so
sequences run up to ~512 tokens. A naive `for t in range(T)` sequential scan
pays that as 512 Python/CUDA-launch-bound steps per layer, per batch, every
epoch -- real overhead at this length (unlike a <=7-step sequence, where it's
free). This module instead scans in chunks of size `chunk_size` (default 32):
within a chunk the recurrence runs sequentially (cheap, small), chunks are
combined via a second short sequential pass over just the O(T/chunk_size)
chunk-boundary states, and a third pass (same cost as the first) fills in
the correct per-timestep states given the right carry-in. Total sequential
depth is `2*chunk_size + T/chunk_size` instead of `T` -- for T=512,
chunk_size=32 that's 80 steps instead of 512, with everything else
vectorized across batch/chunks/channels as ordinary tensor ops.

Interface matches NaturalnessTransformer / BaseTransformerAudioEncoder in the
main TRACE script exactly (forward_tokens, encode_pooled), so it plugs into
the existing fusion-model wrappers with no changes to CatCtxRelEncoder /
CrossAttnFusion.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kpm_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """True means PAD. Local copy to keep this file import-independent."""
    rng = torch.arange(T, device=lengths.device)[None, :]
    return rng >= lengths[:, None]


def _chunk_scan(
    a: torch.Tensor,  # [N, T, d_inner, d_state] decay factors, N = flattened (batch*chunks)
    b: torch.Tensor,  # [N, T, d_inner, d_state] input injections
    init_state: torch.Tensor,  # [N, d_inner, d_state]
) -> torch.Tensor:
    """
    Sequential linear recurrence state_t = a_t * state_{t-1} + b_t over the
    T dimension, for all N "lanes" in parallel. Returns the full per-step
    state sequence [N, T, d_inner, d_state]. This is the shared primitive
    used for both the within-chunk passes in _selective_scan_chunked below;
    T here is the *chunk* size (small), not the full sequence length.
    """
    T = a.shape[1]
    state = init_state
    out = []
    for t in range(T):
        state = a[:, t] * state + b[:, t]
        out.append(state)
    return torch.stack(out, dim=1)


def selective_scan_chunked(
    a: torch.Tensor,  # [B, L, d_inner, d_state]
    b: torch.Tensor,  # [B, L, d_inner, d_state]
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Chunked selective scan: state_t = a_t * state_{t-1} + b_t, state_0 = 0.
    Returns the full per-timestep state sequence [B, L, d_inner, d_state].
    """
    B, L, d_inner, d_state = a.shape
    C = min(chunk_size, L)
    num_chunks = (L + C - 1) // C
    L_pad = num_chunks * C

    if L_pad != L:
        pad = L_pad - L
        a = F.pad(a, (0, 0, 0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))

    def to_chunks(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, num_chunks, C, d_inner, d_state).reshape(B * num_chunks, C, d_inner, d_state)

    a_c, b_c = to_chunks(a), to_chunks(b)

    # Pass 1: within-chunk states assuming zero carry-in.
    zero_init = a_c.new_zeros(B * num_chunks, d_inner, d_state)
    local_states = _chunk_scan(a_c, b_c, zero_init)  # [B*num_chunks, C, d_inner, d_state]
    chunk_local_final = local_states[:, -1].view(B, num_chunks, d_inner, d_state)
    chunk_A = a.view(B, num_chunks, C, d_inner, d_state).prod(dim=2)  # [B, num_chunks, d_inner, d_state]

    # Pass 2: sequential carry across the (small) number of chunks.
    carry = a.new_zeros(B, d_inner, d_state)
    carry_ins = []
    for c in range(num_chunks):
        carry_ins.append(carry)
        carry = chunk_A[:, c] * carry + chunk_local_final[:, c]
    carry_in = torch.stack(carry_ins, dim=1).reshape(B * num_chunks, d_inner, d_state)

    # Pass 3: within-chunk states with the correct carry-in -> true per-step states.
    full_states = _chunk_scan(a_c, b_c, carry_in)  # [B*num_chunks, C, d_inner, d_state]
    full_states = full_states.view(B, num_chunks, C, d_inner, d_state).reshape(B, L_pad, d_inner, d_state)
    return full_states[:, :L]


class _DirectionalSSM(nn.Module):
    """One direction (forward or backward, via input reversal) of the S6 recurrence."""

    def __init__(self, d_inner: int, d_state: int, dt_rank: int, chunk_size: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        """x: [B, L, d_inner] -> y: [B, L, d_inner]"""
        if reverse:
            x = x.flip(dims=[1])

        dt_rank = self.dt_proj.in_features
        dbl = self.x_proj(x)
        dt, Bmat, Cmat = torch.split(dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B,L,d_inner,d_state]
        deltaB_u = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)  # [B,L,d_inner,d_state]

        state_seq = selective_scan_chunked(deltaA, deltaB_u, chunk_size=self.chunk_size)
        y = torch.einsum("bldn,bln->bld", state_seq, Cmat) + x * self.D

        if reverse:
            y = y.flip(dims=[1])
        return y


class SelectiveSSMBlock(nn.Module):
    """
    Bidirectional Mamba-style block: causal depthwise conv -> forward+backward
    selective scan (combined) -> gated output, pre-norm residual. Bidirectional
    because TRACE's task is full-sequence classification (its own LLaMA blocks
    use bidirectional attention too), not causal generation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: Optional[int] = None,
        dropout: float = 0.1,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        rank = dt_rank or max(d_model // 16, 4)

        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel,
            groups=self.d_inner, padding=conv_kernel - 1,
        )
        self.fwd_ssm = _DirectionalSSM(self.d_inner, d_state, rank, chunk_size)
        self.bwd_ssm = _DirectionalSSM(self.d_inner, d_state, rank, chunk_size)
        self.combine = nn.Linear(2 * self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: [B,L,d_model], pad_mask: [B,L] bool, True=PAD."""
        B, L, _ = x.shape
        residual = x
        h = self.norm(x)

        xz = self.in_proj(h)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x_in = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_in = F.silu(x_in)
        x_in = x_in.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        y_fwd = self.fwd_ssm(x_in, reverse=False)
        y_bwd = self.bwd_ssm(x_in, reverse=True)
        y = self.combine(torch.cat([y_fwd, y_bwd], dim=-1))

        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        return residual + out


class NaturalnessMambaAudioEncoder(nn.Module):
    """
    Drop-in replacement for NaturalnessTransformer: same forward_tokens /
    encode_pooled interface, same [B,T] True=PAD masking convention
    (_kpm_from_lengths), so it plugs into NaturalnessFusionLLaMAModel /
    NaturalnessCrossAttnModel with no changes to those wrappers beyond a
    `backbone` switch.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        dropout: float = 0.1,
        chunk_size: int = 32,
        **_ignored,  # absorbs n_heads/max_len etc. passed uniformly by callers
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [SelectiveSSMBlock(d_model, d_state=d_state, dropout=dropout, chunk_size=chunk_size)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)

    def forward_tokens(
        self, x: torch.Tensor, lengths: torch.Tensor, return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        T = x.shape[1]
        pad_mask = _kpm_from_lengths(lengths, T)
        h = self.input_proj(x)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            h = block(h, pad_mask)
        h = self.final_norm(h)
        # No attention maps to return (this is an SSM) -- callers that
        # request return_attn should fall back to None and skip attn dumps.
        return h, (None if not return_attn else [])

    def encode_pooled(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h, _ = self.forward_tokens(x, lengths, return_attn=False)
        kpm = _kpm_from_lengths(lengths, T)
        valid = (~kpm).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (h * valid).sum(dim=1) / denom
