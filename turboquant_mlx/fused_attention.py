"""Fused TurboQuant attention using v3 kernels.

Uses fused Metal kernels from kernels.py to perform Q@K^T directly from 
packed storage. WHT is performed inside the kernel.
"""

import mlx.core as mx
import math
from turboquant_mlx.kernels import (
    packed_fused_qk_scores,
    packed_dequantize,
)


def turboquant_attention(
    queries: mx.array,
    cache,
    attn_scale: float,
    mask=None,
    v_buffer=None,
) -> mx.array:
    """Full attention using fused kernels.

    Args:
        queries: (B, n_heads, 1, dim)
        cache: TurboQuantKVCache with packed K/V
        attn_scale: 1/sqrt(dim)
        mask: optional attention mask
        v_buffer: unused in this version, kept for compatibility

    Returns:
        (B, n_heads, 1, dim) attention output
    """
    B, n_q_heads, S_q, dim = queries.shape
    total = cache.offset
    n_kv_heads = cache.k_packed.shape[1]
    n_rep = n_q_heads // n_kv_heads

    outputs = []
    for b in range(B):
        # --- K attention scores via fused kernel ---
        kp = cache.k_packed[b, :, :total, :]
        kn = cache.k_norms[b, :, :total]

        if n_rep > 1:
            kp = mx.repeat(kp, n_rep, axis=0)
            kn = mx.repeat(kn, n_rep, axis=0)

        q = queries[b, :, 0, :]  # (n_q_heads, dim)

        # Fused scores: codebook lookups + WHT + dot
        scores = packed_fused_qk_scores(
            q, kp, kn,
            cache._k_q.centroids,
            cache._k_q.signs,
            dim, cache.quant_bits,
        )

        scores = scores * attn_scale

        # Mask
        if mask is not None:
            m = mask
            if m.ndim == 4:
                m = m[min(b, m.shape[0] - 1)]
                if m.ndim == 3:
                    m = m[:, 0, :]
                    if m.shape[0] == 1:
                        m = mx.broadcast_to(m, (n_q_heads, total))
            elif m.ndim == 3:
                m = m[min(b, m.shape[0] - 1), 0, :]
                m = mx.broadcast_to(m.reshape(1, -1), (n_q_heads, total))
            scores = scores + m

        weights = mx.softmax(scores, axis=-1)

        # --- V: use pre-dequanted buffer if available, else dequant from packed ---
        if v_buffer is not None:
            v_deq = v_buffer[b]  # (n_kv_heads, total, v_dim)
            if n_rep > 1:
                v_deq = mx.repeat(v_deq, n_rep, axis=0)
        else:
            vp = cache.v_packed[b, :, :total, :]
            vn = cache.v_norms[b, :, :total]
            v_dim = cache._v_dim
            v_deq = packed_dequantize(
                vp, vn,
                cache._v_q.centroids,
                cache._v_q.signs,
                v_dim, cache.quant_bits,
            ).reshape(n_kv_heads, total, v_dim)
            if n_rep > 1:
                v_deq = mx.repeat(v_deq, n_rep, axis=0)

        out = weights[:, None, :] @ v_deq.astype(queries.dtype)
        outputs.append(out)

    return mx.stack(outputs, axis=0)
