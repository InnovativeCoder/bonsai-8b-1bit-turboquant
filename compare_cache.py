import mlx.core as mx
from mlx_lm.models.cache import KVCache
from turboquant_mlx.cache import TurboQuantKVCache

# Configuration for Bonsai-8B / Llama-3-8B style model
num_layers = 32
num_kv_heads = 8 # GQA (Grouped Query Attention)
head_dim = 128
seq_len = 2048 # Hypothetical target context length
bits = 3

print(f"--- Cache Memory Comparison (Bonsai-8B, {seq_len} tokens) ---")

# 1. Standard FP16 Cache (mlx_lm default)
# Each layer stores K and V of shape (1, num_kv_heads, seq_len, head_dim) in float16 (2 bytes)
standard_bytes_per_layer = 1 * num_kv_heads * seq_len * head_dim * 2 * 2 # *2 for K and V
total_standard_mb = (standard_bytes_per_layer * num_layers) / (1024 * 1024)

# 2. TurboQuant 3-bit Cache
# Based on the implementation in cache.py
tq = TurboQuantKVCache(bits=bits)
# We simulate a prefill of seq_len tokens to see the size
dummy_k = mx.zeros((1, num_kv_heads, seq_len, head_dim))
dummy_v = mx.zeros((1, num_kv_heads, seq_len, head_dim))
tq.update_and_fetch(dummy_k, dummy_v)
total_tq_mb = (tq.nbytes * num_layers) / (1024 * 1024)

compression_ratio = total_standard_mb / total_tq_mb

print(f"Standard FP16 Cache:  {total_standard_mb:.2f} MB")
print(f"TurboQuant 3-bit:      {total_tq_mb:.2f} MB")
print(f"Compression Relief:   {compression_ratio:.2f}x less memory")
print("-" * 50)
print(f"Benefit: At {seq_len} tokens, you save {total_standard_mb - total_tq_mb:.2f} MB of VRAM.")
print(f"This allows for much longer context lengths or higher batch sizes on the same hardware.")
