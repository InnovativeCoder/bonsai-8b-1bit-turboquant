import subprocess
import json
import os
import sys

# Configuration
python_path = "/opt/anaconda3/bin/python3"
model_path = "prism-ml/Bonsai-8B-mlx-1bit"
context_lengths = [512, 1024, 2048, 4096]
max_tokens_to_generate = 50

def run_benchmark(context_len, use_turboquant):
    """Run a single benchmark pass in a subprocess."""
    script = f"""
import time
import mlx.core as mx
from mlx_lm import load
import sys
import os
import json

# Ensure local turboquant_mlx is importable
sys.path.insert(0, os.getcwd())

model_path = "{model_path}"
context_len = {context_len}
max_gen = {max_tokens_to_generate}
use_tq = {use_turboquant}

if use_tq:
    from turboquant_mlx.adaptive import make_adaptive_cache
    from turboquant_mlx.patch import apply_patch
    apply_patch()

# Load model
model, tokenizer = load(model_path)

# Prepare context (synthetic)
prompt = "The " * (context_len - 1)
input_ids = mx.array(tokenizer.encode(prompt)[:context_len])[None]

# Setup Cache
if use_tq:
    cache = make_adaptive_cache(len(model.layers), bits=3, fp16_layers=4)
else:
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in range(len(model.layers))]

# 1. Prefill
t0 = time.perf_counter()
logits = model(input_ids, cache=cache)
mx.eval(logits)
prefill_time = time.perf_counter() - t0

# 2. Decode
token = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
t1 = time.perf_counter()
for _ in range(max_gen):
    logits = model(token.reshape(1, 1), cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    tokens.append(token.item())
decode_time = time.perf_counter() - t1

# 3. Memory calculation using built-in nbytes
kv_mem_bytes = sum(c.nbytes for c in cache if hasattr(c, "nbytes"))

results = {{
    "prefill_s": prefill_time,
    "decode_tps": len(tokens) / decode_time,
    "kv_mem_mb": kv_mem_bytes / (1024 * 1024)
}}
print(json.dumps(results))
"""
    try:
        # We use a clean process for every run to avoid Metal / Memory pollution
        result = subprocess.check_output([python_path, "-c", script], stderr=subprocess.STDOUT, text=True)
        # Find the line that is actually JSON
        for line in result.splitlines():
            try:
                return json.loads(line)
            except:
                continue
        return None
    except Exception as e:
        print(f"Error running context {{context_len}} (TQ={{use_turboquant}}): {{e}}")
        return None

def main():
    print(f"Benchmarking Bonsai-8B (1-bit)")
    print(f"{'Context':<10} | {'Mode':<15} | {'TPS':>8} | {'Prefill':>10} | {'KV Mem':>10}")
    print("-" * 65)

    for c_len in context_lengths:
        # 1. Standard
        res_std = run_benchmark(c_len, False)
        if res_std:
            print(f"{c_len:<10} | {'Standard':<15} | {res_std['decode_tps']:>8.2f} | {res_std['prefill_s']:>9.2f}s | {res_std['kv_mem_mb']:>8.1f} MB")
        
        # 2. TurboQuant
        res_tq = run_benchmark(c_len, True)
        if res_tq:
            print(f"{c_len:<10} | {'TurboQuant':<15} | {res_tq['decode_tps']:>8.2f} | {res_tq['prefill_s']:>9.2f}s | {res_tq['kv_mem_mb']:>8.1f} MB")
        
        print("-" * 65)

if __name__ == "__main__":
    main()
