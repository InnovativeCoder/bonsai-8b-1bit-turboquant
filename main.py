import time
import mlx.core as mx
from mlx_lm import load
from turboquant_mlx.adaptive import make_adaptive_cache
from turboquant_mlx.patch import apply_patch

# Apply fused Metal attention patch
apply_patch()

# Load model and tokenizer
model, tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")

# Setup adaptive cache (bits 3, first/last 4 layers FP16)
cache = make_adaptive_cache(len(model.layers), bits=3, fp16_layers=4)

prompt = "Explain quantum computing in simple terms."
input_ids = mx.array(tokenizer.encode(prompt))[None]

print(f"Generating response with TurboQuant optimization...\n")

start_time = time.perf_counter()

# Prefill
logits = model(input_ids, cache=cache)
mx.eval(logits)
token = mx.argmax(logits[:, -1, :], axis=-1)
tokens = [token.item()]

# Manual generation loop
max_tokens = 256
for _ in range(max_tokens - 1):
    logits = model(token.reshape(1, 1), cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    tok_id = token.item()
    tokens.append(tok_id)
    
    # Print incrementally
    print(tokenizer.decode([tok_id]), end="", flush=True)
    
    if tok_id == tokenizer.eos_token_id:
        break

print("\n")
end_time = time.perf_counter()

# Programmatic Calculation
response_text = tokenizer.decode(tokens)
total_tokens = len(tokens)
total_time = end_time - start_time
tokens_per_sec = total_tokens / total_time

print(f"--- Performance Metrics ---")
print(f"Total Tokens: {total_tokens}")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Overall Throughput: {tokens_per_sec:.2f} tokens/sec")