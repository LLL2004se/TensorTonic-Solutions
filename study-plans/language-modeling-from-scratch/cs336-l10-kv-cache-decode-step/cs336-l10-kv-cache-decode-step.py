import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:

    # Append new keys and values to cache
    new_key_cache = torch.cat(
        [key_cache, new_key.unsqueeze(2)], dim=2
    )
    new_value_cache = torch.cat(
        [value_cache, new_value.unsqueeze(2)], dim=2
    )

    B, Hq, D = query.shape
    G = num_query_heads // num_kv_heads

    # Convert to float32 for attention calculations
    q = query.float()
    k = new_key_cache.float()
    v = new_value_cache.float()

    # Map each query head to its corresponding KV head
    q = q.reshape(B, num_kv_heads, G, D)

    # Attention scores: (B, Hkv, G, S+1)
    scores = torch.einsum(
        "bhgd,bhsd->bhgs", q, k
    ) / (D ** 0.5)

    # Softmax across sequence dimension
    attn = torch.softmax(scores, dim=-1)

    # Weighted sum: (B, Hkv, G, D)
    output = torch.einsum(
        "bhgs,bhsd->bhgd", attn, v
    )

    # Restore expected output shape, dtype and device
    output = output.reshape(B, Hq, D).to(
        dtype=query.dtype,
        device=query.device
    )

    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache
    }