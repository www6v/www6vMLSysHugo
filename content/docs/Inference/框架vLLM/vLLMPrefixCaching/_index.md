---
title:  (实现)[vLLM]Prefix Caching + 
date: 2025-05-14 19:53:42
weight: 14
tags:
  - infer
categories: 
  - AIGC
  - infer 
---

<p></p>
<!-- more -->




# Prefix caching

```python
llm.inference(
    input_tokens: list[int],  # N tokens
    previous_kv_cache: list[Tensor],  # N tokens' kv cache ∪ <N
) -> output_tokens, new_kv_cache

output_tokens:  # N' new tokens
new_kv_cache:  # kv cache of N + N' tokens

```

Key: tokens
Value: KV cache tensors

```python
class KVCacheStore:
    def store(tokens, kv_cache_tensors):
        pass

    def retrieve(tokens) -> kv_cache_tensors:
        pass

```

# Prefix-based matching

- Tokens 1: ABCDE -> [KV1, KV2, KV3, KV4, KV5]
- Tokens 2: ABCDF -> [KV1, KV2, KV3, KV4, KV6]

```python
kv_cache_store.store("ABCDE", [KV1, KV2, KV3, KV4, KV5])
kv_cache_store.retrieve("ABCD") -> [KV1, KV2, KV3, KV4]

```

- "Trie"
- "ABCDEF" -> "AB", "CD", "EF" -> list of chunked prefix hashes

```python
prefix_hash = ""
for chunk in chunked_tokens:  # ["AB", "CD", "EF"]
    chunk_hash = hash(prefix_hash + chunk)
    prefix_hash = chunk_hash

# Given chunked prefix hashes, chunked kv cache
# store
for chunk_hash, chunk_kv in zip(...):
    redis.put(chunk_hash, chunk_kv)

# retrieve
for chunk_hash in ...:
    kv_chunk = redis.get(chunk_hash)
    if kv_chunk is None:
        break

```

# Eviction

- LRU, LFU...
- "ABCDEF" --> ["AB", KV1], ["CD", KV2], ["EF", KV3]

# 参考

[**[EP05] vllm从开源到部署，Prefix Caching和开源答疑**](https://www.notion.so/EP05-vllm-Prefix-Caching-1fbbfe21108480db82c8d7cb6573eb5e?pvs=21)
