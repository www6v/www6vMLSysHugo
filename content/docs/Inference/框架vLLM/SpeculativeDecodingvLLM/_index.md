---
title: (实战)[vLLM]投机解码 +
date: 2023-11-18 16:28:40
weight: 3
tags:
  - infer
categories: 
  - AIGC
  - infer 
---

<p></p>
<!-- more -->




# **Speculating with a draft model[1]**

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b", # verify 小模型
    tensor_parallel_size=1,
    speculative_model="facebook/opt-125m",    # draft 小模型
    num_speculative_tokens=5, # 一次生成5个token 
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```powershell
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
       --model facebook/opt-6.7b \
       --seed 42 -tp 1 \
       --speculative_model facebook/opt-125m \
       --use-v2-block-manager \
       --num_speculative_tokens 5 \
       --gpu_memory_utilization 0.8 \
```

- Small LM
    
     DistillSpec
    

# **Speculating by matching n-grams in the prompt[1,2]**

```powershell
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_model="[ngram]", # ngram
    num_speculative_tokens=5,   # 一次生成5个token
    ngram_prompt_lookup_max=4,  
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

- lookahead
    
    

# **Speculating using MLP speculators[1]**

```powershell
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",
    speculative_draft_tensor_parallel_size=1,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

- Medusa

# 参考

1. [Speculative decoding in vLLM](https://docs.vllm.ai/en/stable/models/spec_decode.html)  3种类型

 2. [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)

1xx. [A Hacker’s Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4) v ***

[A Hacker’s Guide to Speculative Decoding in vLLM](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)  pdf

1xx. [Optimizing attention for spec decode can reduce latency / increase throughput](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
