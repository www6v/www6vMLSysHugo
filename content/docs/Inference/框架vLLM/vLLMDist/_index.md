---
title:  (实现)[vLLM]分布式 *
date: 2024-05-14 19:51:55
weight: 11
tags:
  - infer
categories: 
  - AIGC
  - infer 
---

<p></p>
<!-- more -->



## Feature

- Distributed Inference

  - Why distributed inference?

    - Infra-side

      - Communication device:

        - NVLink: direct communication between GPUs
        - Infinity Band: High-speed connection between nodes
        - RDMA: Remote direct memory access
          - RDMA NIC
          - Software solution
          - Key advantage: bypass operating system / zero copy

      - Communication library: 

        ```
        dlms/distributed/device_communicators
        ```

        - PyDCL: communication for NVIDIA
        - shared memory : OS
        - custom allreduce - A kernel jsut for all reduce operation
          - Before:
            - 0 machine: [0]
            - 1 machine: [1]
            - 2 machine: [2]
            - 3 machine: [3]
          - After:
            - 0 machine: [0,1,2,3]
            - 1 machine: [0,1,2,3]
            - 2 machine: [0,1,2,3]
            - 3 machine: [0,1,2,3]
        - torch.distributed : provide wide support to a list of communication library

      - GroupCoordinator

    - Algorithm-side

      - [TP]
      - `vlms/model_executor/models/llama.py`

- Pipeline parallel

  - Much less requirement to device--device connection hardware
  - Cost: not improve latency
    - Tensor parallel: directly improve latency
  - Algorithm-side:
    - Worker in charge of a subset of layers
      - `~~vlms/model_executor/models/llama.py~~`
      - `vlms/model_executor/models/llama.py`
      - self.start_layer --> self.end_layer
      - between workers: communicate IntermediateTensor
      - get_pp_group()
      - `vlms/worker/model_runner.py`: search `get_pp_group()`

- Expert parallel & data parallel (advanced)

  - Why expert parallel:
    - Mistral / Mixtral / Deepseek model: Mixture of Experts (MoE)
      - Only for linear layers
      - Normal MoE: all weights participant in computation
      - MoE: expert as granularity, only a small subset of experts participate the computation, this subset of experts may be different between request
    - Place different experts onto different GPUs --> expert parallel
    - Algorithm:
      - Expert parallel:
        - Shuffle (deepsp communication kernel)
        - Forward
        - Shuffle back
    - TP is for attention, EP is for linear layers.
    - Shared expert will have high load --> duplicate shared expert.

- DP (data parallel)

  - max tp << ep needed
  - tp < # attention head
  - basic linear layer "degree of parallism" >> basic attention layer tp "degree of parallism", parallel request to raise attention "degree of parallism"
  - Difficult to implement in practice:
    - request padding to avoid deadlock.

- Types of distributed inference: TP / PP / EP / DP

- PD Disaggregation



# 代码

### TP

https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py

```python
_TP: Optional[GroupCoordinator] = None  ### TP

def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP
```



```python

class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.
    """

    # available attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_device_communicator: bool  # whether to use device communicator
    device_communicator: DeviceCommunicatorBase  # device communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster
```

------

https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py

```python
    def all_reduce(self,
                   in_tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}")

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllReduce(buffer_type(in_tensor.data_ptr()),
                                buffer_type(out_tensor.data_ptr()),
                                in_tensor.numel(),
                                ncclDataTypeEnum.from_torch(in_tensor.dtype),
                                ncclRedOpTypeEnum.from_torch(op), self.comm,
                                cudaStream_t(stream.cuda_stream))
        return out_tensor

    def all_gather(self,
                   output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}")
        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()), input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype), self.comm,
            cudaStream_t(stream.cuda_stream))
```

### TP in llama

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

```python
class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()  ### 
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size   ### 
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  ### 
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor",
                                             1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
```

### PP

```python
@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers( ## start_layer end_layer
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int] = tuple()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
```

# 参考

[[EP02]分布式推理优化，vllm源码解读](https://www.notion.so/EP02-vllm-1f4bfe21108480de94fecfa7fe5c474e?pvs=21)

