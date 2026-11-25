from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional Eagle/MTP draft model directory.
        num_draft_tokens: Number of draft tokens to verify per step.
        enable_deepseek_v4_mtp: Opt in to classic DeepSeek V4 NextN (C1-C3, K1-K3).
        nextn_branch_draft_lengths: Ordered extra candidate depths for single-owner eager NextN.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        pipeline_parallel_stage: Pipeline stage index for this engine.
        master_addr: Address used to bootstrap distributed communication.
        master_port: TCP port used to bootstrap distributed communication.
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        static_cache_capacity: Physical static KV rows (1 or 2); prefills are separate and decode can batch both rows.
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        enable_prefix_caching: Whether to reuse KV cache across requests.
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_parallel_stage: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    moe_ep_backend: str = "disabled"
    moe_ep_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False
    attn_backend: str = "default"
    use_mla: bool = False
    pre_transpose: bool = False
    weight_load_mode: str = "async"
    skip_load: bool = False
    kv_transfer_config: Optional[KVTransferConfig] = None
    enable_prefix_caching: bool = True
    static_cache_capacity: int = 1
    chunked_prefill_size: int = 0
    enable_deepseek_v4_mtp: bool = False
    nextn_branch_draft_lengths: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.enable_deepseek_v4_mtp and (
            not self.draft_model_path
            or type(self.num_draft_tokens) is not int
            or self.num_draft_tokens not in (1, 2, 3)
            or type(self.max_batch_size) is not int
            or self.max_batch_size not in (1, 2, 3)
            or self.cache_type != "paged"
            or self.attn_backend != "paged-attn"
            or self.block_size != 256
            or self.enable_prefix_caching
            or self.chunked_prefill_size
            or self.skip_load
            or self.pipeline_parallel_size != 1
            or self.moe_ep_backend != "disabled"
            or self.moe_ep_size != 1
            or (self.kv_transfer_config is not None and self.kv_transfer_config.kv_connector)
        ):
            raise ValueError(
                "classic DeepSeek V4 MTP requires a Draft and paged-attn "
                "with 256-token pages, 1-3 requests, 1-3 draft tokens, local KV, "
                "and TP expert placement"
            )
        lengths = self.nextn_branch_draft_lengths
        if lengths is not None:
            if (
                not self.enable_deepseek_v4_mtp
                or self.enable_graph
                or self.max_batch_size != 1
                or not isinstance(lengths, (list, tuple))
                or not lengths
                or any(type(k) is not int or k not in (1, 2, 3) for k in lengths)
                or len(set(lengths)) != len(lengths)
                or min(lengths) > self.num_draft_tokens
            ):
                raise ValueError(
                    "classic NextN serial branches require eager C1 and distinct "
                    "K1/K2/K3 candidate depths with at least one within the draft limit"
                )
            self.nextn_branch_draft_lengths = tuple(lengths)
        if (
            type(self.static_cache_capacity) is not int
            or self.static_cache_capacity < 1
        ):
            raise ValueError("static_cache_capacity must be a positive integer")
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if not 0 <= self.pipeline_parallel_stage < self.pipeline_parallel_size:
            raise ValueError(
                "pipeline_parallel_stage must be in [0, pipeline_parallel_size)"
            )
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")
        if type(self.chunked_prefill_size) is not int or (
            self.chunked_prefill_size < 0
        ):
            raise ValueError("chunked_prefill_size must be a non-negative integer")
