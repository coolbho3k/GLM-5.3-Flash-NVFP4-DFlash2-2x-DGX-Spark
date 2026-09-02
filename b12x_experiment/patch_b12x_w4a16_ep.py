#!/usr/bin/env python3
"""Backport FlashInfer PR #4302 W4A16 expert-map support and wire vLLM."""

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
B12X = SITE / "flashinfer/fused_moe/cute_dsl/b12x_moe.py"
DISPATCH = SITE / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
VLLM = SITE / "vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one EP patch site in {path}, found {count}")
    path.write_text(source.replace(old, new, 1))


replace_once(
    B12X,
    """def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


""",
    """def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _resolve_cuda_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _prepare_ep_expert_map(
    expert_map: torch.Tensor,
    *,
    num_local_experts: int,
    num_experts: int,
    device=None,
) -> torch.Tensor:
    if not isinstance(expert_map, torch.Tensor):
        raise TypeError("expert_map must be a torch.Tensor")
    if expert_map.dtype != torch.int32:
        raise TypeError(
            f"expert_map must have dtype torch.int32, got {expert_map.dtype}"
        )
    if expert_map.ndim != 1 or not expert_map.is_contiguous():
        raise ValueError("expert_map must be a contiguous rank-1 tensor")
    if expert_map.numel() != num_experts:
        raise ValueError(
            f"expert_map must have num_experts={num_experts} entries, "
            f"got {expert_map.numel()}"
        )
    if device is not None:
        expected = _resolve_cuda_device(device)
        if _resolve_cuda_device(expert_map.device) != expected:
            raise ValueError(
                f"expert_map must be on {expected}, got {expert_map.device}"
            )
    if _is_cuda_graph_capturing():
        raise RuntimeError("expert_map must be validated before CUDA graph capture")
    values = expert_map.detach().cpu().tolist()
    invalid = [v for v in values if v < -1 or v >= num_local_experts]
    if invalid:
        raise ValueError(
            "expert_map values must be -1 or valid local expert ids, "
            f"found {invalid[0]} for num_local_experts={num_local_experts}"
        )
    mapped = sorted(v for v in values if v >= 0)
    if mapped != list(range(num_local_experts)):
        raise ValueError(
            "expert_map must map every local expert id exactly once: "
            f"expected {list(range(num_local_experts))}, got {mapped}"
        )
    return expert_map


""",
)
replace_once(
    B12X,
    """    num_local_experts: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
""",
    """    num_local_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
""",
)
replace_once(
    B12X,
    """    if num_local_experts != num_experts:
        raise NotImplementedError(
            f"b12x_fused_moe does not yet support Expert Parallelism "
            f"(num_local_experts={num_local_experts} != num_experts={num_experts}). "
            f"Use a different MoE backend for EP configurations."
        )
""",
    """    if num_local_experts != num_experts or expert_map is not None:
        from .blackwell_sm12x.moe_dispatch import _normalize_quant_mode

        if _normalize_quant_mode(quant_mode, activation_precision) != "w4a16":
            raise NotImplementedError(
                "b12x_fused_moe supports expert_map only with quant_mode='w4a16'"
            )
        if expert_map is None:
            raise ValueError("Expert Parallelism requires expert_map")
        if expert_map.dtype != torch.int32:
            raise TypeError(
                f"expert_map must have dtype torch.int32, got {expert_map.dtype}"
            )
        if expert_map.ndim != 1 or not expert_map.is_contiguous():
            raise ValueError("expert_map must be a contiguous rank-1 tensor")
        if expert_map.numel() != num_experts:
            raise ValueError(
                f"expert_map must have num_experts={num_experts} entries, "
                f"got {expert_map.numel()}"
            )
""",
)
replace_once(
    B12X,
    """        num_local_experts=num_local_experts,
        scatter_output=output,
""",
    """        num_local_experts=num_local_experts,
        expert_map=expert_map,
        scatter_output=output,
""",
)
replace_once(
    B12X,
    """        num_local_experts: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
""",
    """        num_local_experts: Optional[int] = None,
        expert_map: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.bfloat16,
""",
)
replace_once(
    B12X,
    """        self.num_local_experts = num_local_experts or num_experts

        if self.num_local_experts != self.num_experts:
            raise NotImplementedError(
                f"B12xMoEWrapper does not yet support Expert Parallelism "
                f"(num_local_experts={self.num_local_experts} != "
                f"num_experts={self.num_experts}). "
                f"Use a different MoE backend for EP configurations."
            )
        self.output_dtype = output_dtype
""",
    """        self.num_local_experts = (
            num_local_experts if num_local_experts is not None else num_experts
        )
        self.output_dtype = output_dtype
""",
)
replace_once(
    B12X,
    """        self.source_format = source_format

        # Pre-allocated objects. Both workspace slots may be populated so
""",
    """        self.source_format = source_format

        if self.num_local_experts != self.num_experts or expert_map is not None:
            if self.quant_mode != "w4a16":
                raise NotImplementedError(
                    "B12xMoEWrapper supports expert_map only with "
                    "quant_mode='w4a16'"
                )
            if expert_map is None:
                raise ValueError("Expert Parallelism requires expert_map")
            expert_map = _prepare_ep_expert_map(
                expert_map,
                num_local_experts=self.num_local_experts,
                num_experts=self.num_experts,
                device=self.device,
            ).clone()
        self.expert_map = expert_map

        # Pre-allocated objects. Both workspace slots may be populated so
""",
)
replace_once(
    B12X,
    """            num_local_experts=self.num_local_experts,
            scatter_output=moe_output,
""",
    """            num_local_experts=self.num_local_experts,
            expert_map=self.expert_map,
            scatter_output=moe_output,
""",
)

replace_once(
    DISPATCH,
    """    num_local_experts: int,
    scatter_output: torch.Tensor,
    fast_math: bool = True,
""",
    """    num_local_experts: int,
    scatter_output: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    fast_math: bool = True,
""",
)
replace_once(
    DISPATCH,
    """    if int(prepared.num_experts) != int(num_local_experts):
        raise ValueError("num_local_experts must match w1_weight.shape[0] for W4A16.")
    num_tokens = int(topk_ids.size(0))
""",
    """    if int(prepared.num_experts) != int(num_local_experts):
        raise ValueError("num_local_experts must match w1_weight.shape[0] for W4A16.")
    if expert_map is not None:
        if expert_map.dtype != torch.int32:
            raise TypeError(
                f"expert_map must have dtype torch.int32, got {expert_map.dtype}"
            )
        if expert_map.ndim != 1 or not expert_map.is_contiguous():
            raise ValueError("expert_map must be a contiguous rank-1 tensor")
        if int(expert_map.numel()) != int(num_experts):
            raise ValueError(
                f"expert_map must have num_experts={int(num_experts)} entries, "
                f"got {int(expert_map.numel())}"
            )
        if expert_map.device != a.device:
            raise ValueError(
                f"expert_map must be on {a.device}, got {expert_map.device}"
            )
    num_tokens = int(topk_ids.size(0))
""",
)
replace_once(
    DISPATCH,
    """        expert_offsets=workspace.expert_offsets,
        expert_map=workspace.expert_map,
        fast_math=fast_math,
""",
    """        expert_offsets=workspace.expert_offsets,
        expert_map=expert_map if expert_map is not None else workspace.expert_map,
        fast_math=fast_math,
""",
)
replace_once(
    DISPATCH,
    """    num_local_experts: int,
    scatter_output: torch.Tensor,
    input_scales_are_reciprocal: bool = False,
""",
    """    num_local_experts: int,
    scatter_output: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    input_scales_are_reciprocal: bool = False,
""",
)
replace_once(
    DISPATCH,
    """    activation_precision = _activation_precision_from_quant_mode(quant_mode)

    num_tokens = topk_ids.size(0)
""",
    """    activation_precision = _activation_precision_from_quant_mode(quant_mode)

    if expert_map is not None and quant_mode != "w4a16":
        raise NotImplementedError(
            "expert_map-based expert parallelism is only supported for "
            f"quant_mode='w4a16', got quant_mode={quant_mode!r}"
        )

    num_tokens = topk_ids.size(0)
""",
)
replace_once(
    DISPATCH,
    """            num_local_experts=num_local_experts,
            scatter_output=scatter_output,
            fast_math=fast_math,
""",
    """            num_local_experts=num_local_experts,
            scatter_output=scatter_output,
            expert_map=expert_map,
            fast_math=fast_math,
""",
)

replace_once(
    VLLM,
    """        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
""",
    """        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.use_ep = moe_config.moe_parallel_config.use_ep
        # FC2 input scale tensor bound in process_weights_after_loading: the
""",
)
replace_once(
    VLLM,
    """        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False
""",
    """        # Static maps are supported. EPLB may mutate placement after the
        # wrapper has validated and cloned its map, so keep it gated off.
        return not moe_parallel_config.enable_eplb

    def supports_expert_map(self) -> bool:
        return True
""",
)
replace_once(
    VLLM,
    """    def _ensure_wrapper(self) -> None:
        \"\"\"Lazily create B12xMoEWrapper on first use.\"\"\"
        if self._wrapper is not None:
            return

        from flashinfer.fused_moe import B12xMoEWrapper

        self._wrapper = B12xMoEWrapper(
""",
    """    def _ensure_wrapper(self, expert_map: torch.Tensor | None) -> None:
        \"\"\"Lazily create B12xMoEWrapper on first use.\"\"\"
        if self._wrapper is not None:
            return
        if self.use_ep and expert_map is None:
            raise ValueError("FlashInfer B12X EP requires vLLM's expert_map")

        from flashinfer.fused_moe import B12xMoEWrapper

        self._wrapper = B12xMoEWrapper(
""",
)
replace_once(
    VLLM,
    """            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            swiglu_limit=self._swiglu_limit,
""",
    """            num_local_experts=self.num_local_experts,
            expert_map=expert_map if self.use_ep else None,
            activation=self._activation_str,
            swiglu_limit=self._swiglu_limit,
            quant_mode="w4a16" if self.use_ep else "nvfp4",
""",
)
replace_once(
    VLLM,
    """        self._ensure_wrapper()
        wrapper = self._wrapper
""",
    """        self._ensure_wrapper(expert_map)
        wrapper = self._wrapper
""",
)

print("Patched FlashInfer/vLLM B12X W4A16 expert parallelism")
