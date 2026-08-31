#!/usr/bin/env python3
"""Keep replicated hybrid-cache block tables at global-sequence width.

The GPU model runner sizes every cache group's block table with
``block_size * dcp_size`` tokens per table column. That is correct for the
target attention groups whose sequence KV is DCP-sharded, but not for GLM's
replicated recurrent state or the TP-sharded/sequence-replicated DFlash
sliding-window cache.

At DCP=2 the old calculation halves the Mamba table. With a 5,120-token
Mamba block and one speculative block, a 256K configuration gets 27 columns.
``mamba_get_block_table_tensor`` needs columns 26 and 27 as soon as a request
crosses 133,120 tokens, causing a deterministic CUDA gather assertion even
though the KV pool is almost empty. Size replicated groups with DCP=1 while
leaving target attention groups DCP-sharded.
"""

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"
)
MARKER = "LONG-CONTEXT-REPLICATED-GROUPS"


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one model-runner patch anchor, found {count}: "
            f"{old[:120]!r}"
        )
    return text.replace(old, new, 1)


text = TARGET.read_text()
if MARKER in text:
    raise SystemExit(0)

text = replace_once(
    text,
    "from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec\n",
    """from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
)
""",
)

text = replace_once(
    text,
    """        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
""",
    """        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # LONG-CONTEXT-REPLICATED-GROUPS: target attention KV is DCP-
            # sharded, so one local block covers ``block_size * dcp_size``
            # global tokens. Mamba state is sequence-replicated. The DFlash
            # group is a UniformTypeKVCacheSpecs wrapper around exact
            # SlidingWindowSpec members; its KV heads are TP-sharded and its
            # sequence cache is also replicated (the FlashInfer patch makes
            # the same distinction). Their block tables therefore need one
            # column per GLOBAL ``block_size`` tokens.
            inner_specs = getattr(spec, "kv_cache_specs", None)
            replicated_sliding = inner_specs is not None and all(
                type(inner_spec) is SlidingWindowSpec
                for inner_spec in inner_specs.values()
            )
            table_dcp_size = (
                1
                if isinstance(spec, MambaSpec) or replicated_sliding
                else self.dcp_size
            )
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * table_dcp_size
            )
""",
)

TARGET.write_text(text)
