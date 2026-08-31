#!/usr/bin/env python3
"""Add structured, byte-exact KV and GPU-memory accounting to vLLM.

The patch is logging-only and does not alter cache geometry, allocation,
scheduling, or kernels. Records use KV_ACCOUNTING_MEMORY and
KV_ACCOUNTING_CONFIG prefixes followed by one JSON object.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_gpu_worker(path: Path) -> None:
    text = path.read_text()
    if "KV_ACCOUNTING_MEMORY" in text:
        print(f"kv_accounting: {path} already patched")
        return
    text = replace_once(text, "import gc\n", "import gc\nimport json\n", "gpu import")
    anchor = """        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )
"""
    replacement = """        # Machine-readable physical accounting. The identity below must be
        # exact: requested = non-KV + applied graph reserve + available KV.
        kv_memory_accounting = {
            "schema": 1,
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "initial_total_bytes": int(self.init_snapshot.total_memory),
            "initial_free_bytes": int(self.init_snapshot.free_memory),
            "requested_bytes": int(self.requested_memory),
            "weights_bytes": int(profile_result.weights_memory),
            "torch_peak_increase_bytes": int(profile_result.torch_peak_increase),
            "non_torch_increase_bytes": int(profile_result.non_torch_increase),
            "total_consumed_bytes": int(profile_result.total_consumed),
            "transient_peak_headroom_bytes": int(
                profile_result.transient_peak_headroom
            ),
            "non_kv_cache_bytes": int(profile_result.non_kv_cache_memory),
            "cudagraph_measured_bytes": int(cudagraph_memory_estimate),
            "cudagraph_applied_bytes": int(cudagraph_memory_estimate_applied),
            "after_profile_free_bytes": int(free_gpu_memory),
            "after_profile_torch_allocated_bytes": int(
                profile_result.after_profile.torch_allocated
            ),
            "after_profile_torch_reserved_bytes": int(
                profile_result.after_profile.torch_memory
            ),
            "available_kv_bytes": int(self.available_kv_cache_memory_bytes),
        }
        kv_memory_accounting["requested_identity_error_bytes"] = int(
            kv_memory_accounting["requested_bytes"]
            - kv_memory_accounting["non_kv_cache_bytes"]
            - kv_memory_accounting["cudagraph_applied_bytes"]
            - kv_memory_accounting["available_kv_bytes"]
        )
        logger.info(
            "KV_ACCOUNTING_MEMORY %s",
            json.dumps(kv_memory_accounting, sort_keys=True),
        )
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )
"""
    text = replace_once(text, anchor, replacement, "gpu accounting")
    path.write_text(text)
    print(f"kv_accounting: patched {path}")


def patch_kv_utils(path: Path) -> None:
    text = path.read_text()
    if "KV_ACCOUNTING_CONFIG" in text:
        print(f"kv_accounting: {path} already patched")
        return
    text = replace_once(
        text, "import hashlib\n", "import hashlib\nimport json\n", "kv import"
    )
    anchor = """    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs"""
    replacement = """    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    def spec_record(spec: KVCacheSpec) -> dict[str, Any]:
        fields = (
            "block_size", "page_size_bytes", "page_size_padded",
            "unpadded_page_size_bytes", "real_page_size_bytes",
            "compress_ratio", "num_kv_heads", "head_size", "head_size_v",
            "sliding_window", "cache_dtype_str", "model_version",
        )
        record: dict[str, Any] = {"type": type(spec).__name__}
        for field in fields:
            try:
                value = getattr(spec, field)
            except (AttributeError, AssertionError):
                continue
            if value is not None:
                record[field] = str(value) if field == "cache_dtype_str" else value
        dtype = getattr(spec, "dtype", None)
        if dtype is not None:
            record["dtype"] = str(dtype)
        try:
            record["max_request_bytes"] = int(
                spec.max_memory_usage_bytes(vllm_config)
            )
        except (AttributeError, AssertionError, TypeError, ValueError):
            pass
        return record

    groups_record: list[dict[str, Any]] = []
    for group_id, group in enumerate(kv_cache_groups):
        group_record: dict[str, Any] = {
            "group_id": group_id,
            "layer_names": list(group.layer_names),
            "group_spec_type": type(group.kv_cache_spec).__name__,
        }
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            group_record["layer_specs"] = {
                name: spec_record(spec)
                for name, spec in group.kv_cache_spec.kv_cache_specs.items()
            }
        else:
            group_record["spec"] = spec_record(group.kv_cache_spec)
        groups_record.append(group_record)

    pool_bytes_per_block = int(_pool_bytes_per_block(vllm_config, kv_cache_groups))
    tensor_bytes = sum(int(tensor.size) for tensor in kv_cache_tensors)
    expected_tensor_bytes = int(num_blocks * pool_bytes_per_block)
    config_accounting = {
        "schema": 1,
        "available_bytes": int(available_memory),
        "num_blocks": int(num_blocks),
        "pool_bytes_per_block": pool_bytes_per_block,
        "tensor_bytes": tensor_bytes,
        "expected_tensor_bytes": expected_tensor_bytes,
        "tensor_identity_error_bytes": tensor_bytes - expected_tensor_bytes,
        "allocator_tail_bytes": int(available_memory - tensor_bytes),
        "groups": groups_record,
        "tensors": [
            {"size_bytes": int(tensor.size), "shared_by": list(tensor.shared_by)}
            for tensor in kv_cache_tensors
        ],
    }
    logger.info("KV_ACCOUNTING_CONFIG %s", json.dumps(config_accounting, sort_keys=True))
    return kv_cache_config


def unify_hybrid_kv_cache_specs"""
    text = replace_once(text, anchor, replacement, "kv config accounting")
    path.write_text(text)
    print(f"kv_accounting: patched {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vllm_root",
        nargs="?",
        default="/usr/local/lib/python3.12/dist-packages/vllm",
    )
    args = parser.parse_args()
    root = Path(args.vllm_root)
    patch_gpu_worker(root / "v1/worker/gpu_worker.py")
    patch_kv_utils(root / "v1/core/kv_cache_utils.py")


if __name__ == "__main__":
    main()
