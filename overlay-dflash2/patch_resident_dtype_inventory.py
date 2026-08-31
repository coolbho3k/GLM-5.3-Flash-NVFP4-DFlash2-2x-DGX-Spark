#!/usr/bin/env python3
"""Log a deduplicated inventory of resident target and draft tensor storage."""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
)
source = PATH.read_text()
marker = "def _log_resident_dtype_inventory("

helper = r'''
def _log_resident_dtype_inventory(model_runner: Any, rank: int) -> None:
    """Inventory actual resident storages without allocating device tensors.

    Tensor aliases are deduplicated by (device, data pointer, storage bytes).
    This measures live module parameters and buffers rather than checkpoint
    file size. The combined view also deduplicates any target/draft sharing.
    """

    combined_seen: set[tuple[str, int, int]] = set()
    combined_by_dtype: dict[str, int] = {}
    combined_by_device: dict[str, int] = {}
    combined_storage_count = 0
    top_bf16: list[dict[str, Any]] = []

    def inventory(owner: str, module: nn.Module | None) -> dict[str, Any]:
        nonlocal combined_storage_count
        if module is None:
            return {
                "storage_bytes": 0,
                "storage_count": 0,
                "tensor_names": 0,
                "by_dtype": {},
                "by_device": {},
            }

        local_seen: set[tuple[str, int, int]] = set()
        by_dtype: dict[str, int] = {}
        by_device: dict[str, int] = {}
        tensor_names = 0

        for tensor_kind, named_tensors in (
            ("parameter", module.named_parameters(recurse=True)),
            ("buffer", module.named_buffers(recurse=True)),
        ):
            for name, tensor in named_tensors:
                tensor_names += 1
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    continue
                try:
                    storage = tensor.untyped_storage()
                    storage_bytes = int(storage.nbytes())
                    data_ptr = int(storage.data_ptr())
                except (RuntimeError, NotImplementedError):
                    continue
                if storage_bytes <= 0 or data_ptr == 0:
                    continue

                device = str(tensor.device)
                dtype = str(tensor.dtype).removeprefix("torch.")
                key = (device, data_ptr, storage_bytes)
                if key not in local_seen:
                    local_seen.add(key)
                    by_dtype[dtype] = by_dtype.get(dtype, 0) + storage_bytes
                    by_device[device] = by_device.get(device, 0) + storage_bytes

                if key not in combined_seen:
                    combined_seen.add(key)
                    combined_storage_count += 1
                    combined_by_dtype[dtype] = (
                        combined_by_dtype.get(dtype, 0) + storage_bytes
                    )
                    combined_by_device[device] = (
                        combined_by_device.get(device, 0) + storage_bytes
                    )
                    if tensor.dtype == torch.bfloat16:
                        top_bf16.append(
                            {
                                "bytes": storage_bytes,
                                "kind": tensor_kind,
                                "name": f"{owner}.{name}",
                                "shape": list(tensor.shape),
                            }
                        )

        return {
            "storage_bytes": sum(by_dtype.values()),
            "storage_count": len(local_seen),
            "tensor_names": tensor_names,
            "by_dtype": dict(sorted(by_dtype.items())),
            "by_device": dict(sorted(by_device.items())),
        }

    target = inventory("target", model_runner.get_model())
    draft = inventory("draft", model_runner.get_draft_model())
    measured = int(model_runner.model_memory_usage)
    combined_bytes = sum(combined_by_dtype.values())
    top_bf16.sort(key=lambda row: row["bytes"], reverse=True)
    record = {
        "schema": 1,
        "rank": rank,
        "measured_model_memory_bytes": measured,
        "combined_unique_storage_bytes": combined_bytes,
        "combined_storage_count": combined_storage_count,
        "combined_by_dtype": dict(sorted(combined_by_dtype.items())),
        "combined_by_device": dict(sorted(combined_by_device.items())),
        "inventory_coverage_ratio": combined_bytes / measured if measured else None,
        "target": target,
        "draft": draft,
        "top_bf16_storages": top_bf16[:20],
    }
    logger.info("KV_ACCOUNTING_DTYPES %s", json.dumps(record, sort_keys=True))
'''

if marker not in source:
    class_anchor = "class Worker(WorkerBase):\n"
    if source.count(class_anchor) != 1:
        raise RuntimeError(
            f"resident inventory class anchor count {source.count(class_anchor)}"
        )
    source = source.replace(class_anchor, helper + "\n\n" + class_anchor, 1)

call = "        _log_resident_dtype_inventory(self.model_runner, self.rank)\n"
if call not in source:
    load_anchor = '''        if self.vllm_config.weight_transfer_config is not None:
'''
    if source.count(load_anchor) != 1:
        raise RuntimeError(
            f"resident inventory load anchor count {source.count(load_anchor)}"
        )
    source = source.replace(load_anchor, call + "\n" + load_anchor, 1)

PATH.write_text(source)
print(f"resident_dtype_inventory: patched {PATH}")
