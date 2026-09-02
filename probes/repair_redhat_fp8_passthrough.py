#!/usr/bin/env python3
"""Restore native upstream FP8 tensors that Red Hat exported as BF16.

The RedHatAI NVFP4 recipe used by this repository quantizes routed experts
only.  Its source checkpoint is already mixed precision: routed experts,
shared experts, the first three dense MLPs, and selected MLA projections are
stored as block-FP8.  During the experts-only export, all non-target tensors
were materialized as BF16.  This tool repairs that avoidable expansion while
preserving the optimized routed-expert NVFP4 tensors byte-for-byte.

The rewrite is deliberately tensor-byte based.  It never dequantizes or
requantizes a weight:

* every non-target tensor is copied verbatim from the NVFP4 checkpoint;
* each restored weight and block scale is copied verbatim from the pinned
  official FP8 checkpoint; and
* the output uses compressed-tensors mixed-precision metadata with W8A16
  block-FP8 linears, retaining BF16 activations.

Only main-model tensors present in both indexes are eligible.  Routed experts
remain NVFP4, the external DFlash2 drafter is untouched, and tensors that were
BF16 in the official source remain BF16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable


SOURCE_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
SOURCE_INDEX_SHA256 = (
    "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
)
SCALE_INV_SUFFIX = "_scale_inv"
SCALE_SUFFIX = "_scale"
COPY_BUFFER_BYTES = 16 << 20

DENSE_MLP = re.compile(
    r"^model\.language_model\.layers\.[0-2]\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
SHARED_EXPERT = re.compile(
    r"^model\.language_model\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])\."
    r"mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
MLA_PROJECTION = re.compile(
    r"^model\.language_model\.layers\."
    r"(?:3|7|11|15|19|23|27|31|35|39|43)\.self_attn\."
    r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$"
)

FP8_RUNTIME_TARGETS = [
    r"re:(?:.*\.)?layers\.[0-2]\.mlp\.(?:gate_up_proj|down_proj)$",
    (
        r"re:(?:.*\.)?layers\.(?:[3-9]|[1-3][0-9]|4[0-4])\.mlp\."
        r"shared_experts\.(?:gate_up_proj|down_proj)$"
    ),
    (
        r"re:(?:.*\.)?layers\.(?:3|7|11|15|19|23|27|31|35|39|43)\."
        r"self_attn\.(?:fused_qkv_a_proj|q_b_proj|o_proj)$"
    ),
]


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_index(root: Path) -> dict[str, Any]:
    path = root / "model.safetensors.index.json"
    value = read_json(path)
    if "weight_map" not in value:
        raise ValueError(f"missing weight_map in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(COPY_BUFFER_BYTES)
    view = memoryview(buffer)
    with path.open("rb") as handle:
        while amount := handle.readinto(buffer):
            digest.update(view[:amount])
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[OrderedDict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header, object_pairs_hook=OrderedDict)
    return header, 8 + header_length


def tensor_nbytes(entry: dict[str, Any]) -> int:
    start, end = entry["data_offsets"]
    return int(end) - int(start)


def output_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name + SCALE_SUFFIX


def source_scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(weight_name)
    return weight_name + SCALE_INV_SUFFIX


def restore_family(name: str) -> str | None:
    if DENSE_MLP.match(name):
        return "dense_mlp"
    if SHARED_EXPERT.match(name):
        return "shared_expert"
    if MLA_PROJECTION.match(name):
        return "mla_projection"
    return None


def discover_restore_weights(
    source_index: dict[str, str], target_index: dict[str, str]
) -> list[str]:
    source_fp8_weights = {
        name.removesuffix(SCALE_INV_SUFFIX)
        for name in source_index
        if name.endswith(".weight" + SCALE_INV_SUFFIX)
        and name.removesuffix(SCALE_INV_SUFFIX) in source_index
    }
    candidates = sorted(
        name
        for name in source_fp8_weights
        if name in target_index and ".mlp.experts." not in name
    )
    unsupported = [name for name in candidates if restore_family(name) is None]
    if unsupported:
        raise ValueError(
            "official-FP8/current-BF16 candidates outside the reviewed allowlist: "
            + ", ".join(unsupported[:20])
        )
    if len(candidates) != 179:
        raise ValueError(
            f"expected 179 reviewed passthrough weights, found {len(candidates)}"
        )
    counts = Counter(restore_family(name) for name in candidates)
    expected = Counter(
        {"dense_mlp": 9, "shared_expert": 126, "mla_projection": 44}
    )
    if counts != expected:
        raise ValueError(f"unexpected restore-family counts: {dict(counts)}")
    return candidates


def locate_shard(shard: str, roots: Iterable[Path]) -> Path:
    checked = []
    for root in roots:
        path = root / shard
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing source shard {shard}; checked {checked}")


def copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    offset: int,
    length: int,
    buffer: bytearray,
) -> None:
    source.seek(offset)
    remaining = length
    view = memoryview(buffer)
    while remaining:
        amount = min(remaining, len(buffer))
        read = source.readinto(view[:amount])
        if read != amount:
            raise EOFError(f"short tensor read: wanted {amount}, got {read}")
        destination.write(view[:read])
        remaining -= read


def encode_header(header: OrderedDict[str, Any]) -> bytes:
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def make_output_config(
    base_config: dict[str, Any], restore_weights: list[str]
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    quant = config["quantization_config"]
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("base checkpoint is not compressed-tensors")
    if set(quant.get("config_groups", {})) != {"group_0"}:
        raise ValueError("expected exactly the reviewed NVFP4 group_0")

    restored_modules = {name.removesuffix(".weight") for name in restore_weights}
    quant["ignore"] = [
        name for name in quant.get("ignore", []) if name not in restored_modules
    ]
    quant["config_groups"]["group_1"] = {
        "format": "float-quantized",
        # Weight-only FP8 is intentional: BF16 activations preserve the current
        # activation precision and select CompressedTensorsW8A16Fp8.
        "input_activations": None,
        "output_activations": None,
        "targets": FP8_RUNTIME_TARGETS,
        "weights": {
            "actorder": None,
            "block_structure": [128, 128],
            "dynamic": False,
            "group_size": None,
            "num_bits": 8,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "block",
            "symmetric": True,
            "type": "float",
            "zp_dtype": None,
        },
    }
    quant["format"] = "mixed-precision"
    return config


def copy_support_files(base_root: Path, output_root: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
    }
    transient_prefixes = ("nvfp4-build-",)
    for path in base_root.iterdir():
        if not path.is_file() or path.name in excluded:
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        if path.name.startswith(transient_prefixes):
            continue
        shutil.copy2(path, output_root / path.name)


def build_shard(
    *,
    base_path: Path,
    output_path: Path,
    source_roots: list[Path],
    source_index: dict[str, str],
    replacements: set[str],
) -> dict[str, Any]:
    base_header, base_data_start = read_safetensors_header(base_path)
    base_metadata = base_header.get("__metadata__")
    sources: OrderedDict[str, tuple[Path, dict[str, Any], int]] = OrderedDict()
    source_headers: dict[Path, tuple[OrderedDict[str, Any], int]] = {}
    restored = 0

    def official_tensor(name: str) -> tuple[Path, dict[str, Any], int]:
        source_path = locate_shard(source_index[name], source_roots)
        if source_path not in source_headers:
            source_headers[source_path] = read_safetensors_header(source_path)
        source_header, source_data_start = source_headers[source_path]
        if name not in source_header:
            raise KeyError(f"{name} is absent from indexed shard {source_path}")
        return source_path, source_header[name], source_data_start

    for name, entry in base_header.items():
        if name == "__metadata__":
            continue
        if name in replacements:
            scale_in = source_scale_name(name)
            weight_source = official_tensor(name)
            scale_source = official_tensor(scale_in)
            weight_entry = weight_source[1]
            scale_entry = scale_source[1]
            if entry["dtype"] != "BF16":
                raise ValueError(f"repair target is not BF16: {name}")
            if weight_entry["dtype"] != "F8_E4M3" or scale_entry["dtype"] != "F32":
                raise ValueError(f"unexpected official FP8 contract: {name}")
            if entry["shape"] != weight_entry["shape"]:
                raise ValueError(f"shape mismatch for {name}")
            sources[name] = weight_source
            sources[output_scale_name(name)] = scale_source
            restored += 1
        else:
            sources[name] = (base_path, entry, base_data_start)

    output_header: OrderedDict[str, Any] = OrderedDict()
    if base_metadata is not None:
        output_header["__metadata__"] = base_metadata
    offset = 0
    for name, (_, entry, _) in sources.items():
        size = tensor_nbytes(entry)
        output_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size

    temporary = output_path.with_name(f".{output_path.name}.fp8-repair.tmp")
    handles: dict[Path, BinaryIO] = {}
    buffer = bytearray(COPY_BUFFER_BYTES)
    try:
        with temporary.open("wb") as destination:
            destination.write(encode_header(output_header))
            for path, entry, data_start in sources.values():
                handle = handles.get(path)
                if handle is None:
                    handle = path.open("rb")
                    handles[path] = handle
                start, end = entry["data_offsets"]
                copy_range(
                    handle,
                    destination,
                    data_start + int(start),
                    int(end) - int(start),
                    buffer,
                )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output_path)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()

    return {
        "shard": output_path.name,
        "restored_weights": restored,
        "tensors": len(sources),
        "tensor_bytes": offset,
        "file_bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        required=True,
        help="Pinned official source root; repeat for complementary shard roots.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--source-index-sha256", default=SOURCE_INDEX_SHA256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_root = args.base_root.resolve()
    source_roots = [path.resolve() for path in args.source_root]
    output_root = args.output_root.resolve()
    if output_root == base_root or output_root in source_roots:
        raise ValueError("output root must be separate from all inputs")

    base_index_doc = load_index(base_root)
    source_index_path = source_roots[0] / "model.safetensors.index.json"
    source_index_sha256 = sha256_file(source_index_path)
    if source_index_sha256 != args.source_index_sha256:
        raise ValueError(
            "official source index hash differs from the pinned input: "
            f"expected {args.source_index_sha256}, got {source_index_sha256}"
        )
    source_index_doc = load_index(source_roots[0])
    source_index = source_index_doc["weight_map"]
    base_index = base_index_doc["weight_map"]
    restore_weights = discover_restore_weights(source_index, base_index)

    missing_source_indexes = [
        str(root / "model.safetensors.index.json")
        for root in source_roots
        if not (root / "model.safetensors.index.json").is_file()
    ]
    if missing_source_indexes:
        raise FileNotFoundError(missing_source_indexes)
    for root in source_roots[1:]:
        root_index_path = root / "model.safetensors.index.json"
        if sha256_file(root_index_path) != source_index_sha256:
            raise ValueError(f"source index hash differs across roots: {root}")
        if load_index(root)["weight_map"] != source_index:
            raise ValueError(f"source index differs across roots: {root}")

    required_shards = {
        source_index[name] for name in restore_weights
    } | {source_index[source_scale_name(name)] for name in restore_weights}
    for shard in sorted(required_shards):
        locate_shard(shard, source_roots)

    output_root.mkdir(parents=True, exist_ok=True)
    existing_shards = list(output_root.glob("model-*.safetensors"))
    if existing_shards and not args.overwrite:
        raise FileExistsError(
            f"output contains model shards; pass --overwrite: {output_root}"
        )
    copy_support_files(base_root, output_root)

    replacements_by_shard: dict[str, set[str]] = defaultdict(set)
    for name in restore_weights:
        replacements_by_shard[base_index[name]].add(name)

    shard_reports = []
    new_weight_map = dict(base_index)
    for name in restore_weights:
        new_weight_map[output_scale_name(name)] = base_index[name]

    for shard in sorted(set(base_index.values())):
        report = build_shard(
            base_path=base_root / shard,
            output_path=output_root / shard,
            source_roots=source_roots,
            source_index=source_index,
            replacements=replacements_by_shard.get(shard, set()),
        )
        shard_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    base_config = read_json(base_root / "config.json")
    repaired_config = make_output_config(base_config, restore_weights)
    atomic_json(output_root / "config.json", repaired_config)

    total_tensor_bytes = sum(report["tensor_bytes"] for report in shard_reports)
    output_index = json.loads(json.dumps(base_index_doc))
    output_index["weight_map"] = dict(sorted(new_weight_map.items()))
    output_index.setdefault("metadata", {})["total_size"] = total_tensor_bytes
    atomic_json(output_root / "model.safetensors.index.json", output_index)

    family_counts = Counter(restore_family(name) for name in restore_weights)
    base_tensor_bytes = int(base_index_doc.get("metadata", {}).get("total_size", 0))
    manifest = {
        "schema": 1,
        "base_root": str(base_root),
        "source_roots": [str(root) for root in source_roots],
        "source_revision": args.source_revision,
        "source_index_sha256": source_index_sha256,
        "restored_weights": len(restore_weights),
        "restored_scales": len(restore_weights),
        "family_counts": dict(sorted(family_counts.items())),
        "runtime_precision": "FP8 weights / BF16 activations (W8A16)",
        "base_tensor_bytes": base_tensor_bytes,
        "output_tensor_bytes": total_tensor_bytes,
        "tensor_bytes_saved": base_tensor_bytes - total_tensor_bytes,
        "shards": shard_reports,
    }
    atomic_json(output_root / "fp8-passthrough-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
