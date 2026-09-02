#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "probes/repair_nvfp4_w13_shared_scale.py"
PREFIX = "model.language_model.layers.3.mlp.experts.7"


def names(projection: str) -> tuple[str, str, str]:
    base = f"{PREFIX}.{projection}"
    return (
        f"{base}.weight_packed",
        f"{base}.weight_scale",
        f"{base}.weight_global_scale",
    )


def write_checkpoint(root: Path, *, repaired: bool) -> None:
    root.mkdir()
    tensors = {"model.embed_tokens.weight": torch.arange(8, dtype=torch.float32)}
    weight_map = {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
    for projection, seed in (("gate_proj", 10), ("up_proj", 20), ("down_proj", 30)):
        packed, scale, global_scale = names(projection)
        tensors[packed] = torch.arange(8, dtype=torch.uint8).reshape(2, 4) + seed
        tensors[scale] = torch.arange(4, dtype=torch.float32).reshape(2, 2) + seed
        divisor = 4.0 if projection in ("gate_proj", "up_proj") else 8.0
        if not repaired and projection == "up_proj":
            divisor = 5.0
        tensors[global_scale] = torch.tensor(divisor)
        weight_map.update({name: "model-00001-of-00001.safetensors" for name in names(projection)})
    save_file(tensors, root / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        current = base / "current"
        reference = base / "reference"
        output = base / "output"
        write_checkpoint(current, repaired=False)
        write_checkpoint(reference, repaired=True)
        shutil.copytree(current, output)
        subprocess.run(
            [
                "python3",
                str(TOOL),
                "--current-root", str(current),
                "--reference-root", str(reference),
                "--output-root", str(output),
            ],
            check=True,
        )
        shard = "model-00001-of-00001.safetensors"
        with safe_open(current / shard, framework="pt", device="cpu") as old, safe_open(
            reference / shard, framework="pt", device="cpu"
        ) as ref, safe_open(output / shard, framework="pt", device="cpu") as got:
            for projection in ("gate_proj", "up_proj"):
                for name in names(projection):
                    assert torch.equal(got.get_tensor(name), ref.get_tensor(name))
            for name in names("down_proj"):
                assert torch.equal(got.get_tensor(name), old.get_tensor(name))
            assert torch.equal(
                got.get_tensor("model.embed_tokens.weight"),
                old.get_tensor("model.embed_tokens.weight"),
            )
    print("W1/W3 shared-scale repair test passed")


if __name__ == "__main__":
    main()
