#!/usr/bin/env python3
"""Host-only contract checks for the selective MiaAI E2/E3 integration."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "6599585438d6046cb6b4800411b8570971c15dcc"
EXPECTED_HASHES = {
    "vendor/miaai-exl3/exl3.py": "f02574b9e28d56509159ed1e1c091148388466635e46ba5e9bd7f5e755482d47",
    "vendor/miaai-exl3/exl3_fat_gemm.cu": "1442a09d79915206abdc2379c093f9bb25b45cd74d295486c1135bd112cc6ea7",
    "vendor/miaai-exl3/exl3_fat_gemm.cuh": "b5ed5ee3d2b028d4be2091d5c5f14ae13daa2df44246fe9acda08ff92ab3262a",
    "vendor/miaai-exl3/exl3_fat_moe.cu": "21e625aa439367ed13feb88bbb55b72a06f2e5ce56d3615b3d4e1a28cb231e44",
    "vendor/miaai-exl3/exl3_fat_moe.cuh": "0b3ae15e9d42a582c32368ec4f36e386580d4207408cedb0cd44784ee85a8e0e",
    "vendor/miaai-exl3/patch_exl3_fat_kernel.py": "6fb6ed425251e500bc362fc22cd3e5cf1c87a0a8d861a95573297dde8a0c8f0f",
    "overlay-exl3-fp8/patch_spinwait.py": "09ec72e41d48181bb62b84c54d7a45956943271477bd6ca02c6ae221ba0d282c",
}


def test_upstream_hashes() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, (relative, actual, expected)
    provenance = (ROOT / "vendor/miaai-exl3/README.md").read_text()
    docs = (ROOT / "docs/EXL3-FP8-DCP2.md").read_text()
    assert UPSTREAM in provenance
    assert UPSTREAM in docs


def test_extension_patch_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw:
        ext = Path(raw) / "exllamav3_ext"
        quant = ext / "quant"
        quant.mkdir(parents=True)
        bindings = ext / "bindings.cpp"
        bindings.write_text(
            '#include "quant/exl3_moe.cuh"\n'
            'void bind(auto &m) {\n'
            '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
            '}\n'
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "vendor/miaai-exl3/patch_exl3_fat_kernel.py"),
                str(ext),
                str(ROOT / "vendor/miaai-exl3"),
            ],
            check=True,
        )
        patched = bindings.read_text()
        assert patched.count('#include "quant/exl3_fat_gemm.cuh"') == 1
        assert patched.count('m.def("exl3_fat_gemm"') == 1
        assert patched.count('m.def("exl3_fat_gemm_scatter"') == 1
        for symbol in ("gather", "gateup", "down", "tile_rows_gateup", "tile_rows_down"):
            assert patched.count(f'm.def("exl3_fat_moe_{symbol}"') == 1
        for name in (
            "exl3_fat_gemm.cu", "exl3_fat_gemm.cuh",
            "exl3_fat_moe.cu", "exl3_fat_moe.cuh",
        ):
            assert (quant / name).read_bytes() == (
                ROOT / "vendor/miaai-exl3" / name
            ).read_bytes()


def test_recipe_wiring_and_capacity_guardrail() -> None:
    dockerfile = (ROOT / "overlay-exl3-fp8/Dockerfile").read_text()
    launch = (ROOT / "launch-glm53-exl3-fp8-dcp2.sh").read_text()
    cluster = (ROOT / "start-exl3-fp8-dcp2-cluster.sh").read_text()
    selector = (ROOT / "serve-profile.sh").read_text()
    entrypoint = (ROOT / "overlay-exl3-fp8/entrypoint.sh").read_text()
    assert "patch_exl3_fat_kernel.py" in dockerfile
    assert "exl3_fat_gemm_scatter" in dockerfile
    assert f'glm53.miaai-exl3.commit="{UPSTREAM}"' in dockerfile
    assert 'glm53.serving.profile="e3-fp8-dcp2"' in dockerfile
    assert 'ENTRYPOINT ["/opt/glm53/entrypoint.sh"]' in dockerfile
    assert "python3 /opt/glm53/patch_spinwait.py" in entrypoint
    assert 'MAX_NUM_BATCHED_TOKENS:-7168' in launch
    assert 'MAX_NUM_SEQS:-6' in launch
    assert 'BLOCK_SIZE:-2048' in launch
    assert 'ENFORCE_EAGER:-1' in launch
    assert 'GLM53_SPINWAIT_MS:-16' in launch
    assert "launch-glm53-vllm-tp2-dflash2.sh" in launch
    assert "serve-profile.sh" in cluster
    assert "start exl3-fp8-dcp2" in cluster
    shown = subprocess.check_output(
        [str(ROOT / "serve-profile.sh"), "show", "exl3-fp8-dcp2"], text=True
    )
    assert "MAX_NUM_BATCHED_TOKENS             7168" in shown
    assert "EXL3_FAT_GROUPED                   1" in shown
    assert "EXL3_TEMP_ROWS_FUSED               64" in shown
    assert "default_var MAX_NUM_SEQS 6" in selector
    assert "default_var ENFORCE_EAGER 1" in selector
    assert "default_var BLOCK_SIZE 2048" in selector
    assert "7168" in selector


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"selective MiaAI E2/E3 recipe OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
