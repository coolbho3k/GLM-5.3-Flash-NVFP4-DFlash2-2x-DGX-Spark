#!/usr/bin/env python3
"""Host-only contract checks for the selective MiaAI E2 integration."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "eb0469fbb2b49fd7c025f594a3339a121e58f7a9"
EXPECTED_HASHES = {
    "vendor/miaai-exl3/exl3.py": "c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843",
    "vendor/miaai-exl3/exl3_fat_gemm.cu": "1442a09d79915206abdc2379c093f9bb25b45cd74d295486c1135bd112cc6ea7",
    "vendor/miaai-exl3/exl3_fat_gemm.cuh": "b5ed5ee3d2b028d4be2091d5c5f14ae13daa2df44246fe9acda08ff92ab3262a",
    "vendor/miaai-exl3/patch_exl3_fat_kernel.py": "ce2aa43b0560e931a60831c5539e83faaed56e80a58b1bd5fda7610781c234a8",
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
        for name in ("exl3_fat_gemm.cu", "exl3_fat_gemm.cuh"):
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
    assert 'glm53.serving.profile="e2-fp8-dcp2"' in dockerfile
    assert 'ENTRYPOINT ["/opt/glm53/entrypoint.sh"]' in dockerfile
    assert "python3 /opt/glm53/patch_spinwait.py" in entrypoint
    assert 'MAX_NUM_BATCHED_TOKENS:-8192' in launch
    assert 'MAX_NUM_SEQS:-6' in launch
    assert 'BLOCK_SIZE:-2048' in launch
    assert 'ENFORCE_EAGER:-1' in launch
    assert 'GLM53_SPINWAIT_MS:-stock' in launch
    assert "launch-glm53-vllm-tp2-dflash2.sh" in launch
    assert "serve-profile.sh" in cluster
    assert "start exl3-fp8-dcp2" in cluster
    assert "default_var MAX_NUM_BATCHED_TOKENS 8192" in selector
    assert "default_var MAX_NUM_SEQS 6" in selector
    assert "default_var ENFORCE_EAGER 1" in selector
    assert "default_var BLOCK_SIZE 2048" in selector
    assert "7168" not in launch + cluster + selector


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"selective MiaAI E2 recipe OK ({len(tests)} tests)")


if __name__ == "__main__":
    main()
