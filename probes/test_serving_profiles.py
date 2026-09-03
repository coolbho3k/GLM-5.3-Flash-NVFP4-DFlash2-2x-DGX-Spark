#!/usr/bin/env python3
"""Host-only contract tests for the selectable weight/KV profile matrix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "nvfp4-fp4-dcp2": ("glm53-v14:nvfp4-gscale-tooling", "", "marlin", "1"),
    "nvfp4-fp8-dcp2": ("glm53-exl3:e2-fp8-dcp2", "", "marlin", "0"),
    "exl3-fp4-dcp2": ("glm53-exl3:e2-native-fp4-dcp2", "exl3", "auto", "1"),
    "exl3-fp8-dcp2": ("glm53-exl3:e2-fp8-dcp2", "exl3", "auto", "0"),
}


def resolved(profile: str, **overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/home/emi"),
    }
    env.update(overrides)
    output = subprocess.check_output(
        [str(ROOT / "serve-profile.sh"), "show", profile],
        cwd=ROOT,
        env=env,
        text=True,
    )
    values: dict[str, str] = {}
    for line in output.splitlines():
        name, value = line.split(maxsplit=1)
        values[name] = value
    return values


def test_matrix() -> None:
    for profile, (image, quantization, moe, calibrated) in PROFILES.items():
        values = resolved(profile)
        assert values["SERVE_PROFILE"] == profile
        assert values["IMAGE"] == image
        assert values["QUANTIZATION"] == (quantization or "''")
        assert values["MOE_BACKEND"] == moe
        assert values["USE_CALIBRATED_NVFP4_MLA"] == calibrated
        assert values["USE_FP4_INDEXER_CACHE"] == "0"
        assert values["DCP_SIZE"] == "2"
        assert values["BLOCK_SIZE"] == ("2048" if "-fp8-" in profile else "2304")


def test_production_defaults_unchanged() -> None:
    values = resolved("nvfp4-fp4-dcp2")
    assert values["GPU_MEMORY_UTILIZATION"] == "0.87"
    assert values["MAX_MODEL_LEN"] == "1048576"
    assert values["BLOCK_SIZE"] == "2304"
    assert values["MAX_NUM_SEQS"] == "6"
    assert values["MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert values["ENFORCE_EAGER"] == "1"
    assert values["ENABLE_DFLASH"] == "1"
    assert values["DFLASH_TOKENS"] == "7"
    assert values["PREFILL_ADMISSION_POLICY"] == "adaptive"


def test_overrides_win() -> None:
    values = resolved(
        "exl3-fp8-dcp2",
        DCP_SIZE="1",
        ENABLE_DFLASH="0",
        MAX_NUM_SEQS="3",
        ENFORCE_EAGER="0",
        COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"PIECEWISE"}',
    )
    assert values["DCP_SIZE"] == "1"
    assert values["ENABLE_DFLASH"] == "0"
    assert values["MAX_NUM_SEQS"] == "3"
    assert values["ENFORCE_EAGER"] == "0"
    assert "PIECEWISE" in values["COMPILATION_CONFIG"]


def test_recipe_wiring() -> None:
    starter = (ROOT / "start-cluster.sh").read_text()
    launcher = (ROOT / "launch-glm53-vllm-tp2-dflash2.sh").read_text()
    for name in (
        "QUANTIZATION",
        "KV_CACHE_DTYPE",
        "VLLM_ATTENTION_BACKEND",
        "DFLASH_TOKENS",
        "EXL3_FAT_KERNEL",
        "COMPACT_SPEC_REPLAY",
        "COMPILATION_CONFIG",
        "BLOCK_SIZE",
    ):
        assert name in starter
        assert name in launcher
    assert "--quantization" in launcher
    assert "--decode-context-parallel-size" in launcher
    assert "--attention-config" in launcher
    assert '--block-size "$BLOCK_SIZE"' in launcher


def test_fp8_block_size_selects_exact_fit_draft_geometry() -> None:
    # With the 528-byte target MLA record, CLI block size 2048 selects a
    # 4096-token manager block. This produces a DFlash page with an exact
    # multiple of its required 32-token block, avoiding the old 64-token
    # padded fallback (and its roughly 41x wasted bytes per useful page).
    target_page_bytes = 528 * 4096
    draft_block_size = target_page_bytes // 1024
    assert draft_block_size == 2112
    assert draft_block_size % 32 == 0

    old_target_page_bytes = 528 * 5120
    old_draft_block_size = old_target_page_bytes // 1024
    assert old_draft_block_size == 2640
    assert old_draft_block_size % 32 != 0


def test_nvfp4_prepare_uses_published_checkpoint() -> None:
    selector = (ROOT / "serve-profile.sh").read_text()
    preparer = (ROOT / "prepare-nvfp4-weights.sh").read_text()
    assert 'nvfp4-*) "$SCRIPT_DIR/prepare-nvfp4-weights.sh"' in selector
    assert "coolbho3k/GLM-5.3-Flash-NVFP4-Optimized" in preparer
    assert "b919a80ec2fa8959737d37062fb13e7f6a3d11df" in preparer
    assert "b919a80ec2fa8959737d37062fb13e7f6a3d11df" in selector
    assert "glm53-nvfp4-marlin-shared-w13-v1" in preparer
    assert "bf582e4eacc1810f76656d1811693ff6c6737d2a" in preparer
    assert "bf582e4eacc1810f76656d1811693ff6c6737d2a" in (ROOT / "prepare-exl3-weights.sh").read_text()
    assert 'EXPECTED_SHARDS="${EXPECTED_SHARDS:-10}"' in preparer
    assert "54a2a07227b26e9e5930d3546fb64258076d514f8fde15740ba30c5113690502" in preparer
    assert '"${hf_cmd[@]}" download "$MODEL_ID"' in preparer
    assert 'HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"' in preparer


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"serving profile matrix OK ({len(tests)} tests, {len(PROFILES)} profiles)")


if __name__ == "__main__":
    main()
