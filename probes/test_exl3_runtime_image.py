#!/usr/bin/env python3
"""Static image gate shared by EXL3 with either target-KV format."""


def main() -> None:
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.exl3 import (
        EXL3_FAT_DIAG_KEYS,
        EXL3_FAT_DIAG_SCHEMA,
        Exl3Config,
        exl3_fat_diag,
    )

    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is Exl3Config
    import exllamav3_ext

    for symbol in (
        "exl3_moe",
        "exl3_moe_max_concurrency",
        "exl3_fat_gemm",
        "exl3_fat_gemm_scatter",
    ):
        assert hasattr(exllamav3_ext, symbol), symbol
    diag = exl3_fat_diag()
    assert diag["schema"] == EXL3_FAT_DIAG_SCHEMA == 1
    assert set(diag) == set(EXL3_FAT_DIAG_KEYS)
    print("EXL3 routed-expert runtime image composition OK")


if __name__ == "__main__":
    main()
