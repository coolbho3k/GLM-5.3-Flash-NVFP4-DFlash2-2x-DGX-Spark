# MiaAI-Lab EXL3 provenance

These files are vendored from
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at commit eb0469fbb2b49fd7c025f594a3339a121e58f7a9:

- exl3.py
- exl3_fat_gemm.cu
- exl3_fat_gemm.cuh
- patch_exl3_fat_kernel.py
- patch_exl3_ext_aarch64.py
- patch_model_overrides.py
- LICENSE

The upstream numeric spin-wait patch is vendored separately at
`overlay-exl3-fp8/patch_spinwait.py` because it modifies the vLLM runtime,
not ExLlamaV3. The active chat template includes upstream's reasoning-prefix
stability fix from the same revision.

The ExLlamaV3 extension itself is fetched at image-build time from
turboderp-org/exllamav3 commit
c5d9c657966ffeeaa9353f0cc899f18629da4a13. The files here are kept
separate from this repository's FP8 KV, DCP2, and long-context patches so
their provenance remains obvious.
