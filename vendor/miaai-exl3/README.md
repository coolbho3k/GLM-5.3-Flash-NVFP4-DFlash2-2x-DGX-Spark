# MiaAI-Lab EXL3 provenance

These files are vendored from
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at commit 6599585438d6046cb6b4800411b8570971c15dcc:

- exl3.py
- exl3_fat_gemm.cu
- exl3_fat_gemm.cuh
- exl3_fat_moe.cu
- exl3_fat_moe.cuh
- patch_exl3_fat_kernel.py
- patch_exl3_ext_aarch64.py
- patch_model_overrides.py
- LICENSE

The Python overlay retains this recipe's validated decode/no-repack extensions while incorporating MiaAI's E3 grouped-prefill tier.

The upstream numeric spin-wait patch is vendored separately at
`overlay-exl3-fp8/patch_spinwait.py` because it modifies the vLLM runtime,
not ExLlamaV3. The active chat template includes upstream's reasoning-prefix
stability fix from the same revision.

The ExLlamaV3 extension itself is fetched at image-build time from
turboderp-org/exllamav3 commit
c5d9c657966ffeeaa9353f0cc899f18629da4a13. The files here are kept
separate from this repository's FP8 KV, DCP2, and long-context patches so
their provenance remains obvious.

The following optional decode components were subsequently incorporated from
upstream commit 9c0794b68d7fc124f79104409ab434769503fb31:

- `overlay-exl3-fp8/patch_adaptive_k.py`
- `overlay-exl3-fp8/patch_dense_fp8.py`
- the dense-FP8 additions in `exl3.py`

MiaAI Lab is credited as the upstream author of these components and distributes
them under AGPL-3.0. `patch_adaptive_k.py` is copied verbatim. Our
`patch_dense_fp8.py` adds compatibility with the model constructor already used
by this recipe, and the local `exl3.py` combines MiaAI's dense-FP8 additions
with this repository's earlier decode and prefill work. Their complete source,
including those modifications, is present in this repository. A copy of the
AGPL-3.0 is provided as `LICENSE.AGPL-3.0`; the older vendored snapshot retains
its MIT license in `LICENSE`.
