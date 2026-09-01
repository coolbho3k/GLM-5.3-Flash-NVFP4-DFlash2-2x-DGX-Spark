# MiaAI-Lab EXL3 provenance

These files are vendored from
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at commit 493cb88fc69f8ba73ac87404f429d763e2739d89:

- exl3.py
- patch_exl3_ext_aarch64.py
- patch_model_overrides.py
- LICENSE

The ExLlamaV3 extension itself is fetched at image-build time from
turboderp-org/exllamav3 commit
c5d9c657966ffeeaa9353f0cc899f18629da4a13. The files here are kept
separate from this repository's FP8 KV, DCP2, and long-context patches so
their provenance remains obvious.
