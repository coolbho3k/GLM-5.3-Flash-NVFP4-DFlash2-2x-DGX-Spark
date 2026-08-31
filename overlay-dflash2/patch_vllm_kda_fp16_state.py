from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
PATH = ROOT / "model_executor/layers/mamba/mamba_utils.py"

source = PATH.read_text()
old = '''    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, torch.float32)
'''
new = '''    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        # The recurrent Triton kernels accumulate in FP32 regardless of the
        # backing cache dtype.  Persist KDA state in FP16 to halve every dense
        # speculative rollback snapshot while retaining more mantissa bits
        # than BF16.  This is isolated in a capacity experiment image until
        # long-context quality validation is complete.
        return (state_dtype, torch.float16)
'''
if source.count(old) != 1:
    raise RuntimeError(f"expected one KDA dtype definition, found {source.count(old)}")
PATH.write_text(source.replace(old, new))


# FP16 KDA state makes the naturally selected GLM hybrid attention block 4608
# tokens.  That produces a 1296-token DFlash page, which cannot satisfy
# FlashInfer's 32-token split/hash granularity.  Round GLM+DFlash hybrid blocks
# to 1024 tokens: both the FP32 baseline (9216) and FP16 experiment (5120)
# then have exact-fit DFlash pages divisible by 32, allowing the draft cache to
# share the target allocation instead of creating thousands of tiny blocks.
PATH = ROOT / "platforms/interface.py"
source = PATH.read_text()
old = '''            indexer_align = cls._get_indexer_block_alignment(vllm_config)
            if indexer_align:
                attn_block_size = indexer_align * cdiv(attn_block_size, indexer_align)
'''
new = old + '''
            spec_config = vllm_config.speculative_config
            if (
                spec_config is not None
                and "dflash" in str(getattr(spec_config, "method", "")).lower()
                and model_config.architecture
                == "Glm5NextForConditionalGeneration"
            ):
                dflash_exact_fit_alignment = 1024
                attn_block_size = dflash_exact_fit_alignment * cdiv(
                    attn_block_size, dflash_exact_fit_alignment
                )
'''
if source.count(old) != 1:
    raise RuntimeError(
        f"expected one hybrid attention alignment site, found {source.count(old)}"
    )
PATH.write_text(source.replace(old, new))
