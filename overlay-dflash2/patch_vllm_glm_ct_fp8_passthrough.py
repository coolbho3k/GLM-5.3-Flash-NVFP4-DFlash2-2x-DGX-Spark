#!/usr/bin/env python3
"""Keep compressed-tensors block-FP8 GLM attention projections quantized.

The GLM loader has a compatibility path that dequantizes official-checkpoint
``weight_scale_inv`` tensors when a projection is BF16 in the instantiated
model. CompressedTensorsW8A16Fp8 initially registers the equivalent scale as
``weight_scale`` and renames it only after checkpoint loading. Teach the
loader about that temporary name so mixed NVFP4 + W8A16 checkpoints take the
normal quantized loading path instead of being materialized as BF16.

The GLM decoder also used to pass ``quant_config=None`` to all MLA layers
because every MLA projection in the Red Hat export was BF16. The repaired
checkpoint restores the official block-FP8 q_a/q_b/kv_a/o tensors, so pass the
real mixed-precision config through. Its ignore list keeps kv_b, indexer, and
the other genuinely BF16 MLA modules unquantized.
"""

from pathlib import Path


MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py"
)
COMPRESSED_TENSORS = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one loader anchor, found {count}: {old!r}")
    return text.replace(old, new, 1)


source = MODEL.read_text()
source = replace_once(
    source,
    '''                quant_config=None,  # MLA projections are BF16 in checkpoint
''',
    '''                # The mixed checkpoint restores official block-FP8 MLA
                # projections. Its ignore list still protects genuinely BF16
                # MLA modules, so do not disable quantization wholesale here.
                quant_config=quant_config,
''',
)
source = replace_once(
    source,
    '''    target_s = f"{layer_prefix}.{target_base}.weight_scale_inv"
    # If the model actually kept this projection in FP8, let the normal path
    # handle it (it has a weight_scale_inv param).
    if target_s in params_dict:
        return False
''',
    '''    target_scales = (
        f"{layer_prefix}.{target_base}.weight_scale_inv",
        f"{layer_prefix}.{target_base}.weight_scale",
    )
    # Native FP8 and compressed-tensors W8A16 use equivalent scale
    # parameters, but compressed-tensors calls it weight_scale until its
    # post-load conversion. In either case, keep the tensor quantized and let
    # the normal stacked/direct loader path handle it.
    if any(target_scale in params_dict for target_scale in target_scales):
        return False
''',
)
source = replace_once(
    source,
    '''            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)
''',
    '''            # Pad kv_a_proj_with_mqa for NoPE models. For block-FP8,
            # the 64 zero weight rows also require one scale row (128-row
            # blocks); padding the scale tensor by 64 rows is incorrect.
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                if name.endswith(".weight"):
                    pad = torch.zeros(
                        kv_a_pad_size,
                        *loaded_weight.shape[1:],
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    )
                    loaded_weight = torch.cat([loaded_weight, pad], dim=0)
                elif name.endswith((".weight_scale", ".weight_scale_inv")):
                    block_rows = 128
                    target_scale_rows = (
                        self.config.kv_lora_rank + kv_a_pad_size + block_rows - 1
                    ) // block_rows
                    scale_pad_rows = target_scale_rows - loaded_weight.shape[0]
                    if scale_pad_rows < 0:
                        raise ValueError(
                            "kv_a block scale has more rows than its padded target"
                        )
                    if scale_pad_rows:
                        scale_pad = torch.ones(
                            scale_pad_rows,
                            *loaded_weight.shape[1:],
                            dtype=loaded_weight.dtype,
                            device=loaded_weight.device,
                        )
                        loaded_weight = torch.cat(
                            [loaded_weight, scale_pad], dim=0
                        )
''',
)
MODEL.write_text(source)
compile(source, str(MODEL), "exec")

# vLLM's W8A16 branch leaves this value as None when input_quant is None.
# None is falsey, but its block-FP8 post-load hook deliberately asserts the
# singleton False. Normalize the weight-only case to an actual bool.
ct_source = COMPRESSED_TENSORS.read_text()
ct_source = replace_once(
    ct_source,
    '''            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = input_quant and not input_quant.dynamic
''',
    '''            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = bool(
                    input_quant and not input_quant.dynamic
                )
''',
)
COMPRESSED_TENSORS.write_text(ct_source)
compile(ct_source, str(COMPRESSED_TENSORS), "exec")
print("GLM compressed-tensors FP8 passthrough loader patch applied")
