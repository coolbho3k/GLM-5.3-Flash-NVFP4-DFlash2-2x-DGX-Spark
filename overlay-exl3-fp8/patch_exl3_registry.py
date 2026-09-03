"""Register the vendored EXL3 quantization method in this pinned vLLM."""

from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
target = site / "model_executor/layers/quantization/__init__.py"
text = target.read_text()
old_literal = """QuantizationMethods = Literal[
    "awq",
"""
new_literal = """QuantizationMethods = Literal[
    "exl3",
    "awq",
"""
if new_literal not in text:
    if text.count(old_literal) != 1:
        raise RuntimeError("expected one QuantizationMethods Literal header")
    text = text.replace(old_literal, new_literal)
old_lazy = """    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]
"""
new_lazy = """    from .exl3 import Exl3Config
    method_to_config["exl3"] = Exl3Config
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]
"""
if new_lazy not in text:
    if text.count(old_lazy) != 1:
        raise RuntimeError("expected one get_quantization_config trailer")
    text = text.replace(old_lazy, new_lazy)
target.write_text(text)
print("registered EXL3 quantization method")
