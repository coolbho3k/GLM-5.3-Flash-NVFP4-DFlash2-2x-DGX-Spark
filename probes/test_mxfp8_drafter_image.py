#!/usr/bin/env python3
"""Build-time invariants for the optional ModelOpt MXFP8 DFlash2 drafter."""

from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages")
linear_registry = (
    SITE / "vllm/model_executor/kernels/linear/__init__.py"
).read_text()
b12x_kernel = (
    SITE / "vllm/model_executor/kernels/linear/mxfp8/b12x.py"
).read_text()
dflash = (SITE / "vllm/model_executor/models/qwen3_dflash.py").read_text()
dflash2 = (SITE / "vllm/model_executor/models/qwen3_dflash2.py").read_text()
dflash2_speculator = (
    SITE / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
).read_text()
model_runner = (
    SITE / "vllm/v1/worker/gpu/model_runner.py"
).read_text()
gpu_worker = (
    SITE / "vllm/v1/worker/gpu_worker.py"
).read_text()
attention_backend = (
    SITE / "vllm/v1/attention/backend.py"
).read_text()
flashinfer = (
    SITE / "vllm/v1/attention/backends/flashinfer.py"
).read_text()

assert "class B12xMxfp8LinearKernel" in b12x_kernel
assert 'import_module("b12x.gemm.mxfp8_linear")' in b12x_kernel
assert 'os.environ.get(\n        "VLLM_USE_B12X_FP8_GEMM", "0"' in b12x_kernel
assert "from vllm.model_executor.kernels.linear.mxfp8.b12x import" in linear_registry
mx_start = linear_registry.index("_POSSIBLE_MXFP8_KERNELS")
mx_registry = linear_registry[mx_start : mx_start + 600]
assert mx_registry.index("B12xMxfp8LinearKernel") < mx_registry.index(
    "FlashInferCutedslMxfp8LinearKernel"
)

assert "dequantize_mxfp8_rows_torch" in dflash
assert "qkv_weight.dtype == torch.float8_e4m3fn" in dflash
assert "quant_config=quant_config" in dflash2
assert "quant_config=self.quant_config" in dflash2
assert "quant_config=None" not in dflash2
assert "VLLM_DFLASH_ONLY_CUDAGRAPH" in dflash2_speculator
assert "capture_speculator = (" in model_runner
assert "if capture_speculator:" in model_runner
assert "capture_dflash_only = os.getenv(" in gpu_worker
assert "if not self.model_config.enforce_eager or capture_dflash_only:" in gpu_worker
assert "if capture_dflash_only and self.use_v2_model_runner:" in gpu_worker
assert "if self.use_v2_model_runner and not capture_dflash_only:" in gpu_worker
assert "effective_use_dcp = getattr(" in attention_backend
assert "if effective_use_dcp and not supports_dcp_with_varlen:" in attention_backend
assert "if self.kv_cache_spec.sliding_window is not None:" in flashinfer
assert "replicated_sliding_window = (" in flashinfer
assert "and not replicated_sliding_window" in flashinfer
assert "self.use_dcp = False" in flashinfer

print("MXFP8 DFlash2 image invariants: PASS")
