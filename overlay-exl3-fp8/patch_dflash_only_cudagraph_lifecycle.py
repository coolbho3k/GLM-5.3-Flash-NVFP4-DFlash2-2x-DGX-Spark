#!/usr/bin/env python3
"""Let an opted-in DFlash speculator capture while the target stays eager."""

from pathlib import Path
import sys


MODEL_RUNNER = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py")
)
GPU_WORKER = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else MODEL_RUNNER.parent.parent / "gpu_worker.py"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


source = MODEL_RUNNER.read_text()

capture_gate_old = """        capture_decoder = self.cudagraph_manager.needs_capture()
        if not capture_encoder and not capture_decoder:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0
"""
capture_gate_new = """        capture_decoder = self.cudagraph_manager.needs_capture()
        speculator_cudagraph_manager = getattr(
            self.speculator, "query_cudagraph_manager", None
        )
        capture_speculator = (
            speculator_cudagraph_manager is not None
            and speculator_cudagraph_manager.needs_capture()
        )
        if not capture_encoder and not capture_decoder and not capture_speculator:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0
"""

if capture_gate_old in source:
    source = replace_once(source, capture_gate_old, capture_gate_new, "capture gate")
elif "capture_speculator = (" not in source:
    raise RuntimeError("DFlash-only capture gate is neither applicable nor present")

capture_body_old = """                if self.speculator is not None:
                    self.speculator.capture()
                if self.adaptive_verification is not None:
"""
capture_body_new = """                if self.adaptive_verification is not None:
"""
if capture_body_old in source:
    source = replace_once(
        source, capture_body_old, capture_body_new, "nested speculator capture"
    )
elif "if capture_speculator:\n                assert self.speculator is not None" not in source:
    raise RuntimeError("nested speculator capture is neither applicable nor removed")

adaptive_tail = """                    self.adaptive_verification.set_initial_cost_curves(timings)

        end_time = time.perf_counter()
"""
speculator_tail = """                    self.adaptive_verification.set_initial_cost_curves(timings)

            if capture_speculator:
                assert self.speculator is not None
                self.speculator.capture()

        end_time = time.perf_counter()
"""
if adaptive_tail in source:
    source = replace_once(
        source, adaptive_tail, speculator_tail, "speculator capture insertion"
    )
elif "if capture_speculator:\n                assert self.speculator is not None" not in source:
    raise RuntimeError("speculator capture insertion is neither applicable nor present")

MODEL_RUNNER.write_text(source)

worker_source = GPU_WORKER.read_text()
worker_gate_old = """        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()
"""
worker_gate_new = """        cuda_graph_memory_bytes = 0
        capture_dflash_only = os.getenv("VLLM_DFLASH_ONLY_CUDAGRAPH", "0") == "1"
        # V2's synthetic decode warmup is not compatible with replaying the
        # privately-shaped DFlash graphs. Run it eagerly before capture; this
        # also guarantees all draft/target JIT kernels exist before recording.
        if capture_dflash_only and self.use_v2_model_runner:
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        if not self.model_config.enforce_eager or capture_dflash_only:
            cuda_graph_memory_bytes = self.model_runner.capture_model()
"""
if worker_gate_old in worker_source:
    worker_source = replace_once(
        worker_source, worker_gate_old, worker_gate_new, "GPU worker capture gate"
    )
elif "capture_dflash_only = os.getenv(" not in worker_source:
    raise RuntimeError("GPU worker capture gate is neither applicable nor present")

v2_warmup_old = """        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
"""
v2_warmup_new = """        if self.use_v2_model_runner and not capture_dflash_only:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
"""
if v2_warmup_old in worker_source:
    worker_source = replace_once(
        worker_source, v2_warmup_old, v2_warmup_new, "V2 post-capture warmup gate"
    )
elif "if self.use_v2_model_runner and not capture_dflash_only:" not in worker_source:
    raise RuntimeError("V2 post-capture warmup gate is neither applicable nor present")

GPU_WORKER.write_text(worker_source)
print("patched DFlash-only CUDA graph lifecycle")
