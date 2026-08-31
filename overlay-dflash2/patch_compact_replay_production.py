#!/usr/bin/env python3
"""Disable compact-replay diagnostics in an already composed image.

The diagnostic path synchronizes CUDA and copies device values to Python for
the first eight decode steps. It is useful while validating replay semantics,
but must be opt-in in the serving image. One-time runner banners are removed as
well so production logs contain only actionable accounting and errors.
"""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str, label: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source and old not in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: anchor count {count} in {target}")
    target.write_text(source.replace(old, new, 1))


replace(
    "models/glm5next/nvidia/kda.py",
    'os.environ.get("VLLM_COMPACT_REPLAY_DEBUG", "1") == "1"',
    'os.environ.get("VLLM_COMPACT_REPLAY_DEBUG", "0") == "1"',
    "compact replay debug default",
)

replace(
    "v1/worker/gpu_model_runner.py",
    '''            self._compact_replay_layers = get_registered_compact_kda_layers()
            print(
                "COMPACT_RUNNER"
                f" rank={torch.distributed.get_rank()}"
                f" layers={len(self._compact_replay_layers)}"
                f" is_hybrid={self.model_config.is_hybrid}"
            )
''',
    '''            self._compact_replay_layers = get_registered_compact_kda_layers()
''',
    "V1 compact replay banner",
)

replace(
    "v1/worker/gpu/model_runner.py",
    '''        if len(compact_replay_layers) != getattr(
            self, "_compact_replay_layer_count", -1
        ):
            self._compact_replay_layer_count = len(compact_replay_layers)
            print(
                "COMPACT_V2_RUNNER"
                f" rank={torch.distributed.get_rank()}"
                f" layers={len(compact_replay_layers)}"
                , flush=True
            )
''',
    '''        if len(compact_replay_layers) != getattr(
            self, "_compact_replay_layer_count", -1
        ):
            self._compact_replay_layer_count = len(compact_replay_layers)
''',
    "V2 compact replay banner",
)

print("compact_replay_production: debug is opt-in; runner banners removed")
