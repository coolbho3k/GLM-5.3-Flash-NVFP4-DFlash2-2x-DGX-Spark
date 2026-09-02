"""Opt-in vLLM runtime support for capture and calibrated NVFP4 MLA scales.

This module is copied into the candidate serving image.  With neither
``VLLM_NVFP4_MLA_SCALES_FILE`` nor ``VLLM_NVFP4_MLA_CAPTURE_DIR`` set, it
does no I/O, creates no collector, and returns the legacy outer scale 1.0.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)
SCALE_SCHEMA = "glm-nvfp4-mla-gscale-v1"
CAPTURE_SCHEMA = "glm-nvfp4-mla-capture-v1"
ALGORITHM = "nvfp4-e2m1-e4m3-four-over-six"
RECORD_LAYOUT = "glm-zero-rope-288b"


@dataclass(frozen=True)
class LayerRuntime:
    latent_scale: float
    collector: "LayerCollector | None"


_SCALE_LOCK = threading.Lock()
_SCALE_PATH: str | None = None
_SCALE_LAYERS: dict[str, float] | None = None
_COLLECTORS: list["LayerCollector"] = []


def _load_scale_layers() -> dict[str, float]:
    global _SCALE_PATH, _SCALE_LAYERS
    path = os.environ.get("VLLM_NVFP4_MLA_SCALES_FILE")
    if not path:
        return {}
    with _SCALE_LOCK:
        if _SCALE_LAYERS is not None and _SCALE_PATH == path:
            return _SCALE_LAYERS
        artifact = json.loads(Path(path).read_text())
        if artifact.get("schema") != SCALE_SCHEMA:
            raise ValueError(f"unsupported NVFP4 scale schema in {path}")
        if artifact.get("algorithm") != ALGORITHM:
            raise ValueError(f"NVFP4 scale algorithm mismatch in {path}")
        if artifact.get("record_layout") != RECORD_LAYOUT:
            raise ValueError(f"NVFP4 record layout mismatch in {path}")
        layers: dict[str, float] = {}
        for name, entry in artifact.get("layers", {}).items():
            value = float(entry["latent_scale"])
            if not value > 0.0 or not value < float("inf"):
                raise ValueError(f"invalid latent_scale for {name!r}: {value}")
            layers[str(name)] = value
        if not layers:
            raise ValueError(f"no layer scales in {path}")
        _SCALE_PATH = path
        _SCALE_LAYERS = layers
        LOGGER.info("Loaded calibrated NVFP4 MLA scales for %d layers", len(layers))
        return layers


def _rank() -> int:
    for variable in ("RANK", "LOCAL_RANK", "VLLM_RANK"):
        value = os.environ.get(variable)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unnamed"


class _Control:
    """Rate-limited reader for the optional corpus-driver control file."""

    _lock = threading.Lock()
    _checked_at = 0.0
    _mtime_ns = -1
    _value: dict[str, Any] = {
        "bucket": "unlabeled",
        "phase": "auto",
        "enabled": True,
        "flush_epoch": 0,
    }

    @classmethod
    def get(cls) -> dict[str, Any]:
        path_text = os.environ.get("VLLM_NVFP4_MLA_CAPTURE_CONTROL")
        if not path_text:
            return cls._value
        now = time.monotonic()
        with cls._lock:
            if now - cls._checked_at < 0.25:
                return cls._value
            cls._checked_at = now
            path = Path(path_text)
            try:
                mtime = path.stat().st_mtime_ns
                if mtime != cls._mtime_ns:
                    value = json.loads(path.read_text())
                    bucket = _safe_name(str(value.get("bucket", "unlabeled")))
                    phase = str(value.get("phase", "auto"))
                    if phase not in ("auto", "prefill", "decode", "mixed"):
                        raise ValueError(f"invalid capture phase: {phase}")
                    flush_epoch = int(value.get("flush_epoch", 0))
                    if flush_epoch < 0:
                        raise ValueError("flush_epoch must be non-negative")
                    cls._value = {
                        "bucket": bucket,
                        "phase": phase,
                        "enabled": bool(value.get("enabled", True)),
                        "flush_epoch": flush_epoch,
                    }
                    cls._mtime_ns = mtime
            except FileNotFoundError:
                pass
            except Exception:
                LOGGER.exception("Could not read NVFP4 capture control file")
            return cls._value


class _PriorityReservoir:
    """Mergeable uniform reservoir: retain the largest random priorities."""

    def __init__(self, capacity: int, seed: int) -> None:
        import numpy as np

        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.values = np.empty((0, 16), dtype=np.float32)
        self.priorities = np.empty((0,), dtype=np.float64)
        self.seen = 0

    def add(self, values) -> None:
        import numpy as np

        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 16:
            raise ValueError(f"capture groups must be (N, 16), got {array.shape}")
        if not len(array):
            return
        priorities = self.rng.random(len(array))
        self.seen += len(array)
        values_all = np.concatenate((self.values, array), axis=0)
        priorities_all = np.concatenate((self.priorities, priorities), axis=0)
        if len(values_all) > self.capacity:
            keep = np.argpartition(priorities_all, -self.capacity)[-self.capacity:]
            values_all = values_all[keep]
            priorities_all = priorities_all[keep]
        self.values = values_all
        self.priorities = priorities_all


class LayerCollector:
    def __init__(self, layer: str, root: Path) -> None:
        self.layer = layer
        self.root = root
        self.rank = _rank()
        self.pid = os.getpid()
        self.capacity = int(
            os.environ.get("VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_STRATUM", "65536")
        )
        self.groups_per_call = int(
            os.environ.get("VLLM_NVFP4_MLA_CAPTURE_GROUPS_PER_CALL", "1024")
        )
        self.flush_every = int(
            os.environ.get("VLLM_NVFP4_MLA_CAPTURE_FLUSH_EVERY", "128")
        )
        self.decode_threshold = int(
            os.environ.get("VLLM_NVFP4_MLA_CAPTURE_DECODE_THRESHOLD", "64")
        )
        if min(self.capacity, self.groups_per_call, self.flush_every) <= 0:
            raise ValueError("NVFP4 capture limits must be positive")
        self._calls = 0
        self._sample_offset = 0
        self._lock = threading.Lock()
        self._reservoirs: dict[str, _PriorityReservoir] = {}
        self._flush_epoch = 0

    def _reservoir(self, stratum: str) -> _PriorityReservoir:
        existing = self._reservoirs.get(stratum)
        if existing is not None:
            return existing
        seed_bytes = hashlib.sha256(
            f"{self.layer}\0{stratum}\0{self.rank}\0{self.pid}".encode()
        ).digest()
        result = _PriorityReservoir(
            self.capacity, int.from_bytes(seed_bytes[:8], "little")
        )
        self._reservoirs[stratum] = result
        return result

    @torch.no_grad()
    def observe(self, kv_c: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        control = _Control.get()
        flush_epoch = int(control["flush_epoch"])
        if flush_epoch != self._flush_epoch:
            self._flush_epoch = flush_epoch
            self.flush()
        if not control["enabled"] or kv_c.numel() == 0:
            return
        token_count = min(int(kv_c.shape[0]), int(slot_mapping.numel()))
        if token_count == 0:
            return
        slots = slot_mapping.reshape(-1)[:token_count]
        valid_tokens = torch.nonzero(slots >= 0, as_tuple=False).flatten()
        if valid_tokens.numel() == 0:
            return
        groups = kv_c[:token_count].index_select(0, valid_tokens).reshape(-1, 16)
        total_groups = int(groups.shape[0])
        sample_count = min(total_groups, self.groups_per_call)
        stride = max(total_groups // sample_count, 1)
        offset = self._sample_offset % stride
        indices = offset + torch.arange(
            sample_count, device=groups.device, dtype=torch.int64
        ) * stride
        indices.clamp_(max=total_groups - 1)
        self._sample_offset += 104729
        sample = groups.index_select(0, indices).float().cpu().numpy()
        phase = control["phase"]
        if phase == "auto":
            phase = "decode" if int(valid_tokens.numel()) <= self.decode_threshold else "prefill"
        stratum = f"{control['bucket']}__{phase}"
        with self._lock:
            self._reservoir(stratum).add(sample)
            self._calls += 1
            should_flush = self._calls % self.flush_every == 0
        if should_flush:
            self.flush()

    def flush(self) -> None:
        import numpy as np

        with self._lock:
            snapshots = [
                (name, reservoir.values.copy(), reservoir.priorities.copy(), reservoir.seen)
                for name, reservoir in self._reservoirs.items()
            ]
        directory = self.root / f"rank-{self.rank:03d}" / f"pid-{self.pid}"
        directory.mkdir(parents=True, exist_ok=True)
        layer_name = _safe_name(self.layer)
        for stratum, values, priorities, seen in snapshots:
            metadata = {
                "schema": CAPTURE_SCHEMA,
                "algorithm": ALGORITHM,
                "record_layout": RECORD_LAYOUT,
                "layer": self.layer,
                "stratum": stratum,
                "rank": self.rank,
                "pid": self.pid,
                "rows": int(len(values)),
                "seen": int(seen),
            }
            output = directory / f"{layer_name}--{_safe_name(stratum)}.npz"
            temporary = output.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                values=values,
                priorities=priorities,
            )
            temporary.replace(output)


def _flush_all() -> None:
    for collector in list(_COLLECTORS):
        try:
            collector.flush()
        except Exception:
            LOGGER.exception("Could not flush NVFP4 capture for %s", collector.layer)


atexit.register(_flush_all)


def runtime_for_layer(layer: str) -> LayerRuntime:
    layer = str(layer)
    layers = _load_scale_layers()
    latent_scale = 1.0
    if layers:
        strict = os.environ.get("VLLM_NVFP4_MLA_SCALES_STRICT", "1") != "0"
        if layer in layers:
            latent_scale = layers[layer]
        elif strict:
            raise KeyError(f"calibrated NVFP4 artifact has no entry for {layer!r}")
        else:
            LOGGER.warning("No calibrated NVFP4 scale for %s; using 1.0", layer)

    capture_dir = os.environ.get("VLLM_NVFP4_MLA_CAPTURE_DIR")
    collector = LayerCollector(layer, Path(capture_dir)) if capture_dir else None
    if collector is not None:
        _COLLECTORS.append(collector)
    return LayerRuntime(latent_scale=latent_scale, collector=collector)
