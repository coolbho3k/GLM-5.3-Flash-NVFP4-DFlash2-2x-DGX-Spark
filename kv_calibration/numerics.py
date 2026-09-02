"""CPU reference numerics for the GLM NVFP4 latent-cache writer.

The serving format stores each group of 16 values as sixteen signed E2M1
nibbles plus one unsigned E4M3 scale byte.  ``latent_scale`` is the static
per-layer outer scale applied by the reader.  The writer evaluates the
amax/6 and amax/4 scale candidates and retains the lower-SSE reconstruction.

This reference models round-to-nearest-even and satfinite conversion.  The
GPU writer additionally uses ``rcp.approx.ftz`` for reciprocals, so a final
candidate must still pass the supplied GPU parity probe before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


E2M1_POSITIVE = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3_positive_table() -> np.ndarray:
    values: list[float] = []
    for code in range(127):  # positive finite codes 0x00 ... 0x7e
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            value = mantissa * 2.0**-9
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values.append(value)
    return np.asarray(values, dtype=np.float64)


E4M3_POSITIVE = _e4m3_positive_table()


def _nearest_even(values: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest table values and indices; midpoint ties choose even code."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), table[0], table[-1])
    upper = np.searchsorted(table, clipped, side="left")
    upper = np.minimum(upper, len(table) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_error = clipped - table[lower]
    upper_error = table[upper] - clipped
    choose_upper = upper_error < lower_error
    tie = upper_error == lower_error
    choose_upper |= tie & ((upper & 1) == 0)
    index = np.where(choose_upper, upper, lower)
    return table[index], index


def round_e4m3_satfinite(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round non-negative scale values to finite E4M3 and return value/code."""
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValueError("E4M3 scale inputs must be finite and non-negative")
    rounded, code = _nearest_even(array, E4M3_POSITIVE)
    return rounded, code.astype(np.uint8)


def round_e2m1_satfinite(values: np.ndarray) -> np.ndarray:
    """Round signed values to finite E2M1, using RN-even at midpoints."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("E2M1 inputs must be finite")
    magnitude, _ = _nearest_even(np.abs(array), E2M1_POSITIVE)
    # The sign of zero is irrelevant to reconstruction and SSE.
    return np.copysign(magnitude, array)


@dataclass(frozen=True)
class QuantizedGroups:
    reconstruction: np.ndarray
    scale_codes: np.ndarray
    scale_values: np.ndarray
    chose_four: np.ndarray


def quantize_groups_four_over_six(
    groups: np.ndarray,
    *,
    latent_scale: float = 1.0,
) -> QuantizedGroups:
    """Quantize ``(N, 16)`` groups using the serving four-over-six rule."""
    values = np.asarray(groups, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"groups must have shape (N, 16), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("groups contain NaN or infinity")
    latent_scale = float(latent_scale)
    if not np.isfinite(latent_scale) or latent_scale <= 0.0:
        raise ValueError("latent_scale must be finite and positive")

    amax = np.max(np.abs(values), axis=1)
    reconstructions: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    decoded_scales: list[np.ndarray] = []
    errors: list[np.ndarray] = []

    for divisor in (6.0, 4.0):
        encoded, code = round_e4m3_satfinite(amax / (divisor * latent_scale))
        dequant_scale = encoded * latent_scale
        inverse = np.divide(
            1.0,
            dequant_scale,
            out=np.zeros_like(dequant_scale),
            where=dequant_scale != 0.0,
        )
        quantized = round_e2m1_satfinite(values * inverse[:, None])
        reconstruction = quantized * dequant_scale[:, None]
        reconstructions.append(reconstruction)
        codes.append(code)
        decoded_scales.append(dequant_scale)
        errors.append(np.sum((reconstruction - values) ** 2, axis=1))

    # Match the writer: /4 must be strictly better; ties retain /6.
    choose_four = errors[1] < errors[0]
    reconstruction = np.where(
        choose_four[:, None], reconstructions[1], reconstructions[0]
    )
    scale_codes = np.where(choose_four, codes[1], codes[0]).astype(np.uint8)
    scale_values = np.where(
        choose_four, decoded_scales[1], decoded_scales[0]
    )
    return QuantizedGroups(
        reconstruction=reconstruction,
        scale_codes=scale_codes,
        scale_values=scale_values,
        chose_four=choose_four,
    )


def evaluate_groups(
    groups: np.ndarray,
    *,
    latent_scale: float,
    chunk_rows: int = 32768,
    relative_epsilon: float | None = None,
) -> dict[str, float | int]:
    """Evaluate one outer scale without retaining reconstructed activations."""
    values = np.asarray(groups)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"groups must have shape (N, 16), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("at least one group is required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    total_sse = 0.0
    total_energy = 0.0
    chose_four = 0
    zero_scales = 0
    saturated_scales = 0
    group_sse_parts: list[np.ndarray] = []
    group_energy_parts: list[np.ndarray] = []

    for start in range(0, values.shape[0], chunk_rows):
        chunk = values[start : start + chunk_rows].astype(np.float64, copy=False)
        quantized = quantize_groups_four_over_six(
            chunk, latent_scale=latent_scale
        )
        error = quantized.reconstruction - chunk
        group_sse = np.sum(error * error, axis=1)
        group_energy = np.sum(chunk * chunk, axis=1)
        total_sse += float(np.sum(group_sse, dtype=np.float64))
        total_energy += float(np.sum(group_energy, dtype=np.float64))
        chose_four += int(np.count_nonzero(quantized.chose_four))
        zero_scales += int(np.count_nonzero(quantized.scale_codes == 0))
        saturated_scales += int(np.count_nonzero(quantized.scale_codes == 0x7E))
        group_sse_parts.append(group_sse)
        group_energy_parts.append(group_energy)

    group_sse_all = np.concatenate(group_sse_parts)
    group_energy_all = np.concatenate(group_energy_parts)
    if relative_epsilon is None:
        # Avoid letting genuinely zero/near-zero groups dominate the relative
        # diagnostic. This does not affect the primary MSE/NMSE objective.
        relative_epsilon = max(total_energy / values.shape[0] * 1e-12, 1e-30)
    relative = group_sse_all / np.maximum(group_energy_all, relative_epsilon)
    element_count = int(values.size)
    group_count = int(values.shape[0])
    return {
        "groups": group_count,
        "values": element_count,
        "sse": total_sse,
        "signal_energy": total_energy,
        "mse": total_sse / element_count,
        "nmse": total_sse / max(total_energy, 1e-300),
        "mean_group_relative_mse": float(np.mean(relative)),
        "p95_group_relative_mse": float(np.quantile(relative, 0.95)),
        "p99_group_relative_mse": float(np.quantile(relative, 0.99)),
        "four_fraction": chose_four / group_count,
        "zero_scale_fraction": zero_scales / group_count,
        "saturated_scale_fraction": saturated_scales / group_count,
    }
