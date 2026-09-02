"""Offline calibration tools for the 288-byte GLM NVFP4 MLA cache."""

__all__ = ["evaluate_groups", "quantize_groups_four_over_six"]


def __getattr__(name: str):
    """Keep standard-library drivers usable on hosts without NumPy."""
    if name in __all__:
        from . import numerics

        return getattr(numerics, name)
    raise AttributeError(name)
