from __future__ import annotations


ATTRIBUTE_NAMES = ["polyphony_rate", "rhythmic_intensity", "velocity_dynamics", "note_density"]


def compute_attributes_stub(_: list[int]) -> list[list[int]]:
    """
    Stub: later compute per-bar attributes and return shape [bars=24][attrs=4] of integer bins 0..7.
    """
    raise NotImplementedError("compute_attributes_stub is a stub.")

