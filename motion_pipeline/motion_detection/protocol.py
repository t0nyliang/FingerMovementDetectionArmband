"""Protocol definitions for the standalone BNO085 motion stream."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


PROTOCOL = "MOTION_v1"
CHANNEL_NAMES = ("yaw", "pitch", "roll", "ax", "ay", "az")
CHANNEL_COUNT = len(CHANNEL_NAMES)


@dataclass(frozen=True)
class MotionFrame:
    """One timestamped UART-RVC sample from the BNO085."""

    sequence: int
    device_us: int
    values: tuple[float, ...]

    @property
    def vector(self) -> np.ndarray:
        """Return the six channels as a fresh float array."""
        return np.asarray(self.values, dtype=float)


def parse_motion_frame(line: str) -> MotionFrame | None:
    """Parse a MOTION line, ignoring firmware status lines."""
    line = line.strip()
    if not line:
        return None
    if line.startswith("ERROR,"):
        raise RuntimeError(line)

    fields = line.split(",")
    if fields[0] != "MOTION":
        return None
    expected_fields = 3 + CHANNEL_COUNT
    if len(fields) != expected_fields:
        raise RuntimeError(
            f"malformed MOTION; expected {CHANNEL_COUNT} sensor values"
        )

    try:
        sequence = int(fields[1])
        device_us = int(fields[2])
        values = tuple(float(value) for value in fields[3:])
    except ValueError:
        raise RuntimeError("MOTION contains a non-numeric value") from None

    if sequence < 0 or device_us < 0:
        raise RuntimeError("MOTION sequence and timestamp must be non-negative")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("all MOTION values must be finite")
    return MotionFrame(sequence, device_us, values)
