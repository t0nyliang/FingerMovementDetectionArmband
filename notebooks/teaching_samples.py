"""Fixed, armband-shaped captures shared by the teaching notebooks.

The values are generated deterministically from measured-like magnetic baselines,
2 µT-scale noise, slow drift, and held gesture changes. They are not hardware
recordings, but their shape matches the 50 Hz / 12-channel FRAME data expected
by the calibration pipeline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


SAMPLE_HZ = 50
CAPTURE_SAMPLES = 2 * SAMPLE_HZ
CAPTURES_PER_CLASS = 10
LABELS = ("rest", "wrist_up", "spread", "fist")
CHANNEL_NAMES = tuple(
    f"S{sensor}{axis}" for sensor in range(4) for axis in ("x", "y", "z")
)

# Sensor 0 starts near the baseline seen in recordings/; the other positions
# have different static fields because they sit at different places on the arm.
BASELINE = np.array(
    [
        373.0, -423.0, 165.0,
        286.0, -351.0, 129.0,
        194.0, -276.0, 211.0,
        331.0, -188.0, 148.0,
    ],
    dtype=float,
)

GESTURE_CHANGES = {
    "rest": np.zeros(12),
    "wrist_up": np.array(
        [15.0, -7.0, 5.0, 10.0, -9.0, 4.0, 4.0, 2.0, -5.0, -7.0, 3.0, 5.0]
    ),
    "spread": np.array(
        [4.0, 15.0, -9.0, -3.0, 13.0, -10.0, 10.0, 8.0, -4.0, -9.0, 5.0, 3.0]
    ),
    "fist": np.array(
        [-14.0, -11.0, 9.0, -10.0, -14.0, 11.0, -8.0, -9.0, 15.0, 13.0, -5.0, -7.0]
    ),
}


@lru_cache(maxsize=1)
def load_teaching_dataset() -> dict[str, np.ndarray | tuple[str, ...]]:
    """Return 40 fixed two-second captures in the project's FRAME shape."""
    rng = np.random.default_rng(90393)
    time_s = np.arange(CAPTURE_SAMPLES, dtype=float) / SAMPLE_HZ
    captures: list[np.ndarray] = []
    labels: list[str] = []
    capture_number: list[int] = []

    for label in LABELS:
        for example in range(CAPTURES_PER_CLASS):
            # The guide says “move after GO and hold”; the central second is
            # deliberately the stable part used for training windows.
            onset_s = rng.normal(0.36, 0.045)
            envelope = 1.0 / (1.0 + np.exp(-(time_s - onset_s) / 0.055))
            if label == "rest":
                envelope = np.zeros_like(time_s)

            amplitude = rng.normal(1.0, 0.10)
            capture_offset = rng.normal(0.0, 0.9, size=12)
            drift = (
                np.sin(2 * np.pi * (0.22 * time_s + rng.uniform()))
                [:, None]
                * rng.normal(0.0, 0.55, size=(1, 12))
            )
            common_noise = rng.normal(0.0, 0.7, size=(CAPTURE_SAMPLES, 1))
            independent_noise = rng.normal(
                0.0, 2.0, size=(CAPTURE_SAMPLES, 12)
            )

            capture = (
                BASELINE
                + capture_offset
                + envelope[:, None] * amplitude * GESTURE_CHANGES[label]
                + drift
                + common_noise * np.linspace(-0.5, 0.5, 12)
                + independent_noise
            )
            captures.append(capture)
            labels.append(label)
            capture_number.append(example)

    return {
        "time_s": time_s,
        "baseline": BASELINE.copy(),
        "captures": np.stack(captures),
        "labels": np.asarray(labels),
        "capture_number": np.asarray(capture_number),
        "channel_names": CHANNEL_NAMES,
    }
