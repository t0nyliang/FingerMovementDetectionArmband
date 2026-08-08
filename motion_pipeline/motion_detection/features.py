"""Timestamped windows and motion features for BNO085 data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .protocol import CHANNEL_COUNT, MotionFrame


SAMPLE_HZ = 50
WINDOW_SAMPLES = 15
WINDOW_SECONDS = (WINDOW_SAMPLES - 1) / SAMPLE_HZ
MAX_LIVE_GAP_SECONDS = 0.250
UINT32_MASK = 0xFFFFFFFF
BASELINE_FORMAT = "motion_baseline_v1"


def _angle_delta_degrees(angles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return signed shortest-path differences in degrees."""
    return (angles - reference + 180.0) % 360.0 - 180.0


def resample_samples(
    samples: np.ndarray,
    sample_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate motion samples onto the fixed 50 Hz grid."""
    samples = np.asarray(samples, dtype=float)
    sample_times = np.asarray(sample_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"samples must contain rows of {CHANNEL_COUNT} values")
    if sample_times.shape != (len(samples),) or len(samples) < 2:
        raise ValueError("sample times must match at least two sample rows")
    if target_times.ndim != 1 or not len(target_times):
        raise ValueError("target times must be a non-empty vector")
    if not (
        np.isfinite(samples).all()
        and np.isfinite(sample_times).all()
        and np.isfinite(target_times).all()
    ):
        raise ValueError("resampling input must be finite")
    if np.any(np.diff(sample_times) <= 0):
        raise ValueError("sample times must be strictly increasing")

    output = []
    for channel in range(CHANNEL_COUNT):
        source = samples[:, channel]
        if channel < 3:
            # RVC orientation is reported in degrees. Unwrap before interpolation
            # so a -179 -> +179 transition does not create a false 358-degree move.
            source = np.rad2deg(np.unwrap(np.deg2rad(source)))
        output.append(np.interp(target_times, sample_times, source))
    return np.column_stack(output)


class MotionWindowResampler:
    """Build fixed-duration windows from timestamped, slightly irregular frames."""

    def __init__(self) -> None:
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.last_sequence: int | None = None
        self.last_device_us: int | None = None
        self.elapsed_seconds = 0.0

    def _restart(self, frame: MotionFrame) -> None:
        self.frames.clear()
        self.frames.append((0.0, frame.vector))
        self.last_sequence = frame.sequence
        self.last_device_us = frame.device_us
        self.elapsed_seconds = 0.0

    def reset(self) -> None:
        self.frames.clear()
        self.last_sequence = None
        self.last_device_us = None
        self.elapsed_seconds = 0.0

    def add(self, frame: MotionFrame) -> tuple[np.ndarray | None, bool]:
        """Return ``(window, reset)`` for each new physical frame."""
        if self.last_sequence is None or self.last_device_us is None:
            self._restart(frame)
            return None, False

        sequence_delta = (frame.sequence - self.last_sequence) & UINT32_MASK
        time_delta_us = (frame.device_us - self.last_device_us) & UINT32_MASK
        time_delta = time_delta_us / 1_000_000.0
        discontinuity = (
            sequence_delta == 0
            or sequence_delta >= 0x80000000
            or time_delta_us == 0
            or time_delta > MAX_LIVE_GAP_SECONDS
        )
        if discontinuity:
            self._restart(frame)
            return None, True

        self.elapsed_seconds += time_delta
        self.last_sequence = frame.sequence
        self.last_device_us = frame.device_us
        self.frames.append((self.elapsed_seconds, frame.vector))

        window_start = self.elapsed_seconds - WINDOW_SECONDS
        while len(self.frames) > 2 and self.frames[1][0] <= window_start:
            self.frames.popleft()
        # Allow a tiny floating-point tolerance when the oldest sample lands
        # exactly on the left edge of the requested window.
        if self.frames[0][0] - window_start > 1e-9:
            return None, False

        sample_times = np.asarray([item[0] for item in self.frames], dtype=float)
        samples = np.stack([item[1] for item in self.frames])
        target_times = window_start + np.arange(WINDOW_SAMPLES) / SAMPLE_HZ
        return resample_samples(samples, sample_times, target_times), False


@dataclass(frozen=True)
class MotionBaseline:
    """Relaxed-pose reference for the six UART-RVC channels."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (CHANNEL_COUNT,) or not np.isfinite(values).all():
            raise ValueError(f"baseline must contain {CHANNEL_COUNT} finite values")
        object.__setattr__(self, "values", values)


def estimate_baseline(samples: np.ndarray) -> MotionBaseline:
    """Estimate a relaxed baseline using the channel-wise median."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"baseline samples must have {CHANNEL_COUNT} channels")
    if len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("baseline samples must contain at least two finite rows")
    return MotionBaseline(np.median(samples, axis=0))


@dataclass(frozen=True)
class MotionFeatures:
    """Window statistics used by the threshold detector."""

    acceleration_rms: float
    acceleration_peak: float
    orientation_speed_rms: float
    orientation_speed_peak: float
    orientation_offset_rms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "acceleration_rms": self.acceleration_rms,
            "acceleration_peak": self.acceleration_peak,
            "orientation_speed_rms": self.orientation_speed_rms,
            "orientation_speed_peak": self.orientation_speed_peak,
            "orientation_offset_rms": self.orientation_offset_rms,
        }


def compute_features(
    window: np.ndarray,
    baseline: MotionBaseline | np.ndarray,
    sample_hz: float = SAMPLE_HZ,
) -> MotionFeatures:
    """Compute dynamic acceleration and orientation-change statistics."""
    window = np.asarray(window, dtype=float)
    baseline_values = (
        baseline.values if isinstance(baseline, MotionBaseline) else np.asarray(baseline)
    )
    if window.shape != (WINDOW_SAMPLES, CHANNEL_COUNT):
        raise ValueError(
            f"window must have shape ({WINDOW_SAMPLES}, {CHANNEL_COUNT})"
        )
    if baseline_values.shape != (CHANNEL_COUNT,):
        raise ValueError(f"baseline must have shape ({CHANNEL_COUNT},)")
    if sample_hz <= 0 or not np.isfinite(window).all() or not np.isfinite(baseline_values).all():
        raise ValueError("feature input must be finite and sample_hz must be positive")

    acceleration_delta = window[:, 3:] - baseline_values[3:]
    acceleration_magnitude = np.linalg.norm(acceleration_delta, axis=1)

    orientation = window[:, :3]
    unwrapped_orientation = np.column_stack(
        [np.rad2deg(np.unwrap(np.deg2rad(orientation[:, i]))) for i in range(3)]
    )
    orientation_speed = np.linalg.norm(
        np.diff(unwrapped_orientation, axis=0) * sample_hz,
        axis=1,
    )
    orientation_offset = _angle_delta_degrees(orientation, baseline_values[:3])
    orientation_offset_magnitude = np.linalg.norm(orientation_offset, axis=1)

    return MotionFeatures(
        acceleration_rms=float(np.sqrt(np.mean(np.square(acceleration_magnitude)))),
        acceleration_peak=float(np.max(acceleration_magnitude)),
        orientation_speed_rms=float(np.sqrt(np.mean(np.square(orientation_speed)))),
        orientation_speed_peak=float(np.max(orientation_speed)),
        orientation_offset_rms=float(
            np.sqrt(np.mean(np.square(orientation_offset_magnitude)))
        ),
    )


@dataclass(frozen=True)
class MotionThresholds:
    """Thresholds for a normalized, explainable motion score."""

    acceleration_rms: float = 1.0
    acceleration_peak: float = 2.5
    orientation_speed_rms: float = 35.0
    orientation_speed_peak: float = 90.0

    def __post_init__(self) -> None:
        values = (
            self.acceleration_rms,
            self.acceleration_peak,
            self.orientation_speed_rms,
            self.orientation_speed_peak,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("all motion thresholds must be positive and finite")

    def score(self, features: MotionFeatures) -> float:
        """Return the largest threshold ratio; 1.0 means motion is present."""
        return max(
            features.acceleration_rms / self.acceleration_rms,
            features.acceleration_peak / self.acceleration_peak,
            features.orientation_speed_rms / self.orientation_speed_rms,
            features.orientation_speed_peak / self.orientation_speed_peak,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "acceleration_rms": self.acceleration_rms,
            "acceleration_peak": self.acceleration_peak,
            "orientation_speed_rms": self.orientation_speed_rms,
            "orientation_speed_peak": self.orientation_speed_peak,
        }


def save_baseline(baseline: MotionBaseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": BASELINE_FORMAT,
        "sample_hz": SAMPLE_HZ,
        "channels": ["yaw", "pitch", "roll", "ax", "ay", "az"],
        "values": baseline.values.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> MotionBaseline:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != BASELINE_FORMAT:
            raise ValueError
        values = np.asarray(payload["values"], dtype=float)
        return MotionBaseline(values)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid motion baseline; run motion calibration again") from exc
