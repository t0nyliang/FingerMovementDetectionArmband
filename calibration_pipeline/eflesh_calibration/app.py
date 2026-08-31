"""Guided calibration and live use of the teaching KNN pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from typing import Callable, Iterator

import numpy as np

from .knn import (
    CLASSES,
    FILTER_SAMPLES,
    RAW_WINDOW_SAMPLES,
    SAMPLE_HZ,
    WINDOW_SAMPLES,
    feature,
    load_model,
    make_model,
    predict,
    proximity_scores,
    save_model,
)
from .sensor import CHANNEL_COUNT, Sensor, SensorFrame


BASELINE_SAMPLES = 2 * SAMPLE_HZ
CALIBRATION_CAPTURE_SECONDS = 2
CALIBRATION_EDGE_SAMPLES = SAMPLE_HZ // 2
CALIBRATION_CAPTURE_SAMPLES = CALIBRATION_CAPTURE_SECONDS * SAMPLE_HZ
CALIBRATION_CLEAN_SAMPLES = (
    CALIBRATION_CAPTURE_SAMPLES - 2 * CALIBRATION_EDGE_SAMPLES
)
EXAMPLES_PER_CLASS = 10
TRAINING_WINDOW_STRIDE_SAMPLES = SAMPLE_HZ // 10
STABILITY_FRAMES = 2
LIVE_WINDOW_SECONDS = (RAW_WINDOW_SAMPLES - 1) / SAMPLE_HZ
MAX_LIVE_GAP_SECONDS = 0.250
UINT32_MASK = 0xFFFFFFFF
PROFILE_PATH = Path(__file__).resolve().parents[1] / "profile.json"


def read_for_duration(
    sensor: Sensor,
    duration_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[np.ndarray, np.ndarray]:
    """Read fresh frames for a wall-clock duration and return relative times."""
    # Count only frames produced after GO. Without this reset, the three-second
    # countdown can fill the serial buffer with stale data.
    sensor.discard_pending()
    started = clock()
    samples: list[np.ndarray] = []
    sample_times: list[float] = []
    while clock() - started < duration_s:
        sample = sensor.read()
        sample_time = clock() - started
        # A serial read can return immediately when several frames are already
        # buffered.  On clocks with a coarse resolution that gives consecutive
        # frames the same timestamp, which ``np.interp`` cannot accept. Keep
        # the newest frame for that instant instead of passing duplicates on to
        # the resampler.
        if sample_times and sample_time <= sample_times[-1]:
            samples[-1] = sample
            continue
        samples.append(sample)
        sample_times.append(sample_time)
    if len(samples) < 2:
        raise RuntimeError("not enough sensor frames received during capture")
    return np.stack(samples), np.asarray(sample_times, dtype=float)


def resample_samples(
    samples: np.ndarray,
    sample_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate timestamped sensor values onto an explicit time grid."""
    samples = np.asarray(samples, dtype=float)
    sample_times = np.asarray(sample_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError("samples must contain rows of 12 sensor values")
    if sample_times.shape != (len(samples),) or len(samples) < 2:
        raise ValueError("sample times must match at least two sensor rows")
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
    return np.column_stack(
        [
            np.interp(target_times, sample_times, samples[:, channel])
            for channel in range(samples.shape[1])
        ]
    )


def read_calibration_windows(
    sensor: Sensor,
    clock: Callable[[], float] = time.monotonic,
) -> list[np.ndarray]:
    """Capture two real seconds and extract overlapping 300 ms windows."""
    capture, sample_times = read_for_duration(
        sensor,
        CALIBRATION_CAPTURE_SECONDS,
        clock,
    )
    capture = resample_samples(
        capture,
        sample_times,
        np.arange(CALIBRATION_CAPTURE_SAMPLES) / SAMPLE_HZ,
    )
    windows = []
    first_smoothed_index = CALIBRATION_EDGE_SAMPLES
    final_start = (
        CALIBRATION_EDGE_SAMPLES + CALIBRATION_CLEAN_SAMPLES - WINDOW_SAMPLES
    )
    for smoothed_start in range(
        first_smoothed_index,
        final_start + 1,
        TRAINING_WINDOW_STRIDE_SAMPLES,
    ):
        raw_start = smoothed_start - FILTER_SAMPLES + 1
        windows.append(capture[raw_start : raw_start + RAW_WINDOW_SAMPLES])
    return windows


def countdown(message: str) -> None:
    print(message)
    for number in (3, 2, 1):
        print(number, flush=True)
        time.sleep(1)
    print("GO", flush=True)


def collect_baseline(sensor: Sensor) -> np.ndarray:
    countdown("Keep your hand relaxed.")
    capture, _sample_times = read_for_duration(sensor, BASELINE_SAMPLES / SAMPLE_HZ)
    return np.mean(capture, axis=0)


def calibration_instruction(label: str) -> str:
    if label == "rest":
        return "Stay relaxed for the whole capture."
    if label == "wrist_up":
        return "After GO, flex your wrist up and hold it."
    if label == "spread":
        return "After GO, spread and hold your fingers."
    return "After GO, make and hold a fist."


def collect_gesture_features(
    sensor: Sensor,
    label: str,
    baseline: np.ndarray,
) -> list[np.ndarray]:
    """Capture one complete calibration set for a gesture."""
    print(f"\n{label.upper()}: {calibration_instruction(label)}")
    input(f"Press Enter to begin this set of {EXAMPLES_PER_CLASS} examples...")
    features: list[np.ndarray] = []
    for example in range(1, EXAMPLES_PER_CLASS + 1):
        countdown(f"{label} example {example}/{EXAMPLES_PER_CLASS}")
        print(
            f"capturing for {CALIBRATION_CAPTURE_SECONDS} seconds...",
            flush=True,
        )
        windows = read_calibration_windows(sensor)
        features.extend(feature(window, baseline) for window in windows)
        print("captured")
    return features


def calibrate(port: str, profile_path: Path) -> None:
    features: list[np.ndarray] = []
    labels: list[str] = []
    with Sensor(port) as sensor:
        print(f"Connected to {port}.")
        baseline = collect_baseline(sensor)
        for label in CLASSES:
            captured = collect_gesture_features(sensor, label, baseline)
            features.extend(captured)
            labels.extend([label] * len(captured))

    save_model(make_model(np.stack(features), labels), profile_path)
    print(f"\nSaved {len(labels)} examples to {profile_path}")


def recalibrate(port: str, profile_path: Path, gesture: str) -> None:
    """Replace one gesture's examples while preserving all other classes."""
    model = load_model(profile_path)
    existing_features = np.asarray(model["features"], dtype=float)
    existing_labels = list(model["labels"])

    with Sensor(port) as sensor:
        print(f"Connected to {port}.")
        baseline = collect_baseline(sensor)
        replacement = collect_gesture_features(sensor, gesture, baseline)

    kept_features = [
        row for row, label in zip(existing_features, existing_labels) if label != gesture
    ]
    kept_labels = [label for label in existing_labels if label != gesture]
    updated_features = kept_features + replacement
    updated_labels = kept_labels + [gesture] * len(replacement)
    save_model(make_model(np.stack(updated_features), updated_labels), profile_path)
    print(
        f"\nReplaced {existing_labels.count(gesture)} {gesture} examples with "
        f"{len(replacement)} new examples in {profile_path}"
    )


def iter_prediction_results(
    sensor: Sensor, model: dict, baseline: np.ndarray
) -> Iterator[tuple[str, dict[str, float]]]:
    """Predict once per fresh frame using a timestamped 360 ms window."""
    window_builder = LiveWindowResampler()
    stabilizer = LabelStabilizer()
    while True:
        window, reset = window_builder.add(sensor.read_frame())
        if reset:
            stabilizer.reset()
        if window is None:
            continue
        sample = feature(window, baseline)
        label = stabilizer.update(predict(model, sample))
        yield label, proximity_scores(model, sample)


class LabelStabilizer:
    """Change the published label after consecutive matching predictions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.label = "rest"
        self.candidate = "rest"
        self.count = 0

    def update(self, current: str) -> str:
        if current == self.candidate:
            self.count += 1
        else:
            self.candidate = current
            self.count = 1
        if self.count >= STABILITY_FRAMES:
            self.label = self.candidate
        return self.label


class LiveWindowResampler:
    """Build fixed-duration 50 Hz windows from timestamped physical frames."""

    def __init__(self) -> None:
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.last_sequence: int | None = None
        self.last_device_us: int | None = None
        self.elapsed_seconds = 0.0

    def _restart(self, frame: SensorFrame) -> None:
        self.frames.clear()
        self.frames.append((0.0, frame.values))
        self.last_sequence = frame.sequence
        self.last_device_us = frame.device_us
        self.elapsed_seconds = 0.0

    def add(self, frame: SensorFrame) -> tuple[np.ndarray | None, bool]:
        """Add a frame and return (resampled window, discontinuity reset)."""
        if self.last_sequence is None or self.last_device_us is None:
            self._restart(frame)
            return None, False

        sequence_delta = (frame.sequence - self.last_sequence) & UINT32_MASK
        time_delta_us = (frame.device_us - self.last_device_us) & UINT32_MASK
        time_delta = time_delta_us / 1_000_000
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
        self.frames.append((self.elapsed_seconds, frame.values))

        window_start = self.elapsed_seconds - LIVE_WINDOW_SECONDS
        while len(self.frames) > 2 and self.frames[1][0] <= window_start:
            self.frames.popleft()
        if self.frames[0][0] > window_start:
            return None, False

        sample_times = np.asarray([item[0] for item in self.frames], dtype=float)
        samples = np.stack([item[1] for item in self.frames])
        target_times = window_start + np.arange(RAW_WINDOW_SAMPLES) / SAMPLE_HZ
        return resample_samples(samples, sample_times, target_times), False


class OnsetTracker:
    """Fire each gesture once until a rest window rearms all gestures."""

    def __init__(self) -> None:
        self.previous = "rest"
        self.fired_since_rest: set[str] = set()

    def update(self, current: str) -> str | None:
        if current == "rest":
            self.fired_since_rest.clear()
            self.previous = current
            return None
        is_new_onset = current != self.previous and current not in self.fired_since_rest
        if is_new_onset:
            self.fired_since_rest.add(current)
        self.previous = current
        return current if is_new_onset else None


def live(port: str, profile_path: Path) -> None:
    model = load_model(profile_path)
    with Sensor(port) as sensor:
        baseline = collect_baseline(sensor)
        print("Live detection started. Press Ctrl+C to stop.")
        tracker = OnsetTracker()
        for label, _scores in iter_prediction_results(sensor, model, baseline):
            onset = tracker.update(label)
            print(f"{label:6s}" + (f"  ONSET {onset}" if onset else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal four-sensor KNN demo")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "capture examples and save a model"),
        ("live", "classify a timestamp-resampled rolling window"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
        command.add_argument("--profile", type=Path, default=PROFILE_PATH)
    command = commands.add_parser(
        "recalibrate",
        help="replace the saved examples for one gesture",
    )
    command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
    command.add_argument("--profile", type=Path, default=PROFILE_PATH)
    command.add_argument("--gesture", required=True, choices=CLASSES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            calibrate(args.port, args.profile)
        elif args.command == "recalibrate":
            recalibrate(args.port, args.profile, args.gesture)
        else:
            live(args.port, args.profile)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
