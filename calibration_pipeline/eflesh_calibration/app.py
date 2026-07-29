"""Guided calibration and live use of the teaching KNN pipeline."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from typing import Iterator

import numpy as np

from .knn import (
    CLASSES,
    SAMPLE_HZ,
    WINDOW_SAMPLES,
    feature,
    load_model,
    make_model,
    predict,
    proximity_scores,
    save_model,
)
from .sensor import Sensor


BASELINE_SAMPLES = 2 * SAMPLE_HZ
CALIBRATION_CAPTURE_SECONDS = 2
CALIBRATION_EDGE_SAMPLES = SAMPLE_HZ // 2
CALIBRATION_CAPTURE_SAMPLES = CALIBRATION_CAPTURE_SECONDS * SAMPLE_HZ
CALIBRATION_WINDOW_SAMPLES = (
    CALIBRATION_CAPTURE_SAMPLES - 2 * CALIBRATION_EDGE_SAMPLES
)
EXAMPLES_PER_CLASS = 5
PREDICTION_STRIDE_SAMPLES = SAMPLE_HZ // 10
PROFILE_PATH = Path(__file__).resolve().parents[1] / "profile.json"


def read_window(sensor: Sensor, count: int = WINDOW_SAMPLES) -> np.ndarray:
    return np.stack([sensor.read() for _ in range(count)])


def read_calibration_window(sensor: Sensor) -> np.ndarray:
    """Capture two seconds and retain the clean one-second center."""
    capture = read_window(sensor, CALIBRATION_CAPTURE_SAMPLES)
    return capture[CALIBRATION_EDGE_SAMPLES:-CALIBRATION_EDGE_SAMPLES]


def countdown(message: str) -> None:
    print(message)
    for number in (3, 2, 1):
        print(number, flush=True)
        time.sleep(1)
    print("GO", flush=True)


def collect_baseline(sensor: Sensor) -> np.ndarray:
    countdown("Keep your hand relaxed.")
    return np.mean(read_window(sensor, BASELINE_SAMPLES), axis=0)


def calibration_instruction(label: str) -> str:
    if label == "rest":
        return "Stay relaxed for the whole capture."
    if label == "wrist_up":
        return "After GO, flex your wrist up and hold it."
    if label == "spread":
        return "After GO, spread and hold your fingers."
    return "After GO, make and hold a fist."


def calibrate(port: str, profile_path: Path) -> None:
    features: list[np.ndarray] = []
    labels: list[str] = []
    with Sensor(port) as sensor:
        print(f"Connected to {port}.")
        baseline = collect_baseline(sensor)
        for label in CLASSES:
            print(f"\n{label.upper()}: {calibration_instruction(label)}")
            input("Press Enter to begin this set of five examples...")
            for example in range(1, EXAMPLES_PER_CLASS + 1):
                countdown(f"{label} example {example}/{EXAMPLES_PER_CLASS}")
                print(
                    f"capturing for {CALIBRATION_CAPTURE_SECONDS} seconds...",
                    flush=True,
                )
                window = read_calibration_window(sensor)
                features.append(feature(window, baseline))
                labels.append(label)
                print("captured")

    save_model(make_model(np.stack(features), labels), profile_path)
    print(f"\nSaved {len(labels)} examples to {profile_path}")


def iter_prediction_results(
    sensor: Sensor, model: dict, baseline: np.ndarray
) -> Iterator[tuple[str, dict[str, float]]]:
    """Return hard labels and display proximity for a rolling live window."""
    window: deque[np.ndarray] = deque(
        (sensor.read() for _ in range(WINDOW_SAMPLES)),
        maxlen=WINDOW_SAMPLES,
    )
    while True:
        sample = feature(np.stack(window), baseline)
        yield predict(model, sample), proximity_scores(model, sample)
        for _ in range(PREDICTION_STRIDE_SAMPLES):
            window.append(sensor.read())


def iter_predictions(
    sensor: Sensor, model: dict, baseline: np.ndarray
) -> Iterator[str]:
    """Classify a rolling one-second window every 0.1 seconds."""
    for label, _scores in iter_prediction_results(sensor, model, baseline):
        yield label


class OnsetTracker:
    """Fire each gesture once until a rest window rearms all gestures."""

    def __init__(self) -> None:
        self.previous = "rest"
        self.fired_since_rest: set[str] = set()

    def update(self, current: str) -> str | None:
        onset = None
        if current == "rest":
            self.fired_since_rest.clear()
        elif current != self.previous and current not in self.fired_since_rest:
            onset = current
            self.fired_since_rest.add(current)
        self.previous = current
        return onset


def live(port: str, profile_path: Path) -> None:
    model = load_model(profile_path)
    with Sensor(port) as sensor:
        baseline = collect_baseline(sensor)
        print("Live detection started. Press Ctrl+C to stop.")
        tracker = OnsetTracker()
        for label in iter_predictions(sensor, model, baseline):
            onset = tracker.update(label)
            print(f"{label:6s}" + (f"  ONSET {onset}" if onset else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal four-sensor KNN demo")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "capture examples and save a model"),
        ("live", "classify a rolling one-second window at 10 Hz"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
        command.add_argument("--profile", type=Path, default=PROFILE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            calibrate(args.port, args.profile)
        else:
            live(args.port, args.profile)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
