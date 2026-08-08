"""CLI for calibration, recording, and live BNO085 motion detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

from .detector import MotionDetector
from .features import (
    MotionWindowResampler,
    compute_features,
    estimate_baseline,
    load_baseline,
    save_baseline,
)
from .serial_sensor import MotionSensor


DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "baseline.json"


def collect_baseline(sensor: MotionSensor, seconds: float) -> tuple[np.ndarray, int]:
    """Capture a still-pose baseline for the requested wall-clock duration."""
    if seconds <= 0:
        raise ValueError("baseline duration must be positive")
    sensor.discard_pending()
    started = time.monotonic()
    rows = []
    while time.monotonic() - started < seconds:
        rows.append(sensor.read_frame().vector)
    if len(rows) < 2:
        raise RuntimeError("not enough motion frames received for a baseline")
    return np.stack(rows), len(rows)


def calibrate(port: str, output: Path, seconds: float) -> None:
    print(f"Keep the wrist still for {seconds:g} seconds.", flush=True)
    with MotionSensor(port) as sensor:
        samples, count = collect_baseline(sensor, seconds)
    baseline = estimate_baseline(samples)
    save_baseline(baseline, output)
    print(f"Saved baseline from {count} frames to {output}")


def record(port: str, output: Path, seconds: float) -> None:
    """Write raw timestamped samples for threshold inspection."""
    if seconds <= 0:
        raise ValueError("record duration must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with MotionSensor(port) as sensor, output.open("w", newline="", encoding="utf-8") as handle:
        sensor.discard_pending()
        writer = csv.writer(handle)
        writer.writerow(["sequence", "device_us", "yaw", "pitch", "roll", "ax", "ay", "az"])
        started = time.monotonic()
        count = 0
        while time.monotonic() - started < seconds:
            frame = sensor.read_frame()
            writer.writerow([frame.sequence, frame.device_us, *frame.values])
            count += 1
    print(f"Saved {count} motion frames to {output}")


def detect(port: str, baseline_path: Path | None, baseline_seconds: float) -> None:
    with MotionSensor(port) as sensor:
        if baseline_path is None:
            print(
                f"Keep the wrist still for {baseline_seconds:g} seconds.",
                flush=True,
            )
            samples, _count = collect_baseline(sensor, baseline_seconds)
            baseline = estimate_baseline(samples)
        else:
            baseline = load_baseline(baseline_path)
            sensor.discard_pending()

        print("Live motion detection started. Press Ctrl+C to stop.")
        windows = MotionWindowResampler()
        detector = MotionDetector()
        while True:
            frame = sensor.read_frame()
            window, reset = windows.add(frame)
            if reset:
                detector.reset()
                print(f"RESET,{frame.sequence},{frame.device_us}", flush=True)
            if window is None:
                continue

            features = compute_features(window, baseline)
            decision = detector.update(frame, features)
            event = decision.event or ""
            print(
                "STATE,"
                f"{decision.sequence},{decision.device_us},{decision.state.value},"
                f"{decision.score:.3f},{features.acceleration_rms:.3f},"
                f"{features.orientation_speed_rms:.3f},{event}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone BNO085 motion pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate_command = commands.add_parser(
        "calibrate", help="capture and save a relaxed-pose baseline"
    )
    calibrate_command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
    calibrate_command.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    calibrate_command.add_argument("--seconds", type=float, default=2.0)

    record_command = commands.add_parser(
        "record", help="save raw timestamped frames to CSV"
    )
    record_command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
    record_command.add_argument("--output", type=Path, required=True)
    record_command.add_argument("--seconds", type=float, default=10.0)

    detect_command = commands.add_parser(
        "detect", help="detect dynamic motion using a baseline and hysteresis"
    )
    detect_command.add_argument("--port", required=True, help="ESP32 port, such as COM3")
    detect_command.add_argument("--baseline", type=Path)
    detect_command.add_argument("--baseline-seconds", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            calibrate(args.port, args.output, args.seconds)
        elif args.command == "record":
            record(args.port, args.output, args.seconds)
        else:
            detect(args.port, args.baseline, args.baseline_seconds)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
