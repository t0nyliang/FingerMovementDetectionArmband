#!/usr/bin/env python3
"""Live plot of four MLX90393 Bx/By/Bz readings streamed from the ESP32.

Expects FRAME packets containing four XYZ sensor readings from
mlx90393_live.ino. Legacy single-sensor SAMPLE and DATA packets remain readable.

Usage:
    python plot_live.py                  # auto-detect port
    python plot_live.py --port COM5
    python plot_live.py --port COM5 --baud 115200 --window 200
    python plot_live.py --port COM5 --sensor 2
"""
import argparse
import collections
from dataclasses import dataclass
import math
import sys

import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation


SENSOR_COUNT = 4
MUX_CHANNELS = (0, 2, 5, 7)
AXIS_COUNT = 3
FRAME_FIELD_COUNT = 3 + SENSOR_COUNT * AXIS_COUNT
UINT32_MASK = 0xFFFFFFFF
AXIS_LABELS = ("Bx", "By", "Bz")
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
XYZReading = tuple[float, float, float]


@dataclass(frozen=True)
class ParsedPacket:
    source: str
    sequence: int | None
    device_us: int | None
    readings: tuple[XYZReading, ...]


def parse_packet(raw: str) -> ParsedPacket | None:
    """Parse a serial packet and expand legacy readings to four sensor slots."""
    parts = raw.strip().split(",")
    if not parts:
        return None

    try:
        if parts[0] == "FRAME" and len(parts) == FRAME_FIELD_COUNT:
            sequence = int(parts[1])
            device_us = int(parts[2])
            values = tuple(float(value) for value in parts[3:])
            readings = tuple(
                (
                    values[sensor_index * AXIS_COUNT],
                    values[sensor_index * AXIS_COUNT + 1],
                    values[sensor_index * AXIS_COUNT + 2],
                )
                for sensor_index in range(SENSOR_COUNT)
            )
            return ParsedPacket("FRAME", sequence, device_us, readings)

        if parts[0] == "SAMPLE" and len(parts) == 6:
            sequence = int(parts[1])
            device_us = int(parts[2])
            sensor_zero = (float(parts[3]), float(parts[4]), float(parts[5]))
            unavailable = (math.nan, math.nan, math.nan)
            return ParsedPacket(
                "SAMPLE",
                sequence,
                device_us,
                (sensor_zero,) + (unavailable,) * (SENSOR_COUNT - 1),
            )

        if parts[0] == "DATA" and len(parts) == 4:
            sensor_zero = (float(parts[1]), float(parts[2]), float(parts[3]))
            unavailable = (math.nan, math.nan, math.nan)
            return ParsedPacket(
                "DATA",
                None,
                None,
                (sensor_zero,) + (unavailable,) * (SENSOR_COUNT - 1),
            )
    except ValueError:
        return None

    return None


def find_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        sys.exit("No serial ports found. Plug in the ESP32 or pass --port explicitly.")
    if len(ports) == 1:
        return ports[0].device
    print("Multiple serial ports found:")
    for p in ports:
        print(f"  {p.device}  ({p.description})")
    sys.exit("Pass --port <PORT> to pick one.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial port (e.g. COM5). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--window", type=int, default=200, help="Number of samples to show")
    parser.add_argument(
        "--sensor",
        type=int,
        choices=range(SENSOR_COUNT),
        default=None,
        help="Show only this sensor index (0-3); default shows all sensors",
    )
    args = parser.parse_args()

    port = args.port or find_port()
    print(f"Connecting to {port} @ {args.baud} baud...")
    ser = serial.Serial(port, args.baud, timeout=1)

    sensor_values = [
        [collections.deque(maxlen=args.window) for _axis in range(AXIS_COUNT)]
        for _sensor in range(SENSOR_COUNT)
    ]
    ts = collections.deque(maxlen=args.window)
    sample_count = 0
    last_sequence = None
    legacy_warning_shown = False

    if args.sensor is None:
        displayed_sensors = tuple(range(SENSOR_COUNT))
        fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        plot_axes = tuple(axes_grid.flat)
        figure_title = "MLX90393 Live Magnetometer Readout (four sensors)"
    else:
        displayed_sensors = (args.sensor,)
        fig, focused_axis = plt.subplots(figsize=(10, 5))
        plot_axes = (focused_axis,)
        figure_title = f"MLX90393 Live Magnetometer Readout (sensor {args.sensor})"

    lines_by_sensor = {}
    axes_by_sensor = {}
    for sensor_index, ax in zip(displayed_sensors, plot_axes):
        sensor_lines = []
        for label, color in zip(AXIS_LABELS, AXIS_COLORS):
            line, = ax.plot([], [], label=f"{label} (uT)", color=color)
            sensor_lines.append(line)
        lines_by_sensor[sensor_index] = tuple(sensor_lines)
        axes_by_sensor[sensor_index] = ax
        ax.set_xlabel("sample")
        ax.set_ylabel("field (uT)")
        ax.set_title(
            f"Sensor {sensor_index} / mux channel {MUX_CHANNELS[sensor_index]}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(figure_title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    all_lines = tuple(
        line
        for sensor_index in displayed_sensors
        for line in lines_by_sensor[sensor_index]
    )

    def read_available():
        nonlocal sample_count, last_sequence, legacy_warning_shown
        while ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            packet = parse_packet(raw)
            if packet is None:
                if raw and not raw.startswith(("SAMPLE,", "DATA,", "FRAME,")):
                    print(raw)  # surface status/error lines from the sketch
                continue

            if packet.source != "FRAME" and not legacy_warning_shown:
                print(
                    "WARNING: single-sensor packet received; only Sensor 0 has "
                    "data. Upload the four-sensor FRAME firmware for all panels."
                )
                legacy_warning_shown = True

            if packet.sequence is not None and last_sequence is not None:
                expected = (last_sequence + 1) & UINT32_MASK
                if packet.sequence != expected:
                    dropped = (packet.sequence - expected) & UINT32_MASK
                    print(
                        f"Sequence gap: expected {expected}, "
                        f"received {packet.sequence} ({dropped} dropped)"
                    )
            if packet.sequence is not None:
                last_sequence = packet.sequence

            sample_count += 1
            ts.append(sample_count)
            for sensor_index, reading in enumerate(packet.readings):
                for axis, value in enumerate(reading):
                    sensor_values[sensor_index][axis].append(value)

    def update(_frame):
        read_available()
        if not ts:
            return all_lines

        x_min = max(0, ts[0])
        x_max = ts[-1] + 1
        for sensor_index in displayed_sensors:
            ax = axes_by_sensor[sensor_index]
            sensor_lines = lines_by_sensor[sensor_index]
            for axis, line in enumerate(sensor_lines):
                line.set_data(ts, sensor_values[sensor_index][axis])

            ax.set_xlim(x_min, x_max)
            finite_values = [
                value
                for axis_values in sensor_values[sensor_index]
                for value in axis_values
                if math.isfinite(value)
            ]
            if finite_values:
                minimum = min(finite_values)
                maximum = max(finite_values)
                pad = max(1.0, (maximum - minimum) * 0.1)
                ax.set_ylim(minimum - pad, maximum + pad)

        return all_lines

    ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    try:
        plt.show()
    finally:
        ser.close()


if __name__ == "__main__":
    main()
