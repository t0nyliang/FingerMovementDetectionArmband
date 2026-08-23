#!/usr/bin/env python3
"""Live plot of four MLX90393 readings plus BNO085 motion data.

Expects FRAME packets containing four XYZ sensor readings and optional MOTION
packets containing BNO085 UART-RVC yaw, pitch, roll, and acceleration values.
Legacy single-sensor SAMPLE and DATA packets remain readable.

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
MOTION_CHANNEL_COUNT = 6
MOTION_FIELD_COUNT = 3 + MOTION_CHANNEL_COUNT
MOTION_LABELS = ("Yaw", "Pitch", "Roll", "Ax", "Ay", "Az")
MOTION_COLORS = (
    "tab:purple",
    "tab:orange",
    "tab:brown",
    "tab:cyan",
    "tab:pink",
    "tab:olive",
)


@dataclass(frozen=True)
class ParsedPacket:
    source: str
    sequence: int | None
    device_us: int | None
    readings: tuple[XYZReading, ...]


@dataclass(frozen=True)
class MotionPacket:
    source: str
    sequence: int
    device_us: int
    values: tuple[float, ...]


def parse_packet(raw: str) -> ParsedPacket | MotionPacket | None:
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

        if parts[0] == "MOTION" and len(parts) == MOTION_FIELD_COUNT:
            sequence = int(parts[1])
            device_us = int(parts[2])
            values = tuple(float(value) for value in parts[3:])
            if not all(math.isfinite(value) for value in values):
                return None
            return MotionPacket("MOTION", sequence, device_us, values)
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
    motion_values = [
        collections.deque(maxlen=args.window)
        for _channel in range(MOTION_CHANNEL_COUNT)
    ]
    ts = collections.deque(maxlen=args.window)
    motion_ts = collections.deque(maxlen=args.window)
    sample_count = 0
    motion_sample_count = 0
    last_sequence = None
    last_motion_sequence = None
    legacy_warning_shown = False

    if args.sensor is None:
        displayed_sensors = tuple(range(SENSOR_COUNT))
        fig = plt.figure(figsize=(12, 10))
        grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.9))
        plot_axes = tuple(
            fig.add_subplot(grid[row, column])
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        figure_title = "MLX90393 + BNO085 Live Sensor Readout"
        motion_orientation_axis = fig.add_subplot(grid[2, 0])
        motion_acceleration_axis = fig.add_subplot(grid[2, 1])
    else:
        displayed_sensors = (args.sensor,)
        fig = plt.figure(figsize=(10, 9))
        grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 0.8, 0.8))
        plot_axes = (fig.add_subplot(grid[0, 0]),)
        figure_title = f"MLX90393 + BNO085 Readout (sensor {args.sensor})"
        motion_orientation_axis = fig.add_subplot(grid[1, 0])
        motion_acceleration_axis = fig.add_subplot(grid[2, 0])

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

    motion_lines = []
    for channel, (label, color) in enumerate(zip(MOTION_LABELS, MOTION_COLORS)):
        axis = motion_orientation_axis if channel < 3 else motion_acceleration_axis
        units = "deg" if channel < 3 else "m/s²"
        line, = axis.plot([], [], label=f"{label} ({units})", color=color)
        motion_lines.append(line)

    motion_orientation_axis.set_xlabel("motion sample")
    motion_orientation_axis.set_ylabel("orientation (degrees)")
    motion_orientation_axis.set_title("BNO085 orientation")
    motion_orientation_axis.legend(loc="upper right")
    motion_orientation_axis.grid(True, alpha=0.3)

    motion_acceleration_axis.set_xlabel("motion sample")
    motion_acceleration_axis.set_ylabel("acceleration (m/s²)")
    motion_acceleration_axis.set_title("BNO085 acceleration")
    motion_acceleration_axis.legend(loc="upper right")
    motion_acceleration_axis.grid(True, alpha=0.3)

    fig.suptitle(figure_title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    all_lines = tuple(
        line
        for sensor_index in displayed_sensors
        for line in lines_by_sensor[sensor_index]
    ) + tuple(motion_lines)

    def read_available():
        nonlocal sample_count, motion_sample_count, last_sequence
        nonlocal last_motion_sequence, legacy_warning_shown
        while ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            packet = parse_packet(raw)
            if packet is None:
                if raw and not raw.startswith(
                    ("SAMPLE,", "DATA,", "FRAME,", "MOTION,")
                ):
                    print(raw)  # surface status/error lines from the sketch
                continue

            if isinstance(packet, MotionPacket):
                if (
                    last_motion_sequence is not None
                    and packet.sequence != (last_motion_sequence + 1) & UINT32_MASK
                ):
                    expected = (last_motion_sequence + 1) & UINT32_MASK
                    dropped = (packet.sequence - expected) & UINT32_MASK
                    print(
                        f"Motion sequence gap: expected {expected}, "
                        f"received {packet.sequence} ({dropped} dropped)"
                    )
                last_motion_sequence = packet.sequence
                motion_sample_count += 1
                motion_ts.append(motion_sample_count)
                for channel, value in enumerate(packet.values):
                    motion_values[channel].append(value)
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
        if not ts and not motion_ts:
            return all_lines

        if ts:
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

        if motion_ts:
            motion_x_min = max(0, motion_ts[0])
            motion_x_max = motion_ts[-1] + 1
            for channel, line in enumerate(motion_lines):
                line.set_data(motion_ts, motion_values[channel])

            motion_orientation_axis.set_xlim(motion_x_min, motion_x_max)
            motion_acceleration_axis.set_xlim(motion_x_min, motion_x_max)
            for axis, channel_values in (
                (motion_orientation_axis, motion_values[:3]),
                (motion_acceleration_axis, motion_values[3:]),
            ):
                finite_values = [
                    value
                    for values in channel_values
                    for value in values
                    if math.isfinite(value)
                ]
                if finite_values:
                    minimum = min(finite_values)
                    maximum = max(finite_values)
                    pad = max(1.0, (maximum - minimum) * 0.1)
                    axis.set_ylim(minimum - pad, maximum + pad)

        return all_lines

    ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    try:
        plt.show()
    finally:
        ser.close()


if __name__ == "__main__":
    main()
