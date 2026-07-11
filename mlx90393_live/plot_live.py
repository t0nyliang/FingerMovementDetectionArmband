#!/usr/bin/env python3
"""Live plot of MLX90393 Bx/By/Bz streamed over Serial from the ESP32.

Expects lines of the form:  DATA,<x>,<y>,<z>
(as emitted by mlx90393_live.ino). Any other line is ignored, so the
sketch's startup/status messages don't break parsing.

Usage:
    python plot_live.py                  # auto-detect port
    python plot_live.py --port COM5
    python plot_live.py --port COM5 --baud 115200 --window 200
"""
import argparse
import collections
import sys

import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    args = parser.parse_args()

    port = args.port or find_port()
    print(f"Connecting to {port} @ {args.baud} baud...")
    ser = serial.Serial(port, args.baud, timeout=1)

    xs = collections.deque(maxlen=args.window)
    ys = collections.deque(maxlen=args.window)
    zs = collections.deque(maxlen=args.window)
    ts = collections.deque(maxlen=args.window)
    sample_count = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    line_x, = ax.plot([], [], label="Bx (uT)", color="tab:red")
    line_y, = ax.plot([], [], label="By (uT)", color="tab:green")
    line_z, = ax.plot([], [], label="Bz (uT)", color="tab:blue")
    ax.set_xlabel("sample")
    ax.set_ylabel("field (uT)")
    ax.set_title("MLX90393 Live Magnetometer Readout")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    def read_available():
        nonlocal sample_count
        while ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw.startswith("DATA,"):
                if raw:
                    print(raw)  # surface status/error lines from the sketch
                continue
            parts = raw.split(",")
            if len(parts) != 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            sample_count += 1
            ts.append(sample_count)
            xs.append(x)
            ys.append(y)
            zs.append(z)

    def update(_frame):
        read_available()
        if not ts:
            return line_x, line_y, line_z
        line_x.set_data(ts, xs)
        line_y.set_data(ts, ys)
        line_z.set_data(ts, zs)
        ax.set_xlim(max(0, ts[0]), ts[-1] + 1)
        all_vals = list(xs) + list(ys) + list(zs)
        pad = max(1.0, (max(all_vals) - min(all_vals)) * 0.1)
        ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        return line_x, line_y, line_z

    ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    try:
        plt.show()
    finally:
        ser.close()


if __name__ == "__main__":
    main()
