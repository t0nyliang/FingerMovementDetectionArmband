"""Read four MLX90393 sensors from the ESP32 FRAME protocol."""

from __future__ import annotations

import math
import time

import numpy as np


BAUD = 115200
CHANNEL_COUNT = 12


def parse_frame(line: str) -> np.ndarray | None:
    """Return twelve XYZ values from one FRAME line."""
    line = line.strip()
    if line.startswith("ERROR,"):
        raise RuntimeError(line)

    fields = line.split(",")
    if fields[0] != "FRAME":
        return None
    if len(fields) != 3 + CHANNEL_COUNT:
        raise RuntimeError("malformed FRAME; expected 12 sensor values")

    try:
        sequence = int(fields[1])
        timestamp_us = int(fields[2])
        values = np.asarray([float(value) for value in fields[3:]], dtype=float)
    except ValueError:
        raise RuntimeError("FRAME contains a non-numeric value") from None

    if sequence < 0 or timestamp_us < 0:
        raise RuntimeError("FRAME sequence and timestamp must be non-negative")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("all four sensors must provide finite XYZ values")
    return values


class Sensor:
    """Small context manager around a pyserial connection."""

    def __init__(self, port: str) -> None:
        self.port = port
        self.serial = None

    def __enter__(self) -> "Sensor":
        import serial

        self.serial = serial.Serial(self.port, BAUD, timeout=0.25)
        self.serial.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.serial is not None:
            self.serial.close()

    def read(self, timeout_s: float = 5.0) -> np.ndarray:
        if self.serial is None:
            raise RuntimeError("serial port is not open")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            values = parse_frame(raw.decode("utf-8", errors="ignore"))
            if values is not None:
                return values
        raise TimeoutError(
            "no FRAME received; upload mlx90393_live.ino and close Serial Monitor"
        )
