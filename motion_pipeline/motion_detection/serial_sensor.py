"""Serial transport for the standalone BNO085 motion pipeline."""

from __future__ import annotations

import time

from .protocol import MotionFrame, parse_motion_frame


BAUD = 115200


class MotionSensor:
    """Small context manager around the ESP32's MOTION_v1 serial stream."""

    def __init__(self, port: str, baud: int = BAUD) -> None:
        self.port = port
        self.baud = baud
        self.serial = None

    def __enter__(self) -> "MotionSensor":
        import serial

        self.serial = serial.Serial(self.port, self.baud, timeout=0.25)
        self.serial.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.serial is not None:
            self.serial.close()

    def discard_pending(self) -> None:
        if self.serial is None:
            raise RuntimeError("motion serial port is not open")
        self.serial.reset_input_buffer()

    def read_frame(self, timeout_s: float = 2.0) -> MotionFrame:
        if self.serial is None:
            raise RuntimeError("motion serial port is not open")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            frame = parse_motion_frame(raw.decode("utf-8", errors="ignore"))
            if frame is not None:
                return frame
        raise TimeoutError(
            "no MOTION frame received; upload motion_pipeline/firmware/bno085_uart_rvc.ino"
        )
