from __future__ import annotations

import numpy as np
import pytest

from motion_pipeline.motion_detection.protocol import (
    CHANNEL_COUNT,
    MotionFrame,
    parse_motion_frame,
)


def motion_line(values: list[float] | None = None) -> str:
    if values is None:
        values = [10.0, 20.0, 30.0, 0.1, 0.2, 9.8]
    return "MOTION,7,140000," + ",".join(str(value) for value in values)


def test_parse_motion_frame() -> None:
    frame = parse_motion_frame(motion_line())
    assert frame is not None
    assert frame.sequence == 7
    assert frame.device_us == 140000
    np.testing.assert_allclose(frame.vector, [10.0, 20.0, 30.0, 0.1, 0.2, 9.8])
    assert parse_motion_frame("READY,protocol=MOTION_v1") is None


def test_parse_motion_frame_rejects_bad_rows() -> None:
    with pytest.raises(RuntimeError, match=f"expected {CHANNEL_COUNT}"):
        parse_motion_frame("MOTION,1,2,1,2")
    with pytest.raises(RuntimeError, match="non-numeric"):
        parse_motion_frame(motion_line().replace(",10.0,", ",bad,"))
    with pytest.raises(RuntimeError, match="finite"):
        parse_motion_frame(motion_line([0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]))
    with pytest.raises(RuntimeError, match="ERROR"):
        parse_motion_frame("ERROR,bno085=not_found")


def test_motion_frame_validates_constructor_values() -> None:
    with pytest.raises(ValueError, match="6 values"):
        MotionFrame(0, 0, (1.0,))
