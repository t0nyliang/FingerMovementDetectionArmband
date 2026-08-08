from __future__ import annotations

import numpy as np

from motion_pipeline.motion_detection.detector import MotionDetector, MotionState
from motion_pipeline.motion_detection.features import MotionFeatures, MotionThresholds
from motion_pipeline.motion_detection.protocol import MotionFrame


def frame(sequence: int) -> MotionFrame:
    return MotionFrame(sequence, sequence * 20_000, (0.0, 0.0, 0.0, 0.0, 0.0, 9.8))


def features(acceleration_rms: float) -> MotionFeatures:
    return MotionFeatures(acceleration_rms, acceleration_rms, 0.0, 0.0, 0.0)


def test_detector_requires_two_motion_frames_and_four_rest_frames() -> None:
    detector = MotionDetector(
        thresholds=MotionThresholds(acceleration_rms=1.0, acceleration_peak=10.0),
        on_frames=2,
        off_frames=4,
    )
    first = detector.update(frame(0), features(1.1))
    second = detector.update(frame(1), features(1.1))
    assert first.state is MotionState.REST
    assert first.event is None
    assert second.state is MotionState.MOTION
    assert second.event == "motion_onset"

    for sequence in range(2, 5):
        decision = detector.update(frame(sequence), features(0.1))
        assert decision.state is MotionState.MOTION
        assert decision.event is None
    final = detector.update(frame(5), features(0.1))
    assert final.state is MotionState.REST
    assert final.event == "motion_offset"


def test_detector_reset_rearms_motion_onset() -> None:
    detector = MotionDetector(on_frames=1)
    assert detector.update(frame(0), features(2.0)).event == "motion_onset"
    detector.reset()
    decision = detector.update(frame(1), features(2.0))
    assert decision.event == "motion_onset"


def test_detector_rejects_invalid_configuration() -> None:
    try:
        MotionDetector(on_frames=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid on_frames to be rejected")
