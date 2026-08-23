from __future__ import annotations

import numpy as np

from motion_pipeline.motion_detection.detector import MotionDetector
from motion_pipeline.motion_detection.features import MotionFeatures, MotionThresholds


def features(acceleration_rms: float) -> MotionFeatures:
    return MotionFeatures(acceleration_rms, acceleration_rms, 0.0, 0.0, 0.0)


def test_detector_requires_two_motion_frames_and_four_rest_frames() -> None:
    detector = MotionDetector(
        thresholds=MotionThresholds(acceleration_rms=1.0, acceleration_peak=10.0),
        on_frames=2,
        off_frames=4,
    )
    _score, first_event = detector.update(features(1.1))
    _score, second_event = detector.update(features(1.1))
    assert detector.state == "motion"
    assert first_event is None
    assert second_event == "motion_onset"

    for sequence in range(2, 5):
        _score, event = detector.update(features(0.1))
        assert detector.state == "motion"
        assert event is None
    _score, event = detector.update(features(0.1))
    assert detector.state == "rest"
    assert event == "motion_offset"


def test_detector_reset_rearms_motion_onset() -> None:
    detector = MotionDetector(on_frames=1)
    assert detector.update(features(2.0))[1] == "motion_onset"
    detector.reset()
    assert detector.update(features(2.0))[1] == "motion_onset"
