"""Read, characterize, and detect wrist motion from a BNO085 stream."""

from .detector import MotionDecision, MotionDetector, MotionState
from .features import (
    MotionBaseline,
    MotionFeatures,
    MotionThresholds,
    MotionWindowResampler,
    compute_features,
    estimate_baseline,
)
from .protocol import MotionFrame, parse_motion_frame

__all__ = [
    "MotionBaseline",
    "MotionDecision",
    "MotionDetector",
    "MotionFeatures",
    "MotionFrame",
    "MotionState",
    "MotionThresholds",
    "MotionWindowResampler",
    "compute_features",
    "estimate_baseline",
    "parse_motion_frame",
]
