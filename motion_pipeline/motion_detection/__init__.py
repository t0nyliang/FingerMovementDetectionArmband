"""Read, characterize, and detect wrist motion from a BNO085 stream."""

from .detector import MotionDetector
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
    "MotionDetector",
    "MotionFeatures",
    "MotionFrame",
    "MotionThresholds",
    "MotionWindowResampler",
    "compute_features",
    "estimate_baseline",
    "parse_motion_frame",
]
