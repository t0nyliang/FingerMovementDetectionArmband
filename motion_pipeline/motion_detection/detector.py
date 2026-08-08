"""Stateful hysteresis detector for dynamic wrist motion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .features import MotionFeatures, MotionThresholds
from .protocol import MotionFrame


class MotionState(str, Enum):
    REST = "rest"
    MOTION = "motion"


@dataclass(frozen=True)
class MotionDecision:
    sequence: int
    device_us: int
    state: MotionState
    score: float
    event: str | None
    features: MotionFeatures


class MotionDetector:
    """Use separate on/off confirmation counts to avoid chatter."""

    def __init__(
        self,
        thresholds: MotionThresholds | None = None,
        on_frames: int = 2,
        off_frames: int = 4,
        off_score: float = 0.65,
    ) -> None:
        if on_frames < 1 or off_frames < 1 or off_score <= 0:
            raise ValueError("detector counts must be positive and off_score must be positive")
        self.thresholds = thresholds or MotionThresholds()
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.off_score = off_score
        self.reset()

    def reset(self) -> None:
        self.state = MotionState.REST
        self._motion_count = 0
        self._rest_count = 0

    def update(self, frame: MotionFrame, features: MotionFeatures) -> MotionDecision:
        score = self.thresholds.score(features)
        event: str | None = None

        if self.state is MotionState.REST:
            self._rest_count = 0
            if score >= 1.0:
                self._motion_count += 1
            else:
                self._motion_count = 0
            if self._motion_count >= self.on_frames:
                self.state = MotionState.MOTION
                self._motion_count = 0
                event = "motion_onset"
        else:
            self._motion_count = 0
            if score < self.off_score:
                self._rest_count += 1
            else:
                self._rest_count = 0
            if self._rest_count >= self.off_frames:
                self.state = MotionState.REST
                self._rest_count = 0
                event = "motion_offset"

        return MotionDecision(
            sequence=frame.sequence,
            device_us=frame.device_us,
            state=self.state,
            score=score,
            event=event,
            features=features,
        )
