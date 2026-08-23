"""Stateful hysteresis detector for dynamic wrist motion."""

from __future__ import annotations

from .features import MotionFeatures, MotionThresholds


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
            raise ValueError("detector counts and off_score must be positive")
        self.thresholds = thresholds or MotionThresholds()
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.off_score = off_score
        self.reset()

    def reset(self) -> None:
        self.state = "rest"
        self._motion_count = 0
        self._rest_count = 0

    def update(self, features: MotionFeatures) -> tuple[float, str | None]:
        score = self.thresholds.score(features)
        event: str | None = None

        if self.state == "rest":
            self._rest_count = 0
            if score >= 1.0:
                self._motion_count += 1
            else:
                self._motion_count = 0
            if self._motion_count >= self.on_frames:
                self.state = "motion"
                self._motion_count = 0
                event = "motion_onset"
        else:
            self._motion_count = 0
            if score < self.off_score:
                self._rest_count += 1
            else:
                self._rest_count = 0
            if self._rest_count >= self.off_frames:
                self.state = "rest"
                self._rest_count = 0
                event = "motion_offset"

        return score, event
