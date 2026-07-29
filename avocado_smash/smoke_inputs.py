"""Smoke checks for Avocado Smash's three gesture inputs."""

from __future__ import annotations

import pygame

import main


def settle(sensor: main.EfleshKeyboardMirror, frames: int = 24) -> None:
    for _ in range(frames):
        sensor.update(1 / 60)


def main_smoke() -> None:
    assert main.RAW_VALUES == 3
    assert all(len(spec.raw_profile) == main.RAW_VALUES for spec in main.ACTION_SPECS)
    assert tuple(spec.label for spec in main.ACTION_SPECS) == (
        "Wrist Up",
        "Spread",
        "Fist",
    )
    assert tuple(spec.key for spec in main.ACTION_SPECS) == (
        pygame.K_f,
        pygame.K_j,
        pygame.K_k,
    )

    sensor = main.EfleshKeyboardMirror()
    assert sensor.key_down(pygame.K_LEFT) is None
    assert sensor.active_actions == set()

    sensor = main.EfleshKeyboardMirror()
    sensor.set_live_output({"wrist_up": 0.9, "spread": 0.1, "fist": 0.0})
    settle(sensor, 3)
    assert sensor.intent_values[0] > sensor.intent_values[1]

    sensor = main.EfleshKeyboardMirror()
    for action, spec in enumerate(main.ACTION_SPECS):
        returned = sensor.key_down(spec.key)
        settle(sensor, 6)
        assert returned == action
        assert sensor.intent_values[action] > 0.8
        sensor.key_up(spec.key)
        settle(sensor, 24)

    target = main.Target(action=0, y=main.HIT_Y, speed=100.0, rotation=0.0, spin=0.0)
    assert not hasattr(target, "quadrant")

    game = object.__new__(main.AvocadoSmash)
    game.targets = [
        target,
        main.Target(action=1, y=main.HIT_Y, speed=100.0, rotation=0.0, spin=0.0),
    ]
    game.particles = []
    game.score = 0
    game.combo = 0
    game.try_hit(1)
    assert game.targets == [target]
    assert game.score == 160
    assert game.combo == 1

    print("single-lane gesture smoke ok")


if __name__ == "__main__":
    main_smoke()
