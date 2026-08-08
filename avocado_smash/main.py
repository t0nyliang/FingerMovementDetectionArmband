"""Avocado Smash: keyboard-first eFlesh demo game.

The live classifier consumes four three-axis magnetometers and exposes three
gesture scores: Wrist Up, Spread, and Fist. The keyboard mirror remains available
without sensor hardware.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from dataclasses import dataclass

from live_sensor import LiveGestureClient

try:
    import pygame
except ImportError:
    print("Pygame is required. Install it with: python -m pip install -r requirements.txt")
    raise


WIDTH = 1280
HEIGHT = 720
FPS = 60

RAW_MAGNETOMETERS = 1
RAW_CHANNELS = 3
RAW_VALUES = RAW_MAGNETOMETERS * RAW_CHANNELS
RAW_CHANNEL_LABELS = ("bx", "by", "bz")

BOARD_LEFT = 64
BOARD_TOP = 88
BOARD_WIDTH = 820
BOARD_HEIGHT = 476
HIT_Y = BOARD_TOP + 378
HIT_WINDOW = 58
TARGET_SIZE = 78

PANEL_LEFT = 910
PANEL_TOP = 88
PANEL_WIDTH = 314
PANEL_HEIGHT = 476

PALM_GREEN = (74, 115, 55)
AVOCADO_GREEN = (107, 140, 33)
AVOCADO_CREAM = (221, 212, 143)
SHELL_TAN = (205, 169, 137)
PIT_BROWN = (112, 64, 18)

BG = PIT_BROWN
COUNTER = AVOCADO_CREAM
COUNTER_SHADOW = PALM_GREEN
INK = PIT_BROWN
MUTED = PALM_GREEN
WHITE = AVOCADO_CREAM
CHIP_TEXT = AVOCADO_CREAM
AVO_DARK = PALM_GREEN
AVO_MID = AVOCADO_GREEN
AVO_FLESH = AVOCADO_CREAM
PIT = PIT_BROWN
RED = PIT_BROWN
GOLD = SHELL_TAN


@dataclass(frozen=True)
class ActionSpec:
    label: str
    key: int
    key_name: str
    symbol: str
    color: tuple[int, int, int]
    raw_profile: tuple[float, ...]


ACTION_SPECS = (
    ActionSpec(
        label="Wrist Up",
        key=pygame.K_f,
        key_name="F",
        symbol="*",
        color=PALM_GREEN,
        raw_profile=(
            0.10,
            0.20,
            0.90,
        ),
    ),
    ActionSpec(
        label="Spread",
        key=pygame.K_j,
        key_name="J",
        symbol="<>",
        color=AVOCADO_GREEN,
        raw_profile=(
            0.90,
            0.35,
            0.15,
        ),
    ),
    ActionSpec(
        label="Fist",
        key=pygame.K_k,
        key_name="K",
        symbol="[]",
        color=PIT_BROWN,
        raw_profile=(
            0.45,
            0.70,
            0.60,
        ),
    ),
)

ACTION_COUNT = len(ACTION_SPECS)
TRACK_WIDTH = 260
TRACK_CENTER_X = BOARD_LEFT + BOARD_WIDTH // 2


@dataclass
class Target:
    action: int
    y: float
    speed: float
    rotation: float
    spin: float


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]
    radius: float


class EfleshKeyboardMirror:
    """Keyboard-backed mirror that derives three intent values from raw eFlesh data."""

    def __init__(self) -> None:
        self.raw_values = [0.0 for _ in range(RAW_VALUES)]
        self.intent_values = [0.0 for _ in range(ACTION_COUNT)]
        self.active_actions: set[int] = set()
        self.key_to_action = {spec.key: action for action, spec in enumerate(ACTION_SPECS)}
        self.live_enabled = False
        self.live_scores = [0.0 for _ in range(ACTION_COUNT)]

    def set_live_output(self, scores: dict[str, float]) -> None:
        self.live_enabled = True
        for action, spec in enumerate(ACTION_SPECS):
            gesture = spec.label.lower().replace(" ", "_")
            value = float(scores.get(gesture, 0.0))
            self.live_scores[action] = max(0.0, min(1.0, value))

    def key_down(self, key: int) -> int | None:
        action = self.key_to_action.get(key)
        if action is None:
            return None
        self.active_actions.add(action)
        self.intent_values[action] = 1.0
        return action

    def key_up(self, key: int) -> None:
        action = self.key_to_action.get(key)
        if action is not None:
            self.active_actions.discard(action)

    def update(self, dt: float) -> None:
        raw_targets = [0.0 for _ in range(RAW_VALUES)]
        for action in self.active_actions:
            for index, value in enumerate(ACTION_SPECS[action].raw_profile):
                raw_targets[index] = max(raw_targets[index], value)

        for index, value in enumerate(self.raw_values):
            target = raw_targets[index]
            rate = 18.0 if target > value else 7.0
            self.raw_values[index] += (target - value) * min(1.0, dt * rate)

        for action, spec in enumerate(ACTION_SPECS):
            numerator = sum(raw * weight for raw, weight in zip(self.raw_values, spec.raw_profile))
            denominator = sum(weight * weight for weight in spec.raw_profile)
            target = min(1.0, numerator / denominator) if denominator else 0.0
            if self.active_actions:
                if action in self.active_actions:
                    target = max(target, 0.88)
                else:
                    target = min(0.35, target * 0.35)
            if self.live_enabled:
                target = max(target, self.live_scores[action])
            rate = 16.0 if target > self.intent_values[action] else 8.0
            self.intent_values[action] += (target - self.intent_values[action]) * min(1.0, dt * rate)

    def magnetometer_energy(self, magnetometer: int) -> float:
        start = magnetometer * RAW_CHANNELS
        values = self.raw_values[start : start + RAW_CHANNELS]
        return min(1.0, math.sqrt(sum(value * value for value in values) / RAW_CHANNELS))


class AvocadoSmash:
    def __init__(
        self,
        speed_multiplier: float = 1.0,
        sensor_port: str | None = None,
        profile_path: Path | None = None,
    ) -> None:
        self.speed_multiplier = speed_multiplier
        self.live_client = (
            LiveGestureClient(sensor_port, profile_path)
            if sensor_port is not None and profile_path is not None
            else None
        )
        if self.live_client is not None:
            self.live_client.start()

        pygame.init()
        pygame.display.set_caption("Avocado Smash - eFlesh Keyboard Demo")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
        self.clock = pygame.time.Clock()
        pygame.key.set_repeat(0)

        self.font_xl = pygame.font.SysFont("arialblack", 42)
        self.font_lg = pygame.font.SysFont("arialblack", 26)
        self.font_md = pygame.font.SysFont("arial", 22, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 16)
        self.font_tiny = pygame.font.SysFont("consolas", 13)

        self.sensor = EfleshKeyboardMirror()
        self.avocado_sprites = [self._make_avocado_sprite(action) for action in range(ACTION_COUNT)]

        self.state = "menu"
        self.running = True
        self.reset_game()

    def reset_game(self) -> None:
        self.targets: list[Target] = []
        self.particles: list[Particle] = []
        self.score = 0
        self.combo = 0
        self.elapsed = 0.0
        self.spawn_timer = self.scaled_interval(0.35)
        self.spawn_interval = self.scaled_interval(1.05)
        self.message = "READY"
        self.message_timer = 0.0

    def run(self) -> int:
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                self.handle_events()
                self.update_live_sensor()
                self.sensor.update(dt)
                if self.state == "playing":
                    self.update_game(dt)
                elif self.state == "menu":
                    self.update_particles(dt)
                self.draw()
            return 0
        finally:
            if self.live_client is not None:
                self.live_client.stop()
            pygame.quit()

    def update_live_sensor(self) -> None:
        if self.live_client is None:
            return
        action_by_label = {
            spec.label.lower().replace(" ", "_"): action
            for action, spec in enumerate(ACTION_SPECS)
        }
        for label, is_onset, scores in self.live_client.poll():
            self.sensor.set_live_output(scores)
            if is_onset and self.state == "playing":
                action = action_by_label.get(label)
                if action is not None:
                    self.try_hit(action)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key_down(event.key)
            elif event.type == pygame.KEYUP:
                self.sensor.key_up(event.key)

    def handle_key_down(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
            return

        if key in (pygame.K_SPACE, pygame.K_RETURN):
            if self.state in {"menu", "game_over"}:
                self.reset_game()
                self.state = "playing"
            elif self.state == "paused":
                self.state = "playing"
            return

        if key == pygame.K_p:
            if self.state == "playing":
                self.state = "paused"
            elif self.state == "paused":
                self.state = "playing"
            return

        action = self.sensor.key_down(key)
        if action is not None and self.state == "playing":
            self.try_hit(action)

    def update_game(self, dt: float) -> None:
        self.elapsed += dt
        self.message_timer = max(0.0, self.message_timer - dt)

        self.spawn_interval = self.scaled_interval(max(0.52, 1.05 - self.elapsed * 0.007))
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            self.spawn_target()
            self.spawn_timer += self.spawn_interval * random.uniform(0.82, 1.18)

        for target in self.targets:
            target.y += target.speed * dt
            target.rotation += target.spin * dt

        missed = [
            target
            for target in self.targets
            if target.y > HIT_Y + HIT_WINDOW + TARGET_SIZE * 0.5
        ]
        if missed:
            for target in missed:
                self.create_particles(TRACK_CENTER_X, target.y, target.action, gentle=True)
            self.targets = [target for target in self.targets if target not in missed]
            self.combo = 0
            self.flash_message("MISSED")

        self.update_particles(dt)

    def update_particles(self, dt: float) -> None:
        next_particles = []
        for particle in self.particles:
            particle.life -= dt
            particle.vy += 340.0 * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            if particle.life > 0:
                next_particles.append(particle)
        self.particles = next_particles

    def spawn_target(self) -> None:
        action = random.randrange(ACTION_COUNT)
        speed = (132.0 + min(102.0, self.elapsed * 2.8) + random.uniform(-10.0, 16.0)) * self.speed_multiplier
        spin = random.uniform(-76.0, 76.0)
        self.targets.append(
            Target(action, BOARD_TOP - TARGET_SIZE - 12, speed, random.uniform(-14.0, 14.0), spin)
        )

    def try_hit(self, action: int) -> None:
        candidates = [
            target
            for target in self.targets
            if target.action == action and abs(target.y - HIT_Y) <= HIT_WINDOW
        ]

        if not candidates:
            self.combo = 0
            self.score = max(0, self.score - 8)
            self.flash_message("AIR")
            self.create_particles(TRACK_CENTER_X, HIT_Y, action, gentle=True)
            return

        target = min(candidates, key=lambda item: abs(item.y - HIT_Y))
        accuracy = 1.0 - abs(target.y - HIT_Y) / HIT_WINDOW
        gain = 100 + int(accuracy * 60) + self.combo * 12
        self.score += gain
        self.combo += 1
        self.targets.remove(target)
        self.flash_message(f"+{gain}")
        self.create_particles(TRACK_CENTER_X, HIT_Y, action)

    def flash_message(self, message: str) -> None:
        self.message = message
        self.message_timer = 0.75

    def scaled_interval(self, seconds: float) -> float:
        return max(0.16, seconds / self.speed_multiplier)

    def track_rect(self) -> pygame.Rect:
        return pygame.Rect(
            TRACK_CENTER_X - TRACK_WIDTH // 2,
            BOARD_TOP + 10,
            TRACK_WIDTH,
            BOARD_HEIGHT - 20,
        )

    def create_particles(self, x: float, y: float, action: int, gentle: bool = False) -> None:
        count = 8 if gentle else 22
        spread = 110.0 if gentle else 240.0
        for _ in range(count):
            angle = random.uniform(-math.pi, 0)
            speed = random.uniform(40.0, spread)
            color = random.choice((AVO_FLESH, AVO_MID, ACTION_SPECS[action].color, PIT))
            self.particles.append(
                Particle(
                    x=x + random.uniform(-18.0, 18.0),
                    y=y + random.uniform(-12.0, 12.0),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=random.uniform(0.38, 0.82),
                    color=color,
                    radius=random.uniform(2.0, 6.5),
                )
            )

    def draw(self) -> None:
        self.screen.fill(BG)
        self.draw_header()
        self.draw_board()
        self.draw_sensor_panel()
        self.draw_particles()
        if self.state in {"menu", "paused", "game_over"}:
            self.draw_overlay()
        pygame.display.flip()

    def draw_header(self) -> None:
        title = self.font_xl.render("AVOCADO SMASH", True, WHITE)
        self.screen.blit(title, (64, 24))

        stats = f"SCORE {self.score:05d}    COMBO {self.combo:02d}    LIVES ∞"
        if self.speed_multiplier != 1.0:
            stats += f"    SPEED {self.speed_multiplier:.2f}x"
        stats_surf = self.font_md.render(stats, True, COUNTER)
        self.screen.blit(stats_surf, (580, 36))

        if self.message_timer > 0.0:
            alpha = int(255 * min(1.0, self.message_timer / 0.35))
            msg = self.font_lg.render(self.message, True, GOLD)
            msg.set_alpha(alpha)
            self.screen.blit(msg, (1088 - msg.get_width() / 2, 35))

    def draw_board(self) -> None:
        board_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, COUNTER_SHADOW, board_rect.move(0, 8), border_radius=8)
        pygame.draw.rect(self.screen, COUNTER, board_rect, border_radius=8)

        track = self.track_rect()
        pygame.draw.rect(self.screen, SHELL_TAN, track, border_radius=10)
        pygame.draw.line(
            self.screen,
            AVOCADO_GREEN,
            (TRACK_CENTER_X, track.top + 18),
            (TRACK_CENTER_X, track.bottom - 18),
            width=5,
        )

        hint = self.font_sm.render("ONE LANE  •  F / J / K", True, INK)
        self.screen.blit(hint, hint.get_rect(center=(TRACK_CENTER_X, track.top + 28)))

        pygame.draw.line(
            self.screen,
            RED,
            (track.left + 18, HIT_Y),
            (track.right - 18, HIT_Y),
            width=4,
        )
        self.draw_hit_outline()

        for target in self.targets:
            self.draw_target(target)

    def draw_hit_outline(self) -> None:
        center = (TRACK_CENTER_X, HIT_Y)
        outline = pygame.Rect(0, 0, TARGET_SIZE + 14, TARGET_SIZE + 8)
        outline.center = center
        color = PIT_BROWN
        pygame.draw.ellipse(self.screen, AVOCADO_CREAM, outline, width=7)
        pygame.draw.ellipse(self.screen, color, outline, width=4)
        pygame.draw.circle(self.screen, color, center, 15, width=3)
        pygame.draw.line(self.screen, color, (center[0] - 22, center[1]), (center[0] + 22, center[1]), width=3)
        pygame.draw.line(self.screen, color, (center[0], center[1] - 22), (center[0], center[1] + 22), width=3)

    def draw_target(self, target: Target) -> None:
        sprite = self.avocado_sprites[target.action]
        rotated = pygame.transform.rotozoom(sprite, target.rotation, 1.0)
        self.screen.blit(rotated, rotated.get_rect(center=(TRACK_CENTER_X, target.y)))

    def draw_particles(self) -> None:
        for particle in self.particles:
            alpha = max(0.0, min(1.0, particle.life / 0.82))
            radius = max(1, int(particle.radius * alpha))
            pygame.draw.circle(self.screen, particle.color, (int(particle.x), int(particle.y)), radius)

    def draw_sensor_panel(self) -> None:
        panel_rect = pygame.Rect(PANEL_LEFT, PANEL_TOP, PANEL_WIDTH, PANEL_HEIGHT)
        pygame.draw.rect(self.screen, COUNTER_SHADOW, panel_rect.move(0, 8), border_radius=8)
        pygame.draw.rect(self.screen, COUNTER, panel_rect, border_radius=8)

        heading = self.font_lg.render("Gesture values", True, INK)
        self.screen.blit(heading, (PANEL_LEFT + 22, PANEL_TOP + 18))

        y = PANEL_TOP + 72
        bar_width = 178
        bar_height = 18
        for action, spec in enumerate(ACTION_SPECS):
            row_y = y + action * 58
            label = self.font_md.render(spec.label, True, INK)
            self.screen.blit(label, (PANEL_LEFT + 24, row_y - 5))

            key_rect = pygame.Rect(PANEL_LEFT + 106, row_y - 4, 28, 24)
            pygame.draw.rect(self.screen, spec.color, key_rect, border_radius=5)
            key = self.font_tiny.render(spec.key_name, True, CHIP_TEXT)
            self.screen.blit(key, key.get_rect(center=key_rect.center))

            bar_rect = pygame.Rect(PANEL_LEFT + 24, row_y + 26, bar_width + 74, bar_height)
            pygame.draw.rect(self.screen, SHELL_TAN, bar_rect, border_radius=5)
            fill_rect = pygame.Rect(
                bar_rect.left,
                bar_rect.top,
                int(bar_rect.width * self.sensor.intent_values[action]),
                bar_rect.height,
            )
            pygame.draw.rect(self.screen, spec.color, fill_rect, border_radius=5)

        guide_title = self.font_md.render("Three inputs. No aiming.", True, INK)
        self.screen.blit(guide_title, (PANEL_LEFT + 24, PANEL_TOP + 278))
        guide = self.font_sm.render("Match the symbol at the hit line.", True, MUTED)
        self.screen.blit(guide, (PANEL_LEFT + 24, PANEL_TOP + 316))

        if self.live_client is None:
            source_text = "source: keyboard gesture mirror"
        else:
            source_text = f"source: 4 sensors × 3 axes ({self.live_client.status})"
        source = self.font_tiny.render(source_text, True, MUTED)
        self.screen.blit(source, (PANEL_LEFT + 22, PANEL_TOP + PANEL_HEIGHT - 36))

    def draw_overlay(self) -> None:
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((*PIT_BROWN, 132))
        self.screen.blit(shade, (0, 0))

        rect = pygame.Rect(350, 178, 580, 270)
        pygame.draw.rect(self.screen, WHITE, rect, border_radius=8)
        pygame.draw.rect(self.screen, GOLD, rect, width=5, border_radius=8)

        if self.state == "menu":
            title = "AVOCADO SMASH"
            subtitle = "SPACE / ENTER"
            detail = "Press F, J, or K to match the avocado at the hit line."
        elif self.state == "paused":
            title = "PAUSED"
            subtitle = "P / SPACE"
            detail = "The three gesture values still respond while paused."
        else:
            title = "GAME OVER"
            subtitle = f"SCORE {self.score:05d}"
            detail = "SPACE / ENTER"

        title_surf = self.font_xl.render(title, True, INK)
        self.screen.blit(title_surf, title_surf.get_rect(center=(rect.centerx, rect.top + 68)))

        sub_surf = self.font_lg.render(subtitle, True, RED)
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(rect.centerx, rect.top + 128)))

        detail_surf = self.font_md.render(detail, True, MUTED)
        self.screen.blit(detail_surf, detail_surf.get_rect(center=(rect.centerx, rect.top + 184)))

        self.draw_mini_keymap(rect.centerx - 100, rect.top + 214)

    def draw_mini_keymap(self, x: int, y: int) -> None:
        for action, spec in enumerate(ACTION_SPECS):
            base_x = x + action * 82
            label = self.font_tiny.render(spec.label, True, INK)
            self.screen.blit(label, label.get_rect(center=(base_x + 18, y + 7)))

            key_rect = pygame.Rect(base_x + 2, y + 20, 32, 22)
            pygame.draw.rect(self.screen, spec.color, key_rect, border_radius=4)
            key = self.font_tiny.render(spec.key_name, True, CHIP_TEXT)
            self.screen.blit(key, key.get_rect(center=key_rect.center))

    def _make_avocado_sprite(self, action: int) -> pygame.Surface:
        size = TARGET_SIZE
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        rect = pygame.Rect(8, 5, size - 16, size - 10)
        pygame.draw.ellipse(surf, AVO_DARK, rect)
        pygame.draw.ellipse(surf, AVO_MID, rect.inflate(-8, -8))
        pygame.draw.ellipse(surf, AVO_FLESH, rect.inflate(-20, -20))
        pygame.draw.circle(surf, PIT, (size // 2, size // 2 + 10), 12)
        pygame.draw.circle(surf, SHELL_TAN, (size // 2 - 3, size // 2 + 6), 4)

        ribbon = pygame.Rect(8, 12, size - 16, 22)
        pygame.draw.rect(surf, ACTION_SPECS[action].color, ribbon, border_radius=5)
        label = self.font_tiny.render(ACTION_SPECS[action].symbol, True, CHIP_TEXT)
        surf.blit(label, label.get_rect(center=ribbon.center))

        badge_center = (size // 2, size // 2 + 10)
        badge = self.font_sm.render(ACTION_SPECS[action].symbol, True, CHIP_TEXT)
        surf.blit(badge, badge.get_rect(center=badge_center))
        return surf


def positive_speed(value: str) -> float:
    try:
        speed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("speed must be a number") from exc

    if not math.isfinite(speed) or speed <= 0.0:
        raise argparse.ArgumentTypeError("speed must be a finite number greater than 0")
    return speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Avocado Smash.")
    parser.add_argument(
        "--speed",
        type=positive_speed,
        default=1.0,
        help="Game speed multiplier. 1.0 is normal, 1.5 is faster, 0.75 is slower.",
    )
    parser.add_argument(
        "--sensor-port",
        help="Optional ESP32 serial port; the three keyboard gesture controls remain active.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "calibration_pipeline" / "profile.json",
        help="Gesture profile used with --sensor-port.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    game = AvocadoSmash(
        speed_multiplier=args.speed,
        sensor_port=args.sensor_port,
        profile_path=args.profile if args.sensor_port else None,
    )
    return game.run()


if __name__ == "__main__":
    raise SystemExit(main())
