"""Gymnasium environment for Block Blast with CNN observation + reward shaping.

Observation  (6 channels × 8 × 8)
-----------
  Ch 0   – binary grid occupancy (0 empty, 1 filled)
  Ch 1   – row fill fraction (each cell gets its row's fill level, 0→1)
  Ch 2   – column fill fraction
  Ch 3–5 – piece shapes (piece cells drawn in the top-left of the 8×8 grid,
           all zeros when that slot is empty)

Action space
------------
  Discrete(192) = 3 piece-slots × 8 rows × 8 cols
  action = slot·64 + row·8 + col

Reward shaping
--------------
  reward = game_score_delta + α · (Φ(s') − Φ(s))

  where Φ is a potential based on squared row/column fill fractions.
  This gives a small positive signal for "packing blocks toward full lines"
  without distorting the optimal policy (potential-based shaping, Ng 1999).

Action masking
--------------
  ``env.action_masks()`` → bool array of shape (192,).
  Compatible with sb3-contrib MaskablePPO.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import GRID_SIZE, NUM_PIECE_SLOTS, BlockBlastGame

NUM_ACTIONS = NUM_PIECE_SLOTS * GRID_SIZE * GRID_SIZE  # 192
NUM_CHANNELS = 6


class BlockBlastEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 5000,
        shaping_coef: float = 3.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._shaping_coef = shaping_coef

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CHANNELS, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        self.game = BlockBlastGame()
        self._steps = 0
        self._potential = self._board_potential()
        self._renderer = None

    # ── gymnasium API ───────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)
        self.game.reset()
        self._steps = 0
        self._potential = self._board_potential()
        return self._obs(), self._info()

    def step(self, action: int):
        old_potential = self._potential

        slot = action // 64
        row = (action % 64) // GRID_SIZE
        col = action % GRID_SIZE

        score = self.game.place_piece(slot, row, col)
        self._steps += 1

        self._potential = self._board_potential()
        shaping = self._shaping_coef * (self._potential - old_potential)
        reward = float(score) + shaping

        terminated = self.game.game_over
        truncated = self._steps >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        game = self.game
        for slot in range(NUM_PIECE_SLOTS):
            piece = game.pieces[slot]
            if piece is None:
                continue
            base = slot * 64
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if game.can_place(piece, r, c):
                        mask[base + r * GRID_SIZE + c] = True
        return mask

    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from .renderer import Renderer

            self._renderer = Renderer()
        import pygame

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
                raise SystemExit
        self._renderer.draw(self.game)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── observation ─────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        binary = (self.game.grid > 0).astype(np.float32)
        obs = np.zeros(
            (NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
        )

        obs[0] = binary

        row_fills = binary.sum(axis=1) / GRID_SIZE          # (8,)
        obs[1] = row_fills[:, np.newaxis]                    # broadcasts → (8, 8)

        col_fills = binary.sum(axis=0) / GRID_SIZE           # (8,)
        obs[2] = col_fills[np.newaxis, :]                    # broadcasts → (8, 8)

        for i in range(NUM_PIECE_SLOTS):
            piece = self.game.pieces[i]
            if piece is not None:
                for r, c in piece.cells:
                    obs[3 + i, r, c] = 1.0

        return obs

    # ── reward shaping ──────────────────────────────────────────────

    def _board_potential(self) -> float:
        """Quadratic potential on row/column fill fractions.

        High when lines are nearly full (ready to clear) or completely empty;
        low when blocks are scattered.  Squaring emphasises near-complete
        lines so the agent gets a stronger gradient the closer it gets.
        """
        binary = (self.game.grid > 0).astype(np.float32)
        row_fills = binary.sum(axis=1) / GRID_SIZE
        col_fills = binary.sum(axis=0) / GRID_SIZE
        return float((row_fills ** 2).sum() + (col_fills ** 2).sum())

    # ── info ────────────────────────────────────────────────────────

    def _info(self) -> dict:
        return {
            "score": self.game.score,
            "steps": self._steps,
            "streak": self.game.streak,
        }
