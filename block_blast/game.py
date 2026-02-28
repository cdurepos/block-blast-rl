"""Core Block Blast game logic — no rendering, pure state machine.

Scoring mirrors the real Block Blast app:

  1.  Placement:    +1 per block placed on the grid
  2.  Block clears: +10 per unique block removed from completed rows/columns
  3.  Combo bonus:  clearing N lines simultaneously gives +10·(N+1)
      (1 line → +20, 2 → +30, 3 → +40, …)
  4.  Streak:       consecutive placements that each clear ≥1 line build a
      streak counter.  Bonus = max(0, streak − 1) · 10.
      Placing without clearing resets the streak to 0.
"""

from __future__ import annotations

import numpy as np

from .pieces import (
    Piece,
    random_piece,
    random_color,
    NUM_COLORS,
)

GRID_SIZE = 8
NUM_PIECE_SLOTS = 3


class BlockBlastGame:
    """8×8 Block Blast game.

    Lifecycle per turn:
      1.  Three pieces are dealt.
      2.  The player places them one-by-one (any order).
      3.  After each placement, completed rows/columns are cleared.
      4.  Once all three are placed a new set is dealt.
      5.  The game ends when no remaining piece can be placed.
    """

    def __init__(self) -> None:
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.pieces: list[Piece | None] = [None] * NUM_PIECE_SLOTS
        self.piece_colors: list[int] = [0] * NUM_PIECE_SLOTS
        self.score: int = 0
        self.game_over: bool = False
        self.streak: int = 0
        self._deal_pieces()

    # ── public API ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.grid[:] = 0
        self.score = 0
        self.game_over = False
        self.streak = 0
        self._deal_pieces()

    def can_place(self, piece: Piece, row: int, col: int) -> bool:
        for dr, dc in piece.cells:
            r, c = row + dr, col + dc
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
                return False
            if self.grid[r, c] != 0:
                return False
        return True

    def place_piece(self, slot: int, row: int, col: int) -> int:
        """Place piece *slot* at (row, col).  Returns the score gained."""
        piece = self.pieces[slot]
        if piece is None or not self.can_place(piece, row, col):
            return 0

        color = self.piece_colors[slot]
        for dr, dc in piece.cells:
            self.grid[row + dr, col + dc] = color

        self.pieces[slot] = None
        self.piece_colors[slot] = 0

        lines_cleared, blocks_cleared = self._clear_lines()

        # ── scoring (mirrors real Block Blast) ──────────────────────
        score = piece.size                             # 1. placement pts
        if lines_cleared > 0:
            score += blocks_cleared * 10               # 2. clear pts
            score += 10 * (lines_cleared + 1)          # 3. combo bonus
            self.streak += 1
            score += max(0, self.streak - 1) * 10      # 4. streak bonus
        else:
            self.streak = 0

        self.score += score

        if all(p is None for p in self.pieces):
            self._deal_pieces()

        if not self._has_valid_move():
            self.game_over = True

        return score

    def has_valid_move(self) -> bool:
        return self._has_valid_move()

    # ── internals ───────────────────────────────────────────────────

    def _deal_pieces(self) -> None:
        for i in range(NUM_PIECE_SLOTS):
            self.pieces[i] = random_piece()
            self.piece_colors[i] = random_color()

    def _clear_lines(self) -> tuple[int, int]:
        """Return (lines_cleared, unique_blocks_cleared)."""
        rows = [r for r in range(GRID_SIZE) if np.all(self.grid[r] != 0)]
        cols = [c for c in range(GRID_SIZE) if np.all(self.grid[:, c] != 0)]

        cleared: set[tuple[int, int]] = set()
        for r in rows:
            for c in range(GRID_SIZE):
                cleared.add((r, c))
        for c in cols:
            for r in range(GRID_SIZE):
                cleared.add((r, c))

        for r in rows:
            self.grid[r, :] = 0
        for c in cols:
            self.grid[:, c] = 0

        return len(rows) + len(cols), len(cleared)

    def _has_valid_move(self) -> bool:
        for i in range(NUM_PIECE_SLOTS):
            piece = self.pieces[i]
            if piece is None:
                continue
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if self.can_place(piece, r, c):
                        return True
        return False
