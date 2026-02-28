"""Piece definitions for Block Blast.

Each piece is a frozen set of (row, col) offsets from its top-left corner.
Since Block Blast does NOT let the player rotate pieces, every orientation
of a shape is stored as a separate piece.

The full set (37 pieces) mirrors the real game:

  Single         1×1 dot                                 ×1
  Lines          h2 h3 h4 h5, v2 v3 v4 v5              ×8
  Squares        2×2, 3×3                                ×2
  Rectangles     2×3, 3×2                                ×2
  Small corner   triomino L  (3 cells, 2×2)    4 rots    ×4
  Big corner     pentomino L (5 cells, 3×3)    4 rots    ×4
  L-tetromino    (4 cells)                     4 rots    ×4
  J-tetromino    (mirror of L)                 4 rots    ×4
  T-tetromino    (4 cells)                     4 rots    ×4
  S-piece        (4 cells)                     2 rots    ×2
  Z-piece        (4 cells)                     2 rots    ×2
                                                        ──────
                                                          37
"""

import random
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Piece:
    cells: Tuple[Tuple[int, int], ...]

    @property
    def width(self) -> int:
        return max(c for _, c in self.cells) + 1

    @property
    def height(self) -> int:
        return max(r for r, _ in self.cells) + 1

    @property
    def size(self) -> int:
        return len(self.cells)


def _parse(rows: list[str]) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "X"
    )


_SHAPES: dict[str, list[str]] = {
    # ── single ──────────────────────────────────────────────────────
    "dot": ["X"],
    # ── lines ───────────────────────────────────────────────────────
    "h2": ["XX"],
    "v2": ["X", "X"],
    "h3": ["XXX"],
    "v3": ["X", "X", "X"],
    "h4": ["XXXX"],
    "v4": ["X", "X", "X", "X"],
    "h5": ["XXXXX"],
    "v5": ["X", "X", "X", "X", "X"],
    # ── squares ─────────────────────────────────────────────────────
    "sq2": ["XX", "XX"],
    "sq3": ["XXX", "XXX", "XXX"],
    # ── 2×3 / 3×2 rectangle ────────────────────────────────────────
    "r23": ["XXX", "XXX"],
    "r32": ["XX", "XX", "XX"],
    # ── small corner / triomino L  (3 cells, 2×2 bbox) ─────────────
    #     ┘    └    ┌    ┐
    "c1": ["X.", "XX"],
    "c2": [".X", "XX"],
    "c3": ["XX", ".X"],
    "c4": ["XX", "X."],
    # ── big corner / pentomino L  (5 cells, 3×3 bbox) ──────────────
    "bc1": ["X..", "X..", "XXX"],
    "bc2": ["XXX", "X..", "X.."],
    "bc3": ["XXX", "..X", "..X"],
    "bc4": ["..X", "..X", "XXX"],
    # ── L-tetromino  (4 cells, all 4 rotations) ────────────────────
    #   X.    XXX    XX    ..X
    #   X.    X..    .X    XXX
    #   XX           .X
    "L0": ["X.", "X.", "XX"],
    "L1": ["XXX", "X.."],
    "L2": ["XX", ".X", ".X"],
    "L3": ["..X", "XXX"],
    # ── J-tetromino  (mirror of L, 4 rotations) ────────────────────
    #   .X    X..    XX    XXX
    #   .X    XXX    X.    ..X
    #   XX           X.
    "J0": [".X", ".X", "XX"],
    "J1": ["X..", "XXX"],
    "J2": ["XX", "X.", "X."],
    "J3": ["XXX", "..X"],
    # ── T-piece  (4 rotations) ─────────────────────────────────────
    #   XXX    .X    .X.    X.
    #   .X.    XX    XXX    XX
    #          .X           X.
    "T0": ["XXX", ".X."],
    "T1": [".X", "XX", ".X"],
    "T2": [".X.", "XXX"],
    "T3": ["X.", "XX", "X."],
    # ── S-piece  (2 rotations) ─────────────────────────────────────
    #   .XX    X.
    #   XX.    XX
    #          .X
    "S0": [".XX", "XX."],
    "S1": ["X.", "XX", ".X"],
    # ── Z-piece  (2 rotations) ─────────────────────────────────────
    #   XX.    .X
    #   .XX    XX
    #          X.
    "Z0": ["XX.", ".XX"],
    "Z1": [".X", "XX", "X."],
}

ALL_PIECES: list[Piece] = [Piece(_parse(shape)) for shape in _SHAPES.values()]
PIECE_NAMES: list[str] = list(_SHAPES.keys())
MAX_PIECE_DIM = 5
NUM_COLORS = 8


def random_piece() -> Piece:
    return random.choice(ALL_PIECES)


def random_color() -> int:
    """Return a random colour index in [1, NUM_COLORS]."""
    return random.randint(1, NUM_COLORS)
