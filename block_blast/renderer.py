"""PyGame renderer for Block Blast.

Handles all drawing (grid, pieces, ghost preview, score, game-over overlay)
and provides hit-detection helpers so the game loop stays thin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from .game import GRID_SIZE

if TYPE_CHECKING:
    from .game import BlockBlastGame
    from .pieces import Piece

# ── layout constants ────────────────────────────────────────────────
WINDOW_W = 480
WINDOW_H = 720
CELL = 48
GAP = 2
STEP = CELL + GAP  # 50
GRID_PX = GRID_SIZE * CELL + (GRID_SIZE - 1) * GAP  # 398

GRID_X = (WINDOW_W - GRID_PX) // 2  # 41
GRID_Y = 100

TRAY_Y = GRID_Y + GRID_PX + 30
TRAY_H = 100
TRAY_CELL = 22
TRAY_STEP = TRAY_CELL + 2
SLOT_W = WINDOW_W // 3

# ── colours ─────────────────────────────────────────────────────────
BG = (15, 15, 35)
GRID_BG = (22, 28, 55)
EMPTY = (35, 42, 78)
WHITE = (240, 240, 245)
GOLD = (255, 215, 0)

BLOCK_COLORS = [
    None,
    (52, 152, 219),   # 1  blue
    (46, 204, 113),   # 2  green
    (231, 76, 60),    # 3  red
    (243, 156, 18),   # 4  orange
    (155, 89, 182),   # 5  purple
    (26, 188, 156),   # 6  teal
    (241, 196, 15),   # 7  yellow
    (232, 67, 147),   # 8  pink
]


def _lighter(color: tuple[int, ...], amount: int = 40) -> tuple[int, ...]:
    return tuple(min(255, c + amount) for c in color)


def _darker(color: tuple[int, ...], amount: int = 35) -> tuple[int, ...]:
    return tuple(max(0, c - amount) for c in color)


class Renderer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Block Blast")
        self.clock = pygame.time.Clock()
        self._font_big = pygame.font.SysFont("Arial", 42, bold=True)
        self._font_med = pygame.font.SysFont("Arial", 28, bold=True)
        self._font_sm = pygame.font.SysFont("Arial", 20)

    # ── main draw entry-point ───────────────────────────────────────

    def draw(
        self,
        game: BlockBlastGame,
        selected: int | None = None,
        mouse_pos: tuple[int, int] | None = None,
    ) -> None:
        self.screen.fill(BG)
        self._draw_header(game.score, game.streak)
        self._draw_grid(game.grid)
        self._draw_tray(game, selected)

        if selected is not None and mouse_pos is not None:
            piece = game.pieces[selected]
            if piece is not None:
                ci = game.piece_colors[selected]
                self._draw_ghost(game, piece, ci, mouse_pos)

        if game.game_over:
            self._draw_game_over(game.score)

        pygame.display.flip()

    # ── hit detection helpers ───────────────────────────────────────

    def placement_pos(
        self, piece: Piece, mouse_pos: tuple[int, int]
    ) -> tuple[int, int]:
        """Grid (row, col) where *piece* would land, centred on cursor."""
        mx, my = mouse_pos
        fc = (mx - GRID_X) / STEP
        fr = (my - GRID_Y) / STEP
        row = int(round(fr - (piece.height - 1) / 2))
        col = int(round(fc - (piece.width - 1) / 2))
        row = max(0, min(GRID_SIZE - piece.height, row))
        col = max(0, min(GRID_SIZE - piece.width, col))
        return row, col

    def tray_piece_idx(
        self, mouse_pos: tuple[int, int], game: BlockBlastGame
    ) -> int | None:
        mx, my = mouse_pos
        if not (TRAY_Y - 10 <= my <= TRAY_Y + TRAY_H + 10):
            return None
        idx = mx // SLOT_W
        if 0 <= idx < 3 and game.pieces[idx] is not None:
            return idx
        return None

    def is_on_grid(self, pos: tuple[int, int]) -> bool:
        mx, my = pos
        return GRID_X <= mx < GRID_X + GRID_PX and GRID_Y <= my < GRID_Y + GRID_PX

    # ── private drawing helpers ─────────────────────────────────────

    def _draw_header(self, score: int, streak: int) -> None:
        txt = self._font_big.render(f"{score:,}", True, GOLD)
        r = txt.get_rect(center=(WINDOW_W // 2, 40))
        self.screen.blit(txt, r)

        label = self._font_sm.render("SCORE", True, WHITE)
        lr = label.get_rect(center=(WINDOW_W // 2, 70))
        self.screen.blit(label, lr)

        if streak > 1:
            ct = self._font_sm.render(f"streak ×{streak}", True, (255, 180, 50))
            cr = ct.get_rect(center=(WINDOW_W // 2, 90))
            self.screen.blit(ct, cr)

    def _draw_grid(self, grid) -> None:
        bg = pygame.Rect(GRID_X - 5, GRID_Y - 5, GRID_PX + 10, GRID_PX + 10)
        pygame.draw.rect(self.screen, GRID_BG, bg, border_radius=8)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = GRID_X + col * STEP
                y = GRID_Y + row * STEP
                rect = pygame.Rect(x, y, CELL, CELL)
                val = int(grid[row, col])
                if val > 0:
                    clr = BLOCK_COLORS[val]
                    pygame.draw.rect(self.screen, clr, rect, border_radius=5)
                    hl = pygame.Rect(x + 3, y + 3, CELL - 6, CELL // 3)
                    pygame.draw.rect(self.screen, _lighter(clr), hl, border_radius=3)
                    sh = pygame.Rect(x + 3, y + CELL - CELL // 5, CELL - 6, CELL // 5 - 3)
                    pygame.draw.rect(self.screen, _darker(clr), sh, border_radius=3)
                else:
                    pygame.draw.rect(self.screen, EMPTY, rect, border_radius=5)

    def _draw_tray(self, game: BlockBlastGame, selected: int | None) -> None:
        for i in range(3):
            piece = game.pieces[i]
            if piece is None:
                continue
            ci = game.piece_colors[i]
            clr = BLOCK_COLORS[ci]

            pw = piece.width * TRAY_STEP - 2
            ph = piece.height * TRAY_STEP - 2
            ox = i * SLOT_W + (SLOT_W - pw) // 2
            oy = TRAY_Y + (TRAY_H - ph) // 2

            if selected == i:
                hl = pygame.Rect(ox - 6, oy - 6, pw + 12, ph + 12)
                pygame.draw.rect(self.screen, WHITE, hl, width=2, border_radius=6)

            for dr, dc in piece.cells:
                r = pygame.Rect(ox + dc * TRAY_STEP, oy + dr * TRAY_STEP,
                                TRAY_CELL, TRAY_CELL)
                pygame.draw.rect(self.screen, clr, r, border_radius=3)

    def _draw_ghost(
        self,
        game: BlockBlastGame,
        piece: Piece,
        color_idx: int,
        mouse_pos: tuple[int, int],
    ) -> None:
        pr, pc = self.placement_pos(piece, mouse_pos)
        valid = game.can_place(piece, pr, pc)

        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        if valid:
            base = BLOCK_COLORS[color_idx]
            surf.fill((*base, 110))
        else:
            surf.fill((200, 50, 50, 90))

        for dr, dc in piece.cells:
            r, c = pr + dr, pc + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                x = GRID_X + c * STEP
                y = GRID_Y + r * STEP
                self.screen.blit(surf, (x, y))

    def _draw_game_over(self, score: int) -> None:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        t1 = self._font_big.render("GAME OVER", True, (255, 70, 70))
        self.screen.blit(t1, t1.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 40)))

        t2 = self._font_med.render(f"Score  {score:,}", True, GOLD)
        self.screen.blit(t2, t2.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 15)))

        t3 = self._font_sm.render("R  restart   ·   Q  quit", True, WHITE)
        self.screen.blit(t3, t3.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 65)))

    def close(self) -> None:
        pygame.quit()
