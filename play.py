#!/usr/bin/env python3
"""Human-playable Block Blast.

Controls
--------
  Left-click a piece in the tray → select it
  Left-click on the grid          → place the selected piece
  Right-click / Escape            → deselect
  R                               → restart after game-over
  Q                               → quit
"""

import sys

import pygame

from block_blast.game import BlockBlastGame
from block_blast.renderer import Renderer


def main() -> None:
    game = BlockBlastGame()
    renderer = Renderer()
    selected: int | None = None

    running = True
    while running:
        mouse = pygame.mouse.get_pos()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_ESCAPE:
                    selected = None
                elif ev.key == pygame.K_r and game.game_over:
                    game.reset()
                    selected = None

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if game.game_over:
                    continue

                if ev.button == 3:
                    selected = None
                    continue

                if ev.button != 1:
                    continue

                if selected is not None:
                    piece = game.pieces[selected]
                    if piece is not None and renderer.is_on_grid(ev.pos):
                        row, col = renderer.placement_pos(piece, ev.pos)
                        if game.can_place(piece, row, col):
                            game.place_piece(selected, row, col)
                            selected = None
                            continue

                    tray = renderer.tray_piece_idx(ev.pos, game)
                    if tray is not None:
                        selected = tray
                    elif not renderer.is_on_grid(ev.pos):
                        selected = None
                else:
                    tray = renderer.tray_piece_idx(ev.pos, game)
                    if tray is not None:
                        selected = tray

        renderer.draw(game, selected, mouse)
        renderer.clock.tick(60)

    renderer.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
