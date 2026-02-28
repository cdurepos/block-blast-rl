#!/usr/bin/env python3
"""Watch a trained agent (or random baseline) play Block Blast.

Usage
-----
  python evaluate.py                            # trained model
  python evaluate.py --model models/best/best_model
  python evaluate.py --random                   # random baseline
  python evaluate.py --episodes 10 --delay 0.15
"""

import argparse
import time

import numpy as np
import pygame

from block_blast.env import BlockBlastEnv


def random_action(mask: np.ndarray) -> int:
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Block Blast agent")
    ap.add_argument("--model", type=str, default="models/best/best_model.zip")
    ap.add_argument("--random", action="store_true", help="Use random agent")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--delay", type=float, default=0.25,
                    help="Seconds between moves (visual pace)")
    args = ap.parse_args()

    model = None
    if not args.random:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model)

    env = BlockBlastEnv(render_mode="human")
    scores: list[int] = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        total = 0
        steps = 0

        while True:
            mask = env.action_masks()
            if not mask.any():
                break

            if model is not None:
                action, _ = model.predict(
                    obs, action_masks=mask, deterministic=True
                )
            else:
                action = random_action(mask)

            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            time.sleep(args.delay)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    env.close()
                    _print_summary(scores)
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    env.close()
                    _print_summary(scores)
                    return

            if terminated or truncated:
                break

        score = info.get("score", int(total))
        scores.append(score)
        print(f"Episode {ep}: score={score:,}  steps={steps}")
        time.sleep(1.0)

    env.close()
    _print_summary(scores)


def _print_summary(scores: list[int]) -> None:
    if not scores:
        return
    arr = np.array(scores, dtype=float)
    print(f"\n{'='*36}")
    print(f"  Episodes : {len(arr)}")
    print(f"  Mean     : {arr.mean():,.1f}")
    print(f"  Median   : {np.median(arr):,.1f}")
    print(f"  Best     : {arr.max():,.0f}")
    print(f"  Worst    : {arr.min():,.0f}")
    print(f"{'='*36}")


if __name__ == "__main__":
    main()
