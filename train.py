#!/usr/bin/env python3
"""Train a Block Blast agent with MaskablePPO + CNN.

Usage
-----
  python train.py                             # 5 M steps (default)
  python train.py --timesteps 20_000_000      # 20 M steps (recommended)
  python train.py --save models/my_model

Architecture
------------
  Observation : (6, 8, 8) spatial tensor — see env.py docstring
  Policy      : ResNet-style CNN  →  actor/critic MLP heads
  Algorithm   : MaskablePPO with linear LR decay
  Reward      : game score delta + potential-based shaping

Monitoring
----------
  Console : score stats printed every --log-interval steps
  TB      : tensorboard --logdir logs/
  Watch   : python evaluate.py --model models/checkpoints/bb_XXXXX_steps
"""

import argparse
import os
import time
from collections import deque

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from block_blast.env import BlockBlastEnv
from block_blast.feature_extractor import BlockBlastCNN


# ── learning-rate schedule ──────────────────────────────────────────


def linear_schedule(initial: float):
    """Linear LR decay from *initial* → 0 over training."""

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return _schedule


# ── score tracking callback ─────────────────────────────────────────


class ScoreTracker(BaseCallback):
    """Logs raw game scores (not shaped reward) to console + TensorBoard."""

    def __init__(self, log_interval: int = 10_000, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.scores: deque[float] = deque(maxlen=window)
        self._last_log = 0
        self._episodes = 0
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                game_score = info.get("score", ep["r"])
                self.scores.append(game_score)
                self._episodes += 1
                self.logger.record("block_blast/episode_score", game_score)
                self.logger.record("block_blast/episode_length", ep["l"])

        if (
            self.num_timesteps - self._last_log >= self.log_interval
            and self.scores
        ):
            arr = np.array(self.scores)
            sps = self.num_timesteps / max(time.time() - self._t0, 1)
            if self.verbose:
                print(
                    f"[{self.num_timesteps:>10,} steps | {self._episodes:>6,} eps | {sps:>5,.0f} sps]  "
                    f"score  mean={arr.mean():8.1f}  med={np.median(arr):8.1f}  "
                    f"best={arr.max():8.0f}  worst={arr.min():8.0f}"
                )
            self.logger.record("block_blast/mean_score", float(arr.mean()))
            self.logger.record("block_blast/median_score", float(np.median(arr)))
            self.logger.record("block_blast/best_score", float(arr.max()))
            self._last_log = self.num_timesteps

        return True


# ── environment factory ─────────────────────────────────────────────


def make_env(shaping_coef: float = 3.0):
    def _init():
        return Monitor(BlockBlastEnv(shaping_coef=shaping_coef))

    return _init


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Block Blast RL agent")
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--save", type=str, default="models/block_blast_ppo")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shaping", type=float, default=3.0,
                    help="Reward shaping coefficient (0 to disable)")
    ap.add_argument("--log-interval", type=int, default=10_000)
    ap.add_argument("--resume", type=str, default=None,
                    help="Checkpoint path to resume from")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    env = DummyVecEnv([make_env(args.shaping) for _ in range(args.n_envs)])
    eval_env = BlockBlastEnv(shaping_coef=0.0)

    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(
            features_dim=512,
            n_res_blocks=4,
            channels=64,
        ),
        net_arch=dict(pi=[256], vf=[256]),
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=linear_schedule(args.lr),
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/",
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy network: {total_params:,} parameters")

    callbacks = [
        ScoreTracker(log_interval=args.log_interval),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path="models/checkpoints/",
            name_prefix="bb",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="models/best/",
            log_path="logs/eval/",
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]

    print(f"\nTraining for {args.timesteps:,} timesteps across {args.n_envs} envs")
    print(f"Reward shaping coef = {args.shaping}")
    print(f"Score stats every {args.log_interval:,} steps")
    print(f"Checkpoints → models/checkpoints/")
    print(f"TensorBoard → tensorboard --logdir logs/\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(args.save)
    print(f"\nFinal model saved → {args.save}")


if __name__ == "__main__":
    main()
