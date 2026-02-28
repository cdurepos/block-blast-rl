# Block Blast RL

An offline clone of the **Block Blast** puzzle game with a reinforcement-learning
training pipeline designed to produce expert-level play.

## Quick start

```bash
pip install -r requirements.txt

# Play it yourself
python play.py

# Train the RL agent (5 M steps default — 20 M+ recommended for strong play)
python train.py
python train.py --timesteps 20_000_000   # longer run

# Watch the trained agent
python evaluate.py
python evaluate.py --random              # random baseline for comparison

# Monitor training live
tensorboard --logdir logs/
```

## Project layout

```
block_blast/
    pieces.py              – 19 piece types (dot, lines, squares, L-shapes)
    game.py                – game engine with real Block Blast scoring
    renderer.py            – PyGame rendering & hit-detection
    env.py                 – Gymnasium env: CNN observation + reward shaping
    feature_extractor.py   – ResNet-style CNN for 8×8 spatial input
play.py                    – human-playable game
train.py                   – RL training (MaskablePPO + CNN)
evaluate.py                – visualise trained agent or random baseline
```

## Game rules (mirrors real Block Blast)

| Rule | Detail |
|------|--------|
| Grid | 8 × 8 |
| Pieces per turn | 3 (place all before new deal) |
| Clearing | Full rows **and** columns are cleared simultaneously |
| Scoring | +1 per block placed, +10 per block cleared, combo bonus +10·(lines+1), streak bonus |
| Game over | When no remaining piece fits anywhere on the board |

## RL architecture

| Component | Detail |
|-----------|--------|
| **Algorithm** | MaskablePPO (sb3-contrib) — PPO with invalid-action masking |
| **Observation** | 6-channel 8×8 tensor: grid occupancy, row/col fill fractions, piece shapes |
| **Feature extractor** | ResNet CNN (4 residual blocks, 64 channels, 1×1 bottleneck → 512-dim) |
| **Action space** | Discrete(192) = 3 slots × 8 rows × 8 cols, with action masking |
| **Policy heads** | Separate actor [256] and critic [256] MLPs |
| **Reward** | Game score delta + potential-based shaping (squared row/col fill fractions) |
| **LR schedule** | Linear decay from 3e-4 → 0 |
| **Parameters** | ~1.66 M |

### Why this works

- **CNN sees spatial patterns** — "row 3 is 7/8 full" is a visual pattern that convolutions detect naturally, unlike an MLP on a flat vector
- **Reward shaping** bridges the credit-assignment gap — placing a block that partially fills a row gives a small immediate reward, even if the line clear comes 3 moves later
- **Action masking** eliminates wasted exploration on illegal moves, dramatically improving sample efficiency
- **Residual blocks** allow deeper feature extraction without vanishing gradients on the small 8×8 input

## Controls (human play)

| Input | Action |
|-------|--------|
| Left-click piece tray | Select piece |
| Left-click grid | Place selected piece |
| Right-click / Escape | Deselect |
| R | Restart (after game-over) |
| Q | Quit |
