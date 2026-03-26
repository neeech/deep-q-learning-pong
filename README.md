# Atari Pong DQN Experiment

A simple implementation of a Deep Q-Network (DQN) agent playing Atari Pong, inspired by [Mnih et al. (2013)](https://arxiv.org/abs/1312.5602).

## The Experiment

The goal was to see a DQN work in a classic environment using standard techniques:

- **Experience Replay**: To reuse past data and break correlation.
- **Fixed Q-Target Network**: Periodically "freezing" weights to stabilize training targets.
- **Checkpointing**: Saving progress to `checkpoints/`.

## Quick Start

### 1. Install Dependencies
Make sure you have `uv` installed, then run:
```bash
uv sync
```

### 2. Train
To start the training process (run for a few hours and forget):
```bash
uv run dql.py
```

### 3. Play
To see a trained agent in action (requires a checkpoint file):
```bash
uv run play_pong.py --checkpoint <checkpoint_name.pth>
```

## Observations

I built this to put into practice what I've been learning about RL. It's a simple implementation to verify the core concepts as I start exploring this field.
