# Reinforcement Learning GridWorld Example

This repository implements a minimal reinforcement learning playground. An agent learns to navigate a simple 2D grid using either classic Q-learning or a lightweight Deep Q-Network (DQN) built with NumPy.

## Files

- `gridworld_env.py` — Custom GridWorld environment.
- `q_learning_agent.py` — Tabular Q-learning implementation.
- `dqn_agent.py` — DQN agent with a small neural network written in NumPy.
- `train.py` — Script to train an agent and show results. Choose the algorithm with `--algorithm`.

## Usage

Install `matplotlib` if needed:

```bash
pip install matplotlib
```

Run training (choose `qlearning` or `dqn`). The environment size and layout can
be customised using command-line options:

```bash
python train.py --algorithm dqn --grid-size 8 8 --obstacle-density 0.2 \
    --random-start-goal
```

The above example trains a DQN agent on an 8x8 grid with 20% of the cells
randomly blocked and new start/goal locations each episode.

Basic usage without any extras:

```bash
python train.py --algorithm dqn
```

The script prints progress every 100 episodes, plots the learning curve, and
prints the learned policy grid at the end. Use `--visualize` to see an animated
episode demonstrating the learned policy.
