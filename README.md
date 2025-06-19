# Q-learning GridWorld Example

This repository implements a minimal reinforcement learning example: an agent learns to navigate a simple 2D grid using Q-learning.

## Files

- `gridworld_env.py` — Custom GridWorld environment.
- `q_learning_agent.py` — Q-learning algorithm implementation.
- `train.py` — Script to train the agent and show results.

## Usage

Install `matplotlib` if needed:

```bash
pip install matplotlib
```

Run training:

```bash
python train.py
```

The script prints progress every 100 episodes, plots the learning curve, and prints the learned policy grid at the end.
