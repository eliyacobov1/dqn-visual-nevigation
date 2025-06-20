# Reinforcement Learning GridWorld Example

This repository implements a minimal reinforcement learning playground. An agent learns to navigate a simple 2D grid using either classic Q-learning or a lightweight Deep Q-Network (DQN) built with NumPy. It supports DynaCue-style planning with synthetic trajectories and a model-free approach using a replay buffer.

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

Additional flags enable richer visualisations:

- `--heatmap` shows a heat map of how often each cell was visited during training.
- `--q-history` plots how the Q-values for the start state evolve over episodes.
- `--animate` plays back the final trajectory with a red trace.
- `--eval-episodes` evaluates the trained agent for the given number of
  episodes and prints the success rate and average steps.
- `--planning-steps` uses DynaCue-style planning by sampling synthetic
  transitions after each real step.
- `--replay-steps` performs additional updates from a replay buffer
  after each real step.

Example evaluation comparing Q-learning and DQN:

```bash
python train.py --algorithm qlearning --episodes 300 --eval-episodes 50
python train.py --algorithm dqn --episodes 300 --eval-episodes 50
```

The printed metrics make it easy to compare algorithms or different parameter
settings.

### Comparing Dyna-style planning and replay buffer

Run `python compare_methods.py` to train two agents on the default grid:

1. **DynaQ** with `--planning-steps` enabled.
2. **Replay Buffer** with `--replay-steps` enabled.

The script prints the average reward and evaluation success rate for both
approaches so you can see how model-based planning compares with the
model-free alternative.

### Benchmarking

Run `python gridworld_benchmark.py` to train DQN and Q-learning agents on the
GridWorld environment. The benchmark also includes a heuristic agent that
follows a shortest path computed by breadth-first search and a random baseline
that avoids invalid moves. The script reports the success rate and average
steps to reach the goal and plots learning curves for each trained agent.

The environment used for benchmarking is a fixed 6×6 grid with start position
at `(0, 0)` and goal at `(5, 5)`. Four obstacles block cells `(1, 1)`,
`(2, 3)`, `(3, 2)` and `(4, 4)`. This configuration defines the dataset of
state transitions the agents see during training. After 300 training episodes,
evaluation over 50 episodes yields the following metrics:

| Agent      | Success Rate | Avg. Steps |
| ---------- | ------------ | ---------- |
| DQN        | 1.00         | 10.0       |
| Q-Learning | 1.00         | 10.0       |
| Heuristic  | 1.00         | 10.0       |
| Random     | ~0.44        | ~86        |

These results demonstrate that both learning agents reliably solve the grid
while the random policy struggles to reach the goal.
