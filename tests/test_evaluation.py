import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from train import evaluate_agent, run_episode
from q_learning_agent import QLearningAgent
from gridworld_env import GridWorldEnv


def test_evaluate_agent_success():
    env = GridWorldEnv(grid_size=(2, 2), start=(0, 0), goal=(0, 1), obstacles=[])
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, epsilon=0.0)
    # Make agent always move right
    agent.q_table[:, 3] = 1.0
    success_rate, avg_steps = evaluate_agent(env, agent, episodes=5, max_steps=2)
    assert np.isclose(success_rate, 1.0)
    assert avg_steps <= 2
