import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from dqn_agent import DQNAgent
from gridworld_env import GridWorldEnv
from train import run_episode


def test_select_action_greedy():
    agent = DQNAgent(n_states=1, n_actions=2, epsilon=0.0)
    agent.b2 = np.array([1.0, 5.0])  # biases ensure action 1 is best
    action = agent.select_action(0)
    assert action == 1


def test_epsilon_decay_minimum():
    agent = DQNAgent(n_states=1, n_actions=1, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    for _ in range(5):
        agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.1)


def test_run_episode_no_training_does_not_update():
    env = GridWorldEnv(grid_size=(2, 2), start=(0, 0), goal=(0, 1), obstacles=[])
    agent = DQNAgent(n_states=env.n_states, n_actions=env.n_actions, epsilon=0.0)
    w1_before = agent.w1.copy()
    run_episode(env, agent, train=False, max_steps=5)
    assert np.array_equal(agent.w1, w1_before)


def test_update_changes_weights():
    agent = DQNAgent(n_states=2, n_actions=2, lr=0.1, epsilon=0.0)
    w1_before = agent.w1.copy()
    agent.update(0, 1, reward=1, next_state=1, done=False)
    assert not np.array_equal(agent.w1, w1_before)
