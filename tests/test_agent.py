import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from q_learning_agent import QLearningAgent
from train import run_episode
from gridworld_env import GridWorldEnv


def test_select_action_greedy():
    agent = QLearningAgent(n_states=1, n_actions=2, epsilon=0.0)
    agent.q_table[0, 0] = 1
    agent.q_table[0, 1] = 5
    action = agent.select_action(0)
    assert action == 1


def test_update_with_done():
    agent = QLearningAgent(n_states=1, n_actions=2, alpha=0.5, gamma=0.9)
    agent.update(0, 1, reward=1, next_state=0, done=True)
    assert np.isclose(agent.q_table[0, 1], 0.5)


def test_epsilon_decay_minimum():
    agent = QLearningAgent(n_states=1, n_actions=1, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    for _ in range(5):
        agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.1)


def test_run_episode_no_training_does_not_update():
    env = GridWorldEnv(grid_size=(2, 2), start=(0, 0), goal=(0, 1), obstacles=[])
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, epsilon=0.0)
    q_table_before = agent.q_table.copy()
    run_episode(env, agent, train=False, max_steps=5)
    assert np.array_equal(agent.q_table, q_table_before)
    assert np.isclose(agent.epsilon, 0.0)

def test_run_episode_training_decays_epsilon():
    env = GridWorldEnv(grid_size=(2, 2), start=(0, 0), goal=(0, 1), obstacles=[])
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, epsilon=1.0, epsilon_decay=0.5)
    run_episode(env, agent, train=True, max_steps=5)
    assert np.isclose(agent.epsilon, 0.5)
