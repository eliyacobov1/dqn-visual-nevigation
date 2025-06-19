import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gridworld_env import GridWorldEnv


def test_obstacle_penalty():
    env = GridWorldEnv(grid_size=(3, 3), start=(0, 0), goal=(2, 2), obstacles=[(0, 1)])
    env.reset()
    next_state, reward, done, _ = env.step(3)  # Right into obstacle
    assert next_state == env.pos_to_state((0, 0))
    assert reward == -10
    assert done is False


def test_goal_reward_and_done():
    env = GridWorldEnv(grid_size=(3, 3), start=(0, 0), goal=(0, 1), obstacles=[])
    env.reset()
    next_state, reward, done, _ = env.step(3)  # Right to goal
    assert next_state == env.pos_to_state((0, 1))
    assert reward == 20
    assert done is True
