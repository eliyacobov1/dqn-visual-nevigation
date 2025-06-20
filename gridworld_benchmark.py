import numpy as np
import matplotlib.pyplot as plt
from gridworld_env import GridWorldEnv
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from train import run_episode, evaluate_agent, plot_learning_curve, plot_policy


class RandomAgent:
    """Baseline agent that selects valid actions uniformly at random."""

    def __init__(self, env):
        self.n_actions = env.n_actions
        self.env = env

    def select_action(self, state):
        y, x = self.env.state_to_pos(state)
        valid = []
        for action in range(self.n_actions):
            ny, nx = y, x
            if action == 0:
                ny = max(0, y - 1)
            elif action == 1:
                ny = min(self.env.grid_size[0] - 1, y + 1)
            elif action == 2:
                nx = max(0, x - 1)
            elif action == 3:
                nx = min(self.env.grid_size[1] - 1, x + 1)
            if (ny, nx) not in self.env.obstacles:
                valid.append(action)
        if not valid:
            valid = list(range(self.n_actions))
        return np.random.choice(valid)

    def update(self, *args, **kwargs):
        pass

    def decay_epsilon(self):
        pass


class HeuristicAgent(RandomAgent):
    """Agent that follows a shortest path to the goal via BFS."""

    def __init__(self, env):
        super().__init__(env)

    def _best_action(self, state):
        """Return the first action along a shortest path from *state* to the goal."""
        from collections import deque

        start = self.env.state_to_pos(state)
        goal = self.env.goal

        queue = deque([start])
        came_from = {start: (None, None)}  # map pos -> (prev_pos, action)

        while queue:
            pos = queue.popleft()
            if pos == goal:
                break
            y, x = pos
            for action in range(self.n_actions):
                ny, nx = y, x
                if action == 0:
                    ny = max(0, y - 1)
                elif action == 1:
                    ny = min(self.env.grid_size[0] - 1, y + 1)
                elif action == 2:
                    nx = max(0, x - 1)
                elif action == 3:
                    nx = min(self.env.grid_size[1] - 1, x + 1)
                next_pos = (ny, nx)
                if next_pos in came_from:
                    continue
                if next_pos in self.env.obstacles:
                    continue
                came_from[next_pos] = (pos, action)
                queue.append(next_pos)

        if goal not in came_from:
            return np.random.choice(self.n_actions)

        pos = goal
        while came_from[pos][0] != start and came_from[pos][0] is not None:
            pos = came_from[pos][0]
        return came_from[pos][1]

    def select_action(self, state):
        return self._best_action(state)


def train_agent(agent, env, episodes=300):
    rewards = []
    for _ in range(episodes):
        r = run_episode(env, agent, train=True)
        rewards.append(r)
    return rewards


def benchmark(episodes=300, eval_episodes=50):
    env = GridWorldEnv()
    agents = {
        "DQN": DQNAgent(n_states=env.n_states, n_actions=env.n_actions),
        "Q-Learning": QLearningAgent(n_states=env.n_states, n_actions=env.n_actions),
        "Heuristic": HeuristicAgent(env),
        "Random": RandomAgent(env),
    }

    results = {}
    for name, agent in agents.items():
        print(f"Training {name} agent...")
        if name in {"DQN", "Q-Learning"}:
            rewards = train_agent(agent, env, episodes)
        else:
            rewards = []
        success, steps = evaluate_agent(env, agent, episodes=eval_episodes)
        results[name] = {
            "rewards": rewards,
            "success": success,
            "steps": steps,
            "agent": agent,
        }
        if rewards:
            plot_learning_curve(rewards)
            print(f"{name} success: {success:.2f}, avg steps: {steps:.2f}")
            print("Policy:")
            plot_policy(env, agent)
        else:
            print(f"{name} success: {success:.2f}, avg steps: {steps:.2f}")

    # Plot comparison of learning curves
    plt.figure()
    for name, res in results.items():
        if res["rewards"]:
            plt.plot(res["rewards"], label=name)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid()
    plt.show()

    return results


if __name__ == "__main__":
    benchmark()
