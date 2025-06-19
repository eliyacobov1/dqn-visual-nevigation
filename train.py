import argparse
import matplotlib.pyplot as plt
import numpy as np

from gridworld_env import GridWorldEnv
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent


def run_episode(env, agent, train=True, max_steps=100):
    state = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    if train:
        agent.decay_epsilon()
    return total_reward


def plot_learning_curve(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.grid()
    plt.show()


def plot_policy(env, agent):
    directions = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    grid = []
    for y in range(env.grid_size[0]):
        row = []
        for x in range(env.grid_size[1]):
            state = env.pos_to_state((y, x))
            if (y, x) in env.obstacles:
                row.append("#")
            elif (y, x) == env.goal:
                row.append("G")
            else:
                best_action = np.argmax(agent.q_table[state])
                row.append(directions[best_action])
        grid.append(row)
    for row in grid:
        print(" ".join(row))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["qlearning", "dqn"], default="qlearning",
                        help="Learning algorithm to use")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    args = parser.parse_args()

    env = GridWorldEnv()

    if args.algorithm == "dqn":
        agent = DQNAgent(n_states=env.n_states, n_actions=env.n_actions)
    else:
        agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    n_episodes = args.episodes
    rewards = []
    for ep in range(n_episodes):
        ep_reward = run_episode(env, agent, train=True)
        rewards.append(ep_reward)
        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep+1}, Reward: {ep_reward}, Epsilon: {agent.epsilon:.3f}")

    plot_learning_curve(rewards)

    print("Learned Policy:")
    plot_policy(env, agent)
