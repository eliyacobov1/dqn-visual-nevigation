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


def visualize_episode(env, agent, interval=1, max_steps=100):
    """Visualize an episode by rendering the grid every *interval* steps."""
    plt.ion()
    fig, ax = plt.subplots()
    state = env.reset()
    env.render(ax=ax)
    ax.set_title("Start")
    plt.pause(0.5)

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        if step % interval == 0 or done:
            env.render(ax=ax)
            ax.set_title(f"Step {step + 1}")
            plt.pause(0.5)
        state = next_state
        if done:
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["qlearning", "dqn"], default="qlearning",
                        help="Learning algorithm to use")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize a demonstration episode after training")
    parser.add_argument("--grid-size", type=int, nargs=2, metavar=("H", "W"),
                        default=(6, 6), help="Grid dimensions")
    parser.add_argument("--obstacle-density", type=float, default=0.0,
                        help="Fraction of cells that are obstacles")
    parser.add_argument("--random-start-goal", action="store_true",
                        help="Randomize start and goal each episode")
    args = parser.parse_args()

    env = GridWorldEnv(grid_size=tuple(args.grid_size),
                       obstacle_density=args.obstacle_density,
                       random_start_goal=args.random_start_goal)

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

    if args.visualize:
        visualize_episode(env, agent)
