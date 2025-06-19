import argparse
import matplotlib.pyplot as plt
import numpy as np

from gridworld_env import GridWorldEnv
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent


def run_episode(env, agent, train=True, max_steps=100, record_path=False,
                return_details=False):
    """Run a single episode.

    If *record_path* is True, also return the sequence of visited states."""
    state = env.reset()
    total_reward = 0
    path = [state]
    steps = 0
    success = False
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if record_path:
            path.append(state)
        if done:
            success = True
            steps += 1
            break
        steps += 1
    if train:
        agent.decay_epsilon()
    if return_details:
        info = {"reward": total_reward, "success": success, "steps": steps}
        if record_path:
            info["path"] = path
        return info
    if record_path:
        return total_reward, path
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


def plot_visit_heatmap(visit_counts):
    """Plot a heat map showing how often each cell was visited."""
    plt.imshow(visit_counts, cmap="hot", origin="upper")
    plt.title("State Visit Heatmap")
    plt.colorbar(label="Visits")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_q_evolution(q_history, state_index=0):
    """Plot how Q-values for *state_index* evolve over episodes."""
    q_arr = np.array([q[state_index] for q in q_history])
    for action in range(q_arr.shape[1]):
        plt.plot(q_arr[:, action], label=f"Action {action}")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title(f"Q-value Evolution for state {state_index}")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_agent(env, agent, episodes=100, max_steps=100):
    """Evaluate *agent* for a number of episodes.

    Returns the success rate and average steps taken."""
    original_epsilon = getattr(agent, "epsilon", None)
    if original_epsilon is not None:
        agent.epsilon = 0.0
    successes = 0
    total_steps = 0
    for _ in range(episodes):
        info = run_episode(
            env, agent, train=False, max_steps=max_steps, return_details=True
        )
        if info["success"]:
            successes += 1
        total_steps += info["steps"]
    if original_epsilon is not None:
        agent.epsilon = original_epsilon
    success_rate = successes / episodes
    avg_steps = total_steps / episodes
    return success_rate, avg_steps


def animate_trajectory(env, path, interval=200):
    """Animate a trajectory given a sequence of states."""
    plt.ion()
    fig, ax = plt.subplots()
    xs, ys = [], []
    for step, state in enumerate(path):
        y, x = env.state_to_pos(state)
        env.agent_pos = (y, x)
        xs.append(x)
        ys.append(y)
        env.render(ax=ax)
        ax.plot(xs, ys, color="red")
        ax.set_title(f"Step {step}")
        plt.pause(interval / 1000.0)
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
    parser.add_argument("--heatmap", action="store_true",
                        help="Show a heat map of visited cells")
    parser.add_argument("--q-history", action="store_true",
                        help="Plot Q-value evolution for the start state")
    parser.add_argument("--animate", action="store_true",
                        help="Animate the final trajectory using the learned policy")
    parser.add_argument("--grid-size", type=int, nargs=2, metavar=("H", "W"),
                        default=(6, 6), help="Grid dimensions")
    parser.add_argument("--obstacle-density", type=float, default=0.0,
                        help="Fraction of cells that are obstacles")
    parser.add_argument("--random-start-goal", action="store_true",
                        help="Randomize start and goal each episode")
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="Run evaluation for N episodes after training")
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
    visit_counts = np.zeros(env.grid_size, dtype=int)
    q_history = []
    for ep in range(n_episodes):
        record_path = args.heatmap
        result = run_episode(env, agent, train=True, record_path=record_path)
        if record_path:
            ep_reward, path = result
            for state in path:
                y, x = env.state_to_pos(state)
                visit_counts[y, x] += 1
        else:
            ep_reward = result
        rewards.append(ep_reward)
        if args.q_history:
            q_history.append(agent.q_table.copy())
        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep+1}, Reward: {ep_reward}, Epsilon: {agent.epsilon:.3f}")

    if args.eval_episodes > 0:
        success_rate, avg_steps = evaluate_agent(
            env, agent, episodes=args.eval_episodes
        )
        print(
            f"Evaluation over {args.eval_episodes} episodes - "
            f"Success rate: {success_rate:.2f}, Avg steps: {avg_steps:.2f}"
        )

    plot_learning_curve(rewards)

    print("Learned Policy:")
    plot_policy(env, agent)

    if args.heatmap:
        plot_visit_heatmap(visit_counts)

    if args.q_history and q_history:
        plot_q_evolution(q_history, state_index=env.pos_to_state(env.start))
    if args.visualize:
        visualize_episode(env, agent)

    if args.animate:
        _, path = run_episode(env, agent, train=False, record_path=True)
        animate_trajectory(env, path)
