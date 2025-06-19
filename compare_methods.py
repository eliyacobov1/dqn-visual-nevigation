import numpy as np
from gridworld_env import GridWorldEnv
from dqn_agent import DQNAgent
from train import run_episode, evaluate_agent


def run_training(planning_steps=0, replay_steps=0, episodes=200):
    env = GridWorldEnv()
    agent = DQNAgent(n_states=env.n_states, n_actions=env.n_actions)
    rewards = []
    for _ in range(episodes):
        r = run_episode(
            env,
            agent,
            train=True,
            planning_steps=planning_steps,
            replay_steps=replay_steps,
        )
        rewards.append(r)
    success, steps = evaluate_agent(env, agent, episodes=20)
    return np.mean(rewards[-50:]), success, steps


def main():
    dyna_reward, dyna_success, dyna_steps = run_training(planning_steps=5)
    rep_reward, rep_success, rep_steps = run_training(replay_steps=5)
    print("DynaQ:    avg reward %.2f, success %.2f, avg steps %.2f" % (
        dyna_reward, dyna_success, dyna_steps))
    print("Replay:   avg reward %.2f, success %.2f, avg steps %.2f" % (
        rep_reward, rep_success, rep_steps))


if __name__ == "__main__":
    main()
