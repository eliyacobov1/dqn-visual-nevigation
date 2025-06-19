import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
