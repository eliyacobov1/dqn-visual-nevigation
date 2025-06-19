import numpy as np

class DQNAgent:
    """Simple DQN agent with a two-layer neural network implemented in NumPy."""

    def __init__(self, n_states, n_actions, hidden_size=64, lr=0.01, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Xavier initialization
        limit1 = np.sqrt(6 / (n_states + hidden_size))
        self.w1 = np.random.uniform(-limit1, limit1, (n_states, hidden_size))
        self.b1 = np.zeros(hidden_size)

        limit2 = np.sqrt(6 / (hidden_size + n_actions))
        self.w2 = np.random.uniform(-limit2, limit2, (hidden_size, n_actions))
        self.b2 = np.zeros(n_actions)

    def _forward(self, state_index):
        x = np.eye(self.n_states)[state_index]
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0)
        q_values = a1 @ self.w2 + self.b2
        return q_values, (x, z1, a1)

    @property
    def q_table(self):
        states = np.eye(self.n_states)
        z1 = np.maximum(states @ self.w1 + self.b1, 0)
        return z1 @ self.w2 + self.b2

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        q_values, _ = self._forward(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        q_values, cache = self._forward(state)
        next_q_values, _ = self._forward(next_state)
        target = reward + self.gamma * np.max(next_q_values) * (not done)
        td_error = q_values[action] - target

        x, z1, a1 = cache
        grad_z2 = np.zeros_like(q_values)
        grad_z2[action] = td_error
        grad_w2 = a1[:, None] @ grad_z2[None, :]
        grad_b2 = grad_z2

        grad_a1 = grad_z2 @ self.w2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_w1 = x[:, None] @ grad_z1[None, :]
        grad_b1 = grad_z1

        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
