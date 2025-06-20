import numpy as np

class DQNAgent:
    """DQN agent with a tiny neural network and optional replay/target network."""

    def __init__(
        self,
        n_states,
        n_actions,
        hidden_size=64,
        lr=0.01,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=32,
        replay_capacity=1000,
        target_update_freq=50,
        max_grad_norm=1.0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm

        self.replay_buffer = []
        self.update_count = 0

        # Xavier initialization
        limit1 = np.sqrt(6 / (n_states + hidden_size))
        self.w1 = np.random.uniform(-limit1, limit1, (n_states, hidden_size))
        self.b1 = np.zeros(hidden_size)

        limit2 = np.sqrt(6 / (hidden_size + n_actions))
        self.w2 = np.random.uniform(-limit2, limit2, (hidden_size, n_actions))
        self.b2 = np.zeros(n_actions)

        # Target network for more stable updates
        self.w1_target = self.w1.copy()
        self.b1_target = self.b1.copy()
        self.w2_target = self.w2.copy()
        self.b2_target = self.b2.copy()

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

    def _forward_target(self, state_index):
        x = np.eye(self.n_states)[state_index]
        z1 = x @ self.w1_target + self.b1_target
        a1 = np.maximum(z1, 0)
        q_values = a1 @ self.w2_target + self.b2_target
        return q_values

    def _apply_gradients(self, grad_w1, grad_b1, grad_w2, grad_b2):
        if self.max_grad_norm is not None:
            total_norm = np.sqrt(
                np.sum(grad_w1 ** 2)
                + np.sum(grad_b1 ** 2)
                + np.sum(grad_w2 ** 2)
                + np.sum(grad_b2 ** 2)
            )
            if total_norm > self.max_grad_norm and total_norm > 0:
                scale = self.max_grad_norm / total_norm
                grad_w1 *= scale
                grad_b1 *= scale
                grad_w2 *= scale
                grad_b2 *= scale

        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def update(self, state, action, reward, next_state, done):
        """Update the network using a mini-batch from the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        if len(self.replay_buffer) < self.batch_size:
            batch = [self.replay_buffer[-1]]
        else:
            idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in idx]

        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = np.zeros_like(self.b2)

        for s, a, r, ns, d in batch:
            q_values, cache = self._forward(s)
            next_q_values = self._forward_target(ns)
            target = r + self.gamma * np.max(next_q_values) * (not d)
            td_error = q_values[a] - target

            x, z1, a1 = cache
            grad_z2 = np.zeros_like(q_values)
            grad_z2[a] = td_error
            grad_w2 += a1[:, None] @ grad_z2[None, :]
            grad_b2 += grad_z2

            grad_a1 = grad_z2 @ self.w2.T
            grad_z1 = grad_a1 * (z1 > 0)
            grad_w1 += x[:, None] @ grad_z1[None, :]
            grad_b1 += grad_z1

        grad_w1 /= len(batch)
        grad_b1 /= len(batch)
        grad_w2 /= len(batch)
        grad_b2 /= len(batch)

        self._apply_gradients(grad_w1, grad_b1, grad_w2, grad_b2)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.w1_target = self.w1.copy()
            self.b1_target = self.b1.copy()
            self.w2_target = self.w2.copy()
            self.b2_target = self.b2.copy()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
