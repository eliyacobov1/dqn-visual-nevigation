import numpy as np

class GridWorldEnv:
    def __init__(self, grid_size=(6, 6), start=(0, 0), goal=(5, 5), obstacles=None):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 3), (3, 2), (4, 4)]
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.n_actions = 4
        self.n_states = grid_size[0] * grid_size[1]
        self.reset()

    def state_to_pos(self, state):
        return (state // self.grid_size[1], state % self.grid_size[1])

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size[1] + pos[1]

    def reset(self):
        self.agent_pos = self.start
        return self.pos_to_state(self.agent_pos)

    def step(self, action):
        y, x = self.agent_pos
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(self.grid_size[0] - 1, y + 1)
        elif action == 2:  # Left
            x = max(0, x - 1)
        elif action == 3:  # Right
            x = min(self.grid_size[1] - 1, x + 1)
        next_pos = (y, x)
        reward = -1
        done = False

        if next_pos in self.obstacles:
            reward = -10
            next_pos = self.agent_pos  # stay in place
        elif next_pos == self.goal:
            reward = 20
            done = True

        self.agent_pos = next_pos
        return self.pos_to_state(self.agent_pos), reward, done, {}

    def render(self):
        grid = np.full(self.grid_size, ".", dtype=str)
        for oy, ox in self.obstacles:
            grid[oy, ox] = "#"
        gy, gx = self.goal
        grid[gy, gx] = "G"
        ay, ax = self.agent_pos
        grid[ay, ax] = "A"
        for row in grid:
            print(" ".join(row))
        print()
