import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt


class Maze2dOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg

        import gym
        import d4rl

        self.dataset = gym.make(cfg.env_id).get_dataset()
        self.gamma = cfg.gamma
        self.n_frames = cfg.episode_len
        self.total_steps = len(self.dataset["observations"])
        self.dataset["values"] = self.compute_value(self.dataset["rewards"]) * (1 - self.gamma) * 4 - 1

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            value[i] += self.gamma * value[i + 1]
        return value

    def __len__(self):
        return self.total_steps - self.n_frames + 1

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.dataset["observations"][idx : idx + self.n_frames]).float()
        action = torch.from_numpy(self.dataset["actions"][idx : idx + self.n_frames]).float()
        reward = torch.from_numpy(self.dataset["rewards"][idx : idx + self.n_frames]).float()
        value = torch.from_numpy(self.dataset["values"][idx : idx + self.n_frames]).float()

        done = np.zeros(self.n_frames, dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)

        goal = torch.zeros((self.n_frames, 0))

        return observation, action, reward, nonterminal


if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gym

    os.chdir("../..")
    cfg = MagicMock()
    cfg.env_id = "maze2d-medium-v1"
    cfg.episode_len = 600
    cfg.gamma = 1.0
    dataset = Maze2dOfflineRLDataset(cfg)
    o, a, r, n = dataset.__getitem__(10)
    print(o.shape, a.shape, r.shape, n.shape)
    plt.figure()
    plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")

    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]

    maze_string = gym.make(cfg.env_id).str_maze_spec
    grid = convert_maze_string_to_grid(maze_string)

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                plt.gca().add_patch(square)

    start_x, start_y = o[..., 0, :2]
    start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    plt.gca().add_patch(start_circle)
    inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    plt.gca().add_patch(inner_circle)

    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        plt.gca().add_patch(star)

    goal_x, goal_y = o[..., -1, :2]
    goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    plt.gca().add_patch(goal_circle)
    draw_star((goal_x, goal_y), radius=0.08)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("lightgray")
    plt.gca().set_axisbelow(True)
    plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    plt.xlim([0.5, len(grid) + 0.5])
    plt.ylim([0.5, len(grid[0]) + 0.5])
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    plt.grid(True, color="white", which="minor", linewidth=4)
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.show()
    print("Done.")
