import torch
import numpy as np

from utils.logging_utils import make_convergence_animation

file = "/home/boyuanc/Projects/diffuser/logs/maze2d-medium-v1/plans/release_H256_T256_LimitsNormalizer_b1_condFalse/0/maze2d-medium-v1_seed_2_plot_data.pt"
env_id = file.split("/")[-1].split("_")[0]
file = torch.load(file)
actions = file["actions"]
observations = file["observations"]
start = file["start"]
goal = file["goal"]

plan_history = torch.from_numpy(np.concatenate([observations, actions], axis=-1))
plan_history = plan_history.permute(1, 2, 0, 3)
trajectory = plan_history[-1].numpy()
plan_history = [plan_history]
start = start[None, :2]
goal = goal[None, :2]

filename = make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    15,
    "diffuser",
    interval=100,
    plot_end_points=True,
    batch_idx=0,
)
print(f"Saved to {filename}")
