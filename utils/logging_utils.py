from typing import Optional

import wandb

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import matplotlib.animation as animation
from pathlib import Path

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure, universal_image_quality_index
from algorithms.common.metrics import (
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    observation_hat[:context_frames] = observation_gt[:context_frames]
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c = c / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c
        observation_hat[:context_frames, :, i, :, [0, -1]] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
    video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        logger.log({f"{namespace}/{prefix}_{i}": wandb.Video(video[i], fps=24), f"trainer/global_step": step})
        # path = Path(f"outputs/robot_video/video_{i}")
        # path.mkdir(parents=True, exist_ok=True)
        # for t, f in enumerate(video[i]):
        #     (path / "view1").mkdir(parents=True, exist_ok=True)
        #     (path / "view2").mkdir(parents=True, exist_ok=True)
        #     f = f[..., :32]
        #     f = np.transpose(f, (1, 2, 0))
        #     f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(str((path / f"view1/{t}.png").resolve()), f[:32])
        #     cv2.imwrite(str((path / f"view2/{t}.png").resolve()), f[32:])


def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(observation_hat, observation_gt)

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt, data_range=2.0)

    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def get_validation_metrics_for_states(observation_hat, observation_gt):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel)
    :return: a tuple of metrics
    """
    frame, batch, channel = observation_hat.shape

    # reshape to (frame * batch, channel)
    observation_hat = observation_hat.reshape(-1, channel)
    observation_gt = observation_gt.reshape(-1, channel)

    mse = mean_squared_error(observation_hat, observation_gt)
    psnr = peak_signal_noise_ratio(observation_hat, observation_gt)

    output_dict = {
        "mse": mse,
        "psnr": psnr,
    }

    return output_dict


def log_galton_for_states(batch_idx, observations_hat, observations_gt, namespace, step):
    def compute_trajectory(dx):
        """
        Integrate sequence of increments dx to get trajectory x.
        """
        x = [torch.zeros(dx.shape[1], 1, device=dx.device)]
        num_time_steps = dx.shape[0]
        for t in range(num_time_steps):
            x.append(dx[t] + x[-1])
        x = torch.stack(x)
        x = x.permute(1, 0, 2)
        return x

    x_hat = compute_trajectory(observations_hat)
    x_gt = compute_trajectory(observations_gt)

    if wandb.run is not None:
        if batch_idx == 0:
            for i in range(min(4, x_hat.shape[0])):
                # Plot x_i and x_gt_i and log the plot to wandb
                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                ax.plot(x_hat[i, :, 0].detach().cpu().numpy(), linewidth=3, label="x_hat")
                ax.plot(x_gt[i, :, 0].detach().cpu().numpy(), linewidth=3, label="x_gt")
                ax.legend()
                wandb.log({f"{namespace}/x_{i}": wandb.Image(fig)}, step=step)
                plt.close(fig)

    final_x_hat = torch.flatten(x_hat[:, -1]).detach().cpu().numpy()
    final_x_gt = torch.flatten(x_gt[:, -1]).detach().cpu().numpy()
    dx_hat = torch.flatten(torch.diff(x_hat, dim=1)).detach().cpu().numpy()
    dx_gt = torch.flatten(torch.diff(x_gt, dim=1)).detach().cpu().numpy()

    return final_x_hat, final_x_gt, dx_hat, dx_gt


def log_galton_for_pixels(batch_idx, observations_hat, observations_gt, namespace, step):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param namespace: the namespace to log to
    :param step: the step to log to
    """

    observations_hat_np = observations_hat.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    observations_gt_np = observations_gt.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    observations_hat_np = (np.clip(observations_hat_np + 1.0, a_min=0.0, a_max=2.0) / 2 * 255).astype(np.uint8)
    observations_gt_np = (np.clip(observations_gt_np + 1.0, a_min=0.0, a_max=2.0) / 2 * 255).astype(np.uint8)
    x_hat, centers_hat, contours_hat = get_galton_metrics_for_pixels(observations_hat_np)
    x_gt, centers_gt, contours_gt = get_galton_metrics_for_pixels(observations_gt_np)

    if wandb.run is not None:
        if batch_idx == 0:
            observations_hat_np = np.transpose(observations_hat_np, (1, 0, 2, 3, 4))
            observations_gt_np = np.transpose(observations_gt_np, (1, 0, 2, 3, 4))

            # Log average number of balls
            avg_num_predicted_balls = np.mean(
                [len(contours_hat[i][t]) for i in range(len(contours_hat)) for t in range(len(contours_hat[i]))]
            )
            wandb.log({f"{namespace}/avg_num_predicted_balls": avg_num_predicted_balls}, step=step)

            for i in range(min(4, x_hat.shape[0])):
                # Plot x_i and x_gt_i and log the plot to wandb
                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                ax.plot(x_hat[i, :], linewidth=3, label="x_hat")
                ax.plot(x_gt[i, :], linewidth=3, label="x_gt")
                ax.legend()
                wandb.log({f"{namespace}/x_{i}": wandb.Image(fig)}, step=step)
                plt.close(fig)

                # Plot contours and log the plot to wandb
                video = []
                for t in range(len(centers_hat[i])):
                    img_hat = render_ball_contours_and_center(
                        observations_hat_np[i, t], centers_hat[i][t], contours_hat[i][t]
                    )
                    img_gt = render_ball_contours_and_center(
                        observations_gt_np[i, t], centers_gt[i][t], contours_gt[i][t]
                    )
                    img_gt[[0, -1], :, 0] = 255
                    img_gt[:, [0, -1], 0] = 255
                    img = np.concatenate([img_hat, img_gt], axis=1)
                    img = np.transpose(img, (2, 0, 1))
                    video.append(img)
                wandb.log({f"{namespace}/contours_{i}": wandb.Video(np.array(video), fps=5, format="gif")}, step=step)

    final_x_hat = x_hat[:, -1].flatten()
    final_x_gt = x_gt[:, -1].flatten()
    dx_hat = np.diff(x_hat, axis=1).flatten()
    dx_gt = np.diff(x_gt, axis=1).flatten()

    return final_x_hat, final_x_gt, dx_hat, dx_gt


def get_galton_metrics_for_pixels(observations):
    """
    :param observations: tensor of shape (frame, batch, height, width, channel)
    :return: x, centers, contours; all of shape (batch, frame)
    """
    frame, batch, height, width, channel = observations.shape
    middle = [width // 2, height // 2]

    batched_x = []
    batched_center = []
    batched_contours = []
    for b in range(batch):
        x = []
        dx = []
        contours_seq = []
        center_seq = []
        for f in range(frame):
            result = get_ball_center_and_contours(observations[f, b])
            if result is None:
                dx.append(np.nan) if f > 0 else x.append(np.nan)
                center_seq.append([])
                contours_seq.append([])
                continue

            center, contours = result
            center_seq.append(center)
            contours_seq.append(contours)
            center_x_normalized = (center[0] - middle[0]) / middle[0]
            dx.append(center_x_normalized) if f > 0 else x.append(center_x_normalized)

        x = np.insert(np.cumsum(dx), 0, x[0])
        batched_x.append(x)
        batched_center.append(center_seq)
        batched_contours.append(contours_seq)

    batched_x = np.stack(batched_x)

    return batched_x, batched_center, batched_contours


def get_ball_center_and_contours(img):
    """
    :param img: an RGB image of shape (height, width, channel) as uint8
    :return: a tuple of (center, contours) where center is a list of [x, y] and contours is a list of contours
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None
    center = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]

    return center, contours


def plot_ball_contours_and_center(img, center, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_axis_off()
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    ax.imshow(img)
    ax.scatter(center[0], center[1], c="r")
    for contour in contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], c="r", linewidth=3)
    return fig


def render_ball_contours_and_center(img, center, contours):
    len = 3
    color = (0, 255, 0)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if center:
        # Draw cross at center
        img = cv2.line(img, (center[0] - len, center[1] - len), (center[0] + len, center[1] + len), color, 1)
        img = cv2.line(img, (center[0] - len, center[1] + len), (center[0] + len, center[1] - len), color, 1)
        # img = cv2.circle(img, (center[0], center[1]), 1, tuple((255 - c for c in color)), -1)
    if contours:
        for contour in contours:
            img = cv2.drawContours(img, [contour], 0, color, 1)
    return img


def accumulate_and_log_histograms(keys, values, namespace, step):
    """
    Accumulate data from all steps in validation epoch and log histograms to wandb.
    """
    if wandb.run is not None:
        for key, value in zip(keys, values):
            value = np.concatenate(value, axis=0)
            wandb.log({f"{namespace}/{key}": wandb.Histogram(value[~np.isnan(value)])}, step=step)


def log_timeseries_plots(all_preds, truth, context_frames, namespace, step, frequency: Optional[str] = None):
    """
    Plot timeseries and log to wandb.
    :param all_preds: (samples, time, batch, feature)
    :param truth: (time, batch, feature)
    """

    if wandb.run is not None:
        batch_idx = 0  # Adjust this as needed based on the dataset 
        # In pytorch-ts repo, the following values are used - electricity: 0, solar: 3, wikipedia: 0

        preds = all_preds[:, context_frames:, batch_idx, :].detach().cpu().numpy()
        truth = truth[:, batch_idx, :].detach().cpu().numpy()
        seq_length, target_dim = truth.shape

        rows = 4
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(20, 12))
        axx = axs.ravel()

        freq_map = {
            "D": "d",
            "B": "d",
            "30min": "h",
            "H": "h",
        }
        freq_label = freq_map.get(frequency, "")

        for feature in range(min(rows * cols, target_dim)):
            ax = axx[feature]

            ax.plot(truth[:, feature], label="Ground truth trajectory", color='#008000', linewidth=2)
            median_pred = np.median(preds[:, :, feature], axis=0)
            ax.plot(range(context_frames, seq_length), median_pred, label="Median prediction", color='#0000FF', 
                    linewidth=2)

            ax.axvline(x=context_frames, color="dimgray", linestyle="--")
            ax.axvspan(context_frames, seq_length, facecolor="oldlace")

            # 90% prediction interval
            lower_90 = np.percentile(preds[:, :, feature], 5, axis=0)
            upper_90 = np.percentile(preds[:, :, feature], 95, axis=0)
            ax.fill_between(
                range(context_frames, seq_length),
                lower_90,
                upper_90,
                color="lavender",
                label="90% prediction interval",
            )

            # 50% prediction interval
            lower_50 = np.percentile(preds[:, :, feature], 25, axis=0)
            upper_50 = np.percentile(preds[:, :, feature], 75, axis=0)
            ax.fill_between(
                range(context_frames, seq_length),
                lower_50,
                upper_50,
                color="lightsteelblue",
                label="50% prediction interval",
            )

            ax.set_title(f"Feature {feature + 1}", fontsize=16, weight="bold")

            ax.set_xlim([0, seq_length - 1])
            tick_positions = np.linspace(0, seq_length - 1, num=5)
            tick_positions = np.round(tick_positions).astype(int)
            tick_positions[2] = context_frames
            if frequency == "30min":
                tick_labels = [f"{(i - context_frames) / 2:+} {freq_label}" for i in tick_positions]
            else:
                tick_labels = [f"{i - context_frames:+} {freq_label}" for i in tick_positions]
            tick_labels = [r'$t_0$' if x.startswith('+0') else x for x in tick_labels]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            # Thicker black outline
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        # Add a proxy artist for the prediction horizon background color in the legend
        from matplotlib.patches import Patch
        prediction_horizon_patch = Patch(facecolor="oldlace", label="Prediction window")

        # Legend outside the subplots
        handles, labels = ax.get_legend_handles_labels()
        handles.append(prediction_horizon_patch)
        labels.append("Prediction window")
        fig.legend(handles, labels, loc="upper center", ncols=3, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        wandb.log({f"{namespace}/timeseries": wandb.Image(plt)}, step=step if step is not 0 else None)
        plt.close(fig)


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id):
    import gym

    maze_string = gym.make(env_id).str_maze_spec
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None):
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
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        ax.scatter(trajectory[:, batch_idx, 0], trajectory[:, batch_idx, 1], c=np.arange(len(trajectory)), cmap="Reds"),
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename


if __name__ == "__main__":
    # Draw image of a circle in a white background
    img = np.ones((64, 64, 3), np.uint8) * 255
    img = cv2.circle(img, (11, 32), 8, (0, 0, 0), -1)
    img = cv2.circle(img, (50, 32), 6, (220, 220, 220), -1)

    # Get the center of the circle
    result = get_ball_center_and_contours(img)
    if result is None:
        print("No ball found")
        exit()
    center, contours = result

    # fig = plot_ball_contours_center(img, center, contours)
    img = render_ball_contours_and_center(img, center, contours)
    plt.imshow(img)
    plt.show()
