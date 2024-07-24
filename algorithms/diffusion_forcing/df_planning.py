from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb
from PIL import Image

from einops import rearrange

from .df_base import DiffusionForcingBase
from .models.diffusion_transition import DiffusionTransitionModel
from utils.logging_utils import make_convergence_animation, make_mpc_animation, make_trajectory_images


class DiffusionForcingPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        import d4rl
        import gym

        self.env_id = cfg.env_id
        env = gym.make(self.env_id)
        self.action_dim = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        self.use_reward = cfg.use_reward
        self.unstacked_dim = self.observation_dim + self.action_dim + int(self.use_reward)
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.guidance_scale = cfg.guidance_scale
        self.ddim_repeats = cfg.ddim_repeats
        self.gamma = cfg.gamma
        self.repeat_observation = cfg.repeat_observation
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = cfg.observation_mean
        self.observation_std = cfg.observation_std
        self.action_mean = cfg.action_mean
        self.action_std = cfg.action_std
        self.open_loop_horizon = cfg.open_loop_horizon
        self.plot_end_points = cfg.plot_start_goal

        super().__init__(cfg)
        if self.open_loop_horizon % self.frame_stack != 0:
            raise ValueError("open_loop_horizon must be divisible by frame_stack")
        if self.chunk_size % self.frame_stack != 0:
            raise ValueError("chunk_size must be divisible by frame_stack")
        if self.context_frames % self.frame_stack != 0 and self.context_frames >= 0:
            raise ValueError("context_frames must be divisible by frame_stack")

    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        )
        self.init_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.z_shape[0]),
        )
        mean = list(self.observation_mean) * self.repeat_observation + list(self.action_mean)
        std = list(self.observation_std) * self.repeat_observation + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.register_data_mean_std(mean, std)

    def configure_optimizers(self):
        transition_params = tuple(self.transition_model.parameters()) + tuple(self.init_encoder.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def reweigh_loss(self, loss, weight=None):
        loss *= 10.0  # somehow is this very important for maze2d.
        return super().reweigh_loss(loss, weight)

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()

        rewards = rearrange(rewards, "b (t fs) -> t b fs 1", fs=self.frame_stack).contiguous()

        init_obs = observations[:, 0]
        init_z = self.init_encoder(self.split_bundle(self._normalize_x(self.make_bundle(init_obs)))[0])
        observations = observations[:, 1:]
        pad = torch.zeros_like(observations[:, -1:])
        observations = torch.cat([observations, pad], 1)
        observations = rearrange(observations, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)

        actions = rearrange(actions, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = [None for _ in range(n_frames // self.frame_stack)]

        bundles = self.make_bundle(observations, actions, rewards)
        bundles = self._normalize_x(bundles)
        bundles = bundles.flatten(2, 3)
        bundles = bundles.contiguous()

        return bundles, conditions, masks, init_z

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def validation_step(self, batch, batch_idx, namespace="validation"):
        if self.guidance_scale:
            self.interact(batch, namespace=namespace, guidance_scale=self.guidance_scale)
        self.interact(batch, namespace=namespace + "_no_guidance", guidance_scale=0.0)

    def interact(self, batch, namespace="validation", guidance_scale=0.0):
        print("Interacting with environment... This may take a couple minutes.")
        batch_size, n_frames = batch[0].shape[:2]

        r_scale = 1
        g_scale = 1

        from stable_baselines3.common.vec_env import DummyVecEnv
        import gym

        envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
        envs.seed(0)
        terminate = False
        obs = envs.reset()
        goal = np.concatenate(envs.get_attr("goal_locations"))
        goal = torch.Tensor(goal).float().to(self.device)

        obs = torch.from_numpy(obs).float().to(self.device)
        obs = self.split_bundle(self._normalize_x(self.make_bundle(obs)))[0]
        z = self.init_encoder(obs)

        # holds a list of plans during the MPC
        plan_history = []

        # holds the bundle for each time step
        trajectory = [self.make_bundle(obs)]
        steps = 0
        episode_reward = np.zeros(batch_size)
        while not terminate and steps < n_frames:
            if self.chunk_size > 0:
                unstacked_horizon = min(n_frames - steps, self.chunk_size)
            else:
                unstacked_horizon = n_frames - steps
            horizon = np.ceil(unstacked_horizon / self.frame_stack).astype(int)
            plan = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]
            last_plan = [None for _ in range(horizon)]

            plan_history_m = []
            pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
            pyramid = np.zeros((pyramid_height, horizon), dtype=int)
            for m in range(pyramid_height):
                for t in range(horizon):
                    pyramid[m, t] = m - int(t * self.uncertainty_scale)

            # the indexing is slightly different from the paper. The entry in the pyramid is not the noise level 0...K
            # but the index for noise level during DDIM denoising. For example, if there are 10 DDIM steps,
            # a value of 1 means noise level is at 0.9K before this diffusion step. A value of 10 means fully diffused

            for m in range(pyramid_height):
                guidance_consts = []

                # we draw multiple sample trajectories.
                clean_plan_samples = [[] for _ in range(horizon)]
                z_chunk = z.detach()

                with torch.enable_grad():
                    for t in range(horizon):
                        plan[t].requires_grad_(True)
                        last_plan[t] = plan[t]
                        i = max(min(pyramid[m, t], self.sampling_timesteps - 1), 0)
                        # bundle_new: less noisy version of bundle; got by forward diffuse 'denoised'
                        # z_chunk_next: z estimation for next time step
                        # clean: clean version of bundle; x0
                        # guidance_const: guidance constant for this ddim step
                        (
                            bundle_new,
                            z_chunk_next,
                            bundle_clean,
                            guidance_const,
                        ) = self.transition_model.ddim_sample_step(
                            plan[t],
                            z_chunk,
                            None,  # conditions_plan[t],
                            i,
                            return_x_start=True,
                            return_guidance_const=True,
                        )
                        clean_plan_samples[t].append(bundle_clean)
                        guidance_consts.append(guidance_const)

                        # draws more samples to get better gradient for guidance
                        bundle_temp = plan[t]
                        if guidance_scale:
                            for _ in range(self.ddim_repeats - 1):
                                _, _, bundle_clean = self.transition_model.ddim_sample_step(
                                    bundle_temp,
                                    z_chunk,
                                    None,  # conditions_plan[t],
                                    i,
                                    return_x_start=True,
                                )
                                clean_plan_samples[t].append(bundle_clean)
                                times = torch.linspace(
                                    -1,
                                    self.transition_model.num_timesteps - 1,
                                    steps=self.transition_model.sampling_timesteps + 1,
                                )
                                times = list(reversed(times.int().tolist()))
                                time_pairs = list(zip(times[:-1], times[1:]))
                                time = torch.full((batch_size,), time_pairs[i][0]).to(self.device)
                                bundle_temp = self.transition_model.q_sample(bundle_clean, time)

                        z_chunk = z_chunk_next

                        if pyramid[m, t] < self.sampling_timesteps:
                            plan[t] = bundle_new
                        else:
                            plan[t] = plan[t].detach()

                    if guidance_scale:
                        clean_plan_samples = torch.stack([torch.stack(sample) for sample in clean_plan_samples])
                        clean_plan_samples = rearrange(
                            clean_plan_samples,
                            "t r b (fs c) -> (t fs) b r c",
                            fs=self.frame_stack,
                            c=self.unstacked_dim,
                        )

                        weight = [1**j for j in range(horizon * self.frame_stack)]
                        weight = torch.from_numpy(np.array(weight)).float().to(self.device)

                        if unstacked_horizon == n_frames - steps:
                            weight[-1] = 0.0

                        # optional reward shaping so it has better numerical landscape, not used for sparse
                        # we didn't use this in paper but this could be useful for future work
                        episode_return_dense = r_scale * clean_plan_samples[..., -1] * weight[:, None, None]
                        episode_return_dense = episode_return_dense.sum([0, 1]).mean() * 4.0

                        # calculate negative distance to goal, so we can guide by goal like original diffusion planning paper
                        unnormalized_plan_samples = self._unnormalize_x(clean_plan_samples)
                        dist = torch.linalg.norm(unnormalized_plan_samples[..., :2] - goal[None, :, None], dim=-1)
                        episode_return_sparse = -(g_scale * n_frames * dist * weight[:, None, None]).sum(1).mean()
                        episode_return = episode_return_dense if self.use_reward else episode_return_sparse

                        # guidance by episode return
                        grads = torch.autograd.grad([episode_return], last_plan, allow_unused=True)
                        for t in range(horizon):
                            plan[t] = plan[t] + guidance_consts[t] * grads[t] * guidance_scale

                plan_processed = torch.stack(plan)
                plan_processed = rearrange(
                    plan_processed, "t b (fs c) -> (t fs) b c", fs=self.frame_stack, c=self.unstacked_dim
                )
                plan_processed = self._unnormalize_x(plan_processed).cpu()
                plan_history_m.append(plan_processed)

            plan_history.append(plan_history_m)

            # interact with the environment for self.open_loop_horizon steps
            for bundle in plan_history[-1][-1][: self.open_loop_horizon]:
                _, action, _ = self.split_bundle(bundle)
                action = torch.clamp(action, -1, 1)
                obs, reward, done, info = envs.step(np.nan_to_num(action.detach().cpu().numpy()))
                episode_reward += reward
                if done.any():
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [obs, reward, done]]
                bundle = self.make_bundle(obs, action, reward[..., None]).to(self.device)
                trajectory.append(self._normalize_x(bundle))
            steps = len(trajectory[1:])

            # done, exit environment interaction
            if terminate:
                break

            # update z with specifies context frames; if context_frames is 0, we will be fully markovian
            if self.context_frames < 0:
                # use full context
                context_frames = steps
            else:
                context_frames = min(self.context_frames, steps)
                context_frames = context_frames // self.frame_stack * self.frame_stack
            z = self.init_encoder(self.split_bundle(trajectory[-context_frames - 1])[0])
            if context_frames > 0:
                context = torch.stack(trajectory[-context_frames:])
                context = context.view(context_frames // self.frame_stack, self.frame_stack, batch_size, -1)
                context = context.permute(0, 2, 1, 3).contiguous()
                context = context.reshape(context_frames // self.frame_stack, batch_size, -1)
                context_condition = None  # conditions[steps - context_frames : steps]
                for bundle, condition in zip(context, context_condition):
                    z, _, _, _ = self.transition_model(z, bundle, condition, deterministic_t=0)

        episode_reward = np.mean(episode_reward)
        self.log(f"{namespace}/episode_reward", episode_reward)
        print(f"Guidance: {guidance_scale:.2f}, Episode reward: {episode_reward:.1f}")

        # Visualization
        trajectory = torch.stack(trajectory)[:-1]  # last observation is dummy
        trajectory = self._unnormalize_x(trajectory).cpu().numpy()
        goal = goal.cpu().numpy()

        start = trajectory[0, :, :2].tolist()
        goal = goal[:, :2].tolist()

        images = make_trajectory_images(self.env_id, trajectory, batch_size, start, goal)
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_visualization/sample_{i}",
                Image.fromarray(img),
            )

        if self.debug:
            # save diffusion convergence and MPC animations, very slow!
            if "medium" in self.env_id:
                indicies = [2, 4]  # choose start/goal with relatively long distances (harder)
            else:
                indicies = [2, 7]  # choose start/goal with relatively long distances (harder)

            for i in indicies:
                filename = make_convergence_animation(
                    self.env_id,
                    plan_history,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"convergence/{namespace}_{i}": wandb.Video(filename, fps=4),
                        f"trainer/global_step": self.global_step,
                    }
                )

                filename = make_mpc_animation(
                    self.env_id,
                    plan_history,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"mpc/{namespace}_{i}": wandb.Video(filename, fps=24),
                        f"trainer/global_step": self.global_step,
                    }
                )
