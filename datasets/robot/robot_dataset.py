from typing import Sequence
import torch
import torch.nn as nn
import random
import time
import os
import numpy as np
import cv2
import pickle
from omegaconf import DictConfig
from pathlib import Path

from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import ColorJitter, Compose, RandomResizedCrop, Resize

from utils.robot_utils import pack_to_2d


class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.split = split
        self.resolution = cfg.resolution
        self.external_cond_dim = cfg.external_cond_dim
        self.n_frames = cfg.n_frames
        self.action_stack = cfg.action_stack
        self.mean_npy_path = cfg.data_mean
        self.std_npy_path = cfg.data_std
        self.relative_action = cfg.relative_action
        self.color_jitter = cfg.color_jitter
        self.random_crop = cfg.random_crop
        self.action_pad = cfg.action_pad
        self.save_dir = Path(cfg.save_dir)
        if not self.save_dir.exists():
            raise ValueError(f"save_dir {self.save_dir} does not exist")
        self.data_dirs = list(self.save_dir.glob("*"))
        self.data_dirs = sorted([d for d in self.data_dirs if d.is_dir()])
        if self.split == "validation" or self.debug:
            self.data_dirs = self.data_dirs[: int(len(self.data_dirs) * 0.05)]
        elif self.split == "training":
            self.data_dirs = self.data_dirs[int(len(self.data_dirs) * 0.05) :]
        elif self.split == "test":
            self.data_dirs = self.data_dirs[:1]  # test in on real robot, use dummy sample
        else:
            raise ValueError(f"split {self.split} not recognized")
        og_h, og_w = 480, 640
        aspect_ratio = og_w / og_h
        augmentation = []
        if self.random_crop:
            self.resize = Resize(og_h // 4, antialias=True)
            augmentation.append(
                RandomResizedCrop(self.resolution, (0.9, 1.0), (aspect_ratio, aspect_ratio), antialias=True),
            )
        else:
            self.resize = Resize((self.resolution, self.resolution), antialias=True)

        if self.color_jitter:
            augmentation.append(ColorJitter(0.1, 0.1, 0.1, 0.05))

        self.augmentation = Compose(augmentation)

        self.videos = self.read_to_ram(self.data_dirs)
        self.n_cameras = len(self.videos[0])
        if cfg.observation_shape[0] != self.n_cameras * 3 + self.action_stack:
            raise ValueError(
                f"observation_shape {cfg.observation_shape} must have channels equal to num_cameras * 3 + action_stack."
                f" However, num_cameras={self.n_cameras} and action_stack={self.action_stack}"
            )
        self.mean, self.std = self.maybe_save_data_stat()
        self.actions = self.compute_actions(self.data_dirs)

    def read_to_ram(self, data_dirs):
        raw_videos = []
        for data_dir in tqdm(data_dirs, desc="Reading videos to RAM"):
            video_paths = list(data_dir.glob("*.mp4"))
            video_paths = sorted(video_paths)
            videos = []
            for video_path in video_paths:
                video = EncodedVideo.from_path(video_path, decode_audio=False)
                raw_frames = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
                raw_frames = raw_frames.permute(1, 0, 2, 3).contiguous() / 255.0
                videos.append(self.resize(raw_frames))
            raw_videos.append(videos)
        return raw_videos

    def compute_actions(self, data_dirs):
        all_actions = []
        for data_dir in data_dirs:
            traj_path = list(data_dir.glob("*.pkl"))[0]
            with open(traj_path, "rb") as f:
                traj = pickle.load(f)
            ee_pos = np.stack(traj["ee_pos"])
            ee_quat = np.stack(traj["ee_quat"])
            ee_pos_target = np.stack(traj["ee_pos_target"])
            ee_quat_target = np.stack(traj["ee_quat_target"])
            grasp = np.stack(traj["grasp"]).astype(np.float32)
            actions = []
            for t in range(len(ee_pos)):
                if self.relative_action:
                    pos = ee_pos_target[t] - ee_pos[t]
                    quat = R.from_quat(ee_quat[t]).inv() * R.from_quat(ee_quat_target[t])
                    quat = quat.as_quat()
                else:
                    pos = ee_pos_target[t]
                    quat = ee_quat_target[t]
                action = pack_to_2d(pos, quat, grasp[t], self.resolution)
                actions.append(action)
            all_actions.append(np.stack(actions))
        return all_actions

    def __len__(self):
        return len(self.data_dirs) * 10000  # repeating dataset is hacky but easier to control steps

    def __getitem__(self, idx):
        idx = idx % len(self.data_dirs)
        drop_front = np.random.randint(self.action_stack)
        drop_back = 1

        videos = self.videos[idx]
        if self.color_jitter or self.random_crop:
            videos = [self.augmentation(v) for v in videos]
        videos = torch.cat(videos, 1)[drop_front:-drop_back]
        actions = self.actions[idx][drop_front:-drop_back]

        n_stacked = min(len(videos) // self.action_stack, self.n_frames)
        videos = videos[: n_stacked * self.action_stack : self.action_stack]
        actions = actions[: (n_stacked - 1) * self.action_stack]

        actions = actions.reshape(n_stacked - 1, self.action_stack, self.resolution, self.resolution)
        actions = torch.from_numpy(actions).float()
        actions = nn.functional.pad(actions, (0, 0, 0, 0, 0, 0, 1, 0))
        if self.action_pad == "mean":
            actions[0] = torch.from_numpy(self.mean[self.n_cameras * 3 :])
        elif self.action_pad != "zero":
            raise ValueError(f"action_pad {self.action_pad} not recognized")
        frames = torch.cat([videos, actions], axis=1)

        if len(frames) < self.n_frames:
            frames = nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, 0, self.n_frames - n_stacked))

        nonterminals = torch.zeros(self.n_frames, dtype=torch.bool)
        nonterminals[:n_stacked] = True

        return frames, nonterminals

    def maybe_save_data_stat(self):
        mean_npy_path = Path(self.mean_npy_path)
        std_npy_path = Path(self.std_npy_path)
        if not mean_npy_path.exists() or not std_npy_path.exists():
            print("Computing and saving data statistics...")
            mean_pixel = np.ones((3 * self.n_cameras, self.resolution, self.resolution)) * 0.5
            std_pixel = np.ones((3 * self.n_cameras, self.resolution, self.resolution)) * 0.5
            if self.relative_action:
                mean_action = pack_to_2d(np.zeros([0.0, 0.0, 0.0]), np.array([0, 0, 0, 1]), 0.5, self.resolution)
                std_action = pack_to_2d(np.array([0.02, 0.02, 0.02]), np.ones((3, 3)) / 2, 0.5, self.resolution)
            else:
                mean_action = pack_to_2d(
                    np.array([0.6, 0.0, 0.15]), np.array([0.3826834, 0.9238795, 0, 0]), 0.5, self.resolution
                )
                std_action = pack_to_2d(np.array([0.1, 0.1, 0.08]), np.ones((3, 3)) / 2, 0.5, self.resolution)
            mean_action = mean_action[None].repeat(self.action_stack, axis=0)
            std_action = std_action[None].repeat(self.action_stack, axis=0)

            mean = np.concatenate([mean_pixel, mean_action], axis=0)
            std = np.concatenate([std_pixel, std_action], axis=0)
            np.save(mean_npy_path, mean)
            np.save(std_npy_path, std)
        else:
            print("Loading data statistics...")
            mean = np.load(mean_npy_path)
            std = np.load(std_npy_path)

        return mean, std


if __name__ == "__main__":
    # tests robot dataset
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.resolution = 32
    cfg.external_cond_dim = 0
    cfg.n_frames = 50
    cfg.action_stack = 10
    cfg.relative_action = True
    cfg.save_dir = "data/robot_swap"
    cfg.data_mean = f"{cfg.save_dir}/mean_{cfg.action_stack}.npy"
    cfg.data_std = f"{cfg.save_dir}/std_{cfg.action_stack}.npy"
    cfg.debug = True
    cfg.observation_shape = [3 * 2 + cfg.action_stack, cfg.resolution, cfg.resolution]
    cfg.color_jitter = True
    cfg.random_crop = False

    dataset = RobotDataset(cfg, split="training")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)
    for e in range(1000):
        for batch in tqdm(dataloader):
            frames, nonterminals = batch
            time.sleep(0.2)
            # break
