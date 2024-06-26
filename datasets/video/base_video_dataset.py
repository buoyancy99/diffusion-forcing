from typing import Sequence
import torch
import random
import numpy as np
import cv2
from omegaconf import DictConfig
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod, ABC
import json


class BaseVideoDataset(torch.utils.data.Dataset, ABC):
    """
    Base class for video datasets. Videos may be of variable length.

    Folder structure of each dataset:
    - [save_dir] (specified in config, e.g., data/phys101)
        - /[split] (one per split)
            - /data_folder_name (e.g., videos)
            metadata.json
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.external_cond_dim = cfg.external_cond_dim
        self.n_frames = cfg.n_frames if split == "training" else cfg.n_frames * cfg.validation_multiplier
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.split_dir = self.save_dir / f"{split}"

        self.metadata_path = self.save_dir / "metadata.json"

        if not self.metadata_path.exists():
            # Build dataset
            print(f"Creating dataset in {self.save_dir}...")
            self.download_dataset()  # if you downloaded data manually, comment this out
            json.dump(
                {"training": self.get_data_lengths("training"), "validation": self.get_data_lengths("validation")},
                open(self.metadata_path, "w"),
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        self.data_paths = self.get_data_paths(self.split)
        self.clips_per_video = np.clip(np.array(self.metadata[split]) - self.n_frames + 1, a_min=1, a_max=None).astype(
            np.int32
        )
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)
        self.transform = transforms.Resize((self.resolution, self.resolution), antialias=True)

        # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
        random.seed(0)
        self.idx_remap = list(range(self.__len__()))
        random.shuffle(self.idx_remap)

    @abstractmethod
    def download_dataset(self) -> Sequence[int]:
        """
        Download dataset from the internet and build it in save_dir

        Returns a list of video lengths
        """
        raise NotImplementedError

    @abstractmethod
    def get_data_paths(self, split):
        """Return a list of data paths (e.g. xxx.mp4) for a given split"""
        raise NotImplementedError

    def get_data_lengths(self, split):
        """Return a list of num_frames for each data path (e.g. xxx.mp4) for a given split"""
        lengths = []
        for path in self.get_data_paths(split):
            length = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT)
            lengths.append(length)
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    @staticmethod
    def load_video(path: Path):
        """
        Load video from a path
        :param filename: path to the video
        :return: video as a numpy array
        """

        cap = cv2.VideoCapture(str(path))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break

        cap.release()
        frames = np.stack(frames, dtype=np.uint8)
        return np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)

    @staticmethod
    def load_image(filename: Path):
        """
        Load image from a path
        :param filename: path to the image
        :return: image as a numpy array
        """
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.transpose(image, (2, 0, 1))

    def __len__(self):
        return self.clips_per_video.sum()

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        video_idx, frame_idx = self.split_idx(idx)
        video_path = self.data_paths[video_idx]
        video = self.load_video(video_path)[frame_idx : frame_idx + self.n_frames]

        pad_len = self.n_frames - len(video)

        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            nonterminal[-pad_len:] = 0

        video = torch.from_numpy(video / 256.0).float()
        video = self.transform(video)

        if self.external_cond_dim:
            external_cond = np.load(self.condition_dir / f"{video_path.name.replace('.mp4', '.npy')}")
            if len(external_cond) < self.n_frames:
                external_cond = np.pad(external_cond, ((0, pad_len),))
            external_cond = torch.from_numpy(external_cond).float()

            return video, external_cond, nonterminal
        else:
            return video, nonterminal
