import os
from typing import Optional, Union
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

import torch
from pathlib import Path

from algorithms.diffusion_forcing import DiffusionForcingPrediction
from .exp_base import BaseLightningExperiment

from datasets import GluontsDataset
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.torch.batchify import batchify


class SequencePredictionExperiment(BaseLightningExperiment):
    """
    A Sequence Prediction experiment
    """

    compatible_algorithms = dict(
        df_prediction=DiffusionForcingPrediction,
    )

    compatible_datasets = dict(
        # Timeseries datasets
        ts_exchange=GluontsDataset,
        ts_solar=GluontsDataset,
        ts_wikipedia=GluontsDataset,
        ts_taxi=GluontsDataset,
        ts_electricity=GluontsDataset,
        ts_traffic=GluontsDataset,
    )

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, LightningDataModule, TrainDataLoader]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if train_dataset:
            if isinstance(train_dataset, GluontsDataset):
                train_dataloader = TrainDataLoader(
                    train_dataset.dataset,
                    transform=train_dataset.transformation,
                    batch_size=self.cfg.training.batch_size,
                    stack_fn=batchify,
                    num_batches_per_epoch=100,
                    shuffle_buffer_length=2048,
                )
                return train_dataloader
            else:
                return torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.cfg.training.batch_size,
                    num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                    shuffle=shuffle,
                )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, LightningDataModule, InferenceDataLoader]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            if isinstance(validation_dataset, GluontsDataset):
                return InferenceDataLoader(
                    validation_dataset.dataset,
                    transform=validation_dataset.transformation,
                    batch_size=self.cfg.validation.batch_size,
                    stack_fn=batchify,
                )
            else:
                return torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=self.cfg.validation.batch_size,
                    num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                    shuffle=shuffle,
                )
        else:
            return None

    def _build_test_loader(self) -> Optional[InferenceDataLoader]:
        test_dataset = self._build_dataset("test")
        shuffle = (
            False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.validation.data.shuffle
        )
        if test_dataset:
            if isinstance(test_dataset, GluontsDataset):
                return InferenceDataLoader(
                    test_dataset.dataset,
                    transform=test_dataset.transformation,
                    batch_size=self.cfg.test.batch_size,
                    stack_fn=batchify,
                )
            else:
                return torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.cfg.validation.batch_size,
                    num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                    shuffle=shuffle,
                )
        else:
            return None

    def save_mean_std_metadata(self) -> None:
        """
        Save mean, std, and metadata of the timeseries datasets in data folder
        """
        if "ts" not in self.root_cfg.dataset._name:
            return
        cached = True
        if isinstance(self.root_cfg.dataset.data_mean, str):
            cached = cached and Path(self.root_cfg.dataset.data_mean).exists()
        if isinstance(self.root_cfg.dataset.data_std, str):
            cached = cached and Path(self.root_cfg.dataset.data_std).exists()
        if isinstance(self.root_cfg.dataset.metadata, str):
            cached = cached and Path(self.root_cfg.dataset.metadata).exists()
        if not cached:
            _ = self._build_dataset("training")
