import torch
import numpy as np
import json
import pandas as pd

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from omegaconf import DictConfig
from pathlib import Path

from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

from .covariates import fourier_time_features_from_frequency, lags_for_fourier_time_features_from_frequency


class GluontsDataset(torch.utils.data.Dataset):
    """
    Base class for gluonts datasets.
    """

    def __init__(self, cfg: DictConfig, split="training"):
        super().__init__()
        self.split = split

        self.train_sampler = None
        self.validation_sampler = TestSplitSampler()
        self.test_sampler = TestSplitSampler()
        self.metadata_path = Path(cfg.metadata)

        self.dataset, self.transformation = self.get_gluonts_dataset(cfg.gluonts_name)

        if self.split == "training":
            data_mean_path = Path(cfg.data_mean)
            data_std_path = Path(cfg.data_std)
            data_mean_path.parent.mkdir(parents=True, exist_ok=True)
            data_std_path.parent.mkdir(parents=True, exist_ok=True)
            if not data_mean_path.exists() or not data_std_path.exists():
                torch_dataset = torch.from_numpy(self.dataset[0]["target"]).float()
                np.save(data_mean_path, torch.mean(torch_dataset, dim=1).numpy())
                np.save(data_std_path, torch.clamp(torch.std(torch_dataset, dim=1), min=1e-7).numpy())

    def get_gluonts_dataset(self, gluonts_name):
        raw_dataset = get_dataset(gluonts_name, regenerate=False)

        if not self.metadata_path.exists():
            prediction_length = raw_dataset.metadata.prediction_length
            context_length = prediction_length
            frequency = raw_dataset.metadata.freq
            lags_seq = lags_for_fourier_time_features_from_frequency(freq_str=frequency)
            time_features_dim = len(fourier_time_features_from_frequency(freq_str=frequency))
            history_length = prediction_length + max(lags_seq)
            target_dimension = min(2000, int(raw_dataset.metadata.feat_static_cat[0].cardinality))
            num_test_dates = int(len(raw_dataset.test) / len(raw_dataset.train))
            metadata = {
                "prediction_length": prediction_length,
                "context_length": context_length,
                "history_length": history_length,
                "frequency": frequency,
                "lags_seq": lags_seq,
                "time_features_dim": time_features_dim,
                "target_dimension": target_dimension,
                "num_test_dates": num_test_dates,
            }
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            setattr(self, k, v)

        if self.split == "training":
            dataset_grouper = MultivariateGrouper(max_target_dim=self.target_dimension)
            training_data = dataset_grouper(raw_dataset.train)
            self.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_past=self.history_length,
                min_future=self.prediction_length,
            )
            return self._process_training_split(training_data)
        elif self.split == "validation":
            dataset_grouper = MultivariateGrouper(max_target_dim=self.target_dimension)
            validation_data = dataset_grouper(raw_dataset.train)
            return self._process_validation_split(validation_data)
        elif self.split == "test":
            dataset_grouper = MultivariateGrouper(
                num_test_dates=self.num_test_dates, max_target_dim=self.target_dimension
            )
            test_data = dataset_grouper(raw_dataset.test)
            return self._process_test_split(test_data)
        else:
            raise ValueError(f"Split {self.split} not supported.")

    def create_transformation(self) -> Transformation:
        time_features = fourier_time_features_from_frequency(self.frequency)

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": self.test_sampler,
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length if mode == "training" else self.history_length + self.prediction_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def _process_training_split(self, data):
        transform = self.create_transformation() + self.create_instance_splitter(mode="training")
        return data, transform

    def _process_validation_split(self, data):
        def get_time_delta(start):
            frequency_map = {
                "D": pd.Timedelta(days=start),
                "B": pd.offsets.BDay(n=start),
                "H": pd.Timedelta(hours=start),
                "T": pd.Timedelta(minutes=start),
                "30min": pd.Timedelta(minutes=30 * start),
            }
            return frequency_map.get(self.frequency, self.frequency)

        # Randomly select self.num_test_dates blocks, each of length self.history_length + self.prediction_length
        # from the training data to form the validation set

        block_length = self.history_length + self.prediction_length
        num_blocks = self.num_test_dates
        num_time_points = data[0]["target"].shape[1] - block_length
        start_points = np.random.choice(num_time_points, num_blocks, replace=False)

        validation_blocks = []
        for start in start_points:
            block_data = {
                "target": data[0]["target"][:, start : start + block_length],
                "start": data[0]["start"] + get_time_delta(start),
                "feat_static_cat": data[0]["feat_static_cat"],
            }
            validation_blocks.append(block_data)

        # Process in the same way as the test split
        return self._process_test_split(validation_blocks)

    def _process_test_split(self, data):
        for x in data:
            # Append zeros to target, such that test_data.input includes the real targets too
            x["target"] = np.hstack((x["target"], np.zeros((x["target"].shape[0], self.prediction_length))))

        window_length = self.prediction_length
        _, test_template = split(data, offset=-window_length)
        test_data = test_template.generate_instances(window_length)
        data = test_data.input
        transform = self.create_transformation() + self.create_instance_splitter(mode="test")
        return data, transform


if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader, InferenceDataLoader
    from gluonts.torch.batchify import batchify

    # you may need to
    # export PYTHONPATH=$PYTHONPATH:/path/to/diffusion_forcing

    os.chdir("../..")
    dataset_split = "validation"  # training, validation, test
    batch_size = 32
    cfg = MagicMock()
    cfg.gluonts_name = "taxi_30min"
    cfg.metadata = f"data/timeseries/{cfg.gluonts_name}/metadata.json"
    cfg.data_mean = f"data/timeseries/{cfg.gluonts_name}/data_mean.npy"
    cfg.data_std = f"data/timeseries/{cfg.gluonts_name}/data_std.npy"
    dataset = GluontsDataset(cfg, split=dataset_split)

    if dataset_split == "training":
        dataloader = TrainDataLoader(
            dataset.dataset,
            transform=dataset.transformation,
            batch_size=batch_size,
            stack_fn=batchify,
            num_batches_per_epoch=100,
            shuffle_buffer_length=2048,
        )
    elif (dataset_split == "validation") or (dataset_split == "test"):
        dataloader = InferenceDataLoader(
            dataset.dataset,
            transform=dataset.transformation,
            batch_size=batch_size,
            stack_fn=batchify,
        )
    else:
        raise ValueError(f"Split {dataset_split} not supported.")

    for data_entry in dataloader:
        print(data_entry.keys())
        import ipdb

        ipdb.set_trace()
