import torch
from einops import repeat, rearrange

from omegaconf import DictConfig
import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import List
import json

from utils.logging_utils import log_timeseries_plots
from algorithms.common.metrics import crps_quantile_sum
from .df_base import DiffusionForcingBase


class DiffusionForcingPrediction(DiffusionForcingBase):
    """
    Time series prediction model using diffusion forcing
    """

    def __init__(self, cfg: DictConfig):
        self.use_covariates = cfg.use_covariates

        with open(cfg.metadata, "r") as f:
            metadata = json.load(f)

        if "target_dimension" in metadata:
            cfg.x_shape = [metadata["target_dimension"]]

        if "context_length" in metadata:
            cfg.context_frames = metadata["context_length"]

        if "lags_seq" in metadata:
            lags_seq = metadata["lags_seq"]
            self.shifted_lags_seq = [x - 1 for x in lags_seq]
            if self.use_covariates:
                self.external_cond_dim = len(lags_seq) * cfg.x_shape[0] + metadata["time_features_dim"] * 2
            else:
                self.external_cond_dim = 0

        self.prediction_length = metadata["prediction_length"]
        self.history_length = metadata["history_length"]
        self.frequency = metadata["frequency"]

        super().__init__(cfg)

    def _build_model(self):
        super()._build_model()
        self.embed = nn.Embedding(num_embeddings=self.x_shape[0], embedding_dim=1) if self.use_covariates else None

    def training_step(self, batch, batch_idx):
        batch = list(self.get_observations_from_gluonts_dataset(batch))
        nonterminals = torch.ones(batch[0].shape[0], batch[0].shape[1]).to(self.device)
        batch.append(nonterminals)

        return super().training_step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        batch = list(self.get_observations_from_gluonts_dataset(batch))

        if self.calc_crps_sum:
            batch = [repeat(d, "b ... -> (c b) ...", c=self.calc_crps_sum) for d in batch if d is not None]

        nonterminals = torch.ones(batch[0].shape[0], batch[0].shape[1]).to(self.device)
        batch.append(nonterminals)

        return super().validation_step(batch, batch_idx, namespace=namespace)

    def on_validation_epoch_end(self, namespace="validation", log_visualizations=False) -> None:
        if not self.validation_step_outputs:
            return

        # multiple trajectories sampled, compute CRPS_sum and visualize trajectories
        if self.calc_crps_sum:
            all_preds = []
            all_gt = []
            for pred, gt in self.validation_step_outputs:
                all_preds.append(pred.view(pred.shape[0], self.calc_crps_sum, -1, *pred.shape[2:]))
                all_gt.append(gt.view(gt.shape[0], self.calc_crps_sum, -1, *gt.shape[2:]))
            all_preds = torch.cat(all_preds, 2).float().permute(1, 0, 2, 3)
            gt = torch.cat(all_gt, 2).float()[:, 0]
            crps_sum_val = crps_quantile_sum(all_preds[:, self.context_frames :], gt[self.context_frames :])
            self.min_crps_sum = min(self.min_crps_sum, crps_sum_val)
            self.log_dict(
                {f"{namespace}/crps_sum": crps_sum_val, f"{namespace}/min_crps_sum": self.min_crps_sum},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            if log_visualizations:
                log_timeseries_plots(
                    all_preds, gt, self.context_frames, namespace, self.trainer.global_step, self.frequency
                )
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.min_crps_sum = float("inf")  # reset min CRPS_sum
        self.on_validation_epoch_end(namespace="test", log_visualizations=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> tuple:
        """
        Adapted from source:
        https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/time_grad/time_grad_network.py

        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        observations = None
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            if lag_index == 0:
                observations = sequence[:, begin_index:end_index, ...]
            else:
                lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return observations, torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def get_observations_from_gluonts_dataset(self, batch):
        """
        Process batch of GLuonTS time series dataset to get inputs for the model. This includes
        standardizing the data and adding covariates if applicable. Covariates adapted from:
        https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/time_grad/time_grad_network.py
        """

        past_time_feat = batch["past_time_feat"]
        future_time_feat = batch["future_time_feat"]
        past_target_cdf = batch["past_target_cdf"]
        future_target_cdf = batch["future_target_cdf"]
        target_dimension_indicator = batch["target_dimension_indicator"]

        target_dim = past_target_cdf.shape[-1]
        assert target_dim == self.x_shape[0]

        if not self.use_covariates:
            if 0 in future_target_cdf.shape:
                # validation
                sequence = past_target_cdf
            else:
                # training
                sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            observations = sequence[:, -(self.context_frames + self.prediction_length) :, ...]
            return observations, None

        if 0 in future_target_cdf.shape:
            # validation and test
            time_feat = past_time_feat[:, -(self.context_frames + self.prediction_length) :, ...]
            sequence = past_target_cdf
        else:
            # training
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_frames :, ...], future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
        sequence_length = self.history_length + self.prediction_length
        subsequences_length = self.context_frames + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        observations, lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.shifted_lags_seq,
            subsequences_length=subsequences_length,
        )

        num_input_lags = len(self.shifted_lags_seq) - 1

        # (batch_size, target_dim, embed_dim=1)
        index_embeddings = self.embed(target_dimension_indicator)

        # (batch_size, seq_len, target_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, subsequences_length, -1, -1)
            .reshape((-1, subsequences_length, target_dim))
        )

        # (batch_size, subsequences_length, input_dim)
        normalized_input_lags = self._normalize_x(rearrange(lags, "b t c l -> b t l c"))
        normalized_input_lags = rearrange(normalized_input_lags, "b t l c -> b t (c l)")
        normalized_covariates = torch.cat((normalized_input_lags, repeated_index_embeddings, time_feat), dim=-1)
        return observations, normalized_covariates
