from datasets.video import DmlabVideoDataset, MinecraftVideoDataset
from algorithms.diffusion_forcing import DiffusionForcingVideo
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
    )

    compatible_datasets = dict(
        # video datasets
        video_dmlab=DmlabVideoDataset,
        video_minecraft=MinecraftVideoDataset,
    )
