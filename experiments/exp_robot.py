from datasets import RobotDataset
from algorithms.diffusion_forcing import DiffusionForcingRobot
from .exp_base import BaseLightningExperiment


class RobotExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_robot=DiffusionForcingRobot,
    )

    compatible_datasets = dict(
        robot_swap=RobotDataset,
    )

    def save_data_stat(self) -> None:
        self._build_dataset("test")
