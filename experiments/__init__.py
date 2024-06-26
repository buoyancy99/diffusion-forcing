from typing import Optional, Union
from omegaconf import DictConfig
import pathlib
from lightning.pytorch.loggers.wandb import WandbLogger

from .exp_base import BaseExperiment
from .exp_prediction import SequencePredictionExperiment
from .exp_planning import POMDPExperiment
from .exp_video import VideoPredictionExperiment
from .exp_robot import RobotExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment' without .yaml suffix
exp_registry = dict(
    exp_prediction=SequencePredictionExperiment,
    exp_planning=POMDPExperiment,
    exp_video=VideoPredictionExperiment,
    exp_robot=RobotExperiment,
)


def build_experiment(
    cfg: DictConfig, logger: Optional[WandbLogger] = None, ckpt_path: Optional[Union[str, pathlib.Path]] = None
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)
