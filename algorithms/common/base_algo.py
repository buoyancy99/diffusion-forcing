from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig


class BaseAlgo(ABC):
    """
    A base class for generic algorithms.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug

    @abstractmethod
    def run(*args: Any, **kwargs: Any) -> Any:
        """
        Run the algorithm.
        """
        raise NotImplementedError
