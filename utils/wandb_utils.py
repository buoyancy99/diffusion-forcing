from pathlib import Path
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Union
from typing_extensions import override
from functools import wraps
import os
from wandb_osh.hooks import TriggerWandbSyncHook
import time
from lightning.pytorch.loggers.wandb import WandbLogger, _scan_checkpoints, ModelCheckpoint, Tensor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH


if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of creating new version.
    A variable expiration_days can be set to control how long older versions of artifacts are kept.
    By default, the latest version is kept indefinitely, while older versions are kept for 5 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"

            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        for artifact in self._last_artifacts:
            if not self._offline:
                artifact.wait()
            artifact.ttl = timedelta(days=self.expiration_days)
            artifact.save()
        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out
