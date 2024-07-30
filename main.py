"""
Main file for the project. This will create and run new experiments and load checkpoints from wandb. 
Borrowed part of the code from David Charatan and wandb.
"""

import sys
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from utils.print_utils import cyan
from utils.ckpt_utils import download_latest_checkpoint, is_run_id
from utils.cluster_utils import submit_slurm_job
from utils.distributed_utils import is_rank_zero


def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission
    from experiments import build_experiment
    from utils.wandb_utils import OfflineWandbLogger, SpaceEfficientWandbLogger

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(output_dir, target_is_directory=True)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume = cfg.get("resume", None)
        name = f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})" if resume is None else None

        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            log_model="all" if not offline else False,
            config=OmegaConf.to_container(cfg),
            id=resume,
        )
    else:
        logger = None

    # Load ckpt
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if load and not is_run_id(load):
        checkpoint_path = load
    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        checkpoint_path = Path("outputs/downloaded") / run_path / "model.ckpt"

    if checkpoint_path and is_rank_zero:
        print(f"Will load checkpoint from {checkpoint_path}")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


def run_slurm(cfg: DictConfig):
    python_args = " ".join(sys.argv[1:]) + " +_on_compute_node=True"
    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        project_root,
    )

    if "cluster" in cfg and cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
        print("Job submitted to a compute node without internet. This requires manual syncing on login node.")
        osh_command_dir = project_root / ".wandb_osh_command_dir"

        osh_proc = None
        # if click.confirm("Do you want us to run the sync loop for you?", default=True):
        osh_proc = subprocess.Popen(["wandb-osh", "--command-dir", osh_command_dir])
        print(f"Running wandb-osh in background... PID: {osh_proc.pid}")
        print(f"To kill the sync process, run 'kill {osh_proc.pid}' in the terminal.")
        print(
            f"You can manually start a sync loop later by running the following:",
            cyan(f"wandb-osh --command-dir {osh_command_dir}"),
        )

    print(
        "Once the job gets allocated and starts running, we will print a command below "
        "for you to trace the errors and outputs: (Ctrl + C to exit without waiting)"
    )
    msg = f"tail -f {slurm_log_dir}/* \n"
    try:
        while not list(slurm_log_dir.glob("*.out")) and not list(slurm_log_dir.glob("*.err")):
            time.sleep(1)
        print(cyan("To trace the outputs and errors, run the following command:"), msg)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        print(
            cyan("To trace the outputs and errors, manually wait for the job to start and run the following command:"),
            msg,
        )


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
        with open_dict(cfg):
            if cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    if "name" not in cfg:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    if resume:
        load_id = resume
    elif load and is_run_id(load):
        load_id = load
    else:
        load_id = None

    if load_id and "_on_compute_node" not in cfg:
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_latest_checkpoint(run_path, Path("outputs/downloaded"))

    if "cluster" in cfg and not "_on_compute_node" in cfg:
        print(cyan("Slurm detected, submitting to compute node instead of running locally..."))
        run_slurm(cfg)
    else:
        run_local(cfg)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
