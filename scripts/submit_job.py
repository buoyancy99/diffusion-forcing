"""
This script is written for MIT's Supercloud & Satori. Adding your own cluster is easy, just search for Satori in this 
file and modify accordingly.
"""


import getpass
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import click
from colorama import Fore

# This is set below.
REPO_DIR = None

# mit has two clusters, Satori and SuperCloud
Cluster = Literal["Satori", "SuperCloud"]


@lru_cache
def detect_cluster() -> Cluster:
    info = os.popen("sinfo").read()
    if "xeon-g6-volta" in info:
        return "SuperCloud"
    if "sched_system_all_8" in info:
        return "Satori"
    raise Exception("Unable to detect cluster.")


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"


def only_on(option, *clusters: List[Cluster]):
    """Makes a click option only be triggered on certain clusters."""

    def inner_wrapper(fn):
        if detect_cluster() in clusters:
            return option(fn)
        else:
            return fn

    return inner_wrapper


@click.command()
@click.option(
    "--name",
    type=str,
    prompt=cyan("Run name"),
)
@click.option(
    "--arguments",
    type=str,
    prompt=cyan("Arguments, appended to script"),
    default="",
)
@click.option(
    "--num_gpus",
    type=int,
    default=1,
    prompt=cyan("# GPUs"),
)
@click.option(
    "--partition",
    type=str,
    default=lambda: {
        "Satori": "satori-priority-pi-sitzmann",
        "SuperCloud": "",
    }[detect_cluster()],
    prompt=cyan("Partition"),
)
@click.option(
    "--time",
    type=str,
    default=lambda: {
        "Satori": "72:00:00",
        "SuperCloud": "",
    }[detect_cluster()],
    prompt=cyan("Time"),
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=lambda: REPO_DIR / "slurm_logs",
    prompt=cyan("Workspace"),
)
@click.option(
    "--email",
    type=str,
    default=lambda: {
        "Satori": f"{getpass.getuser()}@mit.edu",
        "SuperCloud": {
            # Add yourself here!
            "boyuanc": "boyuanc@mit.edu",
        }.get(getpass.getuser(), ""),
    }[detect_cluster()],
    prompt=cyan("Email"),
)
@click.option(
    "--num_cpus",
    type=int,
    default=lambda: click.get_current_context().params.get("num_gpus") * 4,
    prompt=cyan("# CPUs"),
)
@click.option(
    "--memory",
    type=str,
    default=lambda: f"{click.get_current_context().params.get('num_gpus') * 32}G",
    prompt=cyan("Memory"),
)
@only_on(
    click.option(
        "--env_name",
        type=str,
        default="",  # fixme
        prompt=cyan("Conda environment name"),
        hidden=detect_cluster() != "Satori",
    ),
    "Satori",
)
def start_slurm_job(
    name: str,
    workspace: Path,
    email: str,
    memory: str,
    num_gpus: int,
    num_cpus: int,
    additions: str,
    time: str,
    partition: str,
    env_name: Optional[str] = None,
):
    root = workspace / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{name}"
    root.mkdir(exist_ok=True, parents=True)
    (workspace / "latest").unlink(missing_ok=True)
    (workspace / "latest").symlink_to(root, target_is_directory=True)

    # SuperCloud compute jobs don't have internet access.
    wandb_mode = {
        "Satori": "online",
        "SuperCloud": "offline",
    }[detect_cluster()]

    # SuperCloud GPUs are specified differently.
    gpu_suffix = {
        "Satori": "",
        "SuperCloud": "volta:",
    }[detect_cluster()]

    script = f'python3 -m main +name={name} wandb.mode={wandb_mode} {additions}'

    environment_setup = {
        # Satori environment setup:
        # Load CUDA, then activate the conda environment.
        "Satori": f"""source {Path.home()}/.bashrc
module load cuda/11.2
cd {REPO_DIR}
conda activate {env_name} """,
        # SuperCloud environment setup:
        # Load Anaconda (for PyTorch). Everything else is supposed to be installed via
        # pip install --user, so it should already be there.
        "SuperCloud": f"""module load anaconda/2023a
cd {REPO_DIR}
export WANDB_MODE=offline""",
    }[detect_cluster()]

    slurm_file = f"""#!/bin/bash
#SBATCH -J {name}
#SBATCH -o {root}/out_%j.out
#SBATCH -e {root}/error_%j.err
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:{gpu_suffix}{num_gpus}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --mem={memory}
{f'#SBATCH --time={time}' if time else ''}
{f'#SBATCH --partition={partition}' if partition else ''}

{environment_setup}

{script}
"""

    job_path = root / "job.slurm"
    with job_path.open("w") as f:
        f.write(slurm_file)

    os.system(f"chmod +x {job_path}")
    os.system(f"sbatch {job_path}")

    print(f"{cyan('script:')} {script}")
    if detect_cluster() == "SuperCloud":
        print(
            "Don't forget to run "
            f"{cyan('python3 scripts/wandb_daemon.py [wandb_dir]')} "
            "from a SuperCloud login node to sync this run."
        )


if __name__ == "__main__":
    REPO_DIR = Path.cwd()
    while not (REPO_DIR / ".git").exists():
        REPO_DIR = REPO_DIR.parent
        if REPO_DIR == Path("/"):
            raise Exception("Could not find repo directory!")
    start_slurm_job()
