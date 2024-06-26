import os
from pathlib import Path

import click


@click.command()
@click.argument(
    "dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
def wandb_daemon(dir: Path):
    while True:
        for run_dir in dir.iterdir():
            if not run_dir.is_dir():
                continue
            os.system(f"wandb sync --include-offline {run_dir}")


if __name__ == "__main__":
    wandb_daemon()
