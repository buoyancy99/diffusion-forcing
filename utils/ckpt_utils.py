from pathlib import Path
import wandb


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"


def rewrite_compiled_ckpt(ckpt_path: Path):
    """Rewrite a checkpoint to remove the torch.compile influence"""
    import torch

    ckpt = torch.load(ckpt_path)
    need_conversion = False
    for k in ckpt["state_dict"].keys():
        if "_orig_mod." in k:
            need_conversion = True

    if need_conversion:
        print("Rewriting compiled checkpoint...")
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
        torch.save(ckpt, ckpt_path)
