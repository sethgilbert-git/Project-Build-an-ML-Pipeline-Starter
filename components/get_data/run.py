"""Publish a provided raw data sample to W&B."""
import argparse
from pathlib import Path

import wandb


def go(args: argparse.Namespace) -> None:
    """Upload the selected bundled sample as a raw_data artifact."""
    path = Path(__file__).resolve().parent / "data" / args.sample
    if args.sample not in {"sample1.csv", "sample2.csv"}:
        raise ValueError("Only bundled sample1.csv or sample2.csv is supported.")
    with wandb.init(job_type="download_file") as run:
        run.config.update(vars(args))
        artifact = wandb.Artifact(args.artifact_name, type=args.artifact_type,
                                  description=args.artifact_description)
        artifact.add_file(str(path), name=args.artifact_name)
        run.log_artifact(artifact).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["sample", "artifact_name", "artifact_type", "artifact_description"]:
        parser.add_argument(name, type=str, help=name.replace("_", " "))
    go(parser.parse_args())
