"""Split a cleaned W&B artifact into trainval and untouched test rows."""
import argparse
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import wandb


def go(args: argparse.Namespace) -> None:
    """Publish both splits while recording their common input artifact."""
    with wandb.init(job_type="train_val_test_split") as run:
        run.config.update(vars(args))
        source = run.use_artifact(args.input)
        data = pd.read_csv(source.file())
        trainval, test = train_test_split(
            data, test_size=args.test_size, random_state=args.random_seed,
            stratify=data[args.stratify_by] if args.stratify_by != "none" else None,
        )
        assert set(trainval.index).isdisjoint(test.index)
        with tempfile.TemporaryDirectory() as directory:
            for frame, name in [(trainval, "trainval"), (test, "test")]:
                path = Path(directory) / f"{name}_data.csv"
                frame.to_csv(path, index=False)
                artifact = wandb.Artifact(
                    path.name, type=f"{name}_data", description=f"{name} split",
                    metadata={"input": source.name, "rows": len(frame),
                              "random_seed": args.random_seed},
                )
                artifact.add_file(str(path))
                run.log_artifact(artifact).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Cleaned input artifact reference.")
    parser.add_argument("test_size", type=float, help="Fraction held out for final test.")
    parser.add_argument("--random_seed", type=int, required=True, help="Split seed.")
    parser.add_argument("--stratify_by", type=str, required=True, help="Stratification column.")
    go(parser.parse_args())
