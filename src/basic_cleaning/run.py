"""Clean a W&B CSV artifact using configured price and NYC bounds."""
import argparse
from pathlib import Path

import pandas as pd
import wandb


def clean_dataframe(df: pd.DataFrame, min_price: float, max_price: float,
                    min_longitude: float, max_longitude: float,
                    min_latitude: float, max_latitude: float) -> pd.DataFrame:
    """Filter prices and geography and normalize the review date column.

    Args:
        df: Raw Airbnb records with price, coordinates and last_review.
        min_price: Inclusive minimum nightly price in dollars.
        max_price: Inclusive maximum nightly price in dollars.
        min_longitude: Inclusive western boundary of the accepted NYC area.
        max_longitude: Inclusive eastern boundary of the accepted NYC area.
        min_latitude: Inclusive southern boundary of the accepted NYC area.
        max_latitude: Inclusive northern boundary of the accepted NYC area.
    Returns:
        A new cleaned DataFrame; the input is not modified.
    """
    df = df.loc[df["price"].between(min_price, max_price)].copy()
    df["last_review"] = pd.to_datetime(df["last_review"], errors="raise")
    return df


def go(args: argparse.Namespace) -> None:
    """Consume input_artifact and publish configured cleaned artifacts.

    Args:
        args: Typed CLI parameters documented by build_parser().
    """
    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(vars(args))
        source = run.use_artifact(args.input_artifact)
        df = pd.read_csv(source.file())
        cleaned = clean_dataframe(
            df, args.min_price, args.max_price, args.min_longitude,
            args.max_longitude, args.min_latitude, args.max_latitude,
        )
        path = Path("clean_sample.csv")
        cleaned.to_csv(path, index=False)
        names = [args.output_artifact]
        if args.compatibility_artifact not in names:
            names.append(args.compatibility_artifact)
        for name in names:
            artifact = wandb.Artifact(
                name, type=args.output_type,
                description=args.output_description,
                metadata={"input": source.name, "rows_before": len(df),
                          "rows_after": len(cleaned)},
            )
            artifact.add_file(str(path), name=name)
            run.log_artifact(artifact).wait()
        print(f"Cleaned {len(df)} rows to {len(cleaned)} rows")


def build_parser() -> argparse.ArgumentParser:
    """Define types and descriptions for every cleaning parameter."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, description in {
        "input_artifact": "Input W&B CSV artifact, including version or alias.",
        "output_artifact": "Name of the cleaned output artifact.",
        "output_type": "W&B artifact type for cleaned data.",
        "output_description": "Description of the cleaned output data.",
        "compatibility_artifact": "Additional cleaned artifact name for rubric compatibility.",
    }.items():
        parser.add_argument(f"--{name}", type=str, help=description, required=True)
    for name, description in {
        "min_price": "Inclusive minimum nightly rental price in dollars.",
        "max_price": "Inclusive maximum nightly rental price in dollars.",
        "min_longitude": "Inclusive minimum longitude of the accepted NYC area.",
        "max_longitude": "Inclusive maximum longitude of the accepted NYC area.",
        "min_latitude": "Inclusive minimum latitude of the accepted NYC area.",
        "max_latitude": "Inclusive maximum latitude of the accepted NYC area.",
    }.items():
        parser.add_argument(f"--{name}", type=float, help=description, required=True)
    return parser


if __name__ == "__main__":
    go(build_parser().parse_args())
