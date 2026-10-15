"""Evaluate a promoted model against the held-out test artifact."""
import argparse
import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import wandb


def go(args: argparse.Namespace) -> None:
    """Log test metrics and compare with validation without retuning on test."""
    with wandb.init(job_type="test_model") as run:
        run.config.update(vars(args))
        model_artifact = run.use_artifact(args.mlflow_model)
        test_artifact = run.use_artifact(args.test_dataset)
        model_path = model_artifact.download()
        data = pd.read_csv(test_artifact.file())
        X_test, y_test = data.drop(columns="price"), data["price"]
        model = mlflow.sklearn.load_model(model_path)
        prediction = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, prediction))
        r2 = float(r2_score(y_test, prediction))
        validation_mae = float(model_artifact.metadata["validation_mae"])
        gap = abs(mae - validation_mae) / max(validation_mae, 1e-12)
        metrics = {"mae": mae, "r2": r2, "validation_mae": validation_mae,
                   "relative_mae_gap": gap,
                   "comparable": gap <= args.relative_tolerance,
                   "relative_tolerance": args.relative_tolerance,
                   "test_rows": len(data), "model_version": model_artifact.name,
                   "test_version": test_artifact.name}
        run.summary.update(metrics)
        path = Path("test_metrics.json")
        path.write_text(json.dumps(metrics, indent=2))
        report = wandb.Artifact("test_metrics", type="evaluation", metadata=metrics)
        report.add_file(str(path))
        run.log_artifact(report).wait()
        print(json.dumps(metrics, indent=2))
        if not metrics["comparable"]:
            raise ValueError("Test/validation MAE gap exceeds the configured tolerance. "
                             "Investigate; do not select a model using the test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow_model", type=str, required=True, help="Model with prod alias.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Held-out test artifact.")
    parser.add_argument("--relative_tolerance", type=float, required=True,
                        help="Maximum relative absolute gap between test and validation MAE.")
    go(parser.parse_args())
