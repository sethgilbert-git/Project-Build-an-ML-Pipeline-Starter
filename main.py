"""Hydra-configured MLflow pipeline with explicit W&B artifact lineage."""
import json
import os
import tempfile
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
_steps = ["download", "basic_cleaning", "data_check", "data_split",
          "train_random_forest"]


def test_regression_model(config: DictConfig) -> None:
    """Evaluate the manually promoted production model on held-out test data."""
    mlflow.run(
        str(ROOT / config.main.components_repository / "test_regression_model"),
        "main", env_manager=config.main.env_manager,
        parameters={
            "mlflow_model": config.artifacts.production_model,
            "test_dataset": config.artifacts.test,
            "relative_tolerance": config.modeling.comparison_relative_tolerance,
        },
    )


@hydra.main(version_base=None, config_name="config", config_path=".")
def go(config: DictConfig) -> None:
    """Execute selected steps; every component parameter comes from config."""
    os.environ["WANDB_PROJECT"] = config.main.project_name
    os.environ["WANDB_RUN_GROUP"] = config.main.experiment_name
    steps = _steps if config.main.steps == "all" else config.main.steps.split(",")
    unknown = set(steps) - set(_steps + ["test_regression_model"])
    if unknown:
        raise ValueError(f"Unknown pipeline steps: {sorted(unknown)}")
    components = ROOT / config.main.components_repository
    artifacts = config.artifacts
    env_manager = config.main.env_manager

    def execute(path: Path, parameters: dict) -> None:
        """Run and wait for one MLflow component, propagating failures."""
        mlflow.run(str(path), "main", env_manager=env_manager,
                   parameters=parameters)

    with tempfile.TemporaryDirectory(prefix="nyc-rf-") as temporary:
        if "download" in steps:
            execute(components / "get_data", {
                "sample": config.etl.sample,
                "artifact_name": artifacts.raw_name,
                "artifact_type": artifacts.raw_type,
                "artifact_description": artifacts.raw_description,
            })
        if "basic_cleaning" in steps:
            execute(ROOT / "src/basic_cleaning", {
                "input_artifact": artifacts.raw_input,
                "output_artifact": artifacts.clean_name,
                "output_type": artifacts.clean_type,
                "output_description": artifacts.clean_description,
                "compatibility_artifact": artifacts.clean_compatibility_name,
                **{key: config.etl[key] for key in [
                    "min_price", "max_price", "min_longitude", "max_longitude",
                    "min_latitude", "max_latitude",
                ]},
            })
        if "data_check" in steps:
            execute(ROOT / "src/data_check", {
                "csv": artifacts.clean_input, "ref": artifacts.reference,
                "kl_threshold": config.data_check.kl_threshold,
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
                **{key: config.etl[key] for key in [
                    "min_longitude", "max_longitude", "min_latitude", "max_latitude",
                ]},
            })
        if "data_split" in steps:
            execute(components / "train_val_test_split", {
                "input": artifacts.clean_input,
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            })
        if "train_random_forest" in steps:
            rf_config = str(Path(temporary) / "rf_config.json")
            Path(rf_config).write_text(json.dumps(
                OmegaConf.to_container(config.modeling.random_forest)))
            execute(ROOT / "src/train_random_forest", {
                "trainval_artifact": artifacts.trainval,
                "val_size": config.modeling.val_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
                "rf_config": rf_config,
                "max_tfidf_features": config.modeling.max_tfidf_features,
                "output_artifact": artifacts.model_name,
            })
        if "test_regression_model" in steps:
            test_regression_model(config)


if __name__ == "__main__":
    go()
