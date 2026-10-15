"""Local code tests; W&B uploads are verified separately in the user's account."""
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import main
from src.basic_cleaning.run import build_parser, clean_dataframe
from src.train_random_forest.run import get_inference_pipeline
from src.train_random_forest.feature_engineering import ReviewAgeTransformer

ROOT = Path(__file__).resolve().parents[1]


def test_cleaning_keeps_inclusive_price_limits():
    rows = pd.DataFrame({"price": [10, 350, 9, 351],
                         "longitude": [-74]*4,
                         "latitude": [40.7]*4, "last_review": [None]*4})
    result = clean_dataframe(rows, 10, 350, -74.25, -73.5, 40.5, 41.2)
    assert result.price.tolist() == [10, 350]
    assert len(rows) == 4


def test_cleaning_cli_types():
    parameters = ["--input_artifact", "sample.csv:latest", "--output_artifact",
                  "clean_sample.csv", "--output_type", "clean_sample",
                  "--output_description", "Clean sample", "--compatibility_artifact",
                  "clean_data.csv", "--min_price", "10", "--max_price", "350",
                  "--min_longitude", "-74.25", "--max_longitude", "-73.5",
                  "--min_latitude", "40.5", "--max_latitude", "41.2"]
    args = build_parser().parse_args(parameters)
    assert isinstance(args.min_price, float)
    assert isinstance(args.input_artifact, str)


def test_required_preprocessing_and_model_steps():
    model, _ = get_inference_pipeline({"n_estimators": 2}, 5)
    assert list(model.named_steps) == ["preprocessor", "random_forest"]
    category_pipeline = model["preprocessor"].transformers[1][1]
    assert isinstance(category_pipeline.steps[0][1], SimpleImputer)
    assert isinstance(category_pipeline.steps[1][1], OneHotEncoder)


def test_review_reference_date_is_training_fitted():
    transform = ReviewAgeTransformer().fit([["2019-07-01"], ["2019-07-10"]])
    single = transform.transform([["2019-07-05"]])
    batch = transform.transform([["2019-07-05"], ["2019-08-01"]])
    np.testing.assert_array_equal(single, batch[:1])
    assert single[0, 0] == 5


def test_orchestrator_uses_configured_parameters(monkeypatch):
    config = OmegaConf.load(ROOT / "config.yaml")
    config.main.steps = "all"
    config.etl.min_price = 22.5
    calls = []
    monkeypatch.setattr(main.mlflow, "run", lambda uri, entry, **kwargs:
                        calls.append((Path(uri).name, kwargs)))
    main.go.__wrapped__(config)
    assert [name for name, _ in calls] == ["get_data", "basic_cleaning",
                                         "data_check", "train_val_test_split",
                                         "train_random_forest"]
    assert calls[1][1]["parameters"]["min_price"] == 22.5
    assert calls[2][1]["parameters"]["ref"] == config.artifacts.reference
    assert calls[3][1]["parameters"]["test_size"] == config.modeling.test_size
    assert all(item[1]["env_manager"] == config.main.env_manager for item in calls)


def test_production_test_is_explicit(monkeypatch):
    config = OmegaConf.load(ROOT / "config.yaml")
    calls = []
    monkeypatch.setattr(main.mlflow, "run", lambda uri, entry, **kwargs:
                        calls.append(kwargs["parameters"]))
    main.test_regression_model(config)
    assert calls == [{"mlflow_model": config.artifacts.production_model,
                      "test_dataset": config.artifacts.test,
                      "relative_tolerance": config.modeling.comparison_relative_tolerance}]


def test_unknown_steps_fail_before_running(monkeypatch):
    config = OmegaConf.load(ROOT / "config.yaml")
    config.main.steps = "typo"
    with pytest.raises(ValueError, match="Unknown"):
        main.go.__wrapped__(config)


@pytest.mark.skipif((ROOT / "VERSION").read_text().strip() == "1.0.0",
                    reason="Geographic cleaning is intentionally added in 1.0.1")
def test_geographic_filter_removes_invalid_coordinates():
    rows = pd.DataFrame({"price": [100]*3, "longitude": [-74, 0, np.nan],
                         "latitude": [40.7]*3, "last_review": [None]*3})
    result = clean_dataframe(rows, 10, 350, -74.25, -73.5, 40.5, 41.2)
    assert len(result) == 1
