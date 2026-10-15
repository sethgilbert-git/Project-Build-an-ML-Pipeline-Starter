"""Train, validate and export a Random Forest inference pipeline to W&B."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
import wandb

try:
    from .feature_engineering import ReviewAgeTransformer, reshape_to_1d
except ImportError:
    from feature_engineering import ReviewAgeTransformer, reshape_to_1d


def get_inference_pipeline(rf_config: dict, max_tfidf_features: int):
    """Build preprocessing and regressor; fit both only on training rows."""
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder(
        categories=[["Shared room", "Private room", "Entire home/apt"]],
        handle_unknown="use_encoded_value", unknown_value=-1,
    )
    # Required non-ordinal preprocessing: imputation followed by one-hot encoding.
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    zero_imputed = ["minimum_nights", "number_of_reviews", "reviews_per_month",
                    "calculated_host_listings_count", "availability_365",
                    "longitude", "latitude"]
    date_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="2010-01-01"),
        ReviewAgeTransformer(),
    )
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        FunctionTransformer(reshape_to_1d, validate=False),
        TfidfVectorizer(max_features=max_tfidf_features, stop_words="english"),
    )
    preprocessor = ColumnTransformer([
        ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
        ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0), zero_imputed),
        ("transform_date", date_imputer, ["last_review"]),
        ("transform_name", name_tfidf, ["name"]),
    ], remainder="drop")
    # Required inference pipeline: named preprocessing and Random Forest steps.
    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", RandomForestRegressor(**rf_config)),
    ])
    return sk_pipe, ["room_type", "neighbourhood_group"] + zero_imputed + ["last_review", "name"]


def split_train_validation(data: pd.DataFrame, val_size: float,
                           random_seed: int, stratify_by: str):
    """Split trainval rows; price is never included in model features."""
    X = data.drop(columns="price")
    y = data["price"]
    stratify = X[stratify_by] if stratify_by != "none" else None
    return train_test_split(X, y, test_size=val_size,
                            random_state=random_seed, stratify=stratify)


def save_pipeline(sk_pipe: Pipeline, path: str, X_example: pd.DataFrame) -> None:
    """Export a complete MLflow sklearn model and custom transformer code."""
    # Required MLflow model export, including input example and signature.
    mlflow.sklearn.save_model(
        sk_model=sk_pipe, path=path, serialization_format="cloudpickle",
        code_paths=[str(Path(__file__).with_name("feature_engineering.py"))],
        input_example=X_example,
        signature=infer_signature(X_example, sk_pipe.predict(X_example)),
        pip_requirements=["scikit-learn==1.7.2", "pandas==2.3.2", "numpy==2.2.6"],
    )


def plot_feature_importance(pipe, feat_names):
    """Aggregate one-hot and TF-IDF importance to their source features."""
    preprocessor = pipe.named_steps["preprocessor"]
    importance = pipe.named_steps["random_forest"].feature_importances_
    indices = preprocessor.output_indices_
    values = [float(importance[indices["ordinal_cat"]].sum()),
              float(importance[indices["non_ordinal_cat"]].sum())]
    values.extend(importance[indices["impute_zero"]].tolist())
    values.extend([float(importance[indices["transform_date"]].sum()),
                   float(importance[indices["transform_name"]].sum())])
    fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(values)
    ax.barh(np.asarray(feat_names)[order], np.asarray(values)[order])
    ax.set_xlabel("Sum of impurity-based feature importance")
    fig.tight_layout()
    return fig


def go(args: argparse.Namespace) -> None:
    """Consume trainval data, log validation MAE/R2 and publish an MLflow model."""
    with wandb.init(job_type="train_random_forest") as run:
        run.config.update(vars(args))
        rf_config = json.loads(Path(args.rf_config).read_text())
        rf_config["random_state"] = args.random_seed
        run.config.update(rf_config)
        source = run.use_artifact(args.trainval_artifact)
        run.config.update({"trainval_version": source.name})
        data = pd.read_csv(source.file())
        X_train, X_val, y_train, y_val = split_train_validation(
            data, args.val_size, args.random_seed, args.stratify_by,
        )
        sk_pipe, features = get_inference_pipeline(rf_config, args.max_tfidf_features)
        # Required fit: preprocessing and forest learn on training rows only.
        sk_pipe.fit(X_train, y_train)
        predictions = sk_pipe.predict(X_val)
        mae = float(mean_absolute_error(y_val, predictions))
        r_squared = float(r2_score(y_val, predictions))
        # Required metrics: MAE is in dollars; smaller MAE is better.
        run.summary["mae"] = mae
        run.summary["r2"] = r_squared
        run.summary["train_rows"] = len(X_train)
        run.summary["validation_rows"] = len(X_val)
        export_path = Path("random_forest_dir") / run.id
        save_pipeline(sk_pipe, str(export_path), X_train.iloc[:5])
        artifact = wandb.Artifact(
            args.output_artifact, type="model_export",
            description="MLflow sklearn preprocessing and Random Forest model",
            metadata={**rf_config, "validation_mae": mae, "validation_r2": r_squared,
                      "trainval_version": source.name},
        )
        artifact.add_dir(str(export_path))
        run.log_artifact(artifact).wait()
        fig = plot_feature_importance(sk_pipe, features)
        run.log({"feature_importance": wandb.Image(fig)})
        plt.close(fig)
        print(f"Validation MAE: {mae:.6f} | R2: {r_squared:.6f}")


def build_parser() -> argparse.ArgumentParser:
    """Declare typed, documented component CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, description in {
        "trainval_artifact": "Training/validation W&B CSV artifact reference.",
        "stratify_by": "Column for stratification, or none.",
        "rf_config": "Path to a JSON RandomForestRegressor configuration.",
        "output_artifact": "Name of the exported W&B model artifact.",
    }.items():
        parser.add_argument(f"--{name}", type=str, help=description, required=True)
    parser.add_argument("--val_size", type=float, required=True,
                        help="Fraction of trainval rows used for validation.")
    parser.add_argument("--random_seed", type=int, required=True,
                        help="Seed for both splitting and Random Forest training.")
    parser.add_argument("--max_tfidf_features", type=int, required=True,
                        help="Maximum number of title TF-IDF features.")
    return parser


if __name__ == "__main__":
    go(build_parser().parse_args())
