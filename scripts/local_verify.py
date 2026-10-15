"""Run real local checks and a four-job sweep without claiming W&B evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.basic_cleaning.run import clean_dataframe
from src.train_random_forest.run import (
    get_inference_pipeline, save_pipeline, split_train_validation,
)


def main():
    """Verify sample1, expected sample2 failure, fixed sample2 and model reload."""
    output = ROOT / "local_verification"
    output.mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    settings = config["etl"]
    bounds = {key: settings[key] for key in [
        "min_price", "max_price", "min_longitude", "max_longitude",
        "min_latitude", "max_latitude",
    ]}
    raw1 = pd.read_csv(ROOT / "components/get_data/data/sample1.csv")
    raw2 = pd.read_csv(ROOT / "components/get_data/data/sample2.csv")
    clean1 = clean_dataframe(raw1, **bounds)
    clean2 = clean_dataframe(raw2, **bounds)
    baseline2 = raw2.loc[raw2.price.between(settings["min_price"],
                                          settings["max_price"])].copy()
    baseline2["last_review"] = pd.to_datetime(baseline2["last_review"])
    paths = {}
    for name, data in [("sample1", clean1), ("sample2_before_fix", baseline2),
                       ("sample2_after_fix", clean2)]:
        path = output / f"{name}.csv"
        data.to_csv(path, index=False)
        paths[name] = path
    for name in paths:
        command = [sys.executable, "-m", "pytest", "src/data_check/test_data.py",
                   "-v", "--color=no", "--csv", str(paths[name]),
                   "--ref", str(paths["sample1"]), "--kl_threshold",
                   str(config["data_check"]["kl_threshold"])]
        for key, value in bounds.items():
            command.extend([f"--{key}", str(value)])
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (output / f"{name}_checks.txt").write_text(result.stdout + result.stderr)
        if name == "sample2_before_fix":
            assert result.returncode == 1
            assert "FAILED" in result.stdout and "test_proper_boundaries" in result.stdout
            assert "1 failed, 5 passed" in result.stdout
        else:
            assert result.returncode == 0, result.stdout + result.stderr
        print(f"{name}: pytest exit {result.returncode}", flush=True)
    # Match the real component boundary: training consumes the serialized CSV.
    clean1 = pd.read_csv(paths["sample1"])
    clean2 = pd.read_csv(paths["sample2_after_fix"])
    modeling = config["modeling"]
    trainval, test = train_test_split(
        clean1, test_size=modeling["test_size"],
        random_state=modeling["random_seed"],
        stratify=clean1[modeling["stratify_by"]],
    )
    X_train, X_val, y_train, y_val = split_train_validation(
        trainval, modeling["val_size"], modeling["random_seed"], modeling["stratify_by"],
    )
    assert set(trainval.index).isdisjoint(test.index)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(test.index)
    runs = []
    best_model = None
    for depth in [10, 50]:
        for estimators in [100, 200]:
            rf = dict(modeling["random_forest"], max_depth=depth,
                      n_estimators=estimators, random_state=modeling["random_seed"])
            model, _ = get_inference_pipeline(rf, modeling["max_tfidf_features"])
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
            metrics = {"max_depth": depth, "n_estimators": estimators,
                       "validation_mae": float(mean_absolute_error(y_val, prediction)),
                       "validation_r2": float(r2_score(y_val, prediction))}
            if not runs or metrics["validation_mae"] < min(r["validation_mae"] for r in runs):
                best_model = model
            runs.append(metrics)
            print(metrics, flush=True)
    best = min(runs, key=lambda item: item["validation_mae"])
    # Only the validation-selected model is evaluated on the untouched test set.
    X_test, y_test = test.drop(columns="price"), test["price"]
    prediction = best_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, prediction))
    gap = abs(test_mae - best["validation_mae"]) / best["validation_mae"]
    exported = output / "model_export"
    if exported.exists():
        import shutil
        shutil.rmtree(exported)
    save_pipeline(best_model, str(exported), X_train.iloc[:5])
    restored = mlflow.sklearn.load_model(exported)
    np.testing.assert_allclose(prediction, restored.predict(X_test))
    # Predictions must not depend on the composition of the inference batch.
    np.testing.assert_allclose(restored.predict(X_test.iloc[:1]),
                               restored.predict(X_test.iloc[:15])[:1])
    report = {"evidence_scope": "LOCAL ONLY; no W&B runs or GitHub releases are claimed",
              "sample1_raw_rows": len(raw1), "sample1_clean_rows": len(clean1),
              "sample2_raw_rows": len(raw2), "sample2_before_fix_rows": len(baseline2),
              "sample2_after_fix_rows": len(clean2), "sample2_removed_geo_rows": len(baseline2)-len(clean2),
              "train_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(test),
              "sweep": runs, "best_validation": best,
              "test_mae": test_mae, "test_r2": float(r2_score(y_test, prediction)),
              "relative_mae_gap": gap,
              "comparable": gap <= modeling["comparison_relative_tolerance"],
              "mlflow_reload_predictions_match": True,
              "inference_batch_invariance": True}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    config["modeling"]["random_forest"]["max_depth"] = best["max_depth"]
    config["modeling"]["random_forest"]["n_estimators"] = best["n_estimators"]
    (ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(report, indent=2), flush=True)
    assert report["comparable"], "Investigate the generalization gap."
    # Train sample2 with validation-selected settings to verify fixed-sample training.
    trainval2, _ = train_test_split(clean2, test_size=modeling["test_size"],
                                   random_state=modeling["random_seed"],
                                   stratify=clean2[modeling["stratify_by"]])
    X2, V2, y2, vy2 = split_train_validation(
        trainval2, modeling["val_size"], modeling["random_seed"], modeling["stratify_by"])
    model2, _ = get_inference_pipeline(
        dict(config["modeling"]["random_forest"], random_state=modeling["random_seed"]),
        modeling["max_tfidf_features"])
    model2.fit(X2, y2)
    report["sample2_validation_mae"] = float(mean_absolute_error(vy2, model2.predict(V2)))
    report["sample2_fixed_training_succeeded"] = True
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Fixed sample2 training succeeded", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
