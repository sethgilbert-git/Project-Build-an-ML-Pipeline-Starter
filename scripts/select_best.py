"""Find the lowest validation MAE in the current W&B trainval cohort."""
import argparse
import json
import os
from pathlib import Path

import wandb
import yaml


def main():
    """Recommend a model and optionally promote it and persist its config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), required=False)
    parser.add_argument("--promote", action="store_true",
                        help="Set prod on the best model and update config.yaml.")
    args = parser.parse_args()
    if not args.entity:
        parser.error("Set WANDB_ENTITY or provide --entity with your W&B username/team.")
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    project = config["main"]["project_name"]
    api = wandb.Api()
    trainval = api.artifact(f"{args.entity}/{project}/{config['artifacts']['trainval']}")
    candidates = []
    for run in api.runs(f"{args.entity}/{project}"):
        if run.job_type != "train_random_forest" or run.state != "finished":
            continue
        if run.config.get("trainval_version") != trainval.name:
            continue
        score = run.summary.get("mae")
        if score is not None:
            candidates.append((float(score), run))
    if not candidates:
        raise RuntimeError("No finished training runs for the current trainval artifact.")
    score, best = min(candidates, key=lambda item: item[0])
    artifacts = [a for a in best.logged_artifacts() if a.type == "model_export"]
    if len(artifacts) != 1:
        raise RuntimeError("Expected exactly one model export from the best training run.")
    artifact = artifacts[0]
    print(json.dumps({"run_url": best.url, "run_id": best.id,
                      "artifact": artifact.name, "validation_mae": score,
                      "max_depth": best.config["max_depth"],
                      "n_estimators": best.config["n_estimators"]}, indent=2))
    if args.promote:
        artifact.aliases = sorted(set(artifact.aliases) | {"prod"})
        artifact.save()
        rf = config["modeling"]["random_forest"]
        for key in list(rf):
            if key in best.config:
                rf[key] = best.config[key]
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print("Updated prod alias and config.yaml using validation performance only.")


if __name__ == "__main__":
    main()
