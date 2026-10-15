#!/usr/bin/env bash
# Ordered online stages; stop between stages for the rubric's manual alias actions.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "First run: export WANDB_ENTITY=YOUR_WANDB_USERNAME_OR_TEAM"
  exit 2
fi
export WANDB_PROJECT=nyc_airbnb
if [[ "${WANDB_MODE:-online}" != "online" ]]; then
  echo "These submission stages require online W&B artifact storage."
  exit 2
fi
stage="${1:-}"
case "$stage" in
  first-data)
    python scripts/prepare_release.py 1.0.0
    mlflow run . --env-manager local -P steps=download,basic_cleaning
    echo "In W&B, manually add the reference alias to clean_sample.csv:latest."
    ;;
  train)
    mlflow run . --env-manager local -P steps=data_check,data_split
    mlflow run . --env-manager local -P steps=train_random_forest \
      -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
    python scripts/select_best.py
    echo "Manually add prod to the best model, or run python scripts/select_best.py --promote."
    ;;
  test)
    mlflow run . --env-manager local -P steps=test_regression_model
    ;;
  second-fail)
    repo="${2:?Provide your GitHub HTTPS repository URL as the second argument}"
    # Intentionally do not swallow the required first-release boundary failure.
    mlflow run "$repo" --env-manager local -v 1.0.0 \
      -P hydra_options="etl.sample=sample2.csv main.experiment_name=sample2_before_fix"
    ;;
  second-pass)
    repo="${2:?Provide your GitHub HTTPS repository URL as the second argument}"
    mlflow run "$repo" --env-manager local -v 1.0.1 \
      -P hydra_options="etl.sample=sample2.csv main.experiment_name=sample2_after_fix"
    ;;
  *)
    echo "Usage: bash scripts/run_stage.sh {first-data|train|test|second-fail|second-pass} [repo-url]"
    exit 2
    ;;
esac
