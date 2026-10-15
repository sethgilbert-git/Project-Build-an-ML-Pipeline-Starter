# NYC Airbnb rental-price ML pipeline

GitHub repository: **https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter**

W\&B project: **https://wandb.ai/YOUR\_WANDB\_USERNAME/nyc\_airbnb**

This project estimates historical nightly listing prices using a reusable Hydra-configured MLflow pipeline. W\&B artifacts record the data and model lineage. Components download a bundled sample, clean prices, validate schema/distributions/geography, split trainval/test data, train a Random Forest pipeline, and evaluate the validation-selected production model.

## Run and submit

Follow [SETUP\_AND\_SUBMIT.md](SETUP_AND_SUBMIT.md) in order. This package contains the finished 1.0.1 source plus `scripts/prepare\_release.py`, which prepares the original 1.0.0 behavior before adding the geographic fix in 1.0.1. Publish the releases chronologically. Do not start by publishing the fixed version as 1.0.0.

```bash
conda env create -f environment.yml
conda activate nyc\_airbnb\_dev
export WANDB\_ENTITY=YOUR\_WANDB\_USERNAME\_OR\_TEAM
wandb login
bash scripts/run\_stage.sh first-data
# Manually add reference to clean\_sample.csv:latest in W\&B.
bash scripts/run\_stage.sh train
# Promote the validation-best model and persist its hyperparameters.
python scripts/select\_best.py --promote
bash scripts/run\_stage.sh test
```

The project uses a consistent Python 3.12 environment. Inner MLflow components use that active environment (`env\_manager: local`); the root MLproject also supports Conda. This replaces incompatible and inconsistent Python versions in the starter. Linux/WSL is recommended for the shell wrappers and remote release runs.

## Artifacts and lineage

|Artifact|Producer|Purpose|
|-|-|-|
|sample.csv|download\_file|Versioned raw sample|
|clean\_sample.csv|basic\_cleaning|Canonical cleaned dataset; manually set reference on sample1|
|clean\_data.csv|basic\_cleaning|Additional name required by one rubric section; identical cleaned records|
|trainval\_data.csv|train\_val\_test\_split|Dataset used for train/validation splitting|
|test\_data.csv|train\_val\_test\_split|Held-out test dataset|
|random\_forest\_export|train\_random\_forest|Artifact of type model\_export containing an MLflow sklearn model|
|test\_metrics|test\_model|Final test/validation comparison|

The model collection is named `random\_forest\_export` as instructed; its artifact type is `model\_export`. `reference` and `prod` are artifact **aliases**, which are what `name:alias` resolves. Every producing component uses `run.use\_artifact` for its inputs and `run.log\_artifact` for outputs, so the W\&B Lineage/Graph view includes cleaning, validation, splitting, training, and final evaluation.

## Local validation results

These are actual local measurements, not W\&B run evidence. The online stages and GitHub releases must still be executed in the student's accounts.

The first sample has 20,000 rows and 19,001 price-cleaned rows. There are 12,160 training, 3,040 validation, and 3,801 held-out test rows. Training-only preprocessing imputes and encodes data, creates a stable review-age feature, and extracts title TF-IDF features.

A four-job search compared max\_depth 10/50 and n\_estimators 100/200. The lowest validation MAE was 34.2461 dollars with depth 50 and 200 trees. Held-out test MAE was 33.8417 dollars and R2 was 0.5623. The absolute relative MAE gap was 1.18%, below the configured 15% comparison tolerance. This is consistent with similar validation/test generalization; it does not prove there is no overfitting or guarantee future-market performance.

The second sample contains one out-of-area point after price filtering. Before the geographic fix its boundary check fails, with the other five data checks passing. After the fix all six data checks pass and 46,427 rows remain. Local retraining on the corrected second sample succeeds with validation MAE 32.3596 dollars. The MLflow export reloads with matching predictions, including in a fresh process outside the repository.

Run `python -m pytest tests -v` for code tests, or `python scripts/local\_verify.py` for a full local four-model verification without W\&B uploads. Detailed results are in `local\_verification/metrics.json` and the recorded local check logs. The executed EDA notebook used the bundled first sample locally; rerun it in W\&B mode for submission evidence.

## Limitations and future releases

The model is limited to the configured $10–$350 price range and NYC area. Historical listing prices are not realized booking revenue, and seasonality, fees, amenities and current demand are not represented. Future releases could add temporal holdouts, richer text/amenity features, prediction intervals, drift alerts and comparisons with boosting models. Small geographic segments need separate evaluation. The test set must remain excluded from hyperparameter selection.

## Submission status

Code and local checks are complete. Public W\&B runs/artifacts, the manual reference alias, production alias, online held-out evaluation, lineage review, two GitHub releases, and remote released-pipeline runs remain pending until completed in the student's accounts. Replace this paragraph with verified public links and an accurate completion statement only after those steps succeed.

